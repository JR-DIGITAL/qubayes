# Implementation of Quantum Bayesian Networks including Quantum Rejection Sampling
#
# Copyright (C) 2024
# Florian Krebs, Joanneum Research Forschungsgesellschaft mbH
# Email: florian.krebs@joanneum.at
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pickle
import itertools
import os.path
from itertools import product
from copy import deepcopy
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator, MCMT, MCMTVChain, ZGate, RYGate
from qiskit_aer import AerSimulator
from math import ceil, log2, atan2, sqrt
from qubayes.config import MODEL_FLN
from qubayes.dataset_stats import MusicDataset


def run_circuit(circuit, draw_circuit=False, use_sim=True, shots=1024, verbose=False, seed=42):

    if use_sim:
        # simulator = Aer.get_backend('qasm_simulator')
        simulator = AerSimulator(seed_simulator=seed)
        new_circuit = transpile(circuit, simulator)
        if verbose:
            print('Transpiled circuit:')
            print(dict(new_circuit.count_ops()))
            print(f'depth: {new_circuit.depth()}')
        circuit_params = {'ops': dict(new_circuit.count_ops()),
                          'depth': new_circuit.depth()}
        job = simulator.run(new_circuit, shots=shots)

    if draw_circuit:
        new_circuit.draw(output='mpl')

    return job.result().get_counts(), circuit_params


def angle_from_probability(p0, p1):
    """ Equation 20 from https://arxiv.org/pdf/2004.14803.pdf
        p1 is P(X=1), p0 is P(X=0)"""
    angle = 2 * atan2(sqrt(p1), sqrt(p0))
    return angle


def grover_oracle_from_string(marked_states):
    """Build a Grover oracle for multiple marked states

    Here we assume all input marked states have the same number of bits

    Parameters:
        marked_states (str or list): Marked states of oracle

    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle
    """
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    # Compute the number of qubits in circuit
    num_qubits = len(marked_states[0])

    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if zero_inds:
            qc.x(zero_inds)
        qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
        if zero_inds:
            qc.x(zero_inds)
    # ancillas = qiskit.circuit.AncillaRegister(size=None(r, 'ancillas')
    # qc.add(ancillas)
    return qc


class QBN:

    def __init__(self, graph, qc=None, use_ancillas=False, verbose=False):
        self.graph = graph
        self.qc = qc
        self.bit_assignment = None
        self.use_ancillas = use_ancillas
        self.ancilla_qubits = []
        self.verbose = verbose

        if qc is None:
            self.create_circuit()
        self.n_qubits = self.qc.num_qubits
        self.n_ancillas = len(self.ancilla_qubits)

    def add_measurements(self):
        n_clbits = max(self.bit_assignment.values()) + 1
        c_reg = ClassicalRegister(n_clbits, 'c')
        self.qc.add_register(c_reg)
        for i in self.bit_assignment.values():
            self.qc.measure(i, i)
        return

    def get_qubits_from_attribute(self, attribute):
        qubits = []
        for name, qubit in self.bit_assignment.items():
            if name.split('.')[0] == attribute:
                qubits.append(qubit)
        return qubits

    def create_evidence_states(self, evidence):
        # evidence = dict('artists': 'The Beatles')
        import itertools
        n_state_qubits = self.n_qubits - self.n_ancillas
        lst = list(itertools.product([0, 1], repeat=n_state_qubits))
        ok = np.full(len(lst), True)
        for attr, cat in evidence.items():
            qubits = self.get_qubits_from_attribute(attr)
            bin_str = self.graph.bin_state_from_category(attr, cat)[::-1]  # bin_str is q0 q1 q2
            for i, state in enumerate(lst):
                for e, v in zip(qubits, bin_str):
                    if state[n_state_qubits-e-1] != int(v):
                        ok[i] = False
                        break
        new_list = []
        ancillas = '0' * self.n_ancillas
        for i in np.where(ok)[0]:
            new_list.append(ancillas + ''.join(map(str, lst[i])))
        return new_list

    def perform_sampling(self, shots=1024, seed=42):
        self.add_measurements()
        result, circuit_params = run_circuit(self.qc, shots=shots, seed=seed)
        print(sorted(result.items(), key=lambda item: item[1], reverse=True))
        return result

    def perform_rejection_sampling(self, evidence, iterations=1, shots=1024, verbose=False, seed=42):
        if iterations > 0:
            oracle = grover_oracle_from_string(evidence)
            grover_op = GroverOperator(oracle, state_preparation=self.qc)
            self.qc.compose(grover_op.power(iterations), inplace=True)
        self.add_measurements()

        result, circuit_params = run_circuit(self.qc, draw_circuit=False, shots=shots, seed=seed)
        if verbose:
            print(sorted(result.items(), key=lambda item: item[1], reverse=True))
        accepted = 0
        n_shots = 0
        evidence = [e[self.n_ancillas:] for e in evidence]
        for k, v in result.items():
            n_shots += v
            if k in evidence:
                accepted += v
        acc_rate = accepted / n_shots
        if verbose:
            print(f'Acceptance ratio: {100. * acc_rate:.1f} %')
        return result, circuit_params, acc_rate

    def create_circuit(self, n_ancilla_qubits=1):
        # Create a dictionary to hold which states are assigned to which qubits and counters to track the allocation
        bit_assignment = {}
        next_free_qbit = 0

        # TODO: Allocate ancilla_bits

        # Count number of qbits needed and the remaining bits are ancillas
        needed_qbits = 0
        n_ancilla_qubits = 0
        for name, node in self.graph.nodes.items():
            if self.use_ancillas:
                n_ancillas_i = max(0, node.n_parents() - 1)
                if n_ancillas_i > n_ancilla_qubits:
                    n_ancilla_qubits = n_ancillas_i
            needed_qbits += 1
        ancilla_qubits = list(range(needed_qbits, needed_qbits + n_ancilla_qubits))
        needed_qbits += n_ancilla_qubits
        if self.qc is None:
            qc = QuantumCircuit(needed_qbits)

        for name, node in self.graph.nodes.items():
            if node.name in bit_assignment.keys():
                continue
            n_parents = len(node.parents)
            if not node.has_parents():
                # loop and find the parentless/root nodes, assign their rotations first
                assert node.n_states <= 2
                # root node has 2 states, simple controlled rotate
                qc.ry(angle_from_probability(node.data[0], node.data[1]), next_free_qbit)
                # keep track of what node is what qbit
                bit_assignment[node.name] = next_free_qbit
                next_free_qbit += 1
            else:
                bit_assignment, next_free_qbit, target_qbits = assign_bits(node, bit_assignment, next_free_qbit)
                # get all combinations of parent states, e.g.  (0,0), (0,1), (1,0), (1,1)
                parent_state_total = get_parents_states(node)
                parent_qubits = []
                for p in node.parents:
                    parent_qubits.append(bit_assignment[p])
                qc.barrier()
                # Now encode each column in the probability table into the circuit
                for i, parent_state_combo in enumerate(parent_state_total):
                    # for each parent state, add rotation:
                    # 1) add an x for each qubit of the parent state which is 0
                    # 2) add the conditioned rotation
                    # 3) add an x for each qubit of the parent state which is 0

                    assert len(parent_state_combo) == n_parents == len(parent_qubits)
                    # Flip target bit-string to match Qiskit bit-ordering
                    # rev_target = target[::-1]
                    zero_inds = [parent_qubits[ind] for ind in range(n_parents) if parent_state_combo[ind] == 0]
                    if zero_inds:
                        qc.x(zero_inds)
                    # get probability conditioned on parent_state_combo
                    s0 = parent_state_combo + tuple([0])
                    s1 = parent_state_combo + tuple([1])
                    prob = angle_from_probability(node.data[s0], node.data[s1])
                    if self.use_ancillas:
                        # MCMTVChain requires ancillas but is decomposed into a much shallower circuit
                        #   than the default implementation (MCMT)
                        mcgate = MCMTVChain(RYGate(prob), len(parent_qubits), 1)
                        if mcgate.num_ancilla_qubits > 0:
                            ancilla = ancilla_qubits[:mcgate.num_ancilla_qubits]
                            qc.compose(mcgate, qubits=parent_qubits + target_qbits + ancilla, inplace=True)
                        else:
                            qc.compose(mcgate, qubits=parent_qubits + target_qbits, inplace=True)
                    else:
                        qc.compose(MCMT(RYGate(prob), len(parent_qubits), 1),
                                   qubits=parent_qubits + target_qbits, inplace=True)
                    if zero_inds:
                        qc.x(zero_inds)
                    qc.barrier()

        # Note: qiskit has reverse qubit indices
        if self.verbose:
            for k, v in bit_assignment.items():
                print(f'-Qubit {v}: {k[:2]}')
        self.qc = qc
        self.bit_assignment = bit_assignment
        self.ancilla_qubits = ancilla_qubits


def assign_bits(node, bit_assignment, next_free_qbit):
    if len(node.states) <= 2:
        needed_qbits = 1
        bit_assignment[node.name] = next_free_qbit
        target_qbits = [next_free_qbit]
        next_free_qbit += 1
    else:
        # node has 2+ states
        needed_qbits = ceil(log2(len(node.states)))
        sub_start_qbit = next_free_qbit
        target_qbits = []
        # bit allocation time
        for i in range(needed_qbits):
            bit_assignment[node.name + '_q' + str(i)] = next_free_qbit
            target_qbits.append(next_free_qbit)
            next_free_qbit += 1
    return bit_assignment, next_free_qbit, target_qbits


def get_parents_states(node):
    parent_state_enumeration = [[0, 1] for i in range(len(node.parents))]
    return itertools.product(*parent_state_enumeration)


class Node:
    # A single variable in the Bayesian network
    def __init__(self, name, data=None, states=None, parents=[]):
        """
        # name:    str    name of variable
        # data:    array  state data for the node
        # states:  dict   keys are state names, values are the int each takes on in the data
        # parents: list   strings of names of parent nodes to this node
        """
        
        if (states is None) & (data is not None):
            states = {}
            for i in range(data.shape[0]):
                states.update({name.lower() + str(i): i})
        
        self.name = name
        self.data = data
        self.states = states
        self.parents = parents
        if states is None:
            self.n_states = None
        else:
            self.n_states = len(states)
        # TODO: check if probability is valid

    def has_parents(self):
        return len(self.parents) > 0

    def n_parents(self):
        return len(self.parents)


class Graph:
    # A dictionary of nodes
    def __init__(self, nodes=None, verbose=False):
        self.nodes = nodes
        self.categories = None
        self.verbose = verbose
        if nodes is not None:
            self.set_categories_from_nodes()
        self.original_graph = None

    def set_categories_from_nodes(self):
        self.categories = dict()
        for name, node in self.nodes.items():
            self.categories[name] = node.states
        return

    def set_probabilities(self, ds):
        new_nodes = dict()
        for name, node in self.nodes.items():
            assert name in ds.bins.keys()
            counts = ds.data[name].value_counts().to_dict()

            if node.has_parents():
                n_states = len(ds.bin_names[name])
                shape = tuple([n_states])
                parent_state_enumeration = []
                for p_name in node.parents:
                    n_p_states = len(ds.bin_names[p_name])
                    shape += tuple([n_p_states])
                    parent_state_enumeration.append(list(range(n_p_states)))
                probs = np.zeros(shape)
                parent_state_enumeration = product(*parent_state_enumeration)
                for p_state in parent_state_enumeration:
                    idx = np.full(ds.data.shape[0], True)
                    for i, p_name in enumerate(node.parents):
                        idx = idx & np.array(ds.data[p_name] == ds.bin_names[p_name][p_state[i]])
                    counts = ds.data[name][idx].value_counts().to_dict()
                    n_total = sum(counts.values())
                    if n_total == 0:
                        print(f'{name}: No entries for the given parent configuration. Assigning uniform prior.')
                        for i, bin in enumerate(ds.bin_names[name]):
                            probs[tuple([i]) + p_state] = 1. / float(len(ds.bin_names[name]))
                    else:
                        for i, bin in enumerate(ds.bin_names[name]):
                            if bin in counts.keys():
                                probs[tuple([i]) + p_state] = float(counts[bin]) / float(n_total)
            else:
                probs = np.array(list(counts.values())).astype(float)
                probs /= probs.sum()
            if self.verbose:
                print(name)
                print(ds.bin_names[name])
                print(probs)
            # name, data=None, states=None, parents=
            states = dict()
            for i, bin in enumerate(ds.bin_names[name]):
                states[bin] = i
            new_nodes[name] = Node(name, data=probs, states=states,
                                   parents=node.parents)
        self.nodes = new_nodes
        self.set_categories_from_nodes()

    def save_to_file(self, fln):
        with open(fln, 'wb') as f:
            pickle.dump(self.nodes, f)

    def load_from_file(self, fln):
        with open(fln, 'rb') as f:
            self.nodes = pickle.load(f)
        self.set_categories_from_nodes()

    def bin_state_from_category(self, attribute, category):
        if category in self.categories[attribute].keys():
            n_digits = int(np.ceil(np.log2(len(self.categories[attribute]))))
            bin_str = bin(self.categories[attribute][category])[2:].zfill(n_digits)
            return bin_str[::-1]  # return reversed string to match qiskits requirements: q2 q1 q0
        else:
            raise ValueError(f'Unknown category {category} in {attribute}')

    def binarize(self):
        # store the original graph, as binarization deletes attributes
        self.original_graph = self.nodes
        new_nodes = dict()
        for name, node in self.nodes.items():
            if node.has_parents():
                # check if parents already in new_nodes
                new_parents = []
                for i in new_nodes:
                    for j in node.parents:
                        if j in i:
                            new_parents.append(i)
                if new_parents:
                    node.parents = new_parents

            if node.n_states > 2:
                n_sub_nodes = int(np.ceil(np.log2(node.n_states)))
                for i in range(n_sub_nodes):
                    n_substates = int(2 ** (n_sub_nodes - i - 1))
                    prob = np.zeros(
                        [2] * (i + node.n_parents() + 1))  # last dimension is the child, the other one are the parents
                    parents = []
                    for j in range(i):
                        parents.append(node.name + '.' + str(j))
                    for p_ext in range(max([1, 2 * node.n_parents()])):  # run at least once for non-parent states
                        bin_state_ext = [int(d) for d in str(bin(p_ext))[2:].zfill(node.n_parents())]
                        if i > 0:  # all nodes except the first are conditioned on the previous
                            n_parent_states_int = 2 ** i
                            sum_temp = []
                            last_sum_temp = []
                            if node.has_parents():
                                for j in range(int(2 ** n_sub_nodes / n_substates)):
                                    sum_temp.append(node.data[j * n_substates:(j + 1) * n_substates, p_ext].sum())
                                for j in range(int(2 ** n_sub_nodes / (2 * n_substates))):
                                    last_sum_temp.append(
                                        node.data[j * 2 * n_substates:(j + 1) * 2 * n_substates, p_ext].sum())
                            else:
                                for j in range(int(2 ** n_sub_nodes / n_substates)):
                                    sum_temp.append(node.data[j * n_substates:(j + 1) * n_substates].sum())
                                for j in range(int(2 ** n_sub_nodes / (2 * n_substates))):
                                    last_sum_temp.append(node.data[j * 2 * n_substates:(j + 1) * 2 * n_substates].sum())

                            for p in range(n_parent_states_int):
                                bin_state = [int(d) for d in str(bin(p))[2:].zfill(i)]
                                if node.has_parents():
                                    state_tuple_0 = tuple([0]) + tuple(bin_state) + tuple(bin_state_ext)
                                    if last_sum_temp[p] > 0:
                                        prob[state_tuple_0] = sum_temp[p * 2] / last_sum_temp[p]
                                        state_tuple_1 = tuple([1]) + tuple(bin_state) + tuple(bin_state_ext)
                                        prob[state_tuple_1] = 1 - prob[state_tuple_0]
                                else:
                                    state_tuple_0 = tuple([0]) + tuple(bin_state)
                                    if last_sum_temp[p] > 0:
                                        prob[state_tuple_0] = sum_temp[p * 2] / last_sum_temp[p]
                                        state_tuple_1 = tuple([1]) + tuple(bin_state)
                                        prob[state_tuple_1] = 1 - prob[state_tuple_0]
                        else:
                            if node.has_parents():
                                base_prob = node.data[0:n_substates, p_ext].sum()
                                prob[tuple([0]) + tuple(bin_state_ext)] = base_prob
                                prob[tuple([1]) + tuple(bin_state_ext)] = 1 - base_prob
                            else:
                                base_prob = node.data[0:n_substates].sum()
                                prob = np.array([base_prob, 1 - base_prob])
                    if node.has_parents():
                        parents += node.parents
                    states = {node.name + '.' + str(i) + '_0': 0,
                              node.name + '.' + str(i) + '_1': 1}
                    new_node = Node(node.name + '.' + str(i), prob,
                                    parents=parents, states=states)
                    new_nodes[new_node.name] = new_node
            else:
                if node.has_parents():
                    prob = np.zeros([2] * (node.n_parents() + 1))
                    for p_ext in range(max([1, 2 * node.n_parents()])):
                        bin_state_ext = [int(d) for d in str(bin(p_ext))[2:].zfill(node.n_parents())]
                        state_tuple_0 = tuple(bin_state_ext) + tuple([0])
                        # TODO: node.data can be of ndim > 2, if there was more
                        #  than one parent in the original (non-binarized) graph
                        prob[state_tuple_0] = node.data[0, p_ext]
                        state_tuple_1 = tuple(bin_state_ext) + tuple([1])
                        prob[state_tuple_1] = 1 - prob[state_tuple_0]
                    new_node = Node(node.name, prob, parents=node.parents,
                                    states=node.states)
                    new_nodes[new_node.name] = new_node
                else:
                    new_nodes[node.name] = node
        self.nodes = new_nodes

    def sample_from_graph(self, n_samples):
        samples = np.zeros((len(self.nodes), n_samples), dtype=int)
        names = list(self.nodes.keys())
        for c, (name, node) in enumerate(self.nodes.items()):
            n_bins = node.data.shape[0]
            if node.has_parents():
                for i in range(n_samples):
                    prob = deepcopy(node.data)
                    for p in node.parents:
                        prob = prob[:, samples[names.index(p), i]]
                    samples[c, i] = np.random.choice(n_bins, p=prob)
            else:
                samples[c, :] = np.random.choice(n_bins, size=n_samples, p=node.data, replace=True)
        return samples, names


class Query:

    def __init__(self):
        self.qbn = None
        self.graph_orig = None
        self.target = None
        self.evidence = None
        self.use_ancillas = False
        self.qbn_collapsed = False
        self.verbose = False

    def create_model(self, n_artists=4, n_genres=4, n_tempi=2, n_modes=2, n_time_signature=2, use_ancillas=False, model_fln=None):
        self.use_ancillas = use_ancillas
        nodes = {'artists': Node('artists'),
                 'track_genre': Node('track_genre', parents=list(['artists'])),
                 'tempo': Node('tempo', parents=list(['track_genre'])),
                 'mode': Node('mode', parents=list(['track_genre'])),
                 'time_signature': Node('time_signature', parents=list(['track_genre']))}
        graph = Graph(nodes, verbose=self.verbose)
        if model_fln is not None and os.path.exists(MODEL_FLN):
            graph.load_from_file(MODEL_FLN)
        else:
            bins = {'track_genre': n_genres, 'artists': n_artists, 'tempo': n_tempi,
                    'mode': n_modes, 'time_signature': n_time_signature}
            ds = MusicDataset(bins, verbose=self.verbose)
            graph.set_probabilities(ds)
            graph.save_to_file(MODEL_FLN)
        self.graph_orig = deepcopy(graph)
        graph.binarize()
        self.qbn = QBN(graph, use_ancillas=self.use_ancillas)

    def rebuild_qbn(self):
        graph = deepcopy(self.graph_orig)
        graph.binarize()
        self.qbn = QBN(graph, use_ancillas=self.use_ancillas)

    def load_graph(self, model_fln, use_ancillas=True):
        graph = Graph()
        graph.load_from_file(model_fln)
        self.graph_orig = deepcopy(graph)
        graph.binarize()
        self.qbn = QBN(graph, use_ancillas=use_ancillas)

    def set_bit_string(self, attr, cond_str):
        qubits = self.qbn.get_qubits_from_attribute(attr)
        if attr in self.evidence.keys():
            state_str = self.qbn.graph.bin_state_from_category(attr, self.evidence[attr])
        else:
            state_str = self.qbn.graph.bin_state_from_category(attr, self.target[attr])
        for i in range(len(qubits)):
            cond_str[qubits[i]] = state_str[len(qubits) - i - 1]  # q0 is first
        return cond_str

    def predict_from_samples(self, samples):
        # find out which qubits have fixed values (0 or 1) and which ones have to be marginalized (-1)
        cond_str = np.ones((self.qbn.n_qubits - self.qbn.n_ancillas,), dtype=int) * (-1)
        for attr in self.target.keys():
            cond_str = self.set_bit_string(attr, cond_str)
        for attr in self.evidence.keys():
            cond_str = self.set_bit_string(attr, cond_str)
        cond_str = cond_str[::-1]
        mask = cond_str >= 0

        # nominator: count samples with matching target AND evidence
        nom = 0.
        for k, v in samples.items():
            sample_bits = np.array(list(map(int, k)))
            if np.array_equal(cond_str[mask], sample_bits[mask]):
                nom += v
        if nom == 0:
            return 0.

        # denominator: count samples with matching evidence
        cond_str = np.ones((self.qbn.n_qubits - self.qbn.n_ancillas,), dtype=int) * (-1)
        for attr in self.evidence.keys():
            cond_str = self.set_bit_string(attr, cond_str)
        cond_str = cond_str[::-1]
        mask = cond_str >= 0

        denom = 0.
        for k, v in samples.items():
            sample_bits = np.array(list(map(int, k)))
            if np.array_equal(cond_str[mask], sample_bits[mask]):
                denom += v
        return nom / denom

    def perform_rejection_sampling(self, shots=1024, iterations=1, verbose=False, seed=42):
        if self.qbn_collapsed:
            self.rebuild_qbn()
        evidence = self.qbn.create_evidence_states(self.evidence)
        result, circuit_params, acc_rate = self.qbn.perform_rejection_sampling(
            evidence, iterations=iterations, shots=shots, verbose=verbose, seed=seed)
        self.qbn_collapsed = True
        return self.predict_from_samples(result), acc_rate

    def perform_classical_rejection_sampling(self, shots=1024):
        # Go through graph hierarchy and create samples
        samples, names = self.graph_orig.sample_from_graph(shots)
        # P (A | B) = P(A, B) / P(B)
        # Evidence
        ok = np.full(shots, True)
        for key, val in self.evidence.items():
            node_idx = names.index(key)
            val_idx = self.graph_orig.categories[key][val]
            ok = ok & (samples[node_idx, :] == val_idx)
        denom = sum(ok)

        # Joint
        for key, val in self.target.items():
            node_idx = names.index(key)
            val_idx = self.graph_orig.categories[key][val]
            ok = ok & (samples[node_idx, :] == val_idx)
        nom = sum(ok)
        acc_rate = denom / shots
        return float(nom) / float(denom), acc_rate

    def perform_likelihood_weighted_sampling(self, shots=1024):
        pass

    def get_prior_probability(self, var):

        if var in self.evidence.keys():
            idx = self.graph_orig.nodes[var].states[self.evidence[var]]
        elif var in self.target.keys():
            idx = self.graph_orig.nodes[var].states[self.target[var]]
        else:
            return deepcopy(self.graph_orig.nodes[var].data), None
        return deepcopy(self.graph_orig.nodes[var].data[idx]), idx

    def get_cond_probability(self, var, idx_parents):
        if idx_parents is None:
            idx_parents = np.arange(self.graph_orig.nodes[var].data.shape[1])
        if var in self.evidence.keys():
            idx = self.graph_orig.nodes[var].states[self.evidence[var]]
        elif var in self.target.keys():
            idx = self.graph_orig.nodes[var].states[self.target[var]]
        else:
            return deepcopy(self.graph_orig.nodes[var].data[:, idx_parents]), None
        return deepcopy(self.graph_orig.nodes[var].data[idx, idx_parents]), idx

    def get_true_result(self):
        # This function computes the conditional probability of the corresponding query.
        # Note that this computation depends on the structure of the given Bayesian network,
        # and is therefore only valid for the paper by Krebs et al., 2024.
        # Uses P(A | B, C) = P(A, B, C) / P(B, C)

        # Nominator: compute P(targets, evidence)
        # artist
        nom, idxa = self.get_prior_probability('artists')
        # track_genre
        p, idxg = self.get_cond_probability('track_genre', idxa)
        nom *= p
        if idxa is None:
            nom = nom.sum(axis=0)
        # tempo
        p, idxt = self.get_cond_probability('tempo', idxg)
        nom = p * nom
        if idxt is None:
            nom = nom.sum(axis=0)
        # time signature
        p, idxt = self.get_cond_probability('time_signature', idxg)
        nom = p * nom
        if idxt is None:
            nom = nom.sum(axis=0)
        # mode
        p, idxm = self.get_cond_probability('mode', idxg)
        nom = p * nom
        if idxm is None:
            nom = nom.sum(axis=0)

        # denominator: compute P(evidence)
        denom = 1.
        # artist
        if 'artists' in self.evidence.keys():
            p, idxa = self.get_prior_probability('artists')
            denom *= p
        else:
            denom *= self.graph_orig.nodes['artists'].data
            idxa = None

        # track_genre
        var = 'track_genre'
        if var in self.evidence.keys():
            p, idxg = self.get_cond_probability(var, idxa)
            denom *= p
        else:
            idxg = None
            if idxa is None:
                denom = self.graph_orig.nodes[var].data * denom
            else:
                denom = self.graph_orig.nodes[var].data[:, idxa] * denom
        if idxa is None:
            if idxg is None:
                denom = denom.sum(axis=1)
            else:
                denom = denom.sum()

        # tempo
        var = 'tempo'
        if var in self.evidence.keys():
            p, idxt = self.get_cond_probability(var, idxg)
            denom *= p

        # time_signature
        var = 'time_signature'
        if var in self.evidence.keys():
            p, idxt = self.get_cond_probability(var, idxg)
            denom *= p

        # mode
        var = 'mode'
        if var in self.evidence.keys():
            p, idxm = self.get_cond_probability(var, idxg)
            denom *= p

        return nom.sum() / denom.sum()