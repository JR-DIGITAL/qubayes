"""
This script contains code for generative models.
"""
__author__ = "Florian Krebs"
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import itertools


def counts_to_cpd(counts, reverse=True):
    # counts come from result.get_counts() and are: {'111': 137, '101': 136, '010': 118} S, R, C
    n_dim = len(list(counts.keys())[0])
    n_samples = sum(counts.values())
    lst = list(itertools.product([0, 1], repeat=n_dim))
    cpd = np.zeros((2,) * n_dim, dtype=float) # cpd is C, R, S
    states = np.array(list(counts.keys()))
    for c in lst:
        if reverse:  #
            idx = c[::-1]  #
        else:
            idx = c
        key = "".join([str(x) for x in idx])
        if key in list(counts.keys()):
            cpd[c] = float(counts[key]) / n_samples
    return cpd


class BornMachine(object):

    def __init__(self, n_qubits, n_blocks=0, ansatz_type='RealAmplitudes'):
        # n_blocks is L in the paper
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.params = None
        self.ansatz = None
        self.q_bm = None
        self.ansatz_type = ansatz_type
        self.reset_ansatz()

    def reset_ansatz(self):
        if self.ansatz_type == 'RealAmplitudes':
            self.ansatz = RealAmplitudes(self.n_qubits, reps=self.n_blocks,
                                         entanglement='linear')
        else:
            self.ansatz = EfficientSU2(self.n_qubits, su2_gates=['rz', 'rx'],
                                       reps=self.n_blocks,
                                       entanglement='linear')
        if self.params is None:
            self.params = np.random.normal(0, 0.1, size=self.ansatz.num_parameters)
        param_dict = {param: value for param, value in zip(self.ansatz.parameters, self.params)}
        self.ansatz.assign_parameters(param_dict, inplace=True)

    def print_circuit(self):
        print(self.ansatz.decompose())

    def sample(self, n_samples, return_samples=True):
        # Create a quantum circuit
        circuit = QuantumCircuit(self.n_qubits)
        # Apply a Hadamard gate to each qubit for state preparation
        for qubit in range(self.n_qubits):
            circuit.h(qubit)
        self.reset_ansatz()
        circuit.compose(self.ansatz, inplace=True)
        circuit.measure_all()

        # Simulate the circuit
        simulator = AerSimulator(method='matrix_product_state')
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=n_samples, memory=True).result()
        counts = result.get_counts()
        self.q_bm = counts_to_cpd(counts, reverse=True)
        if return_samples:
            samples = result.get_memory()
            out = np.array([[char == '1' for char in string[::-1]] for string in samples], dtype='int32')
            # check
            unique_rows, counts = np.unique(out, axis=0, return_counts=True)
            if len(counts) > 1:
                np.testing.assert_almost_equal(counts[1]/n_samples, self.q_bm[tuple(unique_rows[1])], decimal=3)
        else:
            out = self.q_bm
        return out


if __name__ == "__main__":
    main()
