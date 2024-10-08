import unittest
from qubayes.qubayes_tools import QBN, Graph, Node
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals
import numpy as np


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        self.n_shots = 4096

    def test_circuit_4_states(self):
        algorithm_globals.random_seed = 42
        probs = np.array([0.51, 0.11, 0.32, 0.06])
        nodes = {'A': Node('A', probs)}
        graph = Graph(nodes)
        graph.binarize()
        qbn = QBN(graph)
        qbn.add_measurements()
        simulator = AerSimulator(seed_simulator=42)
        qc = transpile(qbn.qc, simulator)
        job = simulator.run(qc, shots=self.n_shots)
        results = job.result().get_counts()
        measured = np.array([results['00'], results['10'],
                             results['01'], results['11']]) / self.n_shots
        np.testing.assert_allclose(probs, measured, atol=0.02)

    def test_circuit_cpd_4_states(self):
        algorithm_globals.random_seed = 42
        probs_a = np.array([0.21, 0.79])
        A = Node("A", probs_a,
                 states={"a0": 0, "a1": 1})
        probs_b = np.array([[0.1, 0.4],
                            [0.2, 0.3],
                            [0.3, 0.2],
                            [0.4, 0.1]])
        B = Node("B", probs_b,
                 states={"b0": 0, "b1": 1, "b2": 2, "b3": 3},
                 parents=['A'])

        nodes = {'A': A, 'B': B}
        graph = Graph(nodes)
        graph.binarize()
        qbn = QBN(graph)
        qbn.add_measurements()
        simulator = AerSimulator(seed_simulator=42)
        qc = transpile(qbn.qc, simulator)
        job = simulator.run(qc, shots=self.n_shots)
        results = job.result().get_counts()
        np.testing.assert_allclose(probs_a[0] * probs_b[0, 0], results['000'] / self.n_shots, atol=0.01)  # A=a0, B=b0
        np.testing.assert_allclose(probs_a[0] * probs_b[2, 0], results['010'] / self.n_shots, atol=0.01)  # A=a0, B=b2
        np.testing.assert_allclose(probs_a[1] * probs_b[3, 1], results['111'] / self.n_shots, atol=0.01)  # A=a1, B=b3
        np.testing.assert_allclose(probs_a[1] * probs_b[0, 1], results['001'] / self.n_shots, atol=0.01)  # A=a1, B=b0

    def test_circuit_cpd_4_states_with_4_state_prior(self):
        algorithm_globals.random_seed = 42
        probs_a = np.array([0.21, 0.39, 0.3, 0.1])
        A = Node("A", probs_a,
                 states={"a0": 0, "a1": 1, "a2": 2, "a3": 3})
        probs_b = np.array([[0.1, 0.4, 0.1, 0.6],
                            [0.2, 0.3, 0.1, 0.1],
                            [0.3, 0.2, 0.5, 0.2],
                            [0.4, 0.1, 0.3, 0.1]])
        B = Node("B", probs_b,
                 states={"b0": 0, "b1": 1, "b2": 2, "b3": 3},
                 parents=['A'])

        nodes = {'A': A, 'B': B}
        graph = Graph(nodes)
        graph.binarize()
        qbn = QBN(graph)
        qbn.add_measurements()
        simulator = AerSimulator(seed_simulator=42)
        qc = transpile(qbn.qc, simulator)
        job = simulator.run(qc, shots=self.n_shots)
        results = job.result().get_counts()
        np.testing.assert_allclose(probs_a[0] * probs_b[0, 0], results['0000'] / self.n_shots, atol=0.01)  # A=a0, B=b0
        np.testing.assert_allclose(probs_a[1] * probs_b[1, 1], results['1010'] / self.n_shots, atol=0.01)  # A=a1, B=b1
        np.testing.assert_allclose(probs_a[1] * probs_b[0, 1], results['0010'] / self.n_shots, atol=0.01)  # A=a1, B=b0
        np.testing.assert_allclose(probs_a[3] * probs_b[2, 3], results['0111'] / self.n_shots, atol=0.01)  # A=a3, B=b2

    def test_circuit_cpd_4_states_vchain(self):
        algorithm_globals.random_seed = 42
        probs_a = np.array([0.21, 0.79])
        A = Node("A", probs_a,
                 states={"a0": 0, "a1": 1})
        probs_b = np.array([[0.1, 0.4],
                            [0.2, 0.3],
                            [0.3, 0.2],
                            [0.4, 0.1]])
        B = Node("B", probs_b,
                 states={"b0": 0, "b1": 1, "b2": 2, "b3": 3},
                 parents=['A'])

        nodes = {'A': A, 'B': B}
        graph = Graph(nodes)
        graph.binarize()
        qbn = QBN(graph, use_ancillas=True)
        qbn.add_measurements()
        simulator = AerSimulator(seed_simulator=42)
        qc = transpile(qbn.qc, simulator)
        job = simulator.run(qc, shots=self.n_shots)
        results = job.result().get_counts()
        np.testing.assert_allclose(probs_a[0] * probs_b[0, 0], results['000'] / self.n_shots, atol=0.01)  # A=a0, B=b0
        np.testing.assert_allclose(probs_a[0] * probs_b[2, 0], results['010'] / self.n_shots, atol=0.01)  # A=a0, B=b2
        np.testing.assert_allclose(probs_a[1] * probs_b[3, 1], results['111'] / self.n_shots, atol=0.01)  # A=a1, B=b3
        np.testing.assert_allclose(probs_a[1] * probs_b[0, 1], results['001'] / self.n_shots, atol=0.01)  # A=a1, B=b0

    def test_circuit_cpd_4_states_with_4_state_prior_vchain(self):
            algorithm_globals.random_seed = 42
            probs_a = np.array([0.21, 0.39, 0.3, 0.1])
            A = Node("A", probs_a,
                     states={"a0": 0, "a1": 1, "a2": 2, "a3": 3})
            probs_b = np.array([[0.1, 0.4, 0.1, 0.6],
                                [0.2, 0.3, 0.1, 0.1],
                                [0.3, 0.2, 0.5, 0.2],
                                [0.4, 0.1, 0.3, 0.1]])
            B = Node("B", probs_b,
                     states={"b0": 0, "b1": 1, "b2": 2, "b3": 3},
                     parents=['A'])

            nodes = {'A': A, 'B': B}
            graph = Graph(nodes)
            graph.binarize()
            qbn = QBN(graph, use_ancillas=True)
            qbn.add_measurements()
            simulator = AerSimulator(seed_simulator=42)
            qc = transpile(qbn.qc, simulator)
            job = simulator.run(qc, shots=self.n_shots)
            results = job.result().get_counts()
            np.testing.assert_allclose(probs_a[0] * probs_b[0, 0], results['0000'] / self.n_shots, atol=0.01)  # A=a0, B=b0
            np.testing.assert_allclose(probs_a[1] * probs_b[1, 1], results['1010'] / self.n_shots, atol=0.01)  # A=a1, B=b1
            np.testing.assert_allclose(probs_a[1] * probs_b[0, 1], results['0010'] / self.n_shots, atol=0.01)  # A=a1, B=b0
            np.testing.assert_allclose(probs_a[3] * probs_b[2, 3], results['0111'] / self.n_shots, atol=0.01)  # A=a3, B=b2

if __name__ == '__main__':
    unittest.main()
