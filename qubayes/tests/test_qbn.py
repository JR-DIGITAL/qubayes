import unittest
from qubayes.qubayes_tools import Node, Graph, QBN
import numpy as np


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        self.n_shots = 4096
        self.probs_a = np.array([0.21, 0.39, 0.3, 0.1])
        A = Node("A", self.probs_a,
                 states={"a0": 0, "a1": 1, "a2": 2, "a3": 3})
        self.probs_b = np.array([[0.1, 0.4, 0.1, 0.6],
                            [0.2, 0.3, 0.1, 0.1],
                            [0.3, 0.2, 0.5, 0.2],
                            [0.4, 0.1, 0.3, 0.1]])
        B = Node("B", self.probs_b,
                 states={"b0": 0, "b1": 1, "b2": 2, "b3": 3},
                 parents=['A'])

        self.nodes = {'A': A, 'B': B}

    def test_evidence(self):

        graph = Graph(self.nodes)
        graph.binarize()
        qbn = QBN(graph, use_ancillas=True)
        evidence = {'A': 'a1'}
        evidence = qbn.create_evidence_states(evidence)
        self.assertEqual(evidence, ['000010', '000110', '001010', '001110'])
        evidence = {'B': 'b2'}
        evidence = qbn.create_evidence_states(evidence)
        self.assertEqual(evidence, ['000100', '000101', '000110', '000111'])

    def test_rej_sampling(self):
        graph = Graph(self.nodes)
        graph.binarize()
        qbn = QBN(graph, use_ancillas=False)
        evidence = {'A': 'a1'}
        evidence = qbn.create_evidence_states(evidence)
        results, circuit_params, acc_rate = qbn.perform_rejection_sampling(evidence, iterations=1, shots=self.n_shots)
        np.testing.assert_allclose(results['0010'], 1302, atol=3)
        np.testing.assert_allclose(results['1010'], 1011, atol=3)
        np.testing.assert_allclose(results['1111'], 8, atol=3)

    def test_rej_sampling_ancilla(self):
        graph = Graph(self.nodes)
        graph.binarize()
        qbn = QBN(graph, use_ancillas=True)
        evidence = {'A': 'a1'}
        evidence = qbn.create_evidence_states(evidence)
        results, circuit_params, acc_rate = qbn.perform_rejection_sampling(evidence, iterations=1, shots=self.n_shots)
        np.testing.assert_allclose(results['0010'], 1302, atol=3)
        np.testing.assert_allclose(results['1010'], 1011, atol=3)
        np.testing.assert_allclose(results['1111'], 8, atol=3)


if __name__ == '__main__':
    unittest.main()
