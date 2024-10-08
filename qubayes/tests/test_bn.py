import unittest
from qubayes.qubayes_tools import Node, Graph
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_binarize_4_states(self):
        probs = np.array([0.51, 0.11, 0.32, 0.06])
        AR = Node("Artist", probs,
                  states={"The Beatles": 0, "The Rolling Stones": 1,
                          "The Beach Boys": 2, "The Who": 3})
        nodes = {'Artist': AR}
        graph = Graph(nodes)
        graph.binarize()
        names = list(graph.nodes.keys())
        self.assertAlmostEqual(probs[0],
                               graph.nodes[names[0]].data[0] *
                               graph.nodes[names[1]].data[0, 0])
        self.assertAlmostEqual(probs[1],
                               graph.nodes[names[0]].data[0] *
                               graph.nodes[names[1]].data[0, 1])
        self.assertAlmostEqual(probs[2],
                               graph.nodes[names[0]].data[1] *
                               graph.nodes[names[1]].data[1, 0])
        self.assertAlmostEqual(probs[3],
                               graph.nodes[names[0]].data[1] *
                               graph.nodes[names[1]].data[1, 1])

    def test_binarize_6_states(self):
        probs = np.array([0.21, 0.11, 0.32, 0.06, 0.16, 0.14])
        AR = Node("Artist", probs,
                  states={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5})
        nodes = {'Artist': AR}
        graph = Graph(nodes)
        graph.binarize()
        names = list(graph.nodes.keys())
        self.assertAlmostEqual(probs[0],
                               graph.nodes[names[0]].data[0] *
                               graph.nodes[names[1]].data[0, 0] *
                               graph.nodes[names[2]].data[0, 0, 0])
        self.assertAlmostEqual(probs[1],
                               graph.nodes[names[0]].data[0] *
                               graph.nodes[names[1]].data[0, 0] *
                               graph.nodes[names[2]].data[0, 0, 1])
        self.assertAlmostEqual(probs[4],
                               graph.nodes[names[0]].data[1] *
                               graph.nodes[names[1]].data[1, 0] *
                               graph.nodes[names[2]].data[1, 0, 0])
        self.assertAlmostEqual(probs[5],
                               graph.nodes[names[0]].data[1] *
                               graph.nodes[names[1]].data[1, 0] *
                               graph.nodes[names[2]].data[1, 0, 1])

    def test_binarize_cpd_4_states(self):
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
        names = list(graph.nodes.keys())
        self.assertAlmostEqual(probs_b[0, 0],
                               graph.nodes[names[1]].data[0, 0] *
                               graph.nodes[names[2]].data[0, 0, 0])
        self.assertAlmostEqual(probs_b[0, 1],
                               graph.nodes[names[1]].data[1, 0] *
                               graph.nodes[names[2]].data[0, 1, 0])
        self.assertAlmostEqual(probs_b[2, 1],
                               graph.nodes[names[1]].data[1, 1] *
                               graph.nodes[names[2]].data[1, 1, 0])

    def test_binarize_cpd_2_states_with_4_state_prior(self):
        probs_a = np.array([0.21, 0.39, 0.3, 0.1])
        A = Node("A", probs_a,
                 states={"a0": 0, "a1": 1, "a2": 2, "a3": 3})
        probs_b = np.array([[0.1, 0.3, 0.2, 0.6],
                            [0.9, 0.7, 0.8, 0.4]])
        B = Node("B", probs_b,
                 states={"b0": 0, "b1": 1},
                 parents=['A'])

        nodes = {'A': A, 'B': B}
        graph = Graph(nodes)
        graph.binarize()
        self.assertAlmostEqual(probs_b[0, 0], graph.nodes['B'].data[0, 0, 0])
        self.assertAlmostEqual(probs_b[1, 1], graph.nodes['B'].data[0, 1, 1])
        self.assertAlmostEqual(probs_b[0, 3], graph.nodes['B'].data[1, 1, 0])

    def test_binarize_cpd_4_states_with_4_state_prior(self):
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
        self.assertAlmostEqual(probs_a[2],
                               graph.nodes['A.0'].data[1] *
                               graph.nodes['A.1'].data[1, 0])
        self.assertAlmostEqual(probs_b[0, 0],
                               graph.nodes['B.0'].data[0, 0, 0] *
                               graph.nodes['B.1'].data[0, 0, 0, 0])
        self.assertAlmostEqual(probs_b[0, 2],  # a state is 2 = '10'
                               graph.nodes['B.0'].data[1, 0, 0] *
                               graph.nodes['B.1'].data[0, 1, 0, 0])
        self.assertAlmostEqual(probs_b[2, 1],  # a state is 1 = '01'
                               graph.nodes['B.0'].data[0, 1, 1] *
                               graph.nodes['B.1'].data[1, 0, 1, 0])
        self.assertAlmostEqual(probs_b[2, 3],  # a state is 3 = '11'
                               graph.nodes['B.0'].data[1, 1, 1] *
                               graph.nodes['B.1'].data[1, 1, 1, 0])


if __name__ == '__main__':
    unittest.main()
