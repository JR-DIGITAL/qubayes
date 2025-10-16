import unittest
from qubayes.qubayes_tools import counts_to_cpd
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_counts_to_cpd(self):
        counts = {'111': 137, '101': 136, '010': 118}
        cpd = counts_to_cpd(counts)
        self.assertAlmostEqual(cpd[0, 1, 0], 0.30179028)
        self.assertAlmostEqual(cpd[1, 0, 1], 0.34782608)
        self.assertAlmostEqual(cpd[1, 1, 0], 0)

        counts = {'110': 100, '000': 100}
        cpd = counts_to_cpd(counts)
        self.assertAlmostEqual(cpd[0, 0, 0], 0.5)
        self.assertAlmostEqual(cpd[1, 1, 0], 0)
        cpd = counts_to_cpd(counts, reverse=False)
        self.assertAlmostEqual(cpd[0, 0, 0], 0.5)
        self.assertAlmostEqual(cpd[1, 1, 0], 0.5)

        counts = {'11': 95, '01': 5}
        cpd = counts_to_cpd(counts)
        np.testing.assert_array_equal(cpd, np.array([[0.  , 0.  ],
                                                     [0.05, 0.95]]))
        cpd = counts_to_cpd(counts, reverse=False)
        np.testing.assert_array_equal(cpd, np.array([[0., 0.05],
                                                     [0., 0.95]]))






if __name__ == '__main__':
    unittest.main()
