import unittest

import test
import numpy as np

class GetData(unittest.TestCase):
    def test_mean_pref_score(self):
        """
        """
        pref_ids = np.array([1, 0, 0, 1, 0])
        cov_scores = np.array([0.5, 0.1, 0.3, 0.2, 0.1])

        avg = test.mean_pref_score(cov_scores, pref_ids)

        # edge between home and first uav at [200, 300]
        self.assertEqual(avg, 0.35)

    def test_mean_reg_score(self):
        """
        """
        pref_ids = np.array([1, 0, 0, 1, 0])
        cov_scores = np.array([0.5, 0.2, 0.3, 0.2, 0.1])

        avg = test.mean_reg_score(cov_scores, pref_ids)

        self.assertAlmostEqual(avg, 0.2, 2)

    def test_iq_vals(self):
        ""
        l = [6, 1, 3, 2, 4, 5]

        iq_vals = test.get_interquartile_vals(l)

        self.assertEqual(iq_vals, [2, 3, 4, 5])



if __name__ == '__main__':
    unittest.main()
