import unittest
from types import SimpleNamespace
import numpy as np

from src.calibration.class_thresholds import confidence_order, select_class_thresholds, tail_risk


class ClassThresholdTests(unittest.TestCase):
    def setUp(self):
        self.p = np.r_[np.linspace(.01, .49, 30), np.linspace(.51, .99, 30)]
        self.y = (self.p >= .5).astype(int)
        self.y[30:50] = 0

    def test_exact_budget_and_two_threshold_shape(self):
        human, state = select_class_thresholds(self.p, self.y, self.p, 30, min_support=2)
        self.assertEqual(human.sum(), 30)
        for c, ix in enumerate(confidence_order(self.p)):
            count = state[f"class_{c}_accepted"]
            self.assertGreaterEqual(count, 2)
            self.assertTrue((~human[ix[:count]]).all())
            self.assertTrue(human[ix[count:]].all())
        self.assertGreater(state["class_1_threshold"], state["class_0_threshold"])

    def test_objective_matches_independent_exhaustive_search(self):
        _, state = select_class_thresholds(self.p, self.y, self.p, 30, min_support=2)
        risks = []
        for k0 in range(2, 29):
            k1 = 30-k0
            thresholds = [np.sort(np.maximum(self.p[:30], 1-self.p[:30]))[-k0],
                          np.sort(self.p[30:])[-k1]]
            risk = [tail_risk(self.p, self.y, [thresholds[c]], c, 2)[0][0] for c in (0, 1)]
            risks.append(max(risk))
        self.assertAlmostEqual(state["estimated_worst_class_risk"], min(risks))

    def test_empty_calibration_class_is_flagged_unsupported(self):
        human, state = select_class_thresholds([.1]*10, [0]*10, self.p, 30)
        self.assertEqual(human.sum(), 30)
        self.assertFalse(state["both_tails_supported"])
        self.assertEqual(state["class_1_estimated_risk"], 1.)

    def test_tied_scores_keep_floor_and_reproduce_without_labels(self):
        p = [.1]*20+[.9]*20
        a, sa = select_class_thresholds(p, [0]*20+[1]*20, p, 20, seed=6)
        b, sb = select_class_thresholds(p, [0]*20+[1]*20, p, 20, seed=6)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(sa, sb)
        self.assertGreater(sa["class_0_accepted"], 0)
        self.assertGreater(sa["class_1_accepted"], 0)
        self.assertEqual(sum(sa[f"class_{c}_boundary_accepted"] for c in (0, 1)), 20)

    def test_infeasible_class_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            select_class_thresholds(self.p, self.y, self.p, 1)

    def test_graded_ndcg_uses_linear_gains_and_zero_unjudged_gain(self):
        from scripts.run_class_threshold_experiment import RankingMetrics
        ranking = object.__new__(RankingMetrics)
        discount = 1/np.log2(np.arange(2, 12))
        ranking.binary = SimpleNamespace(discount=discount, query_indices=[np.arange(3)],
            indices=np.asarray([[[0, 1]+[3]*8]]))
        actual = ranking.scores(np.asarray([3, 1, 2]))[0]
        # Retrieved grades are [3,1,unjudged,...]; ideal grades are [3,2,1].
        expected = (3+discount[1])/(3+2*discount[1]+discount[2])
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
