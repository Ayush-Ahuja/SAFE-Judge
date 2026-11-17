"""Regression for losing confidence ordering at the original hard floor."""
import unittest
import numpy as np
from src.baselines.active_learning import LabelOracle
from src.calibration.safe_judge import SafeJudgeConfig, run_safe_judge
from src.calibration.precision_review import run_precision_review
from src.utils.io import pair_key


class PrecisionReviewTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"query_id": "q", "doc_id": str(i), "prob_relevant": p}
                     for i, p in enumerate((.999999, .9999, .999, .995, .5, 0., 1.))]
        self.labels = {pair_key(r): 1 for r in self.pool}
        self.cfg = SafeJudgeConfig(gate_local=False, repair_labels=False, explore_every=0)

    def test_unreviewed_ultraconfident_scores_remain_distinct(self):
        hard = run_precision_review(self.pool, LabelOracle(self.labels, 0), 0, self.cfg)
        soft = run_precision_review(self.pool, LabelOracle(self.labels, 0), 0, self.cfg, epsilon=.001)
        self.assertEqual(len(set(hard[2][:4])), 1)
        self.assertTrue(np.all(np.diff(soft[2][:4]) > 0))
        self.assertTrue(np.isfinite(soft[2]).all())
        self.assertTrue(((soft[2] > 0) & (soft[2] < 1)).all())

    def test_no_epsilon_reproduces_frozen_policy(self):
        old = run_safe_judge(self.pool, LabelOracle(self.labels, 3), 3, self.cfg, 12)
        new = run_precision_review(self.pool, LabelOracle(self.labels, 3), 3, self.cfg, 12)
        for i in (0, 1, 2):
            np.testing.assert_array_equal(old[i], new[i])
        self.assertEqual(old[3], new[3])

    def test_invalid_floor_rejected(self):
        for eps in (0, .5, float("nan"), -1):
            with self.assertRaises(ValueError):
                run_precision_review(self.pool, LabelOracle(self.labels, 0), 0, self.cfg, epsilon=eps)
