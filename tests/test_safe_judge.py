"""SAFE-Judge label isolation, query feedback, ablations and exact costs."""

from dataclasses import replace
import unittest

import numpy as np

from src.baselines.active_learning import LabelOracle
from src.calibration.safe_judge import SafeJudgeModel, SafeJudgeConfig, run_safe_judge
from src.utils.io import pair_key


class SafeJudgeTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"query_id": str(q), "doc_id": str(d), "prob_relevant": (.99 if d % 2 else .05)}
                     for q in range(3) for d in range(16)]
        self.labels = {pair_key(r): int(r["prob_relevant"] >= .5) for r in self.pool}
        for i in (1, 4, 16, 18, 21, 33):
            self.labels[pair_key(self.pool[i])] = 1-self.labels[pair_key(self.pool[i])]

    def test_exact_budgets_and_unique_purchases(self):
        for budget in (0, 1, 6, 20, 47, 48):
            oracle = LabelOracle(self.labels, budget)
            human, predicted, error, log, state = run_safe_judge(self.pool, oracle, budget, seed=9)
            self.assertEqual(int(human.sum()), budget)
            self.assertEqual(len(log), budget)
            self.assertEqual(len(set(oracle.queried)), budget)
            self.assertTrue(np.isfinite(error).all())
            self.assertTrue(np.isin(predicted, [0, 1]).all())

    def test_unpurchased_gold_cannot_change_allocation_or_predictions(self):
        first_oracle = LabelOracle(self.labels, 15)
        first = run_safe_judge(self.pool, first_oracle, 15, seed=9)
        seen = set(first_oracle.queried)
        changed = {key: value if key in seen else 1-value for key, value in self.labels.items()}
        second = run_safe_judge(self.pool, LabelOracle(changed, 15), 15, seed=9)
        for i in (0, 1, 2):
            np.testing.assert_array_equal(first[i], second[i])
        self.assertEqual(first[3:], second[3:])

    def test_zero_budget_reads_no_labels_and_preserves_raw_labels(self):
        class Forbidden(dict):
            def __getitem__(self, key):
                raise AssertionError("An unpurchased label was read")
        result = run_safe_judge(self.pool, LabelOracle(Forbidden(), 0), 0)
        self.assertFalse(result[0].any())
        np.testing.assert_array_equal(result[1], [int(r["prob_relevant"] >= .5) for r in self.pool])

    def test_local_evidence_repairs_a_systematic_query_failure(self):
        pool = [{"query_id": q, "doc_id": str(d), "prob_relevant": .99}
                for q in ("bad", "good") for d in range(30)]
        model = SafeJudgeModel(pool, SafeJudgeConfig())
        for d in range(20):
            model.observe(d, 0)
            model.observe(30+d, 1)
        predicted, _, _, gate = model.predict()
        self.assertTrue((predicted[20:30] == 0).all())
        self.assertTrue((predicted[50:60] == 1).all())
        self.assertGreater(gate[0, 5], .99)
        with self.assertRaises(ValueError):
            model.observe(0, 0)

    def test_no_repair_ablation_retains_original_labels(self):
        pool = [{"query_id": "bad", "doc_id": str(d), "prob_relevant": .99} for d in range(40)]
        model = SafeJudgeModel(pool, replace(SafeJudgeConfig(), repair_labels=False))
        for d in range(30):
            model.observe(d, 0)
        self.assertTrue((model.predict()[0] == 1).all())

    def test_invalid_pool_or_prior_is_rejected(self):
        with self.assertRaises(ValueError):
            SafeJudgeModel([{**self.pool[0], "true_binary": 1}], SafeJudgeConfig())
        with self.assertRaises(ValueError):
            SafeJudgeConfig(local_strength=float("nan"))


if __name__ == "__main__":
    unittest.main()
