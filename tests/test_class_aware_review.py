"""Meaningful allocation, label-isolation and budget checks."""

from dataclasses import replace
from itertools import combinations
import unittest

import numpy as np

from src.baselines.active_learning import LabelOracle
from src.calibration.class_aware_review import acceptance_plan, run_class_aware_review
from src.calibration.safe_judge import SafeJudgeConfig
from src.utils.io import pair_key


class ClassAwareReviewTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"query_id": str(q), "doc_id": str(d),
                      "prob_relevant": .98 if d % 2 else .03}
                     for q in range(3) for d in range(20)]
        self.labels = {pair_key(r): int(r["prob_relevant"] >= .5) for r in self.pool}
        for r in self.pool[:20]:
            if r["prob_relevant"] > .5:
                self.labels[pair_key(r)] = 0

    def test_minimax_matches_all_subsets_independent_of_prefix_assumption(self):
        raw = np.array([0]*4+[1]*4)
        error = np.array([.01, .04, .2, .3, .1, .12, .4, .8])
        reviewed = np.zeros(8, dtype=bool)
        accepted, state = acceptance_plan(raw, error, reviewed, 4, [1, 1], np.arange(8))
        values = []
        for subset in combinations(range(8), 4):
            ix = np.array(subset)
            if len(set(raw[ix])) == 2:
                values.append(max(error[ix][raw[ix] == c].mean() for c in (0, 1)))
        self.assertAlmostEqual(state["estimated_worst_class_risk"], min(values))
        self.assertEqual(accepted.sum(), 4)

    def test_query_objective_matches_exhaustive_weighted_error(self):
        raw = np.array([0]*3+[1]*3)
        error = np.array([.01, .2, .3, .02, .12, .8])
        weights = np.array([20, 1, 1, 20, 1, 1])
        accepted, _ = acceptance_plan(raw, error, np.zeros(6, bool), 3,
            [1, 1], np.arange(6), "query_error", weights)
        best = min(sum((error*weights)[list(s)]) for s in combinations(range(6), 3)
                   if len(set(raw[list(s)])) == 2)
        self.assertAlmostEqual(sum((error*weights)[accepted]), best)

    def test_exact_budget_floors_and_no_label_flips(self):
        for budget in (0, 1, 10, 30, 48):
            oracle = LabelOracle(self.labels, budget)
            human, predicted, _, log, _ = run_class_aware_review(self.pool, oracle, budget,
                minimum_class_coverage=.2, batch_size=4)
            self.assertEqual(human.sum(), budget)
            self.assertEqual(len(set(oracle.queried)), budget)
            self.assertEqual({pair_key(r) for r in log}, set(oracle.queried))
            raw = np.array([int(r["prob_relevant"] >= .5) for r in self.pool])
            np.testing.assert_array_equal(predicted, raw)
            for c in (0, 1):
                self.assertGreaterEqual(sum(~human & (raw == c)), 6)

    def test_unpurchased_gold_cannot_change_result(self):
        oracle = LabelOracle(self.labels, 25)
        first = run_class_aware_review(self.pool, oracle, 25, seed=9, batch_size=4)
        changed = {key: y if key in set(oracle.queried) else 1-y for key, y in self.labels.items()}
        second = run_class_aware_review(self.pool, LabelOracle(changed, 25), 25, seed=9, batch_size=4)
        for i in (0, 1, 2):
            np.testing.assert_array_equal(first[i], second[i])
        self.assertEqual(first[3:], second[3:])

    def test_reviewed_items_cannot_reenter_automatic_plan(self):
        accepted, _ = acceptance_plan([0, 0, 1, 1], [.01, .2, .01, .2],
            [True, False, False, False], 2, [1, 1], np.arange(4))
        self.assertFalse(accepted[0])

    def test_infeasible_floor_or_repairs_are_rejected_before_purchase(self):
        oracle = LabelOracle(self.labels, 59)
        with self.assertRaises(ValueError):
            run_class_aware_review(self.pool, oracle, 59)
        self.assertFalse(oracle.queried)
        with self.assertRaises(ValueError):
            run_class_aware_review(self.pool, LabelOracle(self.labels, 10), 10,
                replace(SafeJudgeConfig(), repair_labels=True))


if __name__ == "__main__":
    unittest.main()
