"""Utility weighting, held-out run semantics and confidence-cache parsing."""

from itertools import combinations
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_confidence_views import read_token_verdicts
from src.baselines.active_learning import LabelOracle
from src.calibration.audit_release import upper_error_total
from src.calibration.diagnostic_review import ReviewConfig, run_diagnostic_review
from src.calibration.diagnostic_utility import run_utility_review
from src.evaluation.retrieval_fidelity import RetrievalFidelity
from src.utils.io import pair_key


class ReviewFidelityTests(unittest.TestCase):
    def test_default_utility_exactly_matches_original_review(self):
        pool = [{"query_id": str(q), "doc_id": str(d), "prob_relevant": .9 if d % 2 else .2}
                for q in range(3) for d in range(10)]
        labels = {pair_key(r): int(r["prob_relevant"] >= .5) for r in pool}
        first = run_diagnostic_review(pool, LabelOracle(labels, 12), 12, seed=19)
        second = run_utility_review(pool, LabelOracle(labels, 12), 12, seed=19)
        for i in (0, 1, 2):
            np.testing.assert_array_equal(first[i], second[i])
        self.assertEqual(first[3:], second[3:])
        with self.assertRaises(ValueError):
            run_utility_review(pool, LabelOracle(labels, 12), 12, utility=np.zeros(len(pool)))

    def test_ranking_utility_ignores_gold_and_human_labels_restore_gold_scores(self):
        rows = [{"query_id": "q", "doc_id": str(i), "true_binary": int(i == 1)} for i in range(1, 5)]
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for system, order in (("bm25_original", [1, 2, 3, 4]), ("bm25_minilm", [4, 3, 2, 1]),
                                  ("splade_original", [1, 2, 3, 4]), ("splade_minilm", [4, 3, 2, 1])):
                path = Path(temp)/(system+".trec")
                path.write_text("".join(f"q Q0 {d} {r} {10-r} {system}\n" for r, d in enumerate(order, 1)))
                paths.append(path)
            original = RetrievalFidelity(rows, paths)
            changed = RetrievalFidelity([{**r, "true_binary": 1-r["true_binary"]} for r in rows], paths)
            np.testing.assert_array_equal(original.review_utility()[0], changed.review_utility()[0])
            self.assertIn("splade_minilm", original.held_out)
            metrics, scores = original.evaluate(np.ones(4, dtype=bool), np.zeros(4, dtype=int))
            np.testing.assert_array_equal(scores, original.gold_scores)
            self.assertAlmostEqual(metrics["public_system_kendall_tau"], 1.)

    def test_low_token_probability_does_not_reverse_the_generated_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"scores.pkl"
            path.write_bytes(pickle.dumps([[('[', .9), ('Guess', .9), ('No', np.float32(.49))]]))
            result = read_token_verdicts(path)
            self.assertEqual(result[0][0], 0)
            self.assertAlmostEqual(result[0][1], .49, places=6)

    def test_token_reader_rejects_executable_globals(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"invalid.pkl"
            path.write_bytes(pickle.dumps(eval))
            with self.assertRaises(ValueError):
                read_token_verdicts(path)

    def test_zero_audit_errors_are_optimistic_for_remaining_error_bounds(self):
        for population in range(1, 15):
            for n in range(population+1):
                remaining = [upper_error_total(population, n, e, .03)-e for e in range(n+1)]
                self.assertTrue(all(a <= b for a, b in zip(remaining, remaining[1:])))

    def test_removing_largest_weights_is_an_optimistic_audit_plan(self):
        weights = np.array([.4, .25, .2, .1, .05])
        for n in range(5):
            for upper in range(6-n):
                optimistic = np.sort(weights)[::-1][n:n+upper].sum()
                for removed in combinations(range(5), n):
                    left = [w for i, w in enumerate(weights) if i not in removed]
                    actual = np.sort(left)[::-1][:upper].sum()
                    self.assertLessEqual(optimistic, actual+1e-15)


if __name__ == "__main__":
    unittest.main()
