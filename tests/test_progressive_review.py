"""Fresh-audit boundaries, optional stopping, label reuse and power checks."""

import unittest
import numpy as np

from src.baselines.active_learning import LabelOracle
from src.calibration.progressive_review import progressive_review
from src.utils.io import pair_key


class ProgressiveReviewTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"query_id": str(i//40), "doc_id": str(i), "prob_relevant": .99 if i % 2 else .01}
                     for i in range(200)]
        self.labels = {pair_key(r): int(r["prob_relevant"] >= .5) for r in self.pool}

    def worker(self, oracle, budget):
        human = np.zeros(len(self.pool), dtype=bool)
        # The first budget pairs contain equal numbers of both classes.
        for i in range(budget):
            oracle.query(pair_key(self.pool[i]))
            human[i] = True
        return human, np.asarray([int(r["prob_relevant"] >= .5) for r in self.pool])

    def test_stops_only_after_a_fresh_audit_and_counts_all_labels_once(self):
        oracle = LabelOracle(self.labels, len(self.pool))
        human, predicted, certificate, looks, ledger = progressive_review(
            self.pool, oracle, self.worker, 0, checkpoints=(0., .25), audit_fraction=.25,
            alpha=.2, class_risk=.3)
        self.assertTrue(certificate["released"])
        self.assertEqual(certificate["stop_step"], 0)
        self.assertEqual(int(human.sum()), len(oracle.queried))
        self.assertEqual(len(ledger), len(set(oracle.queried)))
        self.assertTrue(all(r["reason"] == "quality_check" for r in ledger))
        self.assertLessEqual(certificate["query_auto_error_upper"], .2)

    def test_infeasible_checks_buy_no_audit_labels_before_fallback(self):
        oracle = LabelOracle(self.labels, len(self.pool))
        human, _, certificate, looks, ledger = progressive_review(
            self.pool, oracle, self.worker, 1, checkpoints=(0., .1), audit_fraction=.01)
        self.assertTrue(human.all())
        self.assertFalse(certificate["released"])
        self.assertTrue(all(r["status"] == "skipped_infeasible" for r in looks))
        self.assertTrue(all(r["reason"] != "quality_check" for r in ledger))
        self.assertEqual(len(oracle.queried), len(self.pool))

    def test_all_audit_errors_force_full_review(self):
        wrong = {k: 1-v for k, v in self.labels.items()}
        human, _, certificate, _, ledger = progressive_review(
            self.pool, LabelOracle(wrong, len(self.pool)), self.worker, 2,
            checkpoints=(0., .1), audit_fraction=.25, alpha=.2, class_risk=.3, feasibility_gate=False)
        self.assertTrue(human.all())
        self.assertFalse(certificate["released"])
        self.assertEqual(len(ledger), len(self.pool))

    def test_checkpoint_delta_spending_sums_to_global_budget(self):
        result = progressive_review(self.pool, LabelOracle(self.labels, len(self.pool)), self.worker, 3,
            checkpoints=(0., .1, .25), audit_fraction=.01)
        self.assertAlmostEqual(sum(r["look_delta"] for r in result[3]), .05)


if __name__ == "__main__":
    unittest.main()
