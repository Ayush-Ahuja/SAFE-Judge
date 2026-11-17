"""Finite-population coverage, query weighting and pilot/audit label isolation."""

from itertools import combinations
import unittest

import numpy as np
from scipy.stats import hypergeom

from scripts.run_error_predictor_ablation import complete_budget
from src.calibration.audit_release import build_audit, equal_quotas, select_release, upper_error_total
from src.calibration.candidate_audit import candidate_release
from src.calibration.query_error import fit_error_scores, pilot_sample
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import pair_key


class AuditReleaseTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"query_id": str(q), "doc_id": str(d), "prob_relevant": .8 if d % 2 else .1,
                      "auxiliary_logit": 1. if d % 3 else -1.}
                     for q in range(4) for d in range(12)]
        self.labels = {pair_key(r): int(r["prob_relevant"] >= .5) for r in self.pool}

    def test_exact_upper_bound_has_nominal_coverage_for_all_small_populations(self):
        for population in range(1, 9):
            for n in range(population+1):
                for defects in range(population+1):
                    failure = sum(float(hypergeom.pmf(e, population, defects, n))
                                  for e in range(n+1) if upper_error_total(population, n, e, .1) < defects)
                    with self.subTest(N=population, n=n, M=defects):
                        self.assertLessEqual(failure, .1+1e-12)

    def test_endpoint_and_zero_label_bounds(self):
        self.assertEqual(upper_error_total(20, 0, 0, .05), 20)
        self.assertEqual(upper_error_total(20, 20, 3, .05), 3)
        self.assertEqual(upper_error_total(20, 5, 5, .05), 20)
        with self.assertRaises(ValueError):
            upper_error_total(5, 6, 0, .05)
        self.assertEqual(equal_quotas([1, 7, 3], 8), [1, 4, 3])

    def test_pilot_features_and_predictions_do_not_read_unpurchased_gold(self):
        pilot = pilot_sample(self.pool, 2, 17)
        observed = {key: self.labels[key] for key in pilot}
        observed[pilot[0]] = 1-observed[pilot[0]]
        original = fit_error_scores(self.pool, observed, "query_disagreement")
        changed = [{**r, "true_binary": 1-self.labels[pair_key(r)]} for r in self.pool]
        np.testing.assert_array_equal(original, fit_error_scores(changed, observed, "query_disagreement"))
        self.assertEqual(len(pilot), 8)
        for budget in (8, 16, 47):
            human = complete_budget(self.pool, pilot, original, budget, 17)
            self.assertEqual(int(human.sum()), budget)

    def test_audit_reads_exact_new_labels_and_retains_all_purchases_as_human(self):
        pilot = pilot_sample(self.pool, 2, 17)
        called = []
        def purchase(key):
            self.assertNotIn(key, pilot)
            self.assertNotIn(key, called)
            called.append(key)
            return self.labels[key]
        audit = build_audit(self.pool, np.arange(len(self.pool)), pilot, purchase, 12, 2, 17)
        self.assertEqual(len(called), 12)
        human, certificate = select_release(self.pool, audit, .3, .8, "query_balanced", 0.)
        for i, r in enumerate(self.pool):
            if pair_key(r) in set(pilot+called):
                self.assertTrue(human[i])
        self.assertLessEqual(certificate["certified_query_mean_auto_error_upper"], .3)

    def test_pooled_certificate_does_not_imply_equal_query_certificate(self):
        pool = [{"query_id": q, "doc_id": str(i), "prob_relevant": .9}
                for i, q in enumerate(("small", "large", "large", "large"))]
        audit = {"cells": [{"predicted_class": 1, "rank": 0, "remaining_size": 4,
                             "remaining_error_upper": 1, "query_weighted_error_upper": .5,
                             "remaining_indices": [0, 1, 2, 3]}],
                 "pairs": 4, "class_sizes": {0: 0, 1: 4}, "delta": .05}
        pooled, _ = select_release(pool, audit, .3, .9, "pooled", 0.)
        balanced, _ = select_release(pool, audit, .3, .9, "query_balanced", 0.)
        self.assertEqual(int((~pooled).sum()), 4)
        self.assertEqual(int((~balanced).sum()), 0)

    def test_largest_query_weights_bound_all_compatible_error_placements(self):
        weights = np.array([.5, 1/6, 1/6, 1/6])
        for upper in range(5):
            bound = np.sort(weights)[::-1][:upper].sum()
            for defects in range(upper+1):
                for locations in combinations(range(4), defects):
                    self.assertLessEqual(weights[list(locations)].sum(), bound+1e-15)

    def test_all_gold_audited_leaves_no_automatic_errors(self):
        audit = build_audit(self.pool, np.zeros(len(self.pool)), [], self.labels.__getitem__, len(self.pool), 2, 11)
        for unit in ("pooled", "query_balanced"):
            human, certificate = select_release(self.pool, audit, risk_unit=unit)
            self.assertTrue(human.all())
            self.assertEqual(certificate["certified_query_mean_auto_error_upper"], 0.)

    def test_query_metric_uses_original_population_and_humans_are_correct(self):
        rows = [{"query_id": "a", "doc_id": "a", "prob_relevant": .9, "true_binary": 0},
                {"query_id": "b", "doc_id": "b", "prob_relevant": .9, "true_binary": 1},
                {"query_id": "b", "doc_id": "c", "prob_relevant": .9, "true_binary": 0}]
        metrics, _ = PolicyEvaluator(rows).evaluate([False, False, True])
        self.assertEqual(metrics["query_mean_auto_error_rate"], .5)
        self.assertAlmostEqual(metrics["auto_error_rate"], 1/3)
        self.assertEqual(metrics["human_labels"], 1)

    def test_candidate_audit_caps_purchases_and_never_releases_reviewed_pairs(self):
        acquired = []
        def purchase(key):
            self.assertNotIn(key, acquired)
            acquired.append(key)
            return self.labels[key]
        human, certificate, cells = candidate_release(self.pool, np.zeros(len(self.pool)), [], purchase, 12, 9)
        self.assertEqual(len(acquired), certificate["audit_labels"])
        self.assertLessEqual(len(acquired), 12)
        for i, row in enumerate(self.pool):
            if pair_key(row) in acquired:
                self.assertTrue(human[i])
        self.assertEqual(sum(c["audit_labels"] for c in cells), len(acquired))

    def test_candidate_zero_audit_has_conservative_full_human_fallback(self):
        def forbidden(key):
            self.fail("Zero audit budget read a label")
        human, certificate, _ = candidate_release(self.pool, np.zeros(len(self.pool)), [], forbidden, 0, 9)
        self.assertTrue(human.all())
        self.assertFalse(certificate["released"])


if __name__ == "__main__":
    unittest.main()
