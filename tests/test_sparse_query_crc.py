"""Budget accounting, loss definition, ties, correction and label isolation."""

import argparse
from contextlib import redirect_stdout
import csv
import io
from itertools import combinations
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from scripts.run_sparse_query_crc import execute
from scripts.run_target_pool_sparse_crc import evaluate_allocation
from src.calibration.sparse_query import evaluation_table, query_loss_curves, sampling_plan, select_threshold
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


class SparseQueryCRCTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.array([.5, .8, .9, 1.])
        self.pool = [{"query_id": str(q), "doc_id": str(d), "prob_relevant": .9}
                     for q in range(4) for d in range(6)]

    def test_exact_budget_no_duplicates_and_nested_sample_priorities(self):
        small = sampling_plan(self.pool, ["0", "1"], 4, 42)
        large = sampling_plan(self.pool, ["0", "1", "2", "3"], 16, 42)
        self.assertEqual(sum(map(len, small.values())), 4)
        self.assertEqual(sum(map(len, large.values())), 16)
        self.assertEqual(len({k for v in large.values() for k in v}), 16)
        for q in small:
            self.assertEqual(small[q], large[q][:2])
        reversed_rows = sampling_plan(list(reversed(self.pool)), ["0", "1"], 4, 42)
        self.assertEqual(small, reversed_rows)

    def test_invalid_budgets_and_capacity_fail_instead_of_borrowing_labels(self):
        for ids, budget in ((["0", "1"], 3), (["0"], 7), (["0", "0"], 4), (["0"], 0)):
            with self.subTest(ids=ids, budget=budget), self.assertRaises(ValueError):
                sampling_plan(self.pool, ids, budget, 42)

    def test_unsampled_labels_cannot_be_read_and_cannot_change_threshold(self):
        plan = sampling_plan(self.pool, ["0", "1"], 4, 42)
        observed = {key: 1 for keys in plan.values() for key in keys}
        class PurchasedOnly(dict):
            def __getitem__(self, key):
                if key not in observed:
                    raise AssertionError("An unpurchased label was read")
                return super().__getitem__(key)
        curves = query_loss_curves(self.pool, plan, PurchasedOnly(observed), self.grid)
        changed_pool = [{**r, "true_binary": 0} for r in self.pool]
        np.testing.assert_array_equal(curves, query_loss_curves(changed_pool, plan, observed, self.grid))

    def test_strict_ties_and_probability_one_escalate_at_endpoint(self):
        rows = [{"query_id": "q", "doc_id": str(i), "prob_relevant": p, "true_binary": 0}
                for i, p in enumerate((.8, .9, 1.))]
        table = evaluation_table(rows, self.grid)
        self.assertEqual([r["automatic_labels"] for r in table], [3, 2, 1, 0])
        self.assertEqual(table[-1]["query_mean_auto_error_rate"], 0)
        self.assertIsNone(table[-1]["selective_risk"])
        self.assertIsNone(table[-1]["class_0_coverage"])

    def test_query_weighting_is_not_pooled_pair_weighting_or_selective_risk(self):
        rows = [{"query_id": "a", "doc_id": "0", "prob_relevant": .9, "true_binary": 0}]
        rows += [{"query_id": "b", "doc_id": str(i), "prob_relevant": .6, "true_binary": 1}
                 for i in range(9)]
        metric = evaluation_table(rows, self.grid)[1]
        self.assertAlmostEqual(metric["query_mean_auto_error_rate"], .5)
        self.assertAlmostEqual(metric["auto_error_rate"], .1)
        self.assertAlmostEqual(metric["selective_risk"], 1.)
        self.assertAlmostEqual(metric["coverage"], .1)

    def test_uniform_sample_loss_is_unbiased_at_each_fixed_threshold(self):
        pool = [{"query_id": "q", "doc_id": str(i), "prob_relevant": p}
                for i, p in enumerate((.6, .8, .9, 1.))]
        labels = {pair_key(r): y for r, y in zip(pool, (0, 1, 0, 0))}
        keys = list(labels)
        full = query_loss_curves(pool, {"q": keys}, labels, self.grid)
        samples = [query_loss_curves(pool, {"q": list(subset)}, labels, self.grid)
                   for subset in combinations(keys, 2)]
        np.testing.assert_allclose(np.mean(samples, axis=0), full, atol=1e-15)

    def test_crc_counts_queries_not_sampled_pairs_and_has_safe_fallback(self):
        five = select_threshold(np.zeros((5, 4)), self.grid, .05)
        twenty = select_threshold(np.zeros((20, 4)), self.grid, .05)
        self.assertTrue(five["full_escalation_fallback"])
        self.assertEqual(five["threshold"], 1.)
        self.assertFalse(twenty["full_escalation_fallback"])
        self.assertEqual(twenty["threshold"], .5)
        self.assertAlmostEqual(twenty["selection_criterion"], 1/21)
        boundary = select_threshold(np.zeros((19, 4)), self.grid, .05)
        self.assertFalse(boundary["full_escalation_fallback"])

    def test_least_conservative_valid_threshold_and_empirical_ablation(self):
        losses = np.tile([.1, .02, 0., 0.], (30, 1))
        selected = select_threshold(losses, self.grid, .05)
        naive = select_threshold(losses, self.grid, .05, "empirical")
        self.assertEqual(selected["threshold"], .9)
        self.assertEqual(naive["threshold"], .8)
        self.assertAlmostEqual(selected["selection_criterion"], 1/31)

    def test_invalid_loss_family_is_rejected(self):
        for losses in ([[0, .1, 0, 0]], [[.5, .2, .1, .1]], [[1.1, 0, 0, 0]], [[float('nan')]*4]):
            with self.subTest(losses=losses), self.assertRaises(ValueError):
                select_threshold(losses, self.grid, .05)

    def test_target_pool_purchased_labels_are_human_and_never_double_charged(self):
        rows = [{"query_id": q, "doc_id": d, "prob_relevant": p, "true_binary": y}
                for q, d, p, y in (("a", "1", .99, 0), ("a", "2", .9, 0),
                                   ("b", "3", .7, 1), ("b", "4", .9, 1))]
        metrics = evaluate_allocation(rows, {("a", "1")}, .8)
        self.assertEqual(metrics["calibration_labels"], 1)
        self.assertEqual(metrics["additional_human_labels"], 1)
        self.assertEqual(metrics["total_human_labels"], 2)
        self.assertEqual(metrics["automatic_errors"], 1)
        self.assertEqual(metrics["coverage"], .5)
        self.assertEqual(metrics["query_mean_auto_error_rate"], .25)
        self.assertEqual(evaluate_allocation(rows, {pair_key(r) for r in rows}, .5)["automatic_labels"], 0)
        with self.assertRaises(ValueError):
            evaluate_allocation(rows, {("unknown", "pair")}, .8)

    def test_runner_isolates_held_out_labels_and_records_exact_costs(self):
        settings = {"split_seed": 9, "held_out_queries": 2, "query_counts": [1, 2, 4],
                    "label_budgets": [4], "alphas": [.3], "repetitions": 2,
                    "first_seed": 0, "grid_steps": 10, "full_query_reference": True}
        ids = [str(q) for q in range(6)]
        random.Random(9).shuffle(ids)
        test_ids = ids[:2]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/"config.json", settings)
            selections = []
            for mutate in (False, True):
                manifest, predictions = [], []
                for q in range(6):
                    for d in range(5):
                        gold = int(not (mutate and str(q) in test_ids))
                        row = {"dataset": "fixture", "query_id": str(q), "doc_id": str(d),
                               "query_text": "query", "doc_text": "document", "true_relevance": gold*2,
                               "relevance_threshold": 2, "true_binary": gold}
                        manifest.append(row)
                        predictions.append({**row, "prob_relevant": .9, "predicted_label": 1})
                atomic_text(root/"manifest.jsonl", "".join(json.dumps(r)+"\n" for r in manifest))
                write_csv(root/"predictions.csv", predictions)
                output = root/str(mutate)
                args = argparse.Namespace(config=root/"config.json", manifest=root/"manifest.jsonl",
                                          predictions=root/"predictions.csv", output_dir=output)
                with redirect_stdout(io.StringIO()):
                    metadata = execute(args)
                self.assertEqual(metadata["status"], "complete")
                self.assertFalse(set(metadata["test_query_ids"]) & set(metadata["calibration_pool_query_ids"]))
                with (output/"runs.csv").open(encoding="utf-8", newline="") as stream:
                    runs = list(csv.DictReader(stream))
                selections.append([(r["threshold"], r["selection_criterion"]) for r in runs])
                for run in runs:
                    self.assertEqual(int(run["total_human_labels"]),
                                     int(run["calibration_labels"])+int(run["test_human_labels"]))
                    if run["sampling"] == "sparse_uniform":
                        self.assertEqual(int(run["calibration_labels"]), 4)
                for line in (output/"checksums.sha256").read_text().splitlines():
                    digest, name = line.split("  ", 1)
                    self.assertEqual(file_hash(output/name), digest)
            self.assertEqual(selections[0], selections[1])


if __name__ == "__main__":
    unittest.main()
