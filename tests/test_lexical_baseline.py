"""Budget, leakage, analytical feature and end-to-end baseline checks (CPU)."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.run_lexical_baseline import budget_counts, execute, select_queries
from src.baselines.active_learning import (ActiveLearningConfig, LabelOracle, lexical_features,
                                           run_active_learning, tokenize)
from src.evaluation.budgeted import allocation_metrics, evaluate_runs
from src.utils.io import atomic_text, file_hash, pair_key, read_predictions


def fixture_pool():
    return [{"query_id": "q", "doc_id": f"d{i}", "query_text": "apple fruit",
             "doc_text": "apple " * (i + 1) + "fruit " * (i % 3) + "other " * (8 - i)}
            for i in range(8)]


class LexicalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.pool = fixture_pool()
        self.labels = {pair_key(r): i % 2 for i, r in enumerate(self.pool)}
        self.config = ActiveLearningConfig(seed_size=2, batch_size=2)

    def run_pool(self, budget, seed=1, labels=None, config=None):
        oracle = LabelOracle(self.labels if labels is None else labels, budget)
        rows, log, state = run_active_learning(self.pool, oracle, budget, config or self.config, seed)
        return rows, log, state, oracle

    def test_bm25_matches_hand_calculation(self):
        pool = [{"query_id": "q", "doc_id": "a", "query_text": "apple apple pear",
                 "doc_text": "apple apple"},
                {"query_id": "q", "doc_id": "b", "query_text": "apple apple pear", "doc_text": "pear"}]
        x, stats = lexical_features(pool, self.config)
        expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 2 / 1.5))
        self.assertAlmostEqual(x[0, 0], expected, places=12)
        self.assertEqual(x[0, 1], .5)
        self.assertAlmostEqual(x[0, 2], math.log(3))
        self.assertEqual(stats["document_count"], 2)
        self.assertEqual(stats["average_document_length"], 1.5)

    def test_unique_document_statistics_and_external_reuse(self):
        x, stats = lexical_features(self.pool, self.config)
        repeated = [{**self.pool[0], "query_id": "q2"}]
        _, repeated_stats = lexical_features(self.pool + repeated, self.config)
        self.assertEqual(stats, repeated_stats)
        reused, _ = lexical_features(self.pool, self.config, stats)
        np.testing.assert_array_equal(x, reused)

    def test_empty_tokens_and_unicode(self):
        self.assertEqual(tokenize("CAFÉ_β2,42"), ["café", "β2", "42"])
        pool = [{**self.pool[0], "query_text": "???", "doc_text": "---"}]
        x, _ = lexical_features(pool, self.config)
        np.testing.assert_array_equal(x, [[0, 0, 0]])

    def test_conflicting_and_duplicate_inputs_fail(self):
        for pool in (self.pool + [self.pool[0]], self.pool + [{**self.pool[0], "query_id": "z", "doc_text": "changed"}],
                     [self.pool[0], {**self.pool[1], "query_text": "different query"}]):
            with self.assertRaises(ValueError):
                lexical_features(pool, self.config)
        with self.assertRaisesRegex(ValueError, "Strip gold"):
            run_active_learning([{**r, "true_binary": 0} for r in self.pool], LabelOracle(self.labels, 1), 1)
        with self.assertRaisesRegex(ValueError, "Sort"):
            run_active_learning(list(reversed(self.pool)), LabelOracle(self.labels, 1), 1)

    def test_invalid_external_statistics_fail(self):
        _, stats = lexical_features(self.pool, self.config)
        for broken in ({**stats, "document_count": 0}, {**stats, "average_document_length": float("nan")},
                       {**stats, "tokenizer": "other"}, {**stats, "document_frequencies": {"apple": 99}},
                       {**stats, "document_frequencies": {}}):
            with self.assertRaises(ValueError):
                lexical_features(self.pool, self.config, broken)

    def test_exact_budget_including_seed_and_n_minus_one(self):
        for budget in (0, 1, 2, 3, 7, 8):
            with self.subTest(budget=budget):
                rows, log, _, oracle = self.run_pool(budget)
                self.assertEqual(len(log), budget)
                self.assertEqual(len(set(oracle.queried)), budget)
                self.assertEqual(sum(r["label_source"] == "human" for r in rows), budget)
                self.assertEqual(sum(a["stage"] == "random_seed" for a in log), min(budget, 2))
                for r in rows:
                    expected = self.labels[pair_key(r)] if r["label_source"] == "human" else r["predicted_label"]
                    self.assertEqual(r["final_label"], expected)

    def test_zero_budget_does_not_read_any_label(self):
        class ForbiddenLabels(dict):
            def __getitem__(self, key):
                raise AssertionError("Zero-budget learner accessed gold labels")
        rows, log, state, _ = self.run_pool(0, labels=ForbiddenLabels())
        self.assertFalse(log)
        self.assertEqual(state["model"]["kind"], "untrained")
        self.assertTrue(all(r["prob_relevant"] == .5 for r in rows))

    def test_one_class_seed_never_forces_free_labels(self):
        labels = {key: 0 for key in self.labels}
        rows, log, state, _ = self.run_pool(3, labels=labels)
        self.assertEqual(len(log), 3)
        self.assertEqual(state["model"]["kind"], "single_class_prior")
        self.assertTrue(all(r["prob_relevant"] == .2 for r in rows))

    def test_unqueried_gold_cannot_affect_acquisition_or_model(self):
        rows, log, state, oracle = self.run_pool(4)
        unseen_changed = {key: value if key in oracle.queried else 1 - value
                          for key, value in self.labels.items()}
        other_rows, other_log, other_state, _ = self.run_pool(4, labels=unseen_changed)
        self.assertEqual((rows, log, state), (other_rows, other_log, other_state))

    def test_reproducible_seed_and_fitted_model_serialization(self):
        first = self.run_pool(5)
        second = self.run_pool(5)
        self.assertEqual(first[:3], second[:3])
        rows, _, state, _ = first
        self.assertEqual(state["model"]["kind"], "logistic_regression")
        x, _ = lexical_features(self.pool, self.config)
        scaled = (x - state["scaler_mean"]) / state["scaler_scale"]
        z = scaled @ np.asarray(state["model"]["coefficients"]) + state["model"]["intercept"]
        np.testing.assert_allclose([r["prob_relevant"] for r in rows], 1 / (1 + np.exp(-z)), atol=1e-12)

    def test_acquisition_selects_nearest_half_from_independent_fit(self):
        _, log, _, _ = self.run_pool(4)
        x, _ = lexical_features(self.pool, self.config)
        x = StandardScaler().fit_transform(x)
        seed_indices = [next(i for i, row in enumerate(self.pool) if pair_key(row) == pair_key(a)) for a in log[:2]]
        y = [self.labels[pair_key(self.pool[i])] for i in seed_indices]
        self.assertEqual(set(y), {0, 1})
        independent = LogisticRegression(C=1, solver="lbfgs", max_iter=2000, tol=1e-8).fit(x[seed_indices], y)
        probabilities = independent.predict_proba(x)[:, 1]
        remaining = [i for i in range(8) if i not in seed_indices]
        expected = sorted(remaining, key=lambda i: abs(probabilities[i] - .5))[:2]
        self.assertEqual({pair_key(a) for a in log[2:]}, {pair_key(self.pool[i]) for i in expected})

    def test_budget_validation_and_decimal_floor(self):
        self.assertEqual(budget_counts(100, [.29, 0, 1]), [0, 29, 100])
        for invalid in ([-.1], [1.1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                budget_counts(8, invalid)
        for budget in (-1, 9, True):
            with self.assertRaises(ValueError):
                self.run_pool(budget)
        with self.assertRaises(ValueError):
            replace(self.config, seed_size=0)
        with self.assertRaises(ValueError):
            replace(self.config, regularization_c=float("nan"))

    def test_oracle_rejects_duplicate_and_over_budget_queries(self):
        oracle = LabelOracle(self.labels, 1)
        oracle.query(pair_key(self.pool[0]))
        for key in (pair_key(self.pool[0]), pair_key(self.pool[1])):
            with self.assertRaises(ValueError):
                oracle.query(key)

    def test_allocation_metrics_use_only_automatic_errors(self):
        rows = [
            {"query_id": "q", "true_binary": 1, "predicted_label": 0, "final_label": 1,
             "prob_relevant": .1, "label_source": "human"},
            {"query_id": "q", "true_binary": 0, "predicted_label": 1, "final_label": 1,
             "prob_relevant": .7, "label_source": "automatic"},
            {"query_id": "q", "true_binary": 0, "predicted_label": 0, "final_label": 0,
             "prob_relevant": .2, "label_source": "automatic"}]
        result = allocation_metrics(rows)
        self.assertEqual(result["selective_risk"], .5)
        self.assertEqual(result["auto_error_rate"], 1/3)
        self.assertAlmostEqual(result["mixed_label_accuracy"], 2/3)
        self.assertEqual(result["worst_observed_class_selective_risk"], 1)
        all_human = [{**r, "label_source": "human", "final_label": r["true_binary"]} for r in rows]
        result = allocation_metrics(all_human)
        self.assertIsNone(result["selective_risk"])
        self.assertIsNone(result["automatic_classification"])
        self.assertEqual(result["mixed_label_accuracy"], 1)

    def test_shared_split_rejects_overlap_and_unknown_topics(self):
        manifest = [{**self.pool[0], "query_id": q} for q in ("a", "b", "audit")]
        split = {"calibration_query_ids": ["a"], "test_query_ids": ["b"], "audit_query_ids": ["audit"]}
        self.assertEqual([r["query_id"] for r in select_queries(manifest, split)], ["b"])
        for broken in ({**split, "test_query_ids": ["audit"]}, {**split, "test_query_ids": ["missing"]},
                       {**split, "test_query_ids": ["b", "b"]}):
            with self.assertRaises(ValueError):
                select_queries(manifest, broken)

    def test_ranking_handles_unjudged_docs_and_full_human_endpoint(self):
        rows = [{"query_id": "q", "doc_id": "a", "true_binary": 1, "final_label": 0},
                {"query_id": "q", "doc_id": "b", "true_binary": 0, "final_label": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "good.run", Path(tmp) / "bad.run"]
            atomic_text(paths[0], "q Q0 a 1 3 good\nq Q0 unknown 2 2 good\nq Q0 b 3 1 good\n")
            atomic_text(paths[1], "q Q0 b 1 2 bad\nq Q0 a 2 1 bad\n")
            systems, _, ranking = evaluate_runs(rows, paths)
            self.assertEqual(systems[0]["gold_ndcg_at_10"], 1)
            self.assertEqual(systems[0]["mixed_ndcg_at_10"], .5)
            self.assertEqual(ranking["kendall_tau_b"], -1)
            _, _, ranking = evaluate_runs([{**r, "final_label": r["true_binary"]} for r in rows], paths)
            self.assertEqual(ranking["kendall_tau_b"], 1)

    def test_cli_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = [{**r, "dataset": "trec_dl_2019", "relevance_threshold": 2,
                         "true_binary": self.labels[pair_key(r)],
                         "true_relevance": 2 * self.labels[pair_key(r)]} for r in reversed(self.pool)]
            atomic_text(root / "manifest.jsonl", "".join(json.dumps(r) + "\n" for r in manifest))
            atomic_text(root / "config.json", json.dumps({"seed_size": 2, "batch_size": 2}))
            run_paths = [root / "first.run", root / "second.run"]
            for path, order in zip(run_paths, (self.pool, list(reversed(self.pool)))):
                atomic_text(path, "".join(f"q Q0 {r['doc_id']} {i+1} {8-i} {path.stem}\n"
                                         for i, r in enumerate(order)))
            args = argparse.Namespace(manifest=root / "manifest.jsonl", config=root / "config.json",
                                      output_dir=root / "results", split_json=None, bm25_stats=None,
                                      seeds=[1], budgets=[0, 3, 7, 8], budget_fractions=None, run_files=run_paths)
            from src.utils.io import write_csv
            with patch("scripts.run_lexical_baseline.write_csv", wraps=write_csv) as writer:
                summaries = execute(args)
                curve_calls = [call for call in writer.call_args_list if call.args[0].name == "budget_curve.csv"]
                self.assertEqual(len(curve_calls), 1)
            self.assertEqual(len(summaries), 4)
            self.assertEqual([r["human_labels"] for r in summaries], [0, 3, 7, 8])
            directory = args.output_dir / "seed_1/budget_000003"
            predictions, _ = read_predictions(directory / "predictions.csv")
            self.assertEqual(len(predictions), 8)
            self.assertEqual(len((directory / "mixed.qrels").read_text().splitlines()), 8)
            self.assertTrue((directory / "system_scores.csv").is_file())
            experiment = json.loads((args.output_dir / "experiment.json").read_text())
            self.assertEqual(experiment["status"], "complete")
            for line in (args.output_dir / "checksums.sha256").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(file_hash(args.output_dir / name), digest)
            with self.assertRaises(FileExistsError):
                execute(args)


if __name__ == "__main__":
    unittest.main()
