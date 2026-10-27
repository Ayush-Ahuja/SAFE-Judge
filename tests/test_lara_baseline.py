"""Offline reference fixtures, label isolation and exact-budget tests for LARA."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from scripts.run_lara_baseline import execute, load_inputs
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import BinaryCalibrator, LaraConfig, group_quotas, prepare_probabilities, run_lara
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json


FIXTURE = Path(__file__).parent / "fixtures/lara_reference.json"


class LaraTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.pool = self.fixture["pool"]
        self.labels = {pair_key(r): int(y) for r, y in zip(self.pool, self.fixture["labels"])}

    def run_pool(self, budget, grouping=1, config=None, labels=None, seed=42):
        oracle = LabelOracle(self.labels if labels is None else labels, budget)
        result = run_lara(self.pool, oracle, budget, config or LaraConfig(), seed, grouping)
        return *result, oracle

    def test_all_pinned_upstream_fixtures(self):
        indices = {pair_key(r): i for i, r in enumerate(self.pool)}
        for case in self.fixture["cases"]:
            with self.subTest(mode=case["preprocessing"], groups=case["grouping"], budget=case["budget"]):
                rows, acquisitions, _, _ = self.run_pool(case["budget"], case["grouping"],
                    replace(LaraConfig(), preprocessing=case["preprocessing"]), seed=case["seed"])
                self.assertEqual([indices[pair_key(r)] for r in acquisitions], case["selected_indices"])
                np.testing.assert_allclose([r["prob_relevant"] for r in rows], case["probabilities"], rtol=0, atol=1e-12)
                self.assertEqual([r["final_label"] for r in rows], case["final_labels"])

    def test_first_acquisition_uses_uncertainty_without_free_seed_labels(self):
        _, acquired, state, _ = self.run_pool(1, config=LaraConfig(preprocessing="raw"))
        probability = acquired[0]["input_probability"]
        self.assertAlmostEqual(abs(probability-.5), min(abs(r["prob_relevant"]-.5) for r in self.pool))
        self.assertEqual(state["calibrator"]["updates"], 1)

    def test_zero_budget_never_reads_a_label(self):
        class Forbidden(dict):
            def __getitem__(self, key):
                raise AssertionError("Zero-budget oracle accessed")
        rows, acquired, state, oracle = self.run_pool(0, labels=Forbidden())
        self.assertFalse(acquired)
        self.assertFalse(oracle.queried)
        self.assertTrue(all(r["label_source"] == "automatic" for r in rows))
        self.assertEqual(state["calibrator"]["updates"], 0)

    def test_n_minus_one_keeps_one_automatic_and_n_is_full_human(self):
        for grouping in (1, 3, "topics"):
            for budget in (0, 1, 7, len(self.pool)-1, len(self.pool)):
                rows, log, state, oracle = self.run_pool(budget, grouping)
                self.assertEqual(len(log), budget)
                self.assertEqual(len(set(oracle.queried)), budget)
                self.assertEqual(sum(r["label_source"] == "automatic" for r in rows), len(self.pool)-budget)
                self.assertEqual(sum(state["group_quotas"]), budget)
                if budget == len(self.pool):
                    self.assertTrue(state["full_human_shortcut"])
                    self.assertEqual([r["final_label"] for r in rows], self.fixture["labels"])

    def test_one_pair_zero_budget_is_not_full_human(self):
        row = self.pool[:1]
        predictions, acquired, _ = run_lara(row, LabelOracle({}, 0), 0)
        self.assertFalse(acquired)
        self.assertEqual(predictions[0]["label_source"], "automatic")

    def test_unqueried_gold_has_no_effect_on_calibration_or_allocation(self):
        first = self.run_pool(7, "topics")
        queried = set(first[3].queried)
        changed = {key: value if key in queried else 1-value for key, value in self.labels.items()}
        second = self.run_pool(7, "topics", labels=changed)
        self.assertEqual(first[:3], second[:3])

    def test_constant_probability_normalization_remains_finite(self):
        p, state = prepare_probabilities([.4]*10, LaraConfig())
        np.testing.assert_array_equal(p, [.4]*10)
        self.assertTrue(state["constant_fallback"])
        model = BinaryCalibrator(LaraConfig())
        model.observe(.4, 0)
        self.assertTrue(np.isfinite(model.predict(p)).all())

    def test_reference_initial_strength_stays_fixed_after_updates(self):
        model = BinaryCalibrator(LaraConfig())
        for _ in range(20):
            model.observe(.7, 1)
        self.assertEqual(model.state()["updates"], 20)
        self.assertEqual(model.config.initial_strength, 200)
        self.assertEqual(model.state()["positive_prior"], 21)
        self.assertEqual(model.state()["negative_prior"], 1)

    def test_group_quotas_redistribute_without_exceeding_capacity(self):
        for budget in range(14):
            quotas = group_quotas([1, 2, 10], budget)
            self.assertEqual(sum(quotas), budget)
            self.assertTrue(all(q <= capacity for q, capacity in zip(quotas, [1, 2, 10])))

    def test_global_prefix_and_random_state_are_reproducible(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        _, short, _, _ = self.run_pool(7)
        _, long, _, _ = self.run_pool(12)
        self.assertEqual(short, long[:7])
        self.assertEqual(random.getstate(), python_state)
        actual = np.random.get_state()
        np.testing.assert_array_equal(actual[1], numpy_state[1])
        self.assertEqual(actual[2:], numpy_state[2:])

    def test_invalid_inputs_and_hyperparameters_are_rejected(self):
        for values in ([float("nan")], [float("inf")], [-.1], [1.1], []):
            with self.assertRaises(ValueError):
                prepare_probabilities(values, LaraConfig())
        for config in ({"learning_rate": 0}, {"preprocessing": "unknown"},
                       {"initial_strength": 10000, "transition_scale": 1}, {"smoothness": float("nan")}):
            with self.assertRaises(ValueError):
                LaraConfig(**config)
        with self.assertRaises(ValueError):
            run_lara([{**r, "true_binary": 0} for r in self.pool], LabelOracle(self.labels, 1), 1)
        with self.assertRaises(ValueError):
            self.run_pool(len(self.pool)+1)

    def files(self, root):
        rows = [{"dataset": "trec_dl_2019", "query_id": r["query_id"], "doc_id": r["doc_id"],
                 "query_text": "text " + r["query_id"], "doc_text": "passage " + r["doc_id"],
                 "true_relevance": 2*self.labels[pair_key(r)], "true_binary": self.labels[pair_key(r)],
                 "relevance_threshold": 2} for r in self.pool]
        atomic_text(root / "manifest.jsonl", "".join(json.dumps(r)+"\n" for r in rows))
        write_csv(root / "predictions.csv", [{**r, "prob_relevant": p["prob_relevant"],
                    "predicted_label": int(p["prob_relevant"] >= .5)} for r, p in zip(rows, self.pool)])
        write_json(root / "config.json", {**asdict(LaraConfig()), "groups": [1, "topics"],
                                           "budget_fractions": [0, .1, .5, 1], "seeds": [42]})
        return rows

    def test_official_binary_adapter_normalizes_and_validates_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.files(root)
            official = [{"topic_id": r["query_id"], "doc_id": r["doc_id"], "annotation": r["true_binary"],
                         "prob_yes": p["prob_relevant"]*.8, "prob_no": (1-p["prob_relevant"])*.8}
                        for r, p in zip(rows, self.pool)]
            write_csv(root / "official.csv", official)
            probabilities, _ = load_inputs(root / "official.csv", rows, "official_binary")
            for p in self.pool:
                self.assertAlmostEqual(probabilities[pair_key(p)], p["prob_relevant"])
            for broken in (official[:-1], official+[official[0]],
                           [{**official[0], "annotation": 2}]+official[1:],
                           [{**official[0], "prob_yes": 0, "prob_no": 0}]+official[1:]):
                write_csv(root / "official.csv", broken)
                with self.assertRaises(ValueError):
                    load_inputs(root / "official.csv", rows, "official_binary")

    def test_cli_artifacts_splits_ranking_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.files(root)
            write_json(root / "split.json", {"calibration_query_ids": ["q0"], "test_query_ids": ["q1", "q2"], "audit_query_ids": ["q3"]})
            run_files = [root / "first.run", root / "second.run"]
            for path, order in zip(run_files, (self.pool, list(reversed(self.pool)))):
                atomic_text(path, "".join(f"{r['query_id']} Q0 {r['doc_id']} {i+1} {100-i} {path.stem}\n"
                                          for i, r in enumerate(order)))
            args = argparse.Namespace(manifest=root / "manifest.jsonl", predictions=root / "predictions.csv",
                                      predictions_format="project", config=root / "config.json", split_json=root / "split.json",
                                      output_dir=root / "results", preprocessing=None, groups=None, seeds=None,
                                      budgets=[0, 1, 9, 10], budget_fractions=None, run_files=run_files)
            summaries = execute(args)
            self.assertEqual(len(summaries), 8)
            for point in summaries:
                self.assertEqual(point["total_human_labels"], point["budget"])
            directory = args.output_dir / "groups_1/seed_42/budget_000009"
            predictions, _ = read_predictions(directory / "predictions.csv")
            self.assertEqual(len(predictions), 10)
            self.assertEqual({r["query_id"] for r in predictions}, {"q1", "q2"})
            self.assertEqual(sum(r["label_source"] == "automatic" for r in predictions), 1)
            self.assertTrue((directory / "mixed.qrels").is_file())
            self.assertTrue((directory / "system_scores.csv").is_file())
            for line in (args.output_dir / "checksums.sha256").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(file_hash(args.output_dir / name), digest)
            with self.assertRaises(FileExistsError):
                execute(args)


if __name__ == "__main__":
    unittest.main()
