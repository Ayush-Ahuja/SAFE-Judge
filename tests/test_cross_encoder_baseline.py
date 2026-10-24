"""Offline neural-baseline tests; inference is simulated without model downloads."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_cross_encoder_baseline import execute, read_scores, total_budgets
from scripts.score_cross_encoder import execute as score
from src.baselines.cross_encoder import (CrossEncoderConfig, allocate_by_budget, binary_nll,
                                        fit_temperature, prediction_rows, split_calibration_test,
                                        text_hash, text_pool)
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_json


def fixture_manifest():
    return [{"dataset": "trec_dl_2019", "query_id": q, "doc_id": f"d{i}",
             "query_text": f"query {q}", "doc_text": f"passage {i}",
             "true_relevance": 2 * (i % 2), "true_binary": i % 2, "relevance_threshold": 2}
            for q in ("cal", "test") for i in range(4)]


class FakeEncoder:
    seen = []
    fail_after = None
    batches = 0

    def __init__(self, config, *args):
        self.provenance = {"backend": "synthetic_test_fixture", "resolved_revision": config.revision}

    def score_batch(self, rows):
        type(self).batches += 1
        if self.fail_after is not None and self.batches > self.fail_after:
            raise RuntimeError("simulated interruption")
        self.assert_no_gold(rows)
        self.seen.extend(pair_key(r) for r in rows)
        return [{"query_id": r["query_id"], "doc_id": r["doc_id"],
                 "raw_logit": [-3, 2, 1, -1][int(r["doc_id"][1:])],
                 "input_tokens": 8, "original_tokens": 8, "truncated": False,
                 "text_sha256": text_hash(r)} for r in rows]

    @staticmethod
    def assert_no_gold(rows):
        if any("true_binary" in r or "true_relevance" in r for r in rows):
            raise AssertionError("Gold passed to inference")


class CrossEncoderTests(unittest.TestCase):
    def setUp(self):
        self.config = CrossEncoderConfig()
        FakeEncoder.seen, FakeEncoder.batches, FakeEncoder.fail_after = [], 0, None

    def test_probabilities_use_raw_logits_once_and_remain_stable(self):
        pool = text_pool(fixture_manifest())[:4]
        rows = prediction_rows(pool, [-2000, 0, 2, 2000], 2)
        self.assertEqual(rows[0]["prob_relevant"], 0)
        self.assertEqual(rows[1]["prob_relevant"], .5)
        self.assertAlmostEqual(rows[2]["prob_relevant"], 1/(1+math.exp(-1)))
        self.assertEqual(rows[3]["prob_relevant"], 1)
        self.assertEqual([r["predicted_label"] for r in rows], [0, 1, 1, 1])
        self.assertTrue(math.isfinite(binary_nll([1000, -1000], [0, 1])))

    def test_temperature_matches_analytical_binary_optimum(self):
        fitted = fit_temperature([2, 2, 2, 2], [1, 1, 1, 0])
        self.assertAlmostEqual(fitted["temperature"], 2 / math.log(3), places=9)
        self.assertLess(fitted["calibration_nll_after"], fitted["calibration_nll_before"])
        self.assertFalse(fitted["at_boundary"])

    def test_query_weighting_has_expected_optimum(self):
        z, y, queries = [2]*5, [1, 1, 1, 0, 1], ["a"]*4 + ["b"]
        pairs = fit_temperature(z, y)
        query = fit_temperature(z, y, replace(self.config, calibration_weighting="query"), queries)
        self.assertAlmostEqual(pairs["temperature"], 2 / math.log(4), places=9)
        self.assertAlmostEqual(query["temperature"], 2 / math.log(7), places=9)

    def test_temperature_boundaries_and_zero_scores(self):
        correct = fit_temperature([-2, 2], [0, 1])
        wrong = fit_temperature([-2, 2], [1, 0])
        self.assertEqual(correct["temperature"], self.config.temperature_min)
        self.assertEqual(wrong["temperature"], self.config.temperature_max)
        self.assertTrue(correct["at_boundary"])
        self.assertTrue(wrong["at_boundary"])
        self.assertEqual(fit_temperature([0, 0], [0, 1])["temperature"], 1)

    def test_invalid_scores_labels_and_configuration_fail(self):
        for z, y in (([], []), ([float("nan")], [0]), ([1, 2], [1]), ([1], [2]),
                     ([float("inf")], [0])):
            with self.assertRaises(ValueError):
                fit_temperature(z, y)
        for kwargs in ({"revision": "main"}, {"batch_size": 0}, {"max_length": 513},
                       {"temperature_min": 2}, {"temperature_max": float("nan")},
                       {"calibration_weighting": "unknown"}):
            with self.assertRaises(ValueError):
                replace(self.config, **kwargs)

    def test_temperature_preserves_labels_and_exact_budget_ordering(self):
        pool = text_pool(fixture_manifest())[:4]
        selections = []
        labels = []
        for t in (.01, 1, 1000):
            rows = prediction_rows(pool, [-1000, 1, -2, 1000], t)
            selections.append(allocate_by_budget(rows, 2, 42))
            labels.append([r["predicted_label"] for r in rows])
        self.assertEqual(selections, [selections[0]]*3)
        self.assertEqual(set(selections[0]), {1, 2})
        self.assertEqual(labels, [labels[0]]*3)

    def test_allocation_does_not_consult_gold(self):
        rows = prediction_rows(text_pool(fixture_manifest())[:4], [-3, 2, 1, -1], 2)
        selected = allocate_by_budget(rows, 3, 42)
        self.assertEqual(selected, allocate_by_budget([{**r, "true_binary": 1, "true_relevance": 2} for r in rows], 3, 42))
        for budget in (0, 1, 3, 4):
            self.assertEqual(len(allocate_by_budget(rows, budget, 42)), budget)
        with self.assertRaises(ValueError):
            allocate_by_budget(rows, 5, 42)

    def test_split_and_audit_isolation(self):
        manifest = fixture_manifest() + [{**fixture_manifest()[0], "query_id": "audit"}]
        split = {"calibration_query_ids": ["cal"], "test_query_ids": ["test"], "audit_query_ids": ["audit"]}
        calibration, test = split_calibration_test(manifest, split)
        self.assertEqual({r["query_id"] for r in calibration}, {"cal"})
        self.assertEqual({r["query_id"] for r in test}, {"test"})
        for broken in ({**split, "test_query_ids": ["cal"]}, {**split, "test_query_ids": ["audit"]},
                       {**split, "test_query_ids": ["missing"]}, {**split, "calibration_query_ids": []}):
            with self.assertRaises(ValueError):
                split_calibration_test(manifest, broken)
        with self.assertRaises(ValueError):
            split_calibration_test(manifest)

    def test_separate_source_target_calibration(self):
        all_rows = fixture_manifest()
        source, target = all_rows[:4], [{**r, "dataset": "trec_covid", "relevance_threshold": 1,
                                        "true_relevance": r["true_binary"]} for r in all_rows[4:]]
        calibration, test = split_calibration_test(target, calibration_manifest=source)
        self.assertEqual(calibration, source)
        self.assertEqual(test, target)
        with self.assertRaises(ValueError):
            split_calibration_test(source, calibration_manifest=source)

    def test_total_budgets_charge_calibration_and_keep_endpoints(self):
        self.assertEqual(total_budgets(100, 30, [.29, .5], None, True), [29, 30, 50, 130])
        self.assertEqual(total_budgets(100, 30, None, [30, 129], False), [30, 129])
        for budgets in ([-1], [131], [True]):
            with self.assertRaises(ValueError):
                total_budgets(100, 30, None, budgets, False)

    def setup_files(self, root, manifest=None):
        manifest = fixture_manifest() if manifest is None else manifest
        atomic_text(root / "manifest.jsonl", "".join(json.dumps(r) + "\n" for r in manifest))
        write_json(root / "config.json", {**asdict(replace(self.config, batch_size=2)),
                                          "include_endpoints": True, "seeds": [42]})
        write_json(root / "split.json", {"calibration_query_ids": ["cal"], "test_query_ids": ["test"]})
        return argparse.Namespace(manifest=root / "manifest.jsonl", output_dir=root / "scores",
                                  config=root / "config.json", split_json=None, batch_size=None,
                                  device="cpu", cache_dir=root / "cache", local_files_only=True)

    def test_scoring_checkpoint_resumes_without_rescoring(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.setup_files(Path(tmp))
            FakeEncoder.fail_after = 1
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                score(args, FakeEncoder)
            self.assertEqual(len(FakeEncoder.seen), 2)
            FakeEncoder.fail_after = None
            rows = score(args, FakeEncoder)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(set(FakeEncoder.seen)), 8)
            self.assertEqual(len(FakeEncoder.seen), 8)
            def forbidden(*args):
                raise AssertionError("Complete cache tried loading model")
            score(args, forbidden)
            args.batch_size = 4
            with self.assertRaisesRegex(ValueError, "Experiment changed"):
                score(args, FakeEncoder)

    def test_score_cache_integrity_and_revision_checks(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.setup_files(Path(tmp))
            score(args, FakeEncoder)
            manifest = read_manifest(args.manifest)
            logits = read_scores(args.output_dir, args.manifest, manifest, self.config, manifest)
            self.assertEqual(len(logits), 8)
            with self.assertRaisesRegex(ValueError, "revision"):
                read_scores(args.output_dir, args.manifest, manifest,
                            replace(self.config, revision="a"*40), manifest)
            atomic_text(args.output_dir / "scores.csv", "corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_scores(args.output_dir, args.manifest, manifest, self.config, manifest)

    def test_end_to_end_calibration_inclusive_costs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            score_args = self.setup_files(root)
            score(score_args, FakeEncoder)
            run_files = [root / "a.run", root / "b.run"]
            for path, order in zip(run_files, ([0, 1, 2, 3], [3, 2, 1, 0])):
                atomic_text(path, "".join(f"test Q0 d{i} {rank+1} {4-rank} {path.stem}\n" for rank, i in enumerate(order)))
            args = argparse.Namespace(manifest=score_args.manifest, scores=score_args.output_dir,
                                      calibration_manifest=None, calibration_scores=None,
                                      config=score_args.config, split_json=root / "split.json",
                                      output_dir=root / "results", seeds=[42], budgets=[0, 3, 4, 7, 8],
                                      budget_fractions=None, run_files=run_files)
            summaries = execute(args)
            self.assertEqual([r["status"] for r in summaries[:2]], ["infeasible_calibration_budget"]*2)
            for point in summaries[2:]:
                self.assertEqual(point["total_human_labels"], point["budget"])
                self.assertEqual(point["test_human_labels"], point["budget"] - 4)
            self.assertEqual(summaries[-1]["coverage"], 0)
            self.assertIsNone(summaries[-1]["selective_risk"])
            directory = root / "results/seed_42/budget_000007"
            predictions, _ = read_predictions(directory / "predictions.csv")
            self.assertEqual(len(predictions), 4)
            self.assertTrue((directory / "system_scores.csv").is_file())
            self.assertEqual(len((directory / "mixed.qrels").read_text().splitlines()), 4)
            for line in (root / "results/checksums.sha256").read_text().splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(file_hash(root / "results" / name), digest)
            with self.assertRaises(FileExistsError):
                execute(args)

    def test_test_gold_changes_do_not_change_temperature_or_allocation(self):
        fits, allocations = [], []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for index in range(2):
                root = Path(tmp) / str(index)
                root.mkdir()
                manifest = fixture_manifest()
                if index:
                    manifest = [{**r, "true_binary": 1-r["true_binary"], "true_relevance": 2*(1-r["true_binary"])}
                                if r["query_id"] == "test" else r for r in manifest]
                score_args = self.setup_files(root, manifest)
                score(score_args, FakeEncoder)
                args = argparse.Namespace(manifest=score_args.manifest, scores=score_args.output_dir,
                                          calibration_manifest=None, calibration_scores=None,
                                          config=score_args.config, split_json=root / "split.json",
                                          output_dir=root / "results", seeds=[42], budgets=[6],
                                          budget_fractions=None, run_files=None)
                execute(args)
                fits.append(json.loads((args.output_dir / "temperature.json").read_text()))
                predictions, _ = read_predictions(args.output_dir / "seed_42/budget_000006/predictions.csv")
                allocations.append([(pair_key(r), r["label_source"], r["prob_relevant"]) for r in predictions])
        self.assertEqual(fits[0], fits[1])
        self.assertEqual(allocations[0], allocations[1])


if __name__ == "__main__":
    unittest.main()
