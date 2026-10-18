"""Offline regression checks for the errors that affected the original pilot."""

from collections import namedtuple
from contextlib import chdir, closing, redirect_stdout
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.download_data import export_dataset
from scripts.compare_predictions import compare
from scripts.evaluate_pilot import validate_against_manifest
from scripts.export_failures import sample_failures
from scripts.prepare_pilot import reserve_audit_topics, select_qrels
from scripts.run_pilot import execute, run_lock
from src.assessor.prompts import answer_token_ids, encode_pair, render_prompt
from src.evaluation.metrics import classification_metrics, risk_coverage_table, selection_metrics
from src.utils.data_loader import select_text
from src.utils.io import PredictionStore, atomic_text, read_manifest, read_predictions, write_csv


def manifest_row(qid="q1", did="d1", grade=2):
    return {"dataset": "trec_dl_2019", "query_id": qid, "doc_id": did,
            "query_text": "Why is the sky blue?", "doc_text": "Scattering of sunlight.",
            "true_relevance": grade, "true_binary": int(grade >= 2), "relevance_threshold": 2}


class PilotTests(unittest.TestCase):
    def test_rerun_comparison_distinguishes_probability_and_label_changes(self):
        first = [{"query_id": "q", "doc_id": "a", "true_relevance": 2,
                  "prob_relevant": .7, "predicted_label": 1},
                 {"query_id": "q", "doc_id": "b", "true_relevance": 0,
                  "prob_relevant": .49, "predicted_label": 0}]
        repeat = [{**first[0], "prob_relevant": .71},
                  {**first[1], "prob_relevant": .51, "predicted_label": 1}]
        result = compare(first, repeat)
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["changed_labels"], 1)
        self.assertEqual(result["label_agreement"], .5)
        self.assertAlmostEqual(result["mean_absolute_probability_difference"], .015)
        self.assertAlmostEqual(result["max_absolute_probability_difference"], .02)
        with self.assertRaisesRegex(ValueError, "same nonempty pair set"):
            compare(first, repeat[:1])
        with self.assertRaisesRegex(ValueError, "Gold labels"):
            compare(first, [{**repeat[0], "true_relevance": 0}, repeat[1]])

    def test_ties_are_counted_once_and_empty_risk_is_undefined(self):
        rows = [{"query_id": "q", "doc_id": str(i), "prob_relevant": p,
                 "predicted_label": int(p >= .5), "true_binary": gold, "confidence": max(p, 1-p)}
                for i, (p, gold) in enumerate([(.5, 1), (.8, 0), (.1, 0)])]
        curves = risk_coverage_table(rows)
        combined = next(r for r in curves if r["group"] == "all" and r["threshold"] == .5)
        self.assertEqual(combined["coverage"], 1)
        class_counts = [r["accepted"] for r in curves if r["threshold"] == .5 and r["group"] != "all"]
        self.assertEqual(sum(class_counts), 3)
        self.assertIsNone(selection_metrics(rows, .95)["selective_risk"])
        metrics = classification_metrics(rows)
        self.assertAlmostEqual(metrics["accuracy"], 2/3)
        self.assertAlmostEqual(metrics["macro_f1"], 2/3)
        self.assertAlmostEqual(metrics["brier_score"], .3)

    def test_csv_rejects_duplicates_nan_and_mismatched_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            row = {"query_id": "001", "doc_id": "002", "true_relevance": 2,
                   "predicted_label": 1, "prob_relevant": .7}
            write_csv(path, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_predictions(path)
            rows, duplicates = read_predictions(path, deduplicate_last=True)
            self.assertEqual(duplicates, 1)
            self.assertEqual(rows[0]["query_id"], "001")
            write_csv(path, [{**row, "prob_relevant": "nan"}])
            with self.assertRaises(ValueError):
                read_predictions(path)
            write_csv(path, [{**row, "true_binary": 0}])
            with self.assertRaisesRegex(ValueError, "gold"):
                read_predictions(path)

    def test_manifest_and_completeness_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            row = manifest_row()
            atomic_text(path, json.dumps(row) + "\n")
            self.assertEqual(read_manifest(path), [row])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                validate_against_manifest([], [row])
            self.assertEqual(validate_against_manifest([], [row], True), 1)
            atomic_text(path, (json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_manifest(path)

    def test_checkpoint_batch_is_atomic_and_experiment_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.sqlite3"
            store = PredictionStore(path, {"revision": "abc"})
            first = {"query_id": "q", "doc_id": "d"}
            store.add_batch([first])
            with self.assertRaises(sqlite3.IntegrityError):
                store.add_batch([{"query_id": "q", "doc_id": "new"}, first])
            self.assertEqual(store.rows(), [first])
            store.close()
            with self.assertRaisesRegex(ValueError, "Experiment changed"):
                PredictionStore(path, {"revision": "changed"})

    def test_sampling_excludes_unjudged_and_blind_audit_topics(self):
        Qrel = namedtuple("Qrel", "query_id doc_id relevance")
        qrels = [Qrel(str(i), str(j), j % 3) for i in range(20) for j in range(4)]
        qrels.append(Qrel("0", "unjudged", -1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            excluded = reserve_audit_topics([str(i) for i in range(20)], path)
            selected = select_qrels(qrels, 42, 5, 2, excluded)
            self.assertEqual(len(selected), 10)
            self.assertFalse({q for q, _, _ in selected} & excluded)
            self.assertNotIn("unjudged", [d for _, d, _ in selected])
            self.assertEqual(selected, select_qrels(qrels, 42, 5, 2, excluded))
            with self.assertRaisesRegex(ValueError, "blind-audit"):
                reserve_audit_topics([str(i) for i in range(20)], path, seed=43)

    def test_export_preserves_structured_covid_queries_and_rejects_empty_exports(self):
        Query = namedtuple("Query", "query_id title description narrative")
        Qrel = namedtuple("Qrel", "query_id doc_id relevance")
        dataset = SimpleNamespace(queries_iter=lambda: iter([Query("1", "Origin", "Origin of COVID", "Do not include")]),
                                  qrels_iter=lambda: iter([Qrel("1", "d", 2)]))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            export_dataset(dataset, "trec_covid", tmp)
            path = Path(tmp) / "trec_covid/queries.json"
            self.assertEqual(json.loads(path.read_text())["1"], "Origin\nOrigin of COVID")
            dataset.queries_iter = lambda: iter([])
            with self.assertRaisesRegex(ValueError, "No queries"):
                export_dataset(dataset, "trec_covid", tmp)
            self.assertEqual(json.loads(path.read_text())["1"], "Origin\nOrigin of COVID")

    def test_failure_sample_is_unique_reproducible_and_requires_enough_errors(self):
        rows = [{"query_id": str(i % 10), "doc_id": str(i), "true_binary": i % 2,
                 "predicted_label": 1 - i % 2, "confidence": .7 if i % 3 else .95} for i in range(100)]
        selected = sample_failures(rows)
        self.assertEqual(len({r["doc_id"] for r in selected}), 50)
        self.assertEqual(selected, sample_failures(rows))
        self.assertGreater(len({r["query_id"] for r in selected}), 1)
        with self.assertRaisesRegex(ValueError, "Fewer"):
            sample_failures(rows[:10])

    def test_document_only_truncation_preserves_rubric_query_and_answer_boundary(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(map(ord, text))
            def decode(self, ids, **kwargs):
                return "".join(map(chr, ids))
            def apply_chat_template(self, messages, **kwargs):
                self.thinking = kwargs["enable_thinking"]
                return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"
        tokenizer = Tokenizer()
        row = {**manifest_row(), "doc_text": "z" * 1000}
        budget = len(tokenizer.encode(render_prompt(tokenizer, row["query_text"], "", 2))) + 30
        ids, info = encode_pair(tokenizer, row, budget)
        text = tokenizer.decode(ids)
        self.assertLessEqual(len(ids), budget)
        self.assertIn(row["query_text"], text)
        self.assertIn("grades 0", text)
        self.assertTrue(text.endswith("ASSISTANT:"))
        self.assertTrue(info["truncated"])
        self.assertFalse(tokenizer.thinking)
        with self.assertRaisesRegex(ValueError, "token budget"):
            encode_pair(tokenizer, row, 5)
        with self.assertRaisesRegex(ValueError, "exactly one token"):
            answer_token_ids(tokenizer, ids)

    def test_run_lock_prevents_two_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with run_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "exists"):
                    with run_lock(directory):
                        self.fail("Second writer acquired the lock")
            self.assertFalse((directory / "run.lock").exists())

    def test_interrupted_job_resumes_without_rescoring_cached_pairs(self):
        scored, fail = [], [True]
        cuda = SimpleNamespace(get_device_name=lambda _: "fake GPU", synchronize=lambda: None,
                               get_device_properties=lambda _: SimpleNamespace(total_memory=123),
                               reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 12,
                               max_memory_reserved=lambda: 16)
        class Assessor:
            torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="test"))
            compute_dtype = "float16"
            def __init__(self, *args):
                pass
            def score_batch(self, rows):
                if fail[0] and len(scored) >= 2:
                    raise RuntimeError("simulated OOM")
                scored.extend(r["doc_id"] for r in rows)
                return [{"query_id": r["query_id"], "doc_id": r["doc_id"],
                         "true_relevance": r["true_relevance"], "true_binary": r["true_binary"],
                         "prob_relevant": .8, "predicted_label": 1, "truncated": False,
                         "compute_dtype": self.compute_dtype} for r in rows]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            manifest = directory / "pairs.jsonl"
            rows = [manifest_row(did=str(i)) for i in range(5)]
            atomic_text(manifest, "".join(json.dumps(r) + "\n" for r in rows))
            args = SimpleNamespace(output_dir=directory, manifest=manifest, limit=0)
            assessor = {"model_name": "test/model", "revision": "main", "max_length": 512, "batch_size": 2}
            with patch("src.assessor.model.QwenAssessor", Assessor), patch("src.assessor.model.resolve_revision", return_value="commit"), patch("importlib.metadata.version", return_value="test"):
                with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
                    execute(args, {"seed": 42}, assessor, 2, rows)
                self.assertEqual(len(read_predictions(directory / "predictions.csv")[0]), 2)
                fail[0] = False
                execute(args, {"seed": 42}, assessor, 1, rows)
                execute(args, {"seed": 42}, assessor, 1, rows)
                self.assertEqual(scored, ["0", "1", "2", "3", "4"])
                saved, _ = read_predictions(directory / "predictions.csv")
                self.assertEqual(len(saved), 5)
                sessions = json.loads((directory / "sessions.json").read_text())
                self.assertEqual([s["status"] for s in sessions], ["interrupted", "complete", "complete"])
                self.assertEqual(sessions[0]["complete"], False)
                self.assertEqual(sessions[-1]["complete"], True)
                with self.assertRaisesRegex(ValueError, "Experiment changed"):
                    execute(args, {"seed": 42}, {**assessor, "max_length": 1024}, 1, rows)


class Stage2WorkflowTests(unittest.TestCase):
    def test_batch_benchmark_selects_fastest_fitting_size(self):
        from scripts.run_pilot import benchmark_batches
        class OOM(RuntimeError):
            pass
        clock = [0.0]
        cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                               reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 10,
                               OutOfMemoryError=OOM)
        class Model:
            torch = SimpleNamespace(cuda=cuda)
            def score_batch(self, rows):
                if len(rows) > 2:
                    raise OOM()
                clock[0] += 1
        rows = [{"query_text": "query", "doc_text": "document" * i} for i in range(1, 6)]
        with patch("scripts.run_pilot.time.perf_counter", side_effect=lambda: clock[0]):
            selected, results = benchmark_batches(Model(), rows, 4)
        self.assertEqual(selected, 2)
        self.assertEqual(results[-1]["status"], "out_of_memory")

    def test_query_bootstrap_and_calibration_use_queries(self):
        import math
        from src.evaluation.metrics import query_bootstrap, temperature_calibration
        rows = []
        for q in range(4):
            for gold, probability in ((0, .1), (1, .9)):
                rows.append({"query_id": str(q), "doc_id": str(gold), "true_binary": gold,
                             "true_relevance": gold * 2, "predicted_label": gold,
                             "prob_relevant": probability, "confidence": .9,
                             "no_logit": math.log(1 - probability), "yes_logit": math.log(probability)})
        result = query_bootstrap(rows, samples=100, seed=42)
        self.assertEqual(result["queries"], 4)
        self.assertEqual(result["intervals"]["accuracy"], {"lower": 1.0, "upper": 1.0})
        test, calibrated, split = temperature_calibration(rows, calibration_queries=1)
        self.assertFalse(set(split["calibration_query_ids"]) & set(split["test_query_ids"]))
        changed = [{**r, "true_binary": 1 - r["true_binary"]} if r["query_id"] in split["test_query_ids"] else r for r in rows]
        _, _, other = temperature_calibration(changed, calibration_queries=1)
        self.assertEqual(split["temperature"], other["temperature"])
        self.assertEqual(len(test), len(calibrated))

    def test_oom_backoff_snapshot_and_runtime_resume(self):
        class OOM(RuntimeError):
            pass
        calls = []
        cuda = SimpleNamespace(get_device_name=lambda _: "test GPU", synchronize=lambda: None,
                               get_device_properties=lambda _: SimpleNamespace(total_memory=1000),
                               reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 10,
                               max_memory_reserved=lambda: 20, empty_cache=lambda: None, OutOfMemoryError=OOM)
        class Assessor:
            torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="test"))
            compute_dtype = "float16"
            def __init__(self, *args):
                pass
            def score_batch(self, rows):
                if len(rows) > 1:
                    raise OOM("simulated CUDA OOM")
                calls.extend(r["doc_id"] for r in rows)
                return [{"query_id": r["query_id"], "doc_id": r["doc_id"], "true_relevance": r["true_relevance"],
                         "true_binary": r["true_binary"], "predicted_label": 1, "prob_relevant": .8,
                         "confidence": .8, "truncated": False, "compute_dtype": "float16"} for r in rows]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp)
            out = base / "working"
            out.mkdir()
            manifest = base / "manifest.jsonl"
            rows = [manifest_row(did=str(i)) for i in range(5)]
            atomic_text(manifest, "".join(json.dumps(r) + "\n" for r in rows))
            args = SimpleNamespace(output_dir=out, manifest=manifest, limit=0, auto_batch=True, max_batch_size=2,
                                   max_seconds=0, backup_dir=base / "saved", backup_every_seconds=1, consistency_pairs=2)
            config = {"model_name": "test/model", "revision": "main", "max_length": 512, "batch_size": 2}
            timings = [{"batch_size": 1, "pairs_per_second": 1}, {"batch_size": 2, "pairs_per_second": 2}]
            with patch("src.assessor.model.QwenAssessor", Assessor), patch("src.assessor.model.resolve_revision", return_value="commit"), patch("importlib.metadata.version", return_value="test"), patch("scripts.run_pilot.benchmark_batches", return_value=(2, timings)):
                self.assertTrue(execute(args, {"seed": 42}, config, 2, rows))
                self.assertEqual(calls, ["0", "1", "2", "3", "4", "0", "1"])
                check = json.loads((out / "consistency.json").read_text())
                self.assertEqual(check["label_agreement"], 1)
                self.assertEqual(check["completed_pairs"], 2)
                with closing(sqlite3.connect(base / "saved/checkpoint.sqlite3")) as connection:
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 5)
                self.assertEqual(json.loads((out / "progress.json").read_text())["oom_backoffs"][0]["new_batch_size"], 1)
                # A new output can stop before scoring, then resume without changing identity.
                args.output_dir = base / "limited"
                args.output_dir.mkdir()
                args.backup_dir = base / "limited_saved"
                args.max_seconds = .01
                clock = iter(i / 10 for i in range(1000))
                with patch("scripts.run_pilot.time.perf_counter", side_effect=lambda: next(clock)):
                    self.assertFalse(execute(args, {"seed": 42}, config, 2, rows))
                self.assertEqual(json.loads((args.output_dir / "progress.json").read_text())["status"], "time_budget_exhausted")
                args.max_seconds = 0
                self.assertTrue(execute(args, {"seed": 42}, config, 2, rows))
                with patch("importlib.metadata.version", return_value="different"):
                    with self.assertRaisesRegex(ValueError, "Experiment changed"):
                        execute(args, {"seed": 42}, config, 1, rows)

    def test_complete_evaluation_report_and_review_preservation(self):
        import math
        import subprocess
        import sys
        from scripts.report_stage2 import build_report
        from src.utils.io import file_hash, write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "run"
            out.mkdir()
            manifest = root / "manifest.jsonl"
            rows, scored = [], []
            for index in range(4):
                gold = index % 2
                row = {**manifest_row(qid=str(index // 2), did=str(index)), "true_relevance": gold * 2, "true_binary": gold}
                rows.append(row)
                probability = .9 if gold == 0 else .1
                scored.append({"query_id": row["query_id"], "doc_id": row["doc_id"], "true_relevance": gold * 2,
                               "true_binary": gold, "predicted_label": 1 - gold, "prob_relevant": probability,
                               "confidence": .9, "no_logit": math.log(1 - probability), "yes_logit": math.log(probability),
                               "compute_dtype": "float16", "answer_probability_mass": .8, "truncated": False})
            atomic_text(manifest, "".join(json.dumps(r) + "\n" for r in rows))
            write_csv(out / "predictions.csv", scored)
            write_json(out / "experiment.json", {"manifest_sha256": file_hash(manifest), "dataset": "trec_dl_2019",
                       "relevance_threshold": 2, "resolved_revision": "test", "assessor": {}})
            cli = lambda *args: subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True, text=True)
            cli("scripts/evaluate_pilot.py", "--manifest", manifest, "--predictions", out / "predictions.csv",
                "--output-dir", out / "evaluation", "--bootstrap-samples", 100, "--calibration-queries", 1)
            review = out / "failure_review.csv"
            cli("scripts/export_failures.py", "--manifest", manifest, "--predictions", out / "predictions.csv", "--output", review, "--count", 2)
            with review.open(newline="", encoding="utf-8") as stream:
                annotations = list(csv.DictReader(stream))
            annotations[0].update(failure_category="other", notes="Read manually", reviewer="tester", reviewed="yes")
            write_csv(review, annotations)
            before = review.read_bytes()
            cli("scripts/export_failures.py", "--manifest", manifest, "--predictions", out / "predictions.csv", "--output", review, "--count", 2, "--reuse-existing")
            self.assertEqual(before, review.read_bytes())
            status = build_report(manifest, out, review)
            self.assertTrue(status["complete_inference"])
            self.assertEqual(status["manual_analysis"]["reviewed"], 1)
            self.assertTrue(status["pending_resource_fields"])
            self.assertTrue((out / "stage2_report.md").exists())
            annotations.append(annotations[0])
            write_csv(review, annotations)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                build_report(manifest, out, review)


class ColabNotebookTests(unittest.TestCase):
    def test_storage_restore_manual_review_and_archive(self):
        import os
        import sys
        import types
        import zipfile
        from src.utils.io import write_json
        source_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((source_root / "notebooks/stage2_colab.ipynb").read_text(encoding="utf-8"))
        code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        storage_code = next(c for c in code_cells if "def persist():" in c)
        export_code = code_cells[-1]
        old_cwd, old_path = Path.cwd(), sys.path[:]
        try:
            with tempfile.TemporaryDirectory() as tmp, chdir(old_cwd):
                base = Path(tmp)
                archive_bytes = io.BytesIO()
                with zipfile.ZipFile(archive_bytes, "w") as archive:
                    for name in ("src/utils/io.py", "configs/default.yaml", "requirements-pilot.txt"):
                        archive.write(source_root / name, name)
                downloads = []
                colab = types.ModuleType("google.colab")
                colab.drive = SimpleNamespace(mount=lambda *_: None)
                colab.files = SimpleNamespace(upload=lambda: {"pilot_code.zip": archive_bytes.getvalue()}, download=downloads.append)
                google = types.ModuleType("google")
                google.__path__ = []
                google.colab = colab
                def transform(code, work):
                    return (code.replace('Path("/content/SAFE-Judge")', f'Path({str(work)!r})')
                            .replace('Path("/content/drive/MyDrive/SAFE-Judge")', f'Path({str(base / "drive")!r})')
                            .replace('Path("/content/stage2_artifacts.zip")', f'Path({str(base / "artifacts.zip")!r})'))
                ns = {"Path": Path, "RUN": "test_run", "UPLOAD_CODE": False, "os": os, "sys": sys,
                      "RESOURCE_PLAN": {}, "write_json": write_json}
                with patch.dict(sys.modules, {"google": google, "google.colab": colab}), redirect_stdout(io.StringIO()):
                    exec(transform(storage_code, base / "first_vm"), ns)
                    output, backup = ns["output"], ns["backup"]
                    experiment = {"manifest_sha256": "test"}
                    write_json(output / "experiment.json", experiment)
                    store = PredictionStore(output / "checkpoint.sqlite3", experiment)
                    store.add_batch([{"query_id": "q", "doc_id": "d"}])
                    store.close()
                    ns["manifest"].parent.mkdir(parents=True, exist_ok=True)
                    ns["manifest"].write_text("manifest")
                    ns["config_path"].write_text("config")
                    ns["review"].write_text("blank review")
                    ns["persist"]()
                    (backup / "failure_review.csv").write_text("human annotations")
                    ns["persist"]()
                    self.assertEqual((backup / "failure_review.csv").read_text(), "human annotations")
                    exec(transform(storage_code, base / "second_vm"), ns)
                    self.assertEqual(ns["review"].read_text(), "human annotations")
                    with closing(sqlite3.connect(ns["output"] / "checkpoint.sqlite3")) as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 1)
                    exec(transform(export_code, base / "second_vm"), ns)
                    self.assertTrue(downloads)
                    with zipfile.ZipFile(downloads[-1]) as archive:
                        self.assertIsNone(archive.testzip())
                        self.assertEqual(archive.read("results/stage2/test_run/failure_review.csv"), b"human annotations")
                        self.assertIn("configs/default.yaml", archive.namelist())
                    os.chdir(old_cwd)
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
