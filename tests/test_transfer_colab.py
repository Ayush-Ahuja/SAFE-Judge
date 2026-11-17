"""Offline checks for transfer task semantics, durable resume and packaging."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts.run_transfer_inference import execute
from scripts.transfer_colab_support import archive_results, restore_run, safe_extract, verify_bundle
from src.assessor import prompts, transfer_prompts as transfer
from src.utils.io import atomic_text, read_predictions


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))
    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"


def row(name="dl2020", did="1", grade=2):
    threshold = transfer.TASKS[name]
    return {"dataset": name, "rubric_name": name, "query_id": "q", "doc_id": did,
            "query_text": "What causes rain?", "doc_text": "Water condenses in clouds.",
            "true_relevance": grade, "true_binary": int(grade >= threshold), "relevance_threshold": threshold}


class TransferTests(unittest.TestCase):
    def test_explicit_task_rubrics_match_targets_and_never_consult_gold(self):
        tokenizer = Tokenizer()
        for name in transfer.TASKS:
            original = row(name)
            changed = {**original, "true_relevance": 0, "true_binary": 0, "human_uncertain": True}
            self.assertEqual(transfer.render_prompt(tokenizer, original, original["doc_text"]),
                             transfer.render_prompt(tokenizer, changed, changed["doc_text"]))
        covid = transfer.render_prompt(tokenizer, row("trec_covid"), "text")
        self.assertIn("grade 1 (partially relevant) are No", covid)
        chat = transfer.render_prompt(tokenizer, row("chatreport"), "text")
        self.assertIn("climate disclosure", chat)
        self.assertNotIn("TREC-COVID", chat)
        with self.assertRaisesRegex(ValueError, "matching"):
            transfer.validate_task({**row("trec_covid"), "relevance_threshold": 1})

    def test_dl_prompt_and_truncation_match_frozen_scorer_and_shared_is_identical(self):
        tokenizer = Tokenizer()
        original = {**row(), "doc_text": "z" * 1000}
        empty = transfer.render_prompt(tokenizer, original, "")
        budget = len(empty) + 40
        self.assertEqual(transfer.encode_pair(tokenizer, original, budget),
                         prompts.encode_pair(tokenizer, original, budget))
        ids, info = transfer.encode_pair(tokenizer, original, budget)
        self.assertTrue(info["truncated"])
        self.assertIn(original["query_text"], tokenizer.decode(ids))
        shared = [transfer.render_prompt(tokenizer, {**row(name), "rubric_mode": "shared"}, "text")
                  for name in transfer.TASKS]
        self.assertEqual(len(set(shared)), 1)
        with self.assertRaisesRegex(ValueError, "token budget"):
            transfer.encode_pair(tokenizer, row("chatreport"), 1)

    def test_interruption_restores_drive_snapshot_and_scores_only_remaining_pairs(self):
        calls, fail = [], [True]
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
                if fail[0] and len(calls) >= 2:
                    raise RuntimeError("simulated disconnect")
                calls.extend(r["doc_id"] for r in rows)
                return [{"query_id": r["query_id"], "doc_id": r["doc_id"],
                         "true_relevance": r["true_relevance"], "true_binary": r["true_binary"],
                         "prob_relevant": .8, "predicted_label": 1, "truncated": False,
                         "compute_dtype": self.compute_dtype} for r in rows]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            base = Path(tmp)
            manifest = base / "pairs.jsonl"
            rows = [row("chatreport", str(i), i % 3) for i in range(5)]
            atomic_text(manifest, "".join(json.dumps(r) + "\n" for r in rows))
            out = base / "vm1"
            out.mkdir()
            args = SimpleNamespace(output_dir=out, manifest=manifest, limit=0, backup_dir=base / "drive")
            config = {"model_name": "test/model", "revision": "pinned", "max_length": 512,
                      "batch_size": 2, "rubric_mode": "dataset"}
            with patch("src.assessor.transfer_model.TransferAssessor", Assessor), \
                 patch("src.assessor.model.resolve_revision", return_value="commit"), \
                 patch("importlib.metadata.version", return_value="test"):
                with self.assertRaisesRegex(RuntimeError, "disconnect"):
                    execute(args, {"seed": 42}, config, 2, rows)
                self.assertEqual(len(read_predictions(args.backup_dir / "predictions.csv", 1)[0]), 2)
                fail[0] = False
                args.output_dir = base / "vm2"
                restore_run(args.output_dir, args.backup_dir)
                self.assertTrue(execute(args, {"seed": 42}, config, 1, rows))
                self.assertTrue(execute(args, {"seed": 42}, config, 1, rows))
                self.assertEqual(calls, ["0", "1", "2", "3", "4"])
                self.assertTrue(json.loads((args.backup_dir / "progress.json").read_text())["complete"])
                with self.assertRaisesRegex(ValueError, "Experiment changed"):
                    execute(args, {"seed": 42}, {**config, "rubric_mode": "shared"}, 1, rows)

    def test_bundle_paths_hashes_and_result_archive_preserve_partial_status(self):
        from scripts.transfer_colab_support import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evil = root / "bad.zip"
            with zipfile.ZipFile(evil, "w") as z:
                z.writestr("../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                safe_extract(evil, root / "extract")
            bundle = root / "bundle"
            (bundle / "data/transfer").mkdir(parents=True)
            (bundle / "data/transfer/suite.json").write_text("{}")
            receipt = {"data/transfer/suite.json": sha256(bundle / "data/transfer/suite.json")}
            (bundle / "bundle_hashes.json").write_text(json.dumps(receipt))
            self.assertEqual(verify_bundle(bundle), 1)
            saved = root / "saved"
            saved.mkdir()
            (saved / "progress.json").write_text(json.dumps({"complete": False, "completed_pairs": 2}))
            archive = archive_results(bundle, saved, root / "results.zip")
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(json.loads(z.read("runs/progress.json"))["complete"])
                self.assertIn("inputs/suite.json", z.namelist())
            (bundle / "data/transfer/suite.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
