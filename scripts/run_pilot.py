"""Run from any working directory. One output directory belongs to one experiment."""

import argparse
import gc
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import PredictionStore, file_hash, pair_key, read_manifest, write_csv, write_json

PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers", "huggingface-hub")


def benchmark_batches(model, rows, maximum):
    """Choose throughput using long inputs, without consulting relevance labels."""
    torch = model.torch
    sample = sorted(rows, key=lambda r: len(r["query_text"]) + len(r["doc_text"]), reverse=True)[:32]
    model.score_batch(sample[:1])
    torch.cuda.synchronize()
    measurements = []
    size = 1
    while size <= maximum:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        failed = False
        try:
            for start in range(0, len(sample), size):
                model.score_batch(sample[start:start + size])
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            failed = True
        if failed:
            gc.collect()
            torch.cuda.empty_cache()
            measurements.append({"batch_size": size, "status": "out_of_memory"})
            break
        elapsed = time.perf_counter() - started
        measurements.append({"batch_size": size, "status": "complete", "pairs": len(sample),
                             "seconds": elapsed, "pairs_per_second": len(sample) / elapsed,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        size *= 2
    successful = [r for r in measurements if r["status"] == "complete"]
    if not successful:
        raise RuntimeError("The model does not fit at batch size 1 on this GPU")
    selected = max(successful, key=lambda r: r["pairs_per_second"])["batch_size"]
    return selected, measurements


@contextmanager
def run_lock(directory):
    path = directory / "run.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"{path} exists. Check that no job is running before removing a stale lock.") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
        yield
    finally:
        path.unlink()


def code_hash():
    files = ["scripts/run_pilot.py", "src/assessor/model.py", "src/assessor/prompts.py", "src/utils/io.py"]
    return hashlib.sha256("".join(file_hash(ROOT / name) for name in files).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--batch-size", type=int, help="May be lowered when resuming after OOM")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: score the first N frozen pairs")
    parser.add_argument("--validate-only", action="store_true", help="No GPU/download/dependencies required")
    parser.add_argument("--auto-batch", action="store_true", help="Benchmark batches and reduce the batch after CUDA OOM")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop and export progress after this runtime; 0 disables the limit")
    parser.add_argument("--backup-dir", type=Path, help="Periodic durable snapshots, e.g. a mounted Drive folder")
    parser.add_argument("--backup-every-seconds", type=float, default=120)
    parser.add_argument("--consistency-pairs", type=int, default=0, help="Repeat this many completed pairs at batch size 1")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.max_batch_size < 1 or args.max_seconds < 0 or args.backup_every_seconds <= 0 or args.consistency_pairs < 0:
        parser.error("Invalid batch, runtime, backup interval or consistency count")
    if args.backup_dir and args.output_dir and args.backup_dir.resolve() == args.output_dir.resolve():
        parser.error("Backup and working output directories must differ")
    if args.limit:
        rows = rows[:args.limit]
    print(f"Selected {len(rows)} pairs across {len({r['query_id'] for r in rows})} queries", flush=True)
    if args.validate_only:
        print(f"Manifest SHA256: {file_hash(args.manifest)}")
        return
    if not args.output_dir:
        parser.error("--output-dir is required for inference")
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assessor = config["assessor"]
    batch_size = args.batch_size if args.batch_size is not None else assessor["batch_size"]
    if batch_size < 1 or assessor["max_length"] < 1:
        parser.error("batch size and max_length must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output_dir):
        complete = execute(args, config, assessor, batch_size, rows)
    if not complete:
        raise SystemExit(3)


def execute(args, config, assessor, batch_size, rows):
    from src.assessor.model import QwenAssessor, resolve_revision
    experiment_path = args.output_dir / "experiment.json"
    old = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else None
    requested_revision = assessor.get("revision", "main")
    same_model = old and old["assessor"]["model_name"] == assessor["model_name"] and old["assessor"].get("revision", "main") == requested_revision
    revision = old["resolved_revision"] if same_model else resolve_revision(assessor["model_name"], requested_revision)
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    experiment = {
        "schema_version": 2, "manifest_sha256": file_hash(args.manifest),
        "pipeline_sha256": code_hash(), "resolved_revision": revision, "packages": packages,
        "assessor": {k: v for k, v in assessor.items() if k != "batch_size"},
        "seed": config.get("seed", 42), "limit": args.limit, "expected_pairs": len(rows),
        "dataset": rows[0]["dataset"], "relevance_threshold": rows[0]["relevance_threshold"],
        "tie_rule": "p_relevant >= 0.5 is relevant", "score": "softmax over canonical No/Yes answer tokens",
    }
    if old and old != experiment:
        raise ValueError("Experiment changed. Select a new output directory.")
    backup_dir = getattr(args, "backup_dir", None)
    max_seconds = getattr(args, "max_seconds", 0)
    auto_batch = getattr(args, "auto_batch", False)
    store = PredictionStore(args.output_dir / "checkpoint.sqlite3", experiment)
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "batch_size": batch_size,
               "python": sys.version, "platform": platform.platform(), "status": "running", "packages": packages}
    started, inference_started, inference_elapsed = time.perf_counter(), None, None
    cpu_started = time.process_time()
    session_rows = 0
    model = None
    sessions_path = args.output_dir / "sessions.json"
    history = json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []

    def save_progress(final=False):
        saved = store.rows()
        session.update({"session_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
                        "new_predictions": session_rows, "completed_pairs": len(saved),
                        "expected_pairs": len(rows), "complete": len(saved) == len(rows),
                        "truncated_pairs_total": sum(bool(r["truncated"]) for r in saved),
                        "last_saved_utc": datetime.now(timezone.utc).isoformat()})
        if final:
            session["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if inference_started is not None:
            elapsed = inference_elapsed if inference_elapsed is not None else time.perf_counter() - inference_started
            session.update({"inference_seconds": elapsed, "pairs_per_second": session_rows / elapsed if elapsed else None})
        if saved:
            write_csv(args.output_dir / "predictions.csv", saved)
        write_json(args.output_dir / "progress.json", session)
        write_json(sessions_path, history + [session])
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            store.backup(backup_dir / "checkpoint.sqlite3")
            write_json(backup_dir / "experiment.json", experiment)
            write_json(backup_dir / "sessions.json", history + [session])
            write_json(backup_dir / "progress.json", session)
            if saved:
                write_csv(backup_dir / "predictions.csv", saved)
            for name in ("batch_benchmark.json", "consistency.json", "consistency.csv"):
                source = args.output_dir / name
                if source.exists():
                    from src.utils.io import atomic_text
                    atomic_text(backup_dir / name, source.read_text(encoding="utf-8"))
        return saved

    try:
        write_json(experiment_path, experiment)
        cached = store.rows()
        manifest_map = {pair_key(row): row for row in rows}
        for row in cached:
            original = manifest_map.get(pair_key(row))
            if original is None or original["true_relevance"] != row["true_relevance"]:
                raise ValueError("Checkpoint contains a pair or gold label outside the manifest")
        done = {pair_key(row) for row in cached}
        pending = [row for row in rows if pair_key(row) not in done]
        print(f"Resuming: {len(done)} cached, {len(pending)} remaining", flush=True)
        save_progress()
        if pending:
            model = QwenAssessor(assessor, revision, config.get("seed", 42))
            if any(row.get("compute_dtype") != model.compute_dtype for row in cached):
                raise ValueError("Effective compute dtype changed across GPUs. Use a new output directory.")
            torch = model.torch
            session.update({"gpu": torch.cuda.get_device_name(0),
                            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                            "cuda_runtime": torch.version.cuda, "compute_dtype": model.compute_dtype})
            if auto_batch:
                batch_size, measurements = benchmark_batches(model, pending, getattr(args, "max_batch_size", 8))
                write_json(args.output_dir / "batch_benchmark.json", {"selected_batch_size": batch_size,
                           "selection": "Highest throughput on up to 32 longest inputs by character length; labels unused",
                           "measurements": measurements})
                session["batch_size"] = batch_size
                print(f"Selected batch size {batch_size}", flush=True)
                best_rate = next(r["pairs_per_second"] for r in measurements if r["batch_size"] == batch_size)
                print(f"Estimated remaining inference: {len(pending) / best_rate / 60:.1f} minutes (excludes setup)", flush=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            last_backup = last_print = time.perf_counter()
            cursor = 0
            while cursor < len(pending):
                if max_seconds and time.perf_counter() - started >= max_seconds:
                    session["status"] = "time_budget_exhausted"
                    break
                batch = pending[cursor:cursor + batch_size]
                failed = False
                try:
                    predictions = model.score_batch(batch)
                except getattr(torch.cuda, "OutOfMemoryError", ()):
                    if not auto_batch or batch_size == 1:
                        raise
                    failed = True
                if failed:
                    batch_size = max(1, batch_size // 2)
                    session.setdefault("oom_backoffs", []).append({"completed_pairs": len(done) + session_rows,
                                                                   "new_batch_size": batch_size})
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"CUDA OOM: retrying uncommitted pairs at batch size {batch_size}", flush=True)
                    continue
                if len(predictions) != len(batch) or {pair_key(r) for r in predictions} != {pair_key(r) for r in batch}:
                    raise ValueError("Scorer did not return exactly one prediction per requested pair")
                store.add_batch(predictions)
                cursor += len(batch)
                session_rows += len(predictions)
                now = time.perf_counter()
                if cursor == len(pending) or now - last_print >= 15:
                    print(f"Checkpointed {len(done) + session_rows}/{len(rows)}; batch size {batch_size}", flush=True)
                    last_print = now
                if now - last_backup >= getattr(args, "backup_every_seconds", 120):
                    save_progress()
                    last_backup = time.perf_counter()
            torch.cuda.synchronize()
            inference_elapsed = time.perf_counter() - inference_started
            session["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            session["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            session["final_batch_size"] = batch_size
            complete = cursor == len(pending)
            count = min(getattr(args, "consistency_pairs", 0), len(rows))
            if complete and count:
                from scripts.compare_predictions import compare
                saved = {pair_key(r): r for r in store.rows()}
                targets = rows[:count]
                repeated = []
                check_started = time.perf_counter()
                for row in targets:
                    if max_seconds and time.perf_counter() - started >= max_seconds:
                        break
                    repeated.extend(model.score_batch([row]))
                torch.cuda.synchronize()
                reference = [saved[pair_key(r)] for r in targets[:len(repeated)]]
                check = {"requested_pairs": count, "completed_pairs": len(repeated),
                         "status": "complete" if len(repeated) == count else "time_budget_exhausted",
                         "comparison": "Main-run batches versus batch size 1 in the same model session; not sampled self-consistency",
                         "seconds": time.perf_counter() - check_started}
                if repeated:
                    check.update(compare(reference, repeated))
                    write_csv(args.output_dir / "consistency.csv", repeated)
                write_json(args.output_dir / "consistency.json", check)
        else:
            complete = True
        if complete:
            session["status"] = "complete"
        return complete
    except BaseException as error:
        session.update({"status": "interrupted", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        try:
            save_progress(final=True)
        finally:
            store.close()


if __name__ == "__main__":
    main()
