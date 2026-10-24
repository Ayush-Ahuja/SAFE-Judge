"""Cache pinned MiniLM raw scores once; resumable CPU/CUDA inference."""

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cross_encoder import MiniLMCrossEncoder, load_config, split_calibration_test, text_pool
from src.utils.io import PredictionStore, file_hash, pair_key, read_manifest, write_csv, write_json
from scripts.run_pilot import run_lock


def execute(args, backend_factory=MiniLMCrossEncoder):
    config, _ = load_config(args.config)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    manifest = read_manifest(args.manifest)
    if args.split_json:
        split = json.loads(args.split_json.read_text(encoding="utf-8"))
        calibration, test = split_calibration_test(manifest, split)
        pool = text_pool(calibration + test)
    else:
        pool = text_pool(manifest)
    sources = [Path(__file__), ROOT / "src/baselines/cross_encoder.py", ROOT / "src/utils/io.py"]
    dependencies = {}
    for package in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    experiment = {"kind": "cross_encoder_raw_scores", "config": asdict(config), "device": args.device,
                  "python": platform.python_version(),
                  "manifest_sha256": file_hash(args.manifest),
                  "split_sha256": file_hash(args.split_json) if args.split_json else None,
                  "query_ids": sorted({r["query_id"] for r in pool}), "pairs": len(pool),
                  "dependencies": dependencies,
                  "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources}}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.output_dir / "checkpoint.sqlite3").is_file():
        raise ValueError("An existing nonempty score directory must contain its matching resume checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with run_lock(args.output_dir):
        store = PredictionStore(args.output_dir / "checkpoint.sqlite3", experiment)
        try:
            existing = store.rows()
            completed = {pair_key(r) for r in existing}
            expected = {pair_key(r) for r in pool}
            if not completed <= expected:
                raise ValueError("Checkpoint contains pairs outside the scoring pool")
            pending = [r for r in pool if pair_key(r) not in completed]
            metadata_path = args.output_dir / "experiment.json"
            previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            provenance = previous.get("model", {})
            write_json(metadata_path, {**experiment, "status": "running", "model": provenance})
            if pending:
                model = backend_factory(config, args.device, args.cache_dir, args.local_files_only)
                provenance = model.provenance
                for offset in range(0, len(pending), config.batch_size):
                    batch = pending[offset:offset + config.batch_size]
                    scored = model.score_batch(batch)
                    if [pair_key(r) for r in scored] != [pair_key(r) for r in batch]:
                        raise ValueError("Scorer returned missing, reordered or extra pairs")
                    if any(not math.isfinite(float(r["raw_logit"])) for r in scored):
                        raise ValueError("Scorer returned a nonfinite logit")
                    store.add_batch(scored)
                    if offset == 0 or (offset // config.batch_size + 1) % 10 == 0:
                        print(f"Scored {len(existing) + offset + len(batch)}/{len(pool)} pairs", flush=True)
            rows = store.rows()
            if {pair_key(r) for r in rows} != expected:
                raise ValueError("Incomplete score cache")
            write_csv(args.output_dir / "scores.csv", rows)
            metadata = {**experiment, "status": "complete", "model": provenance,
                        "scores_sha256": file_hash(args.output_dir / "scores.csv"),
                        "elapsed_seconds_this_invocation": time.perf_counter() - start,
                        "resumed_pairs": len(existing),
                        "truncated_pairs": sum(r["truncated"] for r in rows)}
            write_json(metadata_path, metadata)
            print(f"Completed {len(rows)} raw scores: {args.output_dir}", flush=True)
            return rows
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_encoder.json")
    parser.add_argument("--split-json", type=Path, help="Score only the specified calibration/test topics, excluding audit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/huggingface")
    parser.add_argument("--local-files-only", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
