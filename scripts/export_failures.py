"""Prepare examples for human review; never generate manual annotations."""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pilot import validate_against_manifest, validate_experiment_manifest
from src.utils.io import file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json


def sample_failures(rows, count=50, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        if row["predicted_label"] != row["true_binary"]:
            error = "false_positive" if row["predicted_label"] else "false_negative"
            groups[(row["query_id"], error, row["confidence"] >= 0.9)].append(row)
    if sum(map(len, groups.values())) < count:
        raise ValueError(f"Fewer than {count} unique failures; enlarge the pilot or request a smaller draft sample")
    buckets = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        buckets.append(deque(groups[key]))
    rng.shuffle(buckets)
    selected = []
    while len(selected) < count:
        for bucket in buckets:
            if bucket:
                selected.append(bucket.popleft())
                if len(selected) == count:
                    break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true", help="Preserve an existing review with matching input hashes")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        metadata = args.output.with_suffix(".metadata.json")
        if args.reuse_existing and args.output.exists() and metadata.exists():
            old = json.loads(metadata.read_text(encoding="utf-8"))
            if (old["manifest_sha256"] == file_hash(args.manifest)
                    and old["predictions_sha256"] == file_hash(args.predictions)
                    and old["count"] == args.count and old["seed"] == args.seed):
                print(f"Preserved existing review: {args.output}")
                return
        parser.error("Review artifact already exists; refusing to overwrite human annotations")
    if args.count < 1:
        parser.error("--count must be positive")
    manifest = read_manifest(args.manifest)
    rows, _ = read_predictions(args.predictions, manifest[0]["relevance_threshold"])
    validate_against_manifest(rows, manifest)
    validate_experiment_manifest(args.predictions, args.manifest)
    originals = {pair_key(r): r for r in manifest}
    selected = sample_failures(rows, args.count, args.seed)
    review = []
    for row in selected:
        original = originals[pair_key(row)]
        review.append({"query_id": row["query_id"], "doc_id": row["doc_id"],
                       "query_text": original["query_text"], "document_text": original["doc_text"],
                       "model_document_text": row.get("truncated_document_text") or original["doc_text"],
                       "human_grade": row["true_relevance"], "human_binary": row["true_binary"],
                       "predicted_label": row["predicted_label"], "confidence": row["confidence"],
                       "error_type": "false_positive" if row["predicted_label"] else "false_negative",
                       "high_confidence_wrong": row["confidence"] >= 0.9,
                       "input_truncated": row.get("truncated", "unknown"),
                       "failure_category": "", "notes": "", "reviewer": "", "reviewed": ""})
    write_csv(args.output, review)
    write_json(args.output.with_suffix(".metadata.json"), {
        "seed": args.seed, "count": len(review), "manual_reviews_completed": 0,
        "manifest_sha256": file_hash(args.manifest), "predictions_sha256": file_hash(args.predictions),
        "sampling": "Round-robin shuffled query/error-type/confidence strata; diagnostic, not prevalence representative.",
        "annotation": "Human must fill failure_category, notes, reviewer and reviewed. Categories are not inferred.",
    })
    print(f"Prepared {len(review)} examples -> {args.output}. Manual annotation is still required.")


if __name__ == "__main__":
    main()
