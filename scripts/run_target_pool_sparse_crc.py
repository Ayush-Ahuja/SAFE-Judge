"""Descriptive same-query experiment. NO unseen-query CRC guarantee is claimed."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pilot import validate_against_manifest
from src.calibration.sparse_query import group_pool, query_loss_curves, sampling_plan, select_threshold
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json


def evaluate_allocation(rows, acquired, threshold):
    """Purchased samples stay human-labelled, even if confidence exceeds t."""
    if not rows or not 0.5 <= threshold <= 1:
        raise ValueError("Nonempty pool and threshold in [0.5, 1] required")
    known = {pair_key(r) for r in rows}
    if len(known) != len(rows) or not set(acquired) <= known:
        raise ValueError("Invalid target/acquired pairs")
    query_sizes, query_errors = Counter(), Counter()
    class_sizes, class_accepted, class_errors = Counter(), Counter(), Counter()
    accepted = errors = 0
    for row in rows:
        qid, probability = str(row["query_id"]), row["prob_relevant"]
        label = int(probability >= .5)
        query_sizes[qid] += 1
        class_sizes[label] += 1
        if pair_key(row) in acquired or max(probability, 1-probability) <= threshold:
            continue
        accepted += 1
        wrong = int(label != row["true_binary"])
        errors += wrong
        query_errors[qid] += wrong
        class_accepted[label] += 1
        class_errors[label] += wrong
    result = {"target_pairs": len(rows), "target_queries": len(query_sizes),
              "calibration_labels": len(acquired), "automatic_labels": accepted, "automatic_errors": errors,
              "additional_human_labels": len(rows)-len(acquired)-accepted,
              "total_human_labels": len(rows)-accepted, "coverage": accepted/len(rows),
              "auto_error_rate": errors/len(rows), "selective_risk": errors/accepted if accepted else None,
              "query_mean_auto_error_rate": sum(query_errors[q]/m for q, m in query_sizes.items())/len(query_sizes)}
    for label in (0, 1):
        a, e, total = class_accepted[label], class_errors[label], class_sizes[label]
        result.update({f"class_{label}_coverage": a/total if total else None,
                       f"class_{label}_selective_risk": e/a if a else None})
    return result


def execute(args):
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if set(settings) != {"query_counts", "label_budget", "alphas", "repetitions", "first_seed", "grid_steps"}:
        raise ValueError("Unexpected configuration keys")
    for field in ("label_budget", "repetitions", "grid_steps", "first_seed"):
        if type(settings[field]) is not int or settings[field] < (0 if field == "first_seed" else 1):
            raise ValueError(f"Invalid {field}")
    counts, budget = settings["query_counts"], settings["label_budget"]
    if (not counts or len(set(counts)) != len(counts)
            or any(type(n) is not int or n < 1 or budget % n for n in counts)):
        raise ValueError("Use distinct positive query counts dividing the label budget")
    if not settings["alphas"] or any(not 0 < a < 1 for a in settings["alphas"]):
        raise ValueError("Invalid target")
    manifest = read_manifest(args.manifest)
    rows, _ = read_predictions(args.predictions, manifest[0]["relevance_threshold"])
    validate_against_manifest(rows, manifest)
    pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": r["prob_relevant"]} for r in rows]
    groups = group_pool(pool)
    if max(counts) > len(groups) or max(budget//n for n in counts) > min(map(len, groups.values())):
        raise ValueError("Infeasible sampling plan")
    labels = {pair_key(r): r["true_binary"] for r in rows}
    grid = np.linspace(.5, 1., settings["grid_steps"]+1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "settings": settings,
                "protocol": "Same target queries supply calibration labels and remaining pairs; descriptive only",
                "guarantee": "No prospective unseen-query theorem is claimed for this transductive allocation",
                "loss_denominator": "All original pairs per query, including human-labelled calibration samples",
                "cost": "Acquired samples plus additional escalations, each pair charged once",
                "manifest_sha256": file_hash(args.manifest), "predictions_sha256": file_hash(args.predictions),
                "config_sha256": file_hash(args.config), "query_ids": sorted(groups), "pairs": len(rows),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                    (Path(__file__), ROOT/"src/calibration/sparse_query.py", ROOT/"src/utils/io.py",
                     ROOT/"scripts/evaluate_pilot.py")}}
    write_json(args.output_dir/"experiment.json", metadata)
    records, logs = [], []
    for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
        order = sorted(groups)
        random.Random(f"target-query-order/{seed}").shuffle(order)
        for n in counts:
            plan = sampling_plan(pool, order[:n], budget, seed)
            observed = {key: labels[key] for keys in plan.values() for key in keys}
            if len(observed) != budget:
                raise AssertionError("Budget mismatch")
            curves = query_loss_curves(pool, plan, observed, grid)
            logs.append(json.dumps({"seed": seed, "queries": n, "budget": budget,
                                    "samples": {q: [key[1] for key in keys] for q, keys in plan.items()}}))
            for alpha in settings["alphas"]:
                for method in ("crc", "empirical"):
                    selection = select_threshold(curves, grid, alpha, method)
                    metrics = evaluate_allocation(rows, observed, selection["threshold"])
                    if metrics["total_human_labels"] != budget+metrics["additional_human_labels"]:
                        raise AssertionError("Human labels counted twice")
                    records.append({"seed": seed, "alpha": alpha, "method": method,
                                    "labels_per_query": budget//n, **selection, **metrics})
    write_csv(args.output_dir/"runs.csv", records)
    summary = []
    for alpha in settings["alphas"]:
        for method in ("crc", "empirical"):
            for n in counts:
                subset = [r for r in records if r["alpha"] == alpha and r["method"] == method and r["calibration_queries"] == n]
                item = {"alpha": alpha, "method": method, "calibration_queries": n,
                        "calibration_labels": budget, "labels_per_query": budget//n, "repetitions": len(subset),
                        "observed_target_exceedance_fraction": sum(r["query_mean_auto_error_rate"] > alpha for r in subset)/len(subset)}
                for field in ("coverage", "query_mean_auto_error_rate", "auto_error_rate", "selective_risk",
                              "class_0_coverage", "class_1_coverage", "class_0_selective_risk", "class_1_selective_risk",
                              "automatic_labels", "total_human_labels"):
                    values = [r[field] for r in subset if r[field] is not None]
                    for suffix, v in (("mean", np.mean(values) if values else None),
                                      ("p10", np.quantile(values, .1) if values else None),
                                      ("p90", np.quantile(values, .9) if values else None)):
                        item[field+"_"+suffix] = float(v) if v is not None else None
                summary.append(item)
    write_csv(args.output_dir/"summary.csv", summary)
    atomic_text(args.output_dir/"sampling.jsonl", "\n".join(logs)+"\n")
    metadata.update(status="complete", evaluated_policies=len(records))
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(json.dumps({"status": "complete", "policies": len(records), "output": str(args.output_dir)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/pilot/dl2019.jsonl")
    parser.add_argument("--predictions", type=Path, default=ROOT/"results/stage2/dl2019_qwen35_full_v1/predictions.csv")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/target_pool_sparse_crc.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
