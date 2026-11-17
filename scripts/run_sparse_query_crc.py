"""Compare few-query/dense versus many-query/sparse calibration, offline."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pilot import validate_against_manifest
from src.calibration.sparse_query import evaluation_table, group_pool, query_loss_curves, sampling_plan, select_threshold
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json


METRICS = ("coverage", "query_mean_coverage", "query_mean_auto_error_rate", "auto_error_rate",
           "selective_risk", "class_0_coverage", "class_1_coverage", "class_0_selective_risk",
           "class_1_selective_risk", "total_human_labels", "calibration_labels", "threshold",
           "calibration_sample_loss", "full_calibration_loss")


def summarize(records):
    groups = defaultdict(list)
    keys = ("sampling", "method", "alpha", "calibration_queries", "budget_condition")
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, runs in sorted(groups.items()):
        row = dict(zip(keys, key))
        row.update(repetitions=len(runs),
                   zero_coverage_fraction=sum(r["automatic_labels"] == 0 for r in runs)/len(runs),
                   fallback_fraction=sum(r["full_escalation_fallback"] for r in runs)/len(runs),
                   observed_test_target_exceedance_fraction=sum(
                       r["query_mean_auto_error_rate"] > r["alpha"] for r in runs)/len(runs))
        for field in METRICS:
            values = [r[field] for r in runs if r[field] is not None]
            row[field+"_defined_runs"] = len(values)
            for suffix, value in (("mean", np.mean(values) if values else None),
                                  ("p10", np.quantile(values, .1) if values else None),
                                  ("p90", np.quantile(values, .9) if values else None)):
                row[field+"_"+suffix] = float(value) if value is not None else None
        result.append(row)
    return result


def paired_comparisons(records, query_counts):
    groups = defaultdict(dict)
    for row in records:
        if row["sampling"] == "sparse_uniform":
            key = row["seed"], row["alpha"], row["method"], row["budget_condition"]
            groups[key][row["calibration_queries"]] = row
    comparisons = defaultdict(list)
    low, high = min(query_counts), max(query_counts)
    for (seed, alpha, method, budget), runs in groups.items():
        for field in ("coverage", "query_mean_auto_error_rate", "total_human_labels"):
            comparisons[(alpha, method, budget, field)].append(runs[high][field]-runs[low][field])
    return [{"alpha": a, "method": m, "budget_condition": b, "metric": f,
             "comparison": f"{high}_queries_minus_{low}_queries", "paired_repetitions": len(v),
             "difference_mean": float(np.mean(v)), "difference_p10": float(np.quantile(v, .1)),
             "difference_p90": float(np.quantile(v, .9))}
            for (a, m, b, f), v in sorted(comparisons.items())]


def validate_settings(settings):
    required = {"split_seed", "held_out_queries", "query_counts", "label_budgets", "alphas",
                "repetitions", "first_seed", "grid_steps", "full_query_reference"}
    if set(settings) != required:
        raise ValueError("Unexpected or missing configuration keys")
    for field in ("split_seed", "held_out_queries", "repetitions", "first_seed", "grid_steps"):
        if type(settings[field]) is not int or settings[field] < (0 if field in ("split_seed", "first_seed") else 1):
            raise ValueError(f"Invalid {field}")
    for field in ("query_counts", "label_budgets"):
        values = settings[field]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 1 for v in values):
            raise ValueError(f"Invalid {field}")
    if any(g < n or g % n for g in settings["label_budgets"] for n in settings["query_counts"]):
        raise ValueError("Every budget must be divisible by every query count")
    if (not settings["alphas"] or any(not 0 < a < 1 for a in settings["alphas"])
            or type(settings["full_query_reference"]) is not bool):
        raise ValueError("Invalid alphas/reference setting")


def execute(args):
    started = time.perf_counter()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    validate_settings(settings)
    manifest = read_manifest(args.manifest)
    rows, _ = read_predictions(args.predictions, manifest[0]["relevance_threshold"])
    validate_against_manifest(rows, manifest)
    source_path = args.predictions.parent / "experiment.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    if source.get("manifest_sha256") not in (None, file_hash(args.manifest)):
        raise ValueError("Predictions belong to another manifest")
    # Model-side rows deliberately contain no gold fields.
    pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"],
             "prob_relevant": r["prob_relevant"]} for r in rows]
    groups = group_pool(pool)
    ids = sorted(groups)
    random.Random(settings["split_seed"]).shuffle(ids)
    test_ids, calibration_ids = ids[:settings["held_out_queries"]], ids[settings["held_out_queries"]:]
    if not test_ids or max(settings["query_counts"]) > len(calibration_ids):
        raise ValueError("Not enough disjoint calibration/test queries")
    if max(g//n for g in settings["label_budgets"] for n in settings["query_counts"]) > min(
            len(groups[q]) for q in calibration_ids):
        raise ValueError("A query may not contain enough pairs for the planned sample")
    test = [r for r in rows if r["query_id"] in test_ids]
    cal_pool = [r for r in pool if r["query_id"] in calibration_ids]
    grid = np.linspace(.5, 1., settings["grid_steps"]+1)
    # These gold-derived arrays are evaluation/reference only, never sparse policy inputs.
    test_table = evaluation_table(test, grid)
    labels = {pair_key(r): r["true_binary"] for r in rows}
    full_plan = {q: [pair_key(r) for r in groups[q]] for q in sorted(calibration_ids)}
    full_curves = query_loss_curves(cal_pool, full_plan, labels, grid)
    full_curve_by_query = dict(zip(full_plan, full_curves))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT/"src/calibration/sparse_query.py",
                    ROOT/"scripts/evaluate_pilot.py", ROOT/"src/utils/io.py"]
    metadata = {"status": "running", "settings": settings,
                "manifest_sha256": file_hash(args.manifest), "predictions_sha256": file_hash(args.predictions),
                "config_sha256": file_hash(args.config), "python": platform.python_version(), "numpy": np.__version__,
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in source_files},
                "assessor": source.get("assessor"), "dataset": manifest[0]["dataset"],
                "relevance_threshold": manifest[0]["relevance_threshold"],
                "test_query_ids": test_ids, "calibration_pool_query_ids": calibration_ids,
                "test_pairs": len(test), "calibration_pool_pairs": len(cal_pool),
                "acceptance": "confidence > threshold; predicted relevant when p >= 0.5",
                "target": "Expected equal-query automatic mistakes / all query pairs, not selective risk",
                "cost": "Purchased calibration labels + held-out test escalations; excludes unlabelled calibration remainder",
                "repetition_scope": "Acquisition randomness on ONE fixed test split; percentiles are not confidence intervals",
                "reference_scope": "All-pairs CRC spends a variable, larger label budget; diagnostic only",
                "guarantee_scope": "Exploratory retrospective experiment; does not certify this fixed test set or domain shift"}
    write_json(args.output_dir/"experiment.json", metadata)
    records, logs = [], []

    def record(seed, n, budget_condition, sampling, method, alpha, curves, full, spent):
        selection = select_threshold(curves, grid, alpha, method)
        index = selection["threshold_index"]
        metric = test_table[index]
        records.append({"seed": seed, "sampling": sampling, "method": method, "alpha": alpha,
                        "budget_condition": budget_condition, "calibration_labels": spent,
                        "labels_per_query": spent//n if sampling == "sparse_uniform" else None,
                        **selection, **metric, "full_calibration_loss": float(full[:, index].mean()),
                        "total_human_labels": spent+metric["test_human_labels"]})

    for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
        order = sorted(calibration_ids)
        random.Random(f"query-order/{seed}").shuffle(order)
        for n in settings["query_counts"]:
            selected = order[:n]
            full = np.asarray([full_curve_by_query[q] for q in selected])
            for budget in settings["label_budgets"]:
                plan = sampling_plan(cal_pool, selected, budget, seed)
                observed = {key: labels[key] for keys in plan.values() for key in keys}
                if len(observed) != budget:
                    raise AssertionError("Calibration budget mismatch")
                losses = query_loss_curves(cal_pool, plan, observed, grid)
                logs.append(json.dumps({"seed": seed, "queries": n, "budget": budget,
                                        "samples": {q: [key[1] for key in keys] for q, keys in plan.items()}}))
                for alpha in settings["alphas"]:
                    for method in ("crc", "empirical"):
                        record(seed, n, str(budget), "sparse_uniform", method, alpha, losses, full, budget)
            if settings["full_query_reference"]:
                for alpha in settings["alphas"]:
                    record(seed, n, "all_pairs", "all_pairs_reference", "crc", alpha, full, full,
                           sum(len(groups[q]) for q in selected))
        if (seed-settings["first_seed"]+1) % 20 == 0:
            print(f"Completed {seed-settings['first_seed']+1}/{settings['repetitions']} acquisition repetitions", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"summary.csv", summarize(records))
    write_csv(args.output_dir/"paired_comparisons.csv", paired_comparisons(records, settings["query_counts"]))
    write_csv(args.output_dir/"test_threshold_diagnostics.csv", test_table)
    atomic_text(args.output_dir/"sampling.jsonl", "\n".join(logs)+"\n")
    metadata.update(status="complete", evaluated_policies=len(records), elapsed_seconds=time.perf_counter()-started)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(json.dumps({"status": "complete", "policies": len(records), "output": str(args.output_dir)}))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/pilot/dl2019.jsonl")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT/"results/stage2/dl2019_qwen35_full_v1/predictions.csv")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/sparse_query_crc.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
