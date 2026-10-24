"""Calibrate cached MiniLM logits on disjoint queries and evaluate total budgets."""

import argparse
import csv
from dataclasses import asdict
from decimal import Decimal
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cross_encoder import (allocate_by_budget, binary_nll, fit_temperature, load_config,
                                        prediction_rows, split_calibration_test, text_hash)
from src.evaluation.budgeted import allocation_metrics, allocation_per_query, evaluate_runs
from src.evaluation.metrics import classification_metrics, risk_coverage_table
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, write_csv, write_json


def read_scores(directory, manifest_path, manifest, config, required):
    metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("kind") != "cross_encoder_raw_scores":
        raise ValueError("A complete raw cross-encoder cache is required")
    if metadata["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Score cache belongs to a different manifest")
    if file_hash(directory / "scores.csv") != metadata["scores_sha256"]:
        raise ValueError("Score cache checksum mismatch")
    for key in ("model_name", "revision", "max_length"):
        if metadata["config"][key] != getattr(config, key):
            raise ValueError(f"Score cache {key} differs from evaluation configuration")
    expected = {pair_key(r): r for r in manifest}
    scores = {}
    with (directory / "scores.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = pair_key(row)
            if key in scores or key not in expected:
                raise ValueError("Duplicate or unknown score pair")
            score = float(row["raw_logit"])
            if not math.isfinite(score) or row["text_sha256"] != text_hash(expected[key]):
                raise ValueError("Invalid raw score or mismatched input text")
            scores[key] = score
    if len(scores) != metadata["pairs"] or not {pair_key(r) for r in required} <= set(scores):
        raise ValueError("Score cache is incomplete for the requested queries")
    return [scores[pair_key(r)] for r in required]


def total_budgets(test_pairs, calibration_pairs, fractions, budgets, include_endpoints):
    maximum = test_pairs + calibration_pairs
    if budgets is not None:
        if not budgets or any(type(b) is not int or not 0 <= b <= maximum for b in budgets):
            raise ValueError("Total budgets must lie in [0, test + calibration pairs]")
        counts = set(budgets)
    else:
        if not fractions:
            raise ValueError("At least one budget fraction is required")
        values = [Decimal(str(v)) for v in fractions]
        if any(not v.is_finite() or not 0 <= v <= 1 for v in values):
            raise ValueError("Budget fractions must lie in [0, 1]")
        # Same denominator as the lexical baseline: total cost / test-pool size.
        counts = {int(v * test_pairs) for v in values}
    if include_endpoints:
        counts.update((calibration_pairs, maximum))
    return sorted(counts)


CURVE_FIELDS = ["method", "seed", "budget", "status", "pairs", "queries", "calibration_labels",
                "total_human_labels", "total_human_fraction_of_test", "test_human_labels", "coverage",
                "selective_risk", "auto_error_rate", "mixed_label_accuracy",
                "worst_observed_class_selective_risk", "both_predicted_classes_accepted", "kendall_tau_b"]


def execute(args):
    config, settings = load_config(args.config)
    manifest = read_manifest(args.manifest)
    external = args.calibration_manifest is not None
    if external != (args.calibration_scores is not None):
        raise ValueError("Separate calibration manifest and score cache must be supplied together")
    source = read_manifest(args.calibration_manifest) if external else manifest
    split = json.loads(args.split_json.read_text(encoding="utf-8")) if args.split_json else None
    calibration, test = split_calibration_test(manifest, split, source if external else None)
    calibration_logits = read_scores(args.calibration_scores if external else args.scores,
                                     args.calibration_manifest if external else args.manifest,
                                     source, config, calibration)
    test_logits = read_scores(args.scores, args.manifest, manifest, config, test)
    fitted = fit_temperature(calibration_logits, [r["true_binary"] for r in calibration], config,
                             [r["query_id"] for r in calibration])
    seeds = args.seeds if args.seeds is not None else settings.get("seeds", [42])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    endpoints = settings.get("include_endpoints", True)
    if type(endpoints) is not bool:
        raise ValueError("include_endpoints must be boolean")
    budgets = total_budgets(len(test), len(calibration), args.budget_fractions if args.budget_fractions is not None
                            else settings.get("budget_fractions"), args.budgets, endpoints)
    run_files = sorted(args.run_files or [])
    if any(not p.is_file() for p in run_files):
        raise ValueError("Every --run-files argument must be a TREC run file")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "src/baselines/cross_encoder.py", ROOT / "src/evaluation/budgeted.py",
             ROOT / "src/evaluation/metrics.py", ROOT / "src/utils/io.py"]
    input_paths = {"manifest": args.manifest, "scores": args.scores / "scores.csv",
                   "score_metadata": args.scores / "experiment.json", "config": args.config}
    if args.split_json:
        input_paths["split"] = args.split_json
    if external:
        input_paths.update(calibration_manifest=args.calibration_manifest,
                           calibration_scores=args.calibration_scores / "scores.csv",
                           calibration_score_metadata=args.calibration_scores / "experiment.json")
    experiment = {"status": "running", "method": "temperature_scaled_cross_encoder",
                  "python": platform.python_version(),
                  "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
                  "config": asdict(config), "budgets": budgets, "seeds": seeds,
                  "calibration_labels": len(calibration), "test_pairs": len(test),
                  "calibration_query_ids": sorted({r["query_id"] for r in calibration}),
                  "test_query_ids": sorted({r["query_id"] for r in test}),
                  "audit_query_ids": sorted(split.get("audit_query_ids", [])) if split else [],
                  "inputs_sha256": {k: file_hash(p) for k, p in input_paths.items()},
                  "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
                  "runs_sha256": {str(p): file_hash(p) for p in run_files},
                  "budget_accounting": "Total budget = all calibration pairs + test escalations. Fraction denominator = test pairs.",
                  "policy": "Escalate smallest absolute raw logits; positive temperature preserves confidence ordering.",
                  "interpretation": "One explicit query split. No conformal guarantee; seeds only break allocation ties."}
    write_json(args.output_dir / "experiment.json", experiment)
    write_json(args.output_dir / "temperature.json", fitted)
    write_json(args.output_dir / "split.json", {k: experiment[k] for k in ("calibration_query_ids", "test_query_ids", "audit_query_ids")})
    write_csv(args.output_dir / "calibration_labels.csv", [
        {"query_id": r["query_id"], "doc_id": r["doc_id"], "observed_label": r["true_binary"]}
        for r in calibration])
    variants, calibration_metrics = {}, []
    for name, temperature in (("raw", 1.0), ("temperature_scaled", fitted["temperature"])):
        predictions = prediction_rows(test, test_logits, temperature)
        rows = [{**p, "dataset": r["dataset"], "true_relevance": r["true_relevance"],
                 "true_binary": r["true_binary"]} for p, r in zip(predictions, test)]
        variants[name] = rows
        write_csv(args.output_dir / f"{name}_predictions.csv", rows)
        metric = classification_metrics(rows)
        reliability = metric.pop("reliability")
        metric.update(method=name, temperature=temperature,
                      nll=binary_nll(test_logits, [r["true_binary"] for r in test], temperature))
        calibration_metrics.append(metric)
        write_csv(args.output_dir / f"{name}_reliability.csv", reliability)
        sweep = risk_coverage_table(rows)
        for point in sweep:
            # accepted/errors in class-specific sweeps have different denominators.
            point["interpretation"] = "Descriptive held-out threshold sweep; no tuned risk guarantee"
        write_csv(args.output_dir / f"{name}_risk_coverage.csv", sweep)
    write_csv(args.output_dir / "calibration_metrics.csv", calibration_metrics)
    rows = variants["temperature_scaled"]
    summaries = []
    for seed in seeds:
        for budget in budgets:
            scalar = {"method": "temperature_scaled_cross_encoder", "seed": seed, "budget": budget,
                      "pairs": len(test), "queries": len(experiment["test_query_ids"]),
                      "calibration_labels": len(calibration)}
            if budget < len(calibration):
                summaries.append({**scalar, "status": "infeasible_calibration_budget"})
                continue
            acquired = allocate_by_budget(rows, budget - len(calibration), seed)
            final = [{**r, "label_source": "human" if i in acquired else "automatic",
                      "acquisition_rank": acquired.get(i, ""),
                      "final_label": r["true_binary"] if i in acquired else r["predicted_label"]}
                     for i, r in enumerate(rows)]
            directory = args.output_dir / f"seed_{seed}" / f"budget_{budget:06d}"
            write_csv(directory / "predictions.csv", final)
            write_csv(directory / "acquisitions.csv", [
                {"query_id": final[i]["query_id"], "doc_id": final[i]["doc_id"], "acquisition_rank": rank,
                 "observed_label": final[i]["true_binary"], "confidence": final[i]["confidence"]}
                for i, rank in sorted(acquired.items(), key=lambda item: item[1])],
                ["query_id", "doc_id", "acquisition_rank", "observed_label", "confidence"])
            metrics = allocation_metrics(final)
            metrics.update(scalar, status="complete", test_human_labels=metrics["human_labels"],
                           total_human_labels=len(calibration) + metrics["human_labels"],
                           total_human_fraction_of_test=budget / len(test))
            if metrics["total_human_labels"] != budget:
                raise AssertionError("Calibration-inclusive budget accounting mismatch")
            per_query = allocation_per_query(final)
            write_csv(directory / "per_query.csv", per_query)
            metrics["query_mean_auto_error_rate"] = sum(r["auto_error_rate"] for r in per_query) / len(per_query)
            accepted = [r for r in final if r["label_source"] == "automatic"]
            metrics["minimum_accepted_confidence"] = min((r["confidence"] for r in accepted), default=None)
            for name, field in (("mixed", "final_label"), ("gold", "true_binary")):
                atomic_text(directory / f"{name}.qrels", "".join(
                    f"{r['query_id']} 0 {r['doc_id']} {r[field]}\n" for r in final))
            if run_files:
                systems, queries, ranking = evaluate_runs(final, run_files)
                write_csv(directory / "system_scores.csv", systems)
                write_csv(directory / "system_query_scores.csv", queries)
                metrics["ranking"] = ranking
            metrics["kendall_tau_b"] = metrics.get("ranking", {}).get("kendall_tau_b")
            write_json(directory / "summary.json", metrics)
            summaries.append({k: metrics.get(k) for k in CURVE_FIELDS})
            print(json.dumps({k: metrics[k] for k in ("seed", "budget", "total_human_labels", "coverage", "selective_risk")}), flush=True)
    write_csv(args.output_dir / "budget_curve.csv", summaries, CURVE_FIELDS)
    experiment["status"] = "complete"
    write_json(args.output_dir / "experiment.json", experiment)
    outputs = sorted(p for p in args.output_dir.rglob("*") if p.is_file())
    atomic_text(args.output_dir / "checksums.sha256", "".join(
        f"{file_hash(p)}  {p.relative_to(args.output_dir).as_posix()}\n" for p in outputs))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="Complete raw-score directory")
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--calibration-manifest", type=Path, help="Optional separate source-domain calibration collection")
    parser.add_argument("--calibration-scores", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_encoder.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--budgets", type=int, nargs="+")
    group.add_argument("--budget-fractions", type=float, nargs="+")
    parser.add_argument("--run-files", type=Path, nargs="+")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
