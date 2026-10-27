"""Evaluate binary LARA on cached LLM probabilities with exact human budgets."""

import argparse
import csv
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
from scripts.run_lexical_baseline import budget_counts, select_queries
from scripts.evaluate_pilot import validate_against_manifest
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.evaluation.budgeted import allocation_metrics, allocation_per_query, evaluate_runs
from src.evaluation.metrics import classification_metrics
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json


def load_inputs(path, manifest, input_format):
    expected = {pair_key(r): r for r in manifest}
    if input_format == "project":
        rows, _ = read_predictions(path, manifest[0]["relevance_threshold"])
        validate_against_manifest(rows, manifest)
    elif input_format == "official_binary":
        rows, seen = [], set()
        with path.open(encoding="utf-8", newline="") as stream:
            for item in csv.DictReader(stream):
                key = item["topic_id"], item["doc_id"]
                if key in seen or key not in expected:
                    raise ValueError("Duplicate or unknown official input pair")
                seen.add(key)
                label = int(item["annotation"])
                yes, no = float(item["prob_yes"]), float(item["prob_no"])
                if label not in (0, 1) or label != expected[key]["true_binary"]:
                    raise ValueError("Official CSV requires binary annotations matching the manifest")
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (yes, no)) or yes+no == 0:
                    raise ValueError("Official Yes/No probabilities must have positive finite mass")
                rows.append({**expected[key], "prob_relevant": yes/(yes+no)})
        if seen != set(expected):
            raise ValueError("Official prediction CSV must cover the complete manifest")
    else:
        raise ValueError("Unknown prediction format")
    metadata = path.parent / "experiment.json"
    source = json.loads(metadata.read_text(encoding="utf-8")) if metadata.is_file() else {}
    return {pair_key(r): float(r["prob_relevant"]) for r in rows}, source


def parse_grouping(value):
    if value == "topics":
        return value
    if isinstance(value, bool):
        raise ValueError("Grouping must be a positive integer or topics")
    result = int(value)
    if result < 1 or (not isinstance(value, str) and result != value):
        raise ValueError("Grouping must be a positive integer or topics")
    return result


def execute(args):
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    names = set(LaraConfig.__dataclass_fields__)
    if not isinstance(settings, dict) or set(settings)-names-{"groups", "budget_fractions", "seeds"}:
        raise ValueError("Unknown LARA configuration keys")
    config = LaraConfig(**{k: v for k, v in settings.items() if k in names})
    if args.preprocessing is not None:
        config = replace(config, preprocessing=args.preprocessing)
    groupings = [parse_grouping(g) for g in (args.groups if args.groups is not None else settings.get("groups", [1]))]
    seeds = args.seeds if args.seeds is not None else settings.get("seeds", [42])
    if not groupings or len(set(groupings)) != len(groupings):
        raise ValueError("Groupings must be nonempty and unique")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    manifest = read_manifest(args.manifest)
    probabilities, source = load_inputs(args.predictions, manifest, args.predictions_format)
    if source.get("manifest_sha256") not in (None, file_hash(args.manifest)):
        raise ValueError("Assessor experiment belongs to a different manifest")
    split = json.loads(args.split_json.read_text(encoding="utf-8")) if args.split_json else None
    selected = sorted(select_queries(manifest, split), key=pair_key)
    pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": probabilities[pair_key(r)]}
            for r in selected]
    labels = {pair_key(r): int(r["true_binary"]) for r in selected}
    budgets = budget_counts(len(pool), args.budget_fractions if args.budget_fractions is not None
                           else settings.get("budget_fractions"), args.budgets)
    run_files = sorted(args.run_files or [])
    if any(not p.is_file() for p in run_files):
        raise ValueError("Every --run-files argument must be an official TREC run file")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    references = json.loads((ROOT / "configs/lara_reference.json").read_text(encoding="utf-8"))
    source_files = [Path(__file__), ROOT / "src/baselines/lara.py", ROOT / "src/baselines/active_learning.py",
                    ROOT / "scripts/run_lexical_baseline.py", ROOT / "scripts/evaluate_pilot.py",
                    ROOT / "src/evaluation/budgeted.py", ROOT / "src/evaluation/metrics.py", ROOT / "src/utils/io.py"]
    metadata = {"status": "running", "method": "lara_binary", "config": asdict(config),
                "groupings": groupings, "seeds": seeds, "budgets": budgets,
                "pairs": len(pool), "query_ids": sorted({r["query_id"] for r in selected}),
                "dataset": selected[0]["dataset"], "relevance_threshold": selected[0]["relevance_threshold"],
                "manifest_sha256": file_hash(args.manifest), "predictions_sha256": file_hash(args.predictions),
                "config_sha256": file_hash(args.config), "upstream_reference": references,
                "split_sha256": file_hash(args.split_json) if args.split_json else None,
                "assessor": source.get("assessor"), "assessor_revision": source.get("resolved_revision"),
                "predictions_format": args.predictions_format,
                "python": platform.python_version(),
                "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in source_files},
                "runs_sha256": {str(p): file_hash(p) for p in run_files},
                "budget_accounting": "Every acquired label counts once, jointly calibrates and supplies the final human label. No external calibration labels.",
                "protocol": "Transductive pool annotation. Binary author-code semantics; strict N-1 budget fix.",
                "interpretation": "Same-assessor adaptation when using Qwen inputs; not a reproduction of published Llama/TREC-COVID rankings."}
    write_json(args.output_dir / "experiment.json", metadata)
    raw_rows = [{**r, "prob_relevant": probabilities[pair_key(r)],
                 "predicted_label": int(probabilities[pair_key(r)] >= .5)} for r in selected]
    raw_metrics = classification_metrics(raw_rows)
    raw_metrics.pop("reliability")
    write_json(args.output_dir / "raw_llm_reference.json", raw_metrics)
    summaries = []
    for grouping in groupings:
        for seed in seeds:
            for budget in budgets:
                start = time.perf_counter()
                oracle = LabelOracle(labels, budget)
                predictions, acquired, state = run_lara(pool, oracle, budget, config, seed, grouping)
                rows = [{**prediction, "dataset": gold["dataset"], "true_relevance": gold["true_relevance"],
                         "true_binary": gold["true_binary"]} for prediction, gold in zip(predictions, selected)]
                directory = args.output_dir / f"groups_{grouping}" / f"seed_{seed}" / f"budget_{budget:06d}"
                write_csv(directory / "predictions.csv", rows)
                write_csv(directory / "acquisitions.csv", acquired, ["query_id", "doc_id", "acquisition_rank",
                          "group", "stage", "llm_probability", "input_probability", "prob_relevant_before_query", "observed_label"])
                write_json(directory / "model.json", state)
                metrics = allocation_metrics(rows)
                if metrics["human_labels"] != budget or len(oracle.queried) != budget:
                    raise AssertionError("LARA annotation budget mismatch")
                per_query = allocation_per_query(rows)
                write_csv(directory / "per_query.csv", per_query)
                metrics["query_mean_auto_error_rate"] = sum(q["auto_error_rate"] for q in per_query)/len(per_query)
                for name, field in (("mixed", "final_label"), ("gold", "true_binary")):
                    atomic_text(directory / f"{name}.qrels", "".join(
                        f"{r['query_id']} 0 {r['doc_id']} {r[field]}\n" for r in rows))
                if run_files:
                    systems, queries, ranking = evaluate_runs(rows, run_files)
                    write_csv(directory / "system_scores.csv", systems)
                    write_csv(directory / "system_query_scores.csv", queries)
                    metrics["ranking"] = ranking
                metrics.update(method="lara_binary", grouping=grouping, effective_groups=len(state["groups"]),
                               preprocessing=config.preprocessing, seed=seed, budget=budget, total_human_labels=budget,
                               external_calibration_labels=0, elapsed_seconds=time.perf_counter()-start)
                write_json(directory / "summary.json", metrics)
                flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
                flat["kendall_tau_b"] = metrics.get("ranking", {}).get("kendall_tau_b")
                summaries.append(flat)
                print(json.dumps({k: metrics[k] for k in ("grouping", "seed", "budget", "coverage", "selective_risk")}), flush=True)
    write_csv(args.output_dir / "budget_curve.csv", summaries)
    metadata["status"] = "complete"
    write_json(args.output_dir / "experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.rglob("*") if p.is_file())
    atomic_text(args.output_dir / "checksums.sha256", "".join(
        f"{file_hash(p)}  {p.relative_to(args.output_dir).as_posix()}\n" for p in outputs))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-format", choices=("project", "official_binary"), default="project")
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lara.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocessing", choices=("raw", "upstream_smooth"))
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--budgets", nargs="+", type=int)
    budgets.add_argument("--budget-fractions", nargs="+", type=float)
    parser.add_argument("--run-files", type=Path, nargs="+")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
