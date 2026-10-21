"""Run CPU lexical SAL(hybrid) with exact human budgets on a judged manifest."""

import argparse
from dataclasses import asdict
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.active_learning import (ActiveLearningConfig, FEATURE_NAMES, LabelOracle,
                                           lexical_features, run_active_learning)
from src.evaluation.budgeted import allocation_metrics, allocation_per_query, evaluate_runs
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, write_csv, write_json


def select_queries(manifest, split=None):
    """Use exactly the test topics of a shared query split; audit topics stay out."""
    if split is None:
        return manifest
    groups = []
    for name in ("calibration_query_ids", "test_query_ids", "audit_query_ids"):
        ids = split.get(name, [])
        if not isinstance(ids, list) or any(not isinstance(q, str) or not q for q in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{name} must contain unique nonempty string query IDs")
        groups.append(set(ids))
    if not groups[1] or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Split must have nonempty test queries and disjoint calibration/test/audit queries")
    available = {r["query_id"] for r in manifest}
    if not groups[1] <= available:
        raise ValueError("Split requests test queries missing from the manifest")
    return [r for r in manifest if r["query_id"] in groups[1]]


def budget_counts(size, fractions=None, counts=None):
    if counts is not None:
        if not counts or any(type(n) is not int or not 0 <= n <= size for n in counts):
            raise ValueError("Budgets must be integers in [0, pool size]")
        return sorted(set(counts))
    if not fractions:
        raise ValueError("At least one budget fraction is required")
    values = [Decimal(str(value)) for value in fractions]
    if any(not value.is_finite() or not 0 <= value <= 1 for value in values):
        raise ValueError("Budget fractions must be finite and in [0, 1]")
    # Floor in decimal arithmetic so, e.g., 0.29 * 100 is exactly 29.
    return sorted({int(value * size) for value in values})


def execute(args):
    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    config_names = set(ActiveLearningConfig.__dataclass_fields__)
    if set(config_data) - config_names - {"budget_fractions", "seeds"}:
        raise ValueError("Unknown baseline configuration keys")
    config = ActiveLearningConfig(**{k: v for k, v in config_data.items() if k in config_names})
    seeds = args.seeds if args.seeds is not None else config_data.get("seeds", [42])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    manifest = read_manifest(args.manifest)
    split = json.loads(args.split_json.read_text(encoding="utf-8")) if args.split_json else None
    selected = sorted(select_queries(manifest, split), key=pair_key)
    # Deliberately discard gold/LLM fields at the learner boundary.
    pool = [{key: row[key] for key in ("query_id", "doc_id", "query_text", "doc_text")} for row in selected]
    labels = {pair_key(row): row["true_binary"] for row in selected}
    counts = budget_counts(len(pool), args.budget_fractions if args.budget_fractions is not None
                           else config_data.get("budget_fractions"), args.budgets)
    external_stats = json.loads(args.bm25_stats.read_text(encoding="utf-8")) if args.bm25_stats else None
    features, stats = lexical_features(pool, config, external_stats)
    run_paths = sorted(args.run_files or [])
    if any(not p.is_file() for p in run_paths):
        raise ValueError("Every --run-files argument must be a TREC run file")
    # Fail closed on existing output, including partially completed experiments.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "src/baselines/active_learning.py", ROOT / "src/evaluation/budgeted.py",
               ROOT / "src/evaluation/metrics.py", ROOT / "src/utils/io.py"]
    experiment = {"status": "running", "method": "lexical_sal_hybrid", "protocol": "transductive_pool_annotation",
                  "manifest_sha256": file_hash(args.manifest), "config_sha256": file_hash(args.config),
                  "split_sha256": file_hash(args.split_json) if args.split_json else None,
                  "bm25_stats_input_sha256": file_hash(args.bm25_stats) if args.bm25_stats else None,
                  "pool_sha256": hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                  "pairs": len(pool), "query_ids": sorted({r["query_id"] for r in pool}),
                  "dataset": selected[0]["dataset"], "relevance_threshold": selected[0]["relevance_threshold"],
                  "config": asdict(config), "budgets": counts, "seeds": seeds,
                  "python": platform.python_version(),
                  "dependencies": {name: importlib.metadata.version(name) for name in
                                   ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
                  "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                  "runs_sha256": {str(p): file_hash(p) for p in run_paths},
                  "budget_accounting": "All seed and active acquisitions count; no external training/calibration labels.",
                  "interpretation": "SAL(hybrid) with proposal lexical features; no conformal guarantee. Unqueried labels used only for evaluation."}
    write_json(args.output_dir / "experiment.json", experiment)
    write_json(args.output_dir / "bm25_stats.json", stats)
    write_csv(args.output_dir / "features.csv", [{"query_id": r["query_id"], "doc_id": r["doc_id"],
              **dict(zip(FEATURE_NAMES, map(float, feature)))} for r, feature in zip(pool, features)])
    summaries = []
    for seed in seeds:
        for budget in counts:
            start = time.perf_counter()
            oracle = LabelOracle(labels, budget)
            predictions, acquisitions, state = run_active_learning(pool, oracle, budget, config, seed, features)
            rows = [{**prediction, "dataset": gold["dataset"], "true_relevance": gold["true_relevance"],
                     "true_binary": gold["true_binary"]} for prediction, gold in zip(predictions, selected)]
            directory = args.output_dir / f"seed_{seed}" / f"budget_{budget:06d}"
            write_csv(directory / "predictions.csv", rows)
            write_csv(directory / "acquisitions.csv", acquisitions, ["query_id", "doc_id", "acquisition_rank",
                      "round", "stage", "prob_relevant_before_query", "observed_label"])
            write_json(directory / "model.json", state)
            metrics = allocation_metrics(rows)
            per_query = allocation_per_query(rows)
            metrics["query_mean_auto_error_rate"] = sum(q["auto_error_rate"] for q in per_query) / len(per_query)
            if metrics["human_labels"] != budget or len(oracle.queried) != budget:
                raise AssertionError("Budget accounting mismatch")
            write_csv(directory / "per_query.csv", per_query)
            for name, field in (("mixed", "final_label"), ("gold", "true_binary")):
                atomic_text(directory / f"{name}.qrels", "".join(
                    f"{r['query_id']} 0 {r['doc_id']} {r[field]}\n" for r in rows))
            if run_paths:
                systems, queries, ranking = evaluate_runs(rows, run_paths)
                write_csv(directory / "system_scores.csv", systems)
                write_csv(directory / "system_query_scores.csv", queries)
                metrics["ranking"] = ranking
            metrics.update({"method": "lexical_sal_hybrid", "seed": seed, "budget": budget,
                            "elapsed_seconds": time.perf_counter() - start})
            write_json(directory / "summary.json", metrics)
            scalar = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
            scalar["kendall_tau_b"] = metrics.get("ranking", {}).get("kendall_tau_b")
            summaries.append(scalar)
            print(json.dumps({k: metrics[k] for k in ("seed", "budget", "coverage", "selective_risk")}), flush=True)
    write_csv(args.output_dir / "budget_curve.csv", summaries)
    experiment["status"] = "complete"
    write_json(args.output_dir / "experiment.json", experiment)
    outputs = sorted(p for p in args.output_dir.rglob("*") if p.is_file())
    atomic_text(args.output_dir / "checksums.sha256", "".join(
        f"{file_hash(p)}  {p.relative_to(args.output_dir).as_posix()}\n" for p in outputs))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lexical_active_learning.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--split-json", type=Path, help="Shared split with test_query_ids and optional calibration/audit IDs")
    parser.add_argument("--bm25-stats", type=Path, help="Optional corpus statistics using the documented tokenizer")
    parser.add_argument("--seeds", type=int, nargs="+")
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument("--budget-fractions", type=float, nargs="+")
    budgets.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--run-files", type=Path, nargs="+", help="Optional official TREC runs for binary nDCG@10/tau")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
