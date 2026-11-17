"""Post-hoc error decomposition and fixed-component ablations at 50% review."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import COHORTS, FIRST_SEED, load_cohort
from src.baselines.active_learning import LabelOracle
from src.calibration.safe_judge import SafeJudgeConfig, run_safe_judge
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

EXTRACTED = ROOT / "results/transfer/qwen35_transfer_v1_dataset_full"
COMPARISON = ROOT / "results/experiments/qwen_transfer_comparison_v1"
OUTPUT = ROOT / "reports/qwen_transfer_diagnosis_v1"


def component_ablation(cohort, method, seed, config_values):
    rows = load_cohort(EXTRACTED, cohort)
    pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    config = SafeJudgeConfig(**config_values)
    if method == "shared_only":
        config = replace(config, global_only=True)
    elif method == "without_query_size_weight":
        config = replace(config, query_weighted=False)
    else:
        raise ValueError("Unsupported component ablation")
    budget = len(rows) // 2
    oracle = LabelOracle(labels, budget)
    human, predicted, _, purchased, _ = run_safe_judge(pool, oracle, budget, config, seed)
    if len(purchased) != len(set(oracle.queried)) or len(purchased) != budget:
        raise ValueError("Invalid ablation budget")
    metrics, _ = PolicyEvaluator(rows).evaluate(human, predicted)
    return {"dataset": cohort, "method": method, "seed": seed, **metrics}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((COMPARISON / "experiment.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Require completed main comparison")
    config_path = ROOT / "configs/safe_judge.json"
    config = json.loads(config_path.read_text())["review_config"]
    if config != metadata["identity"]["selected_policy"]["review_config"]:
        raise ValueError("Selected policy changed")
    diagnostic, decomposition = [], []
    for cohort in COHORTS:
        rows = load_cohort(EXTRACTED, cohort)
        p = np.asarray([r["prob_relevant"] for r in rows])
        y = np.asarray([r["true_binary"] for r in rows])
        raw = p >= .5
        wrong = raw != y
        confidence = np.maximum(p, 1-p)
        index = {pair_key(r): i for i, r in enumerate(rows)}
        query_ids = sorted({r["query_id"] for r in rows})
        sizes = [sum(r["query_id"] == q for r in rows) for q in query_ids]
        info = {"dataset": cohort, "pairs": len(rows), "queries": len(query_ids),
                "raw_errors": int(wrong.sum()), "minimum_query_pairs": min(sizes), "maximum_query_pairs": max(sizes),
                "uncertainty_error_auc": float(roc_auc_score(wrong, 1-confidence))}
        for threshold in (.9, .95):
            subset = confidence >= threshold
            key = str(threshold).replace(".", "_")
            info[f"confidence_at_least_{key}_pairs"] = int(subset.sum())
            info[f"confidence_at_least_{key}_errors"] = int((wrong & subset).sum())
            info[f"confidence_at_least_{key}_error_rate"] = float(wrong[subset].mean())
        diagnostic.append(info)
        for method in ("selected_local", "lara_1_raw", "raw_confidence"):
            for path in sorted((COMPARISON / "allocations" / cohort / method).glob(f"*_budget_{len(rows)//2}.json.gz")):
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    allocation = json.load(stream)
                human = np.zeros(len(rows), dtype=bool)
                sentinels = []
                for purchase in allocation["purchases"]:
                    i = index[pair_key(purchase)]
                    human[i] = True
                    if purchase.get("stage") == "confident_sentinel":
                        sentinels.append(i)
                automatic = ~human
                final = np.asarray(allocation["automatic_predicted_labels"])
                removed = int((human & wrong).sum())
                fixed = int((automatic & wrong & (final == y)).sum())
                introduced = int((automatic & ~wrong & (final != y)).sum())
                errors = int((automatic & (final != y)).sum())
                if errors != int(wrong.sum()) - removed - fixed + introduced:
                    raise ValueError("Error decomposition does not balance")
                if errors != allocation["record"]["automatic_errors"]:
                    raise ValueError("Error decomposition differs from main experiment")
                decomposition.append({"dataset": cohort, "method": method, "seed": allocation["record"]["seed"],
                    "budget": len(rows)//2, "raw_errors": int(wrong.sum()), "raw_errors_reviewed": removed,
                    "remaining_raw_errors": int((automatic & wrong).sum()), "automatic_corrections": fixed,
                    "automatic_new_errors": introduced, "final_automatic_errors": errors,
                    "sentinel_labels": len(sentinels), "sentinel_errors_found": int(wrong[sentinels].sum()) if sentinels else 0})
    write_csv(OUTPUT / "input_diagnostics.csv", diagnostic)
    write_csv(OUTPUT / "error_decomposition.csv", decomposition)
    jobs = [(cohort, "shared_only", seed, config) for cohort in COHORTS for seed in range(FIRST_SEED, FIRST_SEED+10)]
    jobs += [("dl2020_all", "without_query_size_weight", seed, config) for seed in range(FIRST_SEED, FIRST_SEED+10)]
    ablations = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(component_ablation, *job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            ablations.append(result)
            print(f"{number}/{len(jobs)} ablations | {result['dataset']} | {result['method']} | {result['seed']}", flush=True)
    ablations.sort(key=lambda r: (r["dataset"], r["method"], r["seed"]))
    write_csv(OUTPUT / "component_ablations.csv", ablations)
    summary = []
    for cohort, method in sorted({(r["dataset"], r["method"]) for r in ablations}):
        subset = [r for r in ablations if r["dataset"] == cohort and r["method"] == method]
        summary.append({"dataset": cohort, "method": method, "seeds": len(subset),
                        "mean_query_auto_error": float(np.mean([r["query_mean_auto_error_rate"] for r in subset])),
                        "mean_pair_auto_error": float(np.mean([r["auto_error_rate"] for r in subset])),
                        "mean_automatic_errors": float(np.mean([r["automatic_errors"] for r in subset]))})
    write_csv(OUTPUT / "component_summary.csv", summary)
    write_json(OUTPUT / "experiment.json", {"status": "complete", "scope": "Post-hoc diagnosis at 50% human review. Two pre-existing configuration switches, no parameter search or replacement of main results.",
        "seeds": list(range(FIRST_SEED, FIRST_SEED+10)), "comparison_metadata_sha256": file_hash(COMPARISON / "experiment.json"),
        "selected_config_sha256": file_hash(config_path), "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
            (Path(__file__), ROOT / "src/calibration/safe_judge.py", ROOT / "src/evaluation/review_metrics.py")},
        "ablations": len(ablations), "allocation_decompositions": len(decomposition),
        "sentinel_scope": "Sentinel cost is measured. Sentinel removal is not evaluated in these ablations.",
        "interpretation": "Removing the local update and removing the query-size weight isolate components conditional on this implementation and these fixed pools."})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(OUTPUT.iterdir())
                if p.is_file() and p.name != "checksums.sha256"))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
