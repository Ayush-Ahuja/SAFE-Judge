"""Apply the existing progressive quality controller to the new Qwen cohorts."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import load_cohort
from scripts.run_progressive_review import make_snapshots
from src.baselines.active_learning import LabelOracle
from src.calibration.diagnostic_review import ReviewConfig
from src.calibration.progressive_review import progressive_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

EXTRACTED = ROOT / "results/transfer/qwen35_transfer_v1_dataset_full"
OUTPUT = ROOT / "results/experiments/qwen_proposal_audit_v1"
COHORTS = ("dl2020_all", "trec_covid_locked", "chatreport_all")
METHODS = ("selected_local", "lara_1_raw", "raw_confidence")
CHECKPOINTS = (.1, .25, .4, .5)


def evaluate_job(cohort, method, seed, selected):
    path = OUTPUT / "allocations" / cohort / method / f"seed_{seed}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)["record"]
    rows = load_cohort(EXTRACTED, cohort)
    pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    budgets = [int(fraction*len(rows)) for fraction in CHECKPOINTS]
    snapshots = make_snapshots(pool, labels, method, budgets, seed, ReviewConfig(**selected))
    def work(local, budget):
        human, predicted, keys = snapshots[budget]
        for key in keys:
            local.query(key)
        return human.copy(), predicted.copy()
    oracle = LabelOracle(labels, len(rows))
    human, predicted, certificate, inspections, ledger = progressive_review(
        pool, oracle, work, seed, checkpoints=CHECKPOINTS, audit_fraction=.05,
        alpha=.05, class_risk=.20, delta=.05, minimum_class_coverage=.05, feasibility_gate=True)
    metrics, _ = PolicyEvaluator(rows).evaluate(human, predicted)
    if len(ledger) != len(set(oracle.queried)) or metrics["human_labels"] != len(ledger):
        raise ValueError("Audit union-cost accounting mismatch")
    violation = certificate["released"] and (metrics["query_mean_auto_error_rate"] > .05 or any(
        metrics[f"class_{c}_selective_risk"] is None or metrics[f"class_{c}_selective_risk"] > .20 for c in (0, 1)))
    record = {"dataset": cohort, "method": method, "seed": seed, **certificate, **metrics,
              "work_labels": sum(r["reason"] == "work" for r in ledger),
              "quality_check_labels": sum(r["reason"] == "quality_check" for r in ledger),
              "fallback_labels": sum(r["reason"] == "full_review_fallback" for r in ledger),
              "skipped_checks": sum(r["status"] == "skipped_infeasible" for r in inspections),
              "observed_target_violation": bool(violation)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump({"record": record, "looks": inspections, "ledger": ledger,
                   "automatic_predicted_labels": np.asarray(predicted, dtype=int).tolist()}, stream, allow_nan=False)
    temporary.replace(path)
    return record


def main():
    selected = json.loads((ROOT / "configs/diagnostic_review_selected.json").read_text())["review_config"]
    identity = {"cohorts": COHORTS, "methods": METHODS, "seeds": list(range(6000, 6010)),
                "checkpoints": CHECKPOINTS, "audit_fraction": .05, "alpha": .05, "class_risk": .20,
                "delta": .05, "minimum_class_coverage": .05, "selected": selected,
                "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                    (Path(__file__), ROOT / "scripts/run_progressive_review.py", ROOT / "src/calibration/progressive_review.py",
                     ROOT / "src/calibration/audit_release.py", ROOT / "src/calibration/diagnostic_review.py")}}
    identity = json.loads(json.dumps(identity))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "design.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Audit design changed")
    write_json(path, identity)
    records = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(evaluate_job, cohort, method, seed, selected)
                for cohort in COHORTS for method in METHODS for seed in range(6000, 6010)]
        for index, job in enumerate(as_completed(jobs), 1):
            records.append(job.result())
            if index % 10 == 0:
                print(f"Progressive audits {index}/{len(jobs)} complete", flush=True)
    records.sort(key=lambda r: (r["dataset"], r["method"], r["seed"]))
    write_csv(OUTPUT / "runs.csv", records)
    summary = []
    for cohort in COHORTS:
        for method in METHODS:
            subset = [r for r in records if r["dataset"] == cohort and r["method"] == method]
            summary.append({"dataset": cohort, "method": method, "runs": len(subset),
                "releases": sum(r["released"] for r in subset), "mean_coverage": float(np.mean([r["coverage"] for r in subset])),
                "mean_human_labels": float(np.mean([r["human_labels"] for r in subset])),
                "mean_quality_check_labels": float(np.mean([r["quality_check_labels"] for r in subset])),
                "observed_violations": sum(r["observed_target_violation"] for r in subset)})
    write_csv(OUTPUT / "summary.csv", summary)
    write_json(OUTPUT / "experiment.json", {"status": "complete", "design_sha256": file_hash(path), "runs": len(records),
        "scope": "Existing fixed-pool progressive audit controller. Distinct from proposal's expected future-query CRC. Five-percent within-class coverage is an additional operational criterion. All existing benchmark outcomes have been inspected."})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(OUTPUT).as_posix()}\n" for p in sorted(OUTPUT.rglob("*"))
                if p.is_file() and p.name != "checksums.sha256"))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
