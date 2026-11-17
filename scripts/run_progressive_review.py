"""Simulate review-in-batches, fresh QC, and stop-on-certificate label completion."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_diagnostic_confirmation import load_scored
from scripts.run_error_predictor_ablation import aggregate, complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.safe_judge import SafeJudgeModel, SafeJudgeConfig, run_safe_judge
from src.calibration.progressive_review import progressive_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def make_snapshots(pool, labels, method, budgets, seed, selected):
    """Each snapshot uses ONLY its own work labels; no audit outcome is input.

    Simulate the work paths once for both audit-gate variants. Replaying their
    exact purchase requests through each global cache charges each label once.
    """
    index = {pair_key(r): i for i, r in enumerate(pool)}
    snapshots = {}
    if method == "selected_local":
        _, _, _, acquired, _ = run_safe_judge(pool, LabelOracle(labels, max(budgets)), max(budgets), selected, seed)
        model = SafeJudgeModel(pool, selected)
        processed = 0
        for budget in sorted(budgets):
            for row in acquired[processed:budget]:
                model.observe(index[pair_key(row)], int(row["observed_label"]))
            processed = budget
            predictions = model.predict()[0]
            snapshots[budget] = model.reviewed.copy(), predictions.copy(), [pair_key(r) for r in acquired[:budget]]
    elif method == "raw_confidence":
        p = np.asarray([r["prob_relevant"] for r in pool])
        for budget in budgets:
            human = complete_budget(pool, [], 1-np.maximum(p, 1-p), budget, seed)
            snapshots[budget] = human, (p >= .5).astype(int), [pair_key(r) for i, r in enumerate(pool) if human[i]]
    else:
        grouping = "topics" if "topics" in method else 1
        preprocessing = "upstream_smooth" if "smooth" in method else "raw"
        for budget in budgets:
            oracle = LabelOracle(labels, budget)
            rows, _, _ = run_lara(pool, oracle, budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
            human, predicted = np.zeros(len(pool), dtype=bool), np.zeros(len(pool), dtype=int)
            for r in rows:
                i = index[pair_key(r)]
                human[i], predicted[i] = r["label_source"] == "human", r["predicted_label"]
            snapshots[budget] = human, predicted, oracle.queried.copy()
    return snapshots


def execute(args):
    settings = json.loads(args.config.read_text())
    selected = SafeJudgeConfig(**json.loads((ROOT/"configs/safe_judge.json").read_text())["review_config"])
    known, known_meta = load_datasets(ROOT)
    confirmation_path = ROOT/"results/experiments/diagnostic_confirmation_v1/experiment.json"
    confirmation_bytes = confirmation_path.read_bytes()
    confirmation = json.loads(confirmation_bytes)
    dl, _ = load_scored(ROOT/"data/pilot/dl2019.jsonl", ROOT/"results/baselines/cross_encoder/dl2019_minilm_scores_v1")
    covid, _ = load_scored(ROOT/"data/research/trec_covid_5000.jsonl", ROOT/"results/experiments/trec_covid_minilm_v1")
    bio = [r for r in covid if r["query_id"] in confirmation["test_query_ids"]["biomedical_ce_locked"]]
    platt = confirmation["platt_parameters"]["target"]
    cohorts = {"dl2019_qwen": known["dl2019"], "chatreport_phi3": known["chatreport"],
               "web_ce_heldout": [r for r in dl if r["query_id"] in confirmation["test_query_ids"]["web_ce_heldout"]],
               "biomedical_ce_locked": bio,
               "biomedical_target_platt": [{**r, "prob_relevant": float(expit(platt["slope"]*r["raw_logit"]+platt["intercept"]))} for r in bio]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output_dir/"configuration.json")
    (args.output_dir/"cohort_snapshot.json").write_bytes(confirmation_bytes)
    sources = [Path(__file__), ROOT/"src/calibration/progressive_review.py", ROOT/"src/calibration/audit_release.py",
               ROOT/"src/calibration/safe_judge.py", ROOT/"scripts/experiment_inputs.py",
               ROOT/"src/evaluation/review_metrics.py", ROOT/"scripts/run_diagnostic_confirmation.py",
               ROOT/"src/baselines/lara.py", ROOT/"src/baselines/active_learning.py"]
    metadata = {"status": "running", "settings": settings, "known_inputs": known_meta,
                "cohort_sha256": file_hash(args.output_dir/"cohort_snapshot.json"),
                "selected_config_sha256": file_hash(ROOT/"configs/safe_judge.json"),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "scope": "Adaptive stopping on a fixed labelled evaluation pool, fresh random quality checks at each look, delta split across all looks and classes",
                "work_replay": "Each work snapshot is fitted only on its own requested labels; future work labels and QC outcomes are excluded from that snapshot. Costs count the union of all work and QC purchases up to stopping.",
                "external_labels": {"biomedical_target_platt": confirmation["target_calibration_labels"]}}
    write_json(args.output_dir/"experiment.json", metadata)
    records, looks, logs, per_query = [], [], [], []
    started = time.perf_counter()
    for cohort in settings["cohorts"]:
        rows = sorted(cohorts[cohort], key=pair_key)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        evaluator = PolicyEvaluator(rows)
        budgets = [int(f*len(rows)) for f in settings["checkpoints"]]
        external = confirmation["target_calibration_labels"] if cohort == "biomedical_target_platt" else 0
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
            for method in settings["methods"]:
                snapshots = make_snapshots(pool, labels, method, budgets, seed, selected)
                def worker(local, budget):
                    human, predicted, requested = snapshots[budget]
                    for key in requested:
                        local.query(key)
                    return human.copy(), predicted.copy()
                for gate in settings["feasibility_gate"]:
                    oracle = LabelOracle(labels, len(rows))
                    human, predicted, certificate, inspections, ledger = progressive_review(
                        pool, oracle, worker, seed, checkpoints=tuple(settings["checkpoints"]),
                        audit_fraction=settings["audit_fraction"], alpha=settings["alpha"],
                        class_risk=settings["class_risk"], delta=settings["delta"],
                        minimum_class_coverage=settings["minimum_class_coverage"], feasibility_gate=gate)
                    metrics, queries = evaluator.evaluate(human, predicted)
                    if metrics["human_labels"] != len(oracle.queried) or len(ledger) != len(oracle.queried):
                        raise AssertionError("Progressive completion cost mismatch")
                    ids = {"dataset": cohort, "seed": seed, "method": method, "feasibility_gate": gate,
                           "budget": "adaptive", "pilot_labels": 0}
                    violation = certificate["released"] and (metrics["query_mean_auto_error_rate"] > settings["alpha"] or any(
                        metrics[f"class_{c}_selective_risk"] is None or metrics[f"class_{c}_selective_risk"] > settings["class_risk"] for c in (0, 1)))
                    records.append({**ids, **certificate, **metrics, "external_calibration_labels": external,
                                    "total_human_labels": metrics["human_labels"]+external,
                                    "work_labels": sum(r["reason"] == "work" for r in ledger),
                                    "quality_check_labels": sum(r["reason"] == "quality_check" for r in ledger),
                                    "fallback_labels": sum(r["reason"] == "full_review_fallback" for r in ledger),
                                    "skipped_checks": sum(r["status"] == "skipped_infeasible" for r in inspections),
                                    "observed_target_violation": bool(violation)})
                    looks.extend({**ids, **r} for r in inspections)
                    per_query.extend({**ids, **q} for q in queries)
                    logs.append(json.dumps({**ids, "ledger": ledger}))
            print(f"Progressive review: {cohort}, seed {seed} complete", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", per_query)
    write_csv(args.output_dir/"looks.csv", looks)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "method", "feasibility_gate")))
    atomic_text(args.output_dir/"label_ledger.jsonl", "\n".join(logs)+"\n")
    metadata.update(status="complete", policies=len(records), elapsed_seconds=time.perf_counter()-started)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} progressive stopping policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/progressive_review.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
