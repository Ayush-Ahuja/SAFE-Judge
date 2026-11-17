"""Evaluate the development-selected simpler policy without tuning on COVID."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_diagnostic_confirmation import load_scored
from scripts.run_error_predictor_ablation import aggregate
from src.baselines.active_learning import LabelOracle
from src.calibration.diagnostic_review import ReviewConfig, run_diagnostic_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(args):
    settings = json.loads(args.config.read_text())
    confirmation = ROOT/"results/experiments/diagnostic_confirmation_v1/experiment.json"
    metadata = json.loads(confirmation.read_text())
    dl, _ = load_scored(ROOT/"data/pilot/dl2019.jsonl", ROOT/"results/baselines/cross_encoder/dl2019_minilm_scores_v1")
    covid, _ = load_scored(ROOT/"data/research/trec_covid_5000.jsonl", ROOT/"results/experiments/trec_covid_minilm_v1")
    cohorts = {"web_ce_heldout": [r for r in dl if r["query_id"] in metadata["test_query_ids"]["web_ce_heldout"]],
               "biomedical_ce_locked": [r for r in covid if r["query_id"] in metadata["test_query_ids"]["biomedical_ce_locked"]]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output_dir/"configuration.json")
    report = {"status": "running", "settings": settings, "config_sha256": file_hash(args.config),
              "cohort_source_sha256": file_hash(confirmation),
              "test_query_ids": metadata["test_query_ids"],
              "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__),
                  ROOT/"src/calibration/diagnostic_review.py", ROOT/"scripts/run_diagnostic_confirmation.py",
                  ROOT/"src/evaluation/review_metrics.py")},
              "scope": "Secondary hypothesis selected on known discovery data before inspecting biomedical policy outcomes; target reviews are paid adaptation"}
    write_json(args.output_dir/"experiment.json", report)
    config = ReviewConfig(**settings["review_config"])
    records, queries, logs = [], [], []
    start = time.perf_counter()
    for cohort, rows in cohorts.items():
        rows = sorted(rows, key=pair_key)
        evaluator = PolicyEvaluator(rows)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
            for fraction in settings["human_budget_fractions"]:
                budget = int(len(pool)*fraction)
                human, predicted, error, acquired, state = run_diagnostic_review(pool, LabelOracle(labels, budget), budget, config, seed)
                metrics, qrows = evaluator.evaluate(human, predicted)
                ids = {"dataset": cohort, "seed": seed, "budget": budget, "method": "selected_local", "pilot_labels": 0}
                records.append({**ids, **metrics, "total_human_labels": metrics["human_labels"],
                                "automatic_repairs": state["automatic_repairs"]})
                queries.extend({**ids, **q} for q in qrows)
                logs.append(json.dumps({**ids, "acquisitions": acquired}))
            print(f"Selected policy: {cohort}, seed {seed} complete", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", queries)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    atomic_text(args.output_dir/"acquisitions.jsonl", "\n".join(logs)+"\n")
    report.update(status="complete", policies=len(records), elapsed_seconds=time.perf_counter()-start)
    write_json(args.output_dir/"experiment.json", report)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} selected-policy allocations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/diagnostic_review_selected.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
