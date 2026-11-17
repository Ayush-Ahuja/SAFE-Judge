"""Compare human-inspired review variants under exact, seed-inclusive budgets."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_error_predictor_ablation import aggregate, complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.safe_judge import SafeJudgeConfig, run_safe_judge
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(args):
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    datasets, provenance = load_datasets(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output_dir/"configuration.json")
    sources = [Path(__file__), ROOT/"src/calibration/safe_judge.py", ROOT/"scripts/experiment_inputs.py",
               ROOT/"src/evaluation/review_metrics.py", ROOT/"src/baselines/lara.py", ROOT/"src/baselines/active_learning.py"]
    metadata = {"status": "running", "settings": settings, "inputs": provenance,
                "config_sha256": file_hash(args.config),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "scope": "Exploratory target-pool review; no risk guarantee or blind-test claim",
                "budget": "Every sentinel, exploratory review and exploitation review charged once"}
    write_json(args.output_dir/"experiment.json", metadata)
    shared = {k: settings[k] for k in ("shared_strength", "local_strength", "local_prior_probability", "repair_threshold")}
    base = SafeJudgeConfig(**shared)
    records, queries, logs = [], [], []
    start = time.perf_counter()
    for dataset, rows in datasets.items():
        rows = sorted(rows, key=pair_key)
        pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": r["prob_relevant"]} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        index = {pair_key(r): i for i, r in enumerate(rows)}
        evaluator = PolicyEvaluator(rows)
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
            for fraction in settings["human_budget_fractions"]:
                budget = int(len(pool)*fraction)
                def record(method, human, predicted=None, error=None, state=None, elapsed=0.):
                    metrics, per_query = evaluator.evaluate(human, predicted)
                    if metrics["human_labels"] != budget:
                        raise AssertionError("Review budget mismatch")
                    identifier = {"dataset": dataset, "seed": seed, "budget": budget, "method": method, "pilot_labels": 0}
                    extras = {"elapsed_seconds": elapsed, "automatic_repairs": state.get("automatic_repairs", 0) if state else 0,
                              "sentinel_labels": state.get("sentinel_labels", 0) if state else 0,
                              "exploration_labels": state.get("exploration_labels", 0) if state else 0,
                              "automatic_error_brier": float(np.mean((error[~human]-(predicted[~human] != evaluator.y[~human]))**2)) if error is not None and (~human).any() else None}
                    records.append({**identifier, **metrics, **extras})
                    queries.extend({**identifier, **q} for q in per_query)
                p = np.asarray([r["prob_relevant"] for r in rows])
                record("raw_confidence", complete_budget(pool, [], 1-np.maximum(p, 1-p), budget, seed))
                for method, overrides in settings["variants"].items():
                    oracle = LabelOracle(labels, budget)
                    began = time.perf_counter()
                    human, predicted, error, acquired, state = run_safe_judge(pool, oracle, budget, replace(base, **overrides), seed)
                    record(method, human, predicted, error, state, time.perf_counter()-began)
                    logs.append(json.dumps({"dataset": dataset, "seed": seed, "budget": budget,
                                            "method": method, "acquisitions": acquired}))
                for preprocessing in ("raw", "upstream_smooth"):
                    for grouping in (1, "topics"):
                        began = time.perf_counter()
                        predictions, _, _ = run_lara(pool, LabelOracle(labels, budget), budget,
                                                    LaraConfig(preprocessing=preprocessing), seed, grouping)
                        human, predicted = np.zeros(len(pool), dtype=bool), np.zeros(len(pool), dtype=int)
                        for r in predictions:
                            i = index[pair_key(r)]
                            human[i], predicted[i] = r["label_source"] == "human", r["predicted_label"]
                        record(f"lara_{grouping}_{preprocessing}", human, predicted, elapsed=time.perf_counter()-began)
            print(f"{dataset}: seed {seed} complete", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", queries)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    atomic_text(args.output_dir/"acquisitions.jsonl", "\n".join(logs)+"\n")
    metadata.update(status="complete", policies=len(records), elapsed_seconds=time.perf_counter()-start)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} diagnostic-review policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/diagnostic_review.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
