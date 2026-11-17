"""Second exploratory audit design: freeze one candidate per class before auditing."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_error_predictor_ablation import aggregate
from src.baselines.active_learning import LabelOracle
from src.calibration.candidate_audit import candidate_release
from src.calibration.query_error import pilot_sample, fit_error_scores
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(args):
    datasets, provenance = load_datasets(ROOT)
    settings = {"first_seed": 2000, "repetitions": 100, "audit_fractions": [.5, .7],
                "candidate_fractions": {"0": .25, "1": .15}, "minimum_candidate_size": 70,
                "pilot_per_query": {"dl2019": 10, "chatreport": 2},
                "scores": ["confidence", "query_disagreement"],
                "design_budget_limits": {"dl2019": [430, 926, 1852], "chatreport": [66, 165, 330]},
                "alpha": .05, "class_risk": .2, "delta": .05, "minimum_class_coverage": .05}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/"scripts/experiment_inputs.py", ROOT/"scripts/run_error_predictor_ablation.py",
               ROOT/"src/calibration/query_error.py", ROOT/"src/calibration/audit_release.py",
               ROOT/"src/calibration/candidate_audit.py", ROOT/"src/evaluation/review_metrics.py"]
    metadata = {"status": "running", "settings": settings, "datasets": provenance,
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "design_history": "Second exploratory design after observing v1 over-fragmentation and exhausted small classes; fresh seeds 2000-2099",
                "scope": "Fixed-pool joint query-auto-error and class-selective-risk certificate; no future-domain claim",
                "budget": "Limit on pilot+audit purchases; unused budget is retained, completion escalations charged separately"}
    write_json(args.output_dir/"experiment.json", metadata)
    records, cells = [], []
    for dataset, rows in datasets.items():
        rows = sorted(rows, key=pair_key)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant", "auxiliary_logit")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        evaluator = PolicyEvaluator(rows)
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
            for score in settings["scores"]:
                pilot = [] if score == "confidence" else pilot_sample(pool, settings["pilot_per_query"][dataset], seed)
                observed = {key: labels[key] for key in pilot}
                p = np.asarray([r["prob_relevant"] for r in pool])
                risk = 1-np.maximum(p, 1-p) if score == "confidence" else fit_error_scores(pool, observed, score)
                for budget in settings["design_budget_limits"][dataset]:
                    for fraction in settings["audit_fractions"]:
                        oracle = LabelOracle(labels, budget)
                        for key in pilot:
                            oracle.query(key)
                        human, certificate, diagnostics = candidate_release(pool, risk, pilot, oracle.query,
                            budget-len(pilot), seed, fraction, settings["alpha"], settings["class_risk"], settings["delta"])
                        metrics, _ = evaluator.evaluate(human)
                        ids = {"dataset": dataset, "seed": seed, "design_budget": budget,
                               "audit_fraction": fraction, "method": score, "pilot_labels": len(pilot)}
                        records.append({**ids, **certificate, **metrics, "design_labels": len(oracle.queried),
                            "observed_target_violation": metrics["query_mean_auto_error_rate"] > .05 or any(
                                metrics[f"class_{c}_selective_risk"] is not None and metrics[f"class_{c}_selective_risk"] > .2 for c in (0, 1))})
                        cells.extend({**ids, **cell} for cell in diagnostics)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "design_budget", "audit_fraction", "method")))
    write_csv(args.output_dir/"audit_cells.csv", cells)
    metadata.update(status="complete", policies=len(records))
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} candidate-audit allocations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
