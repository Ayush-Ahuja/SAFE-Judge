"""Compare pilot-guided, class-conditional and query-weighted finite-pool release."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_error_predictor_ablation import aggregate
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.audit_release import build_audit, select_release
from src.calibration.query_error import fit_error_scores, pilot_sample
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(args):
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    datasets, provenance = load_datasets(ROOT, not args.partial_negative)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output_dir/"configuration.json")
    sources = [Path(__file__), ROOT/"scripts/experiment_inputs.py", ROOT/"scripts/run_error_predictor_ablation.py",
               ROOT/"src/calibration/query_error.py", ROOT/"src/calibration/audit_release.py",
               ROOT/"src/evaluation/review_metrics.py", ROOT/"src/baselines/lara.py",
               ROOT/"src/baselines/active_learning.py"]
    metadata = {"status": "running", "settings": settings, "datasets": provenance,
                "config_sha256": file_hash(args.config),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "scope": "Fixed-pool random-audit certificate; not future-query or domain-transfer certification",
                "guarantee": "With probability >=1-delta over the fresh within-cell audit, the selected union respects the stated finite-pool risk bounds",
                "assumptions": "Fixed target labels, correct human oracle, uniform without-replacement audit conditional on pilot, pre-audit strata/quotas, simultaneous cell bounds",
                "confidence_scope": "Each specified procedure separately; comparing many procedures post hoc is exploratory",
                "cost": "Pilot + audit + completion escalations, counted once; matched LARA receives exactly that TOTAL label count",
                "matched_lara_scope": "All four raw/normalized x global/per-topic variants for query_disagreement, four bins/class, query-balanced mode, first 20 seeds"}
    write_json(args.output_dir/"experiment.json", metadata)
    records, per_query, matched, logs, pilots = [], [], [], [], []
    started = time.perf_counter()
    for dataset, rows in datasets.items():
        rows = sorted(rows, key=pair_key)
        evaluator = PolicyEvaluator(rows)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant", "auxiliary_logit")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        indices = {pair_key(r): i for i, r in enumerate(rows)}
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["audit_repetitions"]):
            for score_method in settings["audit_score_methods"]:
                pilot = [] if score_method == "confidence" else pilot_sample(pool, settings["audit_pilot_labels_per_query"], seed)
                observed = {key: labels[key] for key in pilot}
                if pilot:
                    risk = fit_error_scores(pool, observed, score_method, settings["query_feature_scale"])
                else:
                    p = np.asarray([r["prob_relevant"] for r in pool])
                    risk = 1-np.maximum(p, 1-p)
                pilots.append(json.dumps({"dataset": dataset, "seed": seed, "score": score_method, "pilot_pairs": pilot}))
                for budget in settings["audit_budgets"][dataset]:
                    for bins in settings["audit_bins_per_class"]:
                        oracle = LabelOracle(labels, budget)
                        for key in pilot:
                            if oracle.query(key) != observed[key]:
                                raise AssertionError("Pilot labels changed")
                        audit = build_audit(pool, risk, pilot, oracle.query, budget-len(pilot), bins, seed, settings["delta"])
                        if len(oracle.queried) != budget:
                            raise AssertionError("Design/audit label budget mismatch")
                        id_fields = {"dataset": dataset, "seed": seed, "design_budget": budget,
                                     "bins_per_class": bins, "score": score_method, "pilot_labels": len(pilot)}
                        for h, cell in enumerate(audit["cells"]):
                            logs.append({**id_fields, "cell": h, **{k: v for k, v in cell.items()
                                         if k not in ("indices", "remaining_indices")}})
                        for unit in settings["audit_risk_units"]:
                            human, certificate = select_release(pool, audit, settings["alpha"], settings["class_selective_risk_target"], unit, settings["minimum_class_coverage"])
                            metrics, queries = evaluator.evaluate(human)
                            identifiers = {**id_fields, "risk_unit": unit, "method": "audited_"+score_method,
                                           "budget": budget}
                            actual = metrics["query_mean_auto_error_rate"] if unit == "query_balanced" else metrics["auto_error_rate"]
                            class_violation = any(metrics[f"class_{c}_selective_risk"] is not None and
                                                  metrics[f"class_{c}_selective_risk"] > settings["class_selective_risk_target"] for c in (0, 1))
                            records.append({**identifiers, **certificate, **metrics,
                                            "audit_labels": budget-len(pilot),
                                            "completion_labels": metrics["human_labels"]-budget,
                                            "observed_target_violation": actual > settings["alpha"] or class_violation})
                            per_query.extend({**identifiers, **q} for q in queries)
                            if (score_method == "query_disagreement" and bins == 4 and unit == "query_balanced"
                                    and seed < settings["first_seed"]+settings["repetitions"]):
                                total_budget = metrics["human_labels"]
                                for preprocessing in ("raw", "upstream_smooth"):
                                    for grouping in (1, "topics"):
                                        label_oracle = LabelOracle(labels, total_budget)
                                        predictions, _, _ = run_lara(pool, label_oracle, total_budget,
                                            LaraConfig(preprocessing=preprocessing), seed, grouping)
                                        lara_human, lara_pred = np.zeros(len(pool), dtype=bool), np.zeros(len(pool), dtype=int)
                                        for r in predictions:
                                            i = indices[pair_key(r)]
                                            lara_human[i], lara_pred[i] = r["label_source"] == "human", r["predicted_label"]
                                        lara_metrics, _ = evaluator.evaluate(lara_human, lara_pred)
                                        if lara_metrics["human_labels"] != total_budget:
                                            raise AssertionError("Matched LARA cost differs")
                                        matched.append({**id_fields, "method": f"lara_{grouping}_{preprocessing}",
                                                        "matched_human_labels": total_budget, **lara_metrics})
            if (seed-settings["first_seed"]+1) % 10 == 0:
                print(f"{dataset}: completed {seed-settings['first_seed']+1}/{settings['audit_repetitions']} audit seeds", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", per_query)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "design_budget", "bins_per_class", "score", "risk_unit", "method")))
    write_csv(args.output_dir/"matched_lara.csv", matched)
    write_csv(args.output_dir/"matched_lara_summary.csv", aggregate(matched, ("dataset", "design_budget", "method")))
    write_csv(args.output_dir/"audit_cells.csv", logs)
    atomic_text(args.output_dir/"pilot_samples.jsonl", "\n".join(pilots)+"\n")
    metadata.update(status="complete", policies=len(records), matched_lara_policies=len(matched),
                    elapsed_seconds=time.perf_counter()-started)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} certified allocations and {len(matched)} matched LARA allocations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/error_predictor_ablation.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial-negative", action="store_true")
    execute(parser.parse_args())
