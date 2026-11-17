"""Expected query-CRC transfer baseline on the same frozen cross-encoder."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_diagnostic_confirmation import load_scored
from scripts.run_error_predictor_ablation import aggregate
from src.calibration.sparse_query import evaluation_table, query_loss_curves, sampling_plan, select_threshold
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(output):
    confirmation = json.loads((ROOT/"results/experiments/diagnostic_confirmation_v1/experiment.json").read_text())
    dl, dl_input = load_scored(ROOT/"data/pilot/dl2019.jsonl", ROOT/"results/baselines/cross_encoder/dl2019_minilm_scores_v1")
    covid, bio_input = load_scored(ROOT/"data/research/trec_covid_5000.jsonl", ROOT/"results/experiments/trec_covid_minilm_v1")
    target = [r for r in covid if r["query_id"] in confirmation["test_query_ids"]["biomedical_ce_locked"]]
    grid = np.unique(np.r_[expit(np.linspace(0, 20, 401)), 1.])
    variants = {"raw_source43": (dl, target, 0),
                "raw_target10": ([r for r in covid if r["query_id"] in confirmation["target_calibration_query_ids"]], target, 0)}
    parameters = confirmation["platt_parameters"]["source"]
    def transformed(rows):
        return [{**r, "prob_relevant": float(expit(parameters["slope"]*r["raw_logit"]+parameters["intercept"]))} for r in rows]
    variants["source_platt_source33"] = (transformed([r for r in dl if r["query_id"] not in confirmation["source_calibration_query_ids"]]),
                                            transformed(target), confirmation["source_calibration_labels"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "settings": {"first_seed": 9000, "repetitions": 100, "labels_per_calibration_query": 10,
                "alphas": [.05, .10], "grid": "sigmoid of margins 0:0.05:20 plus full-escalation endpoint"},
                "inputs": {"source": dl_input, "target": bio_input},
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__), ROOT/"src/calibration/sparse_query.py",
                    ROOT/"scripts/run_diagnostic_confirmation.py")},
                "scope": "CRC expected-risk argument requires exchangeable new query blocks. Source-to-biomedical transfer is an empirical stress test and has no automatic target guarantee.",
                "independence": "Platt fitting uses 10 source queries; its CRC calibration uses the other 33 source queries"}
    write_json(output/"experiment.json", metadata)
    records = []
    for name, (calibration, evaluation, fitting_cost) in variants.items():
        cal_pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in calibration]
        labels = {pair_key(r): r["true_binary"] for r in calibration}
        qids = sorted({r["query_id"] for r in calibration})
        metrics = evaluation_table(evaluation, grid)
        for seed in range(9000, 9100):
            budget = 10*len(qids)
            plan = sampling_plan(cal_pool, qids, budget, seed)
            observed = {key: labels[key] for keys in plan.values() for key in keys}
            losses = query_loss_curves(cal_pool, plan, observed, grid)
            for alpha in (.05, .10):
                selection = select_threshold(losses, grid, alpha)
                result = metrics[selection["threshold_index"]]
                defined_risks = [result[f"class_{c}_selective_risk"] for c in (0, 1) if result[f"class_{c}_selective_risk"] is not None]
                records.append({"dataset": "biomedical_ce_locked", "seed": seed, "method": name,
                    "budget": budget, "alpha": alpha, "pilot_labels": 0, **selection, **result,
                    "pairs": result["test_pairs"], "human_labels": result["test_human_labels"],
                    "external_calibration_labels": fitting_cost+budget,
                    "total_human_labels": fitting_cost+budget+result["test_human_labels"],
                    "tail20_query_auto_error": None, "worst_query_auto_error": None,
                    "worst_defined_class_selective_risk": max(defined_risks) if defined_risks else None,
                    "both_classes_accepted": all(result[f"class_{c}_accepted"] > 0 for c in (0, 1)),
                    "observed_target_exceedance": result["query_mean_auto_error_rate"] > alpha})
    write_csv(output/"runs.csv", records)
    write_csv(output/"summary.csv", aggregate(records, ("dataset", "method", "alpha")))
    metadata.update(status="complete", policies=len(records))
    write_json(output/"experiment.json", metadata)
    outputs = sorted(p for p in output.iterdir() if p.is_file())
    atomic_text(output/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} CRC transfer checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
