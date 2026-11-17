"""Climate-only sensitivity: partial relevance counts as negative, not positive."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_error_predictor_ablation import aggregate, complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.query_error import fit_error_scores, pilot_sample
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(output):
    datasets, provenance = load_datasets(ROOT, partial_positive=False)
    rows = sorted(datasets["chatreport"], key=pair_key)
    pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant", "auxiliary_logit")} for r in rows]
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    index = {pair_key(r): i for i, r in enumerate(rows)}
    evaluator = PolicyEvaluator(rows)
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((ROOT/"configs/policy_ideas_screening.json").read_text())
    metadata = {"status": "running", "settings": settings, "datasets": provenance,
                "scope": "Exploratory relevance-definition sensitivity; same prompts and cached model answers",
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__),
                    ROOT/"scripts/experiment_inputs.py", ROOT/"scripts/run_error_predictor_ablation.py",
                    ROOT/"src/calibration/query_error.py", ROOT/"src/evaluation/review_metrics.py",
                    ROOT/"src/baselines/lara.py")}}
    write_json(output/"experiment.json", metadata)
    records = []
    for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
        candidates = []
        for k in settings["pilot_labels_per_query"]:
            pilot = pilot_sample(pool, k, seed)
            observed = {key: labels[key] for key in pilot}
            for method in settings["learned_scores"]:
                candidates.append((f"error_{method}_k{k}", pilot, fit_error_scores(pool, observed, method)))
        for fraction in settings["human_budget_fractions"]:
            budget = int(len(pool)*fraction)
            p = np.asarray([r["prob_relevant"] for r in pool])
            all_candidates = [("raw_confidence", [], 1-np.maximum(p, 1-p)),
                              ("random", [], np.random.default_rng(seed).random(len(pool)))]+candidates
            for method, pilot, score in all_candidates:
                if len(pilot) <= budget:
                    human = complete_budget(pool, pilot, score, budget, seed)
                    metric, _ = evaluator.evaluate(human)
                    records.append({"dataset": "chatreport_partial_negative", "seed": seed, "budget": budget,
                                    "method": method, "pilot_labels": len(pilot), **metric})
            for preprocessing in ("raw", "upstream_smooth"):
                for grouping in (1, "topics"):
                    predictions, _, _ = run_lara(pool, LabelOracle(labels, budget), budget,
                                                LaraConfig(preprocessing=preprocessing), seed, grouping)
                    human, predicted = np.zeros(len(pool), dtype=bool), np.zeros(len(pool), dtype=int)
                    for r in predictions:
                        i = index[pair_key(r)]
                        human[i], predicted[i] = r["label_source"] == "human", r["predicted_label"]
                    metric, _ = evaluator.evaluate(human, predicted)
                    records.append({"dataset": "chatreport_partial_negative", "seed": seed, "budget": budget,
                                    "method": f"lara_{grouping}_{preprocessing}", "pilot_labels": 0, **metric})
    write_csv(output/"runs.csv", records)
    write_csv(output/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    metadata.update(status="complete", policies=len(records))
    write_json(output/"experiment.json", metadata)
    outputs = sorted(p for p in output.iterdir() if p.is_file())
    atomic_text(output/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} label-definition allocations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
