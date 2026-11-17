"""Exploratory, exactly matched human-budget comparisons of error-score ideas."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.query_error import fit_error_scores, pilot_sample
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def aggregate(rows, group_fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_fields)].append(r)
    fields = ("coverage", "query_mean_auto_error_rate", "auto_error_rate", "selective_risk",
              "tail20_query_auto_error", "worst_query_auto_error", "class_0_coverage", "class_1_coverage",
              "class_0_selective_risk", "class_1_selective_risk", "worst_defined_class_selective_risk",
              "system_ranking_kendall_tau", "human_labels", "pilot_labels")
    result = []
    for key, records in sorted(groups.items()):
        r = dict(zip(group_fields, key))
        r["repetitions"] = len(records)
        r["both_classes_accepted_fraction"] = np.mean([x["both_classes_accepted"] for x in records]).item()
        for field in fields:
            values = [x[field] for x in records if x.get(field) is not None]
            r[field+"_defined_runs"] = len(values)
            for name, v in (("mean", np.mean(values) if values else None),
                            ("p10", np.quantile(values, .1) if values else None),
                            ("p90", np.quantile(values, .9) if values else None)):
                r[field+"_"+name] = float(v) if v is not None else None
        result.append(r)
    return result


def complete_budget(pool, pilot, scores, budget, seed):
    indices = {pair_key(r): i for i, r in enumerate(pool)}
    selected = {indices[key] for key in pilot}
    if len(selected) > budget or not 0 <= budget <= len(pool):
        raise ValueError("Infeasible budget")
    tie = list(range(len(pool)))
    random.Random(f"allocation-ties/{seed}").shuffle(tie)
    priority = {i: rank for rank, i in enumerate(tie)}
    order = sorted((i for i in range(len(pool)) if i not in selected), key=lambda i: (-scores[i], priority[i]))
    selected.update(order[:budget-len(selected)])
    human = np.zeros(len(pool), dtype=bool)
    human[list(selected)] = True
    return human


def execute(args):
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    datasets, provenance = load_datasets(ROOT, not args.partial_negative)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/"scripts/experiment_inputs.py", ROOT/"src/calibration/query_error.py",
               ROOT/"src/evaluation/review_metrics.py", ROOT/"src/baselines/lara.py", ROOT/"src/baselines/active_learning.py"]
    metadata = {"status": "running", "settings": settings, "datasets": provenance,
                "config_sha256": file_hash(args.config),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "scope": "Exploratory transductive allocation; every pilot and escalation label charged; no formal risk claim",
                "label_use": "No unpurchased gold reaches the score learner; all gold is available only to evaluation",
                "selection": "Fixed settings before policy outcomes; all specified candidates retained, no best-seed selection"}
    write_json(args.output_dir/"experiment.json", metadata)
    records, per_query, acquisitions = [], [], []
    start = time.perf_counter()
    for dataset, rows in datasets.items():
        rows = sorted(rows, key=pair_key)
        evaluator = PolicyEvaluator(rows)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant", "auxiliary_logit")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        index = {pair_key(r): i for i, r in enumerate(rows)}
        raw_risk = 1-np.maximum([r["prob_relevant"] for r in rows], [1-r["prob_relevant"] for r in rows])

        def record(seed, budget, method, pilot_count, human, predicted=None):
            metrics, queries = evaluator.evaluate(human, predicted)
            if metrics["human_labels"] != budget:
                raise AssertionError("Exact human budget violated")
            identifiers = {"dataset": dataset, "seed": seed, "budget": budget, "method": method,
                           "pilot_labels": pilot_count}
            records.append({**identifiers, **metrics})
            per_query.extend({**identifiers, **q} for q in queries)

        for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
            learned = []
            for k in settings["pilot_labels_per_query"]:
                pilot = pilot_sample(pool, k, seed)
                observed = {key: labels[key] for key in pilot}
                acquisitions.append(json.dumps({"dataset": dataset, "seed": seed, "k": k,
                                                 "pairs": pilot}))
                for score in settings["learned_scores"]:
                    scores = fit_error_scores(pool, observed, score, settings["query_feature_scale"])
                    learned.append((f"error_{score}_k{k}", pilot, scores))
            for fraction in settings["human_budget_fractions"]:
                budget = int(len(pool)*fraction)
                for method, scores in (("raw_confidence", raw_risk),
                                       ("random", np.random.default_rng(seed).random(len(pool)))):
                    record(seed, budget, method, 0, complete_budget(pool, [], scores, budget, seed))
                for method, pilot, scores in learned:
                    if len(pilot) <= budget:
                        record(seed, budget, method, len(pilot), complete_budget(pool, pilot, scores, budget, seed))
                for preprocessing in ("raw", "upstream_smooth"):
                    for grouping in (1, "topics"):
                        oracle = LabelOracle(labels, budget)
                        predictions, _, _ = run_lara(pool, oracle, budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
                        human = np.zeros(len(pool), dtype=bool)
                        predicted = np.zeros(len(pool), dtype=int)
                        for r in predictions:
                            i = index[pair_key(r)]
                            human[i], predicted[i] = r["label_source"] == "human", r["predicted_label"]
                        record(seed, budget, f"lara_{grouping}_{preprocessing}", 0, human, predicted)
            if (seed-settings["first_seed"]+1) % 5 == 0:
                print(f"{dataset}: completed {seed-settings['first_seed']+1}/{settings['repetitions']} seeds", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", per_query)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    atomic_text(args.output_dir/"pilot_samples.jsonl", "\n".join(acquisitions)+"\n")
    metadata.update(status="complete", policies=len(records), elapsed_seconds=time.perf_counter()-start)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} policies in {metadata['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/error_predictor_ablation.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial-negative", action="store_true", help="Climate label-definition sensitivity: partial counts as negative")
    execute(parser.parse_args())
