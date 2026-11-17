"""Query-membership controls and retrieval-comparison-aware review priorities."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
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
from src.calibration.diagnostic_review import ReviewConfig
from src.calibration.diagnostic_utility import run_utility_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.evaluation.retrieval_fidelity import RetrievalFidelity
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(args):
    settings = json.loads(args.config.read_text())
    selected_path = ROOT/"configs/diagnostic_review_selected.json"
    config = ReviewConfig(**json.loads(selected_path.read_text())["review_config"])
    datasets, provenance = load_datasets(ROOT)
    rows = sorted(datasets["dl2019"], key=pair_key)
    pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    evaluator = PolicyEvaluator(rows)
    runs = sorted((ROOT/"results/experiments/public_dl19_rankings_v1/runs").glob("*.trec"))
    if len(runs) != 6:
        raise ValueError("Require the six public retrieval/reranking runs")
    fidelity = RetrievalFidelity(rows, runs)
    impact, design = fidelity.review_utility()
    counts = {q: len(ix) for q, ix in evaluator.groups.items()}
    query_weights = np.asarray([1/counts[r["query_id"]] for r in rows])
    impact = impact*query_weights.sum()/impact.sum()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output_dir/"configuration.json")
    sources = [Path(__file__), ROOT/"src/calibration/diagnostic_utility.py", ROOT/"src/calibration/diagnostic_review.py",
               ROOT/"src/evaluation/retrieval_fidelity.py", ROOT/"src/evaluation/review_metrics.py",
               ROOT/"scripts/experiment_inputs.py", ROOT/"src/baselines/lara.py"]
    metadata = {"status": "running", "settings": settings, "inputs": provenance["dl2019"],
                "selected_config_sha256": file_hash(selected_path), "config_sha256": file_hash(args.config),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "retrieval_run_sha256": {p.name: file_hash(p) for p in runs},
                "utility_design_systems": design, "utility_held_out_systems": fidelity.held_out,
                "gold_ndcg": dict(zip(fidelity.system_names, fidelity.gold_scores.tolist())),
                "scope": "Exploratory DL2019 mechanism study; six public top-100 systems, not all official TREC runs",
                "utility": "Model estimated error times mixture of equal-query label loss and top-10 rank-disagreement discount. No gold labels or IDCG used in utility.",
                "sham_control": "Randomize query membership, preserve query-size histogram, disable query-size acquisition weights for both real and sham controls"}
    write_json(args.output_dir/"experiment.json", metadata)
    records, per_query, systems, logs = [], [], [], []
    started = time.perf_counter()
    index = {pair_key(r): i for i, r in enumerate(rows)}
    for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
        fake_queries = [r["query_id"] for r in pool]
        random.Random(f"sham-query-membership/{seed}").shuffle(fake_queries)
        fake_pool = [{"query_id": fake_queries[i], "doc_id": r["query_id"]+"::"+r["doc_id"],
                      "prob_relevant": r["prob_relevant"]} for i, r in enumerate(pool)]
        fake_labels = {pair_key(r): labels[pair_key(pool[i])] for i, r in enumerate(fake_pool)}
        for fraction in settings["human_budget_fractions"]:
            budget = int(len(rows)*fraction)
            def record(method, human, predicted):
                metrics, qrows = evaluator.evaluate(human, predicted)
                retrieval, scores = fidelity.evaluate(human, predicted)
                metrics.update(retrieval)
                metrics["system_ranking_kendall_tau"] = retrieval["public_system_kendall_tau"]
                if metrics["human_labels"] != budget:
                    raise AssertionError("Mechanism comparison budget mismatch")
                ids = {"dataset": "dl2019", "seed": seed, "budget": budget, "method": method, "pilot_labels": 0}
                records.append({**ids, **metrics})
                per_query.extend({**ids, **q} for q in qrows)
                systems.extend({**ids, "system": system, "gold_ndcg": float(gold), "mixed_ndcg": float(mixed),
                                "held_out_from_utility": system in fidelity.held_out}
                               for system, gold, mixed in zip(fidelity.system_names, fidelity.gold_scores, scores))
            for mixture in settings["ranking_mixtures"]:
                weight = (1-mixture)*query_weights+mixture*impact
                human, predicted, _, acquired, _ = run_utility_review(pool, LabelOracle(labels, budget), budget, config, seed, weight)
                name = f"ranking_mixture_{mixture}"
                record(name, human, predicted)
                logs.append(json.dumps({"seed": seed, "budget": budget, "method": name, "acquisitions": acquired}))
            for method in settings["mechanism_variants"]:
                shuffled = method.startswith("shuffled")
                current_pool, current_labels = (fake_pool, fake_labels) if shuffled else (pool, labels)
                current_config = replace(config, global_only="shared" in method, query_weighted=False)
                human, predicted, _, acquired, _ = run_utility_review(current_pool, LabelOracle(current_labels, budget), budget,
                                                                  current_config, seed, np.ones(len(rows)))
                record(method, human, predicted)
                logs.append(json.dumps({"seed": seed, "budget": budget, "method": method, "acquisitions": acquired}))
            p = np.asarray([r["prob_relevant"] for r in pool])
            record("raw_confidence", complete_budget(pool, [], 1-np.maximum(p, 1-p), budget, seed), (p >= .5).astype(int))
            for grouping, preprocessing in ((1, "raw"), ("topics", "upstream_smooth")):
                predictions, _, _ = run_lara(pool, LabelOracle(labels, budget), budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
                human, predicted = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=int)
                for r in predictions:
                    i = index[pair_key(r)]
                    human[i], predicted[i] = r["label_source"] == "human", r["predicted_label"]
                record(f"lara_{grouping}_{preprocessing}", human, predicted)
        print(f"Mechanism/ranking study: seed {seed} complete", flush=True)
    summary = aggregate(records, ("dataset", "budget", "method"))
    for r in summary:
        subset = [x for x in records if x["budget"] == r["budget"] and x["method"] == r["method"]]
        for field in ("public_system_ndcg_mae", "held_out_system_ndcg_mae", "maximum_system_rank_drop"):
            r[field+"_mean"] = float(np.mean([x[field] for x in subset]))
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"summary.csv", summary)
    write_csv(args.output_dir/"per_query.csv", per_query)
    write_csv(args.output_dir/"system_scores.csv", systems)
    atomic_text(args.output_dir/"acquisitions.jsonl", "\n".join(logs)+"\n")
    metadata.update(status="complete", policies=len(records), elapsed_seconds=time.perf_counter()-started)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} mechanism/ranking allocations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/diagnostic_mechanisms.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
