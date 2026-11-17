"""Post-diagnostic numerical ablation; all epsilon settings retained."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_confidence_views import read_token_verdicts
from scripts.run_diagnostic_confirmation import load_scored
from scripts.run_error_predictor_ablation import aggregate
from src.baselines.active_learning import LabelOracle
from src.calibration.safe_judge import SafeJudgeConfig
from src.calibration.precision_review import run_precision_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import file_hash, pair_key, write_csv, write_json


def execute(output):
    data, provenance = load_datasets(ROOT)
    views = {"dl2019": data["dl2019"]}
    for model in ("phi3", "llama"):
        tokens = read_token_verdicts(ROOT/f"cache/research/diras_795ef8d/small_llm_results/gc_{model}_2e.pkl")
        views[model+"_token"] = [{**r, "prob_relevant": max(.500001, c) if y else 1-max(.500001, c)}
                                for r, (y, c) in zip(data["chatreport"], tokens)]
    covid, _ = load_scored(ROOT/"data/research/trec_covid_5000.jsonl", ROOT/"results/experiments/trec_covid_minilm_v1")
    cohort = json.loads((ROOT/"results/experiments/diagnostic_confirmation_v1/experiment.json").read_text())
    views["biomedical_ce_locked"] = [r for r in covid if r["query_id"] in cohort["test_query_ids"]["biomedical_ce_locked"]]
    cfg = SafeJudgeConfig(**json.loads((ROOT/"configs/safe_judge.json").read_text())["review_config"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "first_seed": 10000, "repetitions": 10,
                "epsilons": [None, .01, .001, .000001], "human_fraction": .5,
                "scope": "Exploratory numerical correction after token-confidence counterexample; no locked confirmation claim",
                "inputs": provenance, "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                    (Path(__file__), ROOT/"src/calibration/precision_review.py", ROOT/"src/calibration/safe_judge.py",
                     ROOT/"scripts/run_confidence_views.py", ROOT/"scripts/run_diagnostic_confirmation.py")}}
    write_json(output/"experiment.json", metadata)
    records, queries = [], []
    for name, rows in views.items():
        rows = sorted(rows, key=pair_key)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        evaluator = PolicyEvaluator(rows)
        budget = len(rows)//2
        for seed in range(10000, 10010):
            for eps in metadata["epsilons"]:
                human, predicted, _, _, _ = run_precision_review(pool, LabelOracle(labels, budget), budget, cfg, seed, eps)
                metric, qrows = evaluator.evaluate(human, predicted)
                ids = {"dataset": name, "seed": seed, "budget": budget, "method": "hard_floor" if eps is None else f"soft_{eps}", "pilot_labels": 0}
                records.append({**ids, **metric})
                queries.extend({**ids, **q} for q in qrows)
        print(f"Precision ablation: {name} complete", flush=True)
    write_csv(output/"runs.csv", records)
    write_csv(output/"per_query.csv", queries)
    write_csv(output/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    metadata.update(status="complete", policies=len(records))
    write_json(output/"experiment.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
