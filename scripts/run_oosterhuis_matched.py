"""Compare annotation policies and metric estimators on common binary top-10 pools."""
from collections import defaultdict
from dataclasses import replace
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.oosterhuis_inputs import binary_top10, load_reference
from scripts.run_error_predictor_ablation import complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.diagnostic_review import ReviewConfig, run_diagnostic_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def reference_dataset(rows, dataset_class, observed=None):
    """Build upstream inputs with only paid labels, or placeholder unseen labels."""
    qids = sorted({r["query_id"] for r in rows})
    segments = {q: i for i, q in enumerate(qids)}
    p = np.asarray([r["prob_relevant"] for r in rows])
    gold = np.asarray([observed[pair_key(r)] if observed is not None else 0 for r in rows])
    return dataset_class(scores=np.asarray([r["retrieval_score"] for r in rows]),
                         true_label_dist=np.column_stack([1-gold, gold]),
                         pred_label_dist=np.column_stack([1-p, p]),
                         query_segments=np.asarray([segments[r["query_id"]] for r in rows]),
                         ranks=np.asarray([r["rank"] for r in rows]))


def execute(output):
    settings, cache, (datasets, methods, _) = load_reference()
    design = settings["matched_experiment"]
    config_path = ROOT/"configs/diagnostic_review_selected.json"
    review_config = ReviewConfig(**json.loads(config_path.read_text())["review_config"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "settings": design, "reference_revision": settings["revision"],
                "reference_file_sha256": {r["path"]: r["sha256"] for r in settings["files"]},
                "config_sha256": file_hash(ROOT/"configs/oosterhuis_reference.json"),
                "selected_policy_sha256": file_hash(config_path),
                "scope": "Exploratory common-input comparison, binary relevance grade>=2 and BM25 top10 only. Same Flan-UL2 inputs and exact purchased-label budgets for every method. Oosterhuis PPI/CRC produce metric intervals, so automatic label quality is undefined for those outputs.",
                "interval_mapping": "Estimate unseen-query mean using upstream method, then combine its bounds with the known human-query contribution to obtain a full-pool DCG interval. Interval coverage is empirical on this fixed pool; no transferred finite-pool guarantee asserted.",
                "label_isolation": "Oosterhuis calibration queries are purchased through an oracle. Unseen-query dataset contains zero-label placeholders. Only the evaluator reads unseen gold.",
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__),
                    ROOT/"scripts/oosterhuis_inputs.py", ROOT/"src/calibration/diagnostic_review.py",
                    ROOT/"src/baselines/lara.py", ROOT/"src/evaluation/review_metrics.py")}, "datasets": {}}
    write_json(output/"experiment.json", metadata)
    records, queries, ledgers = [], [], []
    started = time.perf_counter()
    for name in ("trecdl", "robust04"):
        original = datasets.SegmentedRankingDataset.load(cache/"data"/(name+".npz"))
        rows = sorted(binary_top10(original, design["binary_relevance_threshold"]), key=pair_key)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
        qids = sorted({r["query_id"] for r in rows})
        group_indices = {q: [i for i, r in enumerate(rows) if r["query_id"] == q] for q in qids}
        assert all(len(indices) == 10 for indices in group_indices.values())
        labels = {pair_key(r): r["true_binary"] for r in rows}
        gold = np.asarray([r["true_binary"] for r in rows])
        probabilities = np.asarray([r["prob_relevant"] for r in rows])
        predicted = (probabilities >= .5).astype(int)
        evaluator = PolicyEvaluator(rows)
        discounts = 1/np.log2(np.asarray([r["rank"] for r in rows])+1)
        def score(values):
            return float(np.dot(values, discounts)/len(qids))
        truth = score(gold)
        metadata["datasets"][name] = {"queries": len(qids), "pairs": len(rows), "true_binary_dcg10": truth,
                                      "raw_hard_label_dcg10": score(predicted), "raw_probability_dcg10": score(probabilities)}
        for seed in range(design["first_seed"], design["first_seed"]+design["repetitions"]):
            shuffled_qids = qids.copy()
            random.Random(seed).shuffle(shuffled_qids)
            for fraction in design["human_fractions"]:
                calibration_count = int(len(qids)*fraction)
                calibration_qids = set(shuffled_qids[:calibration_count])
                calibration_mask = np.asarray([r["query_id"] in calibration_qids for r in rows])
                budget = int(calibration_mask.sum())
                ids = {"dataset": name, "seed": seed, "requested_human_fraction": fraction,
                       "budget": budget, "human_fraction": budget/len(rows)}
                def record_annotation(method, human, prediction):
                    metric, qrows = evaluator.evaluate(human, prediction)
                    assert metric["human_labels"] == budget
                    mixed_score = score(np.where(human, gold, prediction))
                    records.append({**ids, "method": method, "output_type": "mixed_labels", "pairs": len(rows),
                        "queries": len(qids), "human_labels": budget, "dcg_estimate": mixed_score, "true_dcg": truth,
                        "dcg_absolute_error": abs(mixed_score-truth), "lower": None, "upper": None,
                        "interval_width": None, "interval_covers": None,
                        "automatic_label_coverage": metric["coverage"], "query_auto_error": metric["query_mean_auto_error_rate"],
                        "selective_risk": metric["selective_risk"], "class_0_coverage": metric["class_0_coverage"],
                        "class_1_coverage": metric["class_1_coverage"]})
                    queries.extend({**ids, "method": method, **r} for r in qrows)
                record_annotation("raw_confidence", complete_budget(pool, [], 1-np.maximum(probabilities, 1-probabilities), budget, seed), predicted)
                record_annotation("random_query_labels", calibration_mask, predicted)
                oracle = LabelOracle(labels, budget)
                human, local_predicted, _, acquired, _ = run_diagnostic_review(pool, oracle, budget, review_config, seed)
                record_annotation("local_diagnosis", human, local_predicted)
                ledgers.append(json.dumps({**ids, "method": "local_diagnosis", "purchases": acquired}))
                index = {pair_key(r): i for i, r in enumerate(rows)}
                for grouping in (1, "topics"):
                    for preprocessing in ("raw", "upstream_smooth"):
                        oracle = LabelOracle(labels, budget)
                        result, acquisitions, _ = run_lara(pool, oracle, budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
                        human = np.zeros(len(rows), dtype=bool)
                        lara_predicted = np.zeros(len(rows), dtype=int)
                        for row in result:
                            i = index[pair_key(row)]
                            human[i] = row["label_source"] == "human"
                            lara_predicted[i] = row["predicted_label"]
                        method = f"lara_{grouping}_{preprocessing}"
                        record_annotation(method, human, lara_predicted)
                        ledgers.append(json.dumps({**ids, "method": method, "purchases": acquisitions}))
                oracle = LabelOracle(labels, budget)
                calibration_rows = [r for r in rows if r["query_id"] in calibration_qids]
                unseen_rows = [r for r in rows if r["query_id"] not in calibration_qids]
                observed = {pair_key(r): oracle.query(pair_key(r)) for r in calibration_rows}
                validation = reference_dataset(calibration_rows, datasets.SegmentedRankingDataset, observed)
                unseen = reference_dataset(unseen_rows, datasets.SegmentedRankingDataset)
                known_weight = calibration_count/len(qids)
                known_contribution = known_weight*validation.true_dcg()
                for method, function in (("oosterhuis_ppi", methods.ppi), ("oosterhuis_crc", methods.crc_bootstrap),
                                         ("gold_bootstrap", methods.bootstrap)):
                    # Each method gets a fresh copy to keep its RNG independent.
                    lower, upper = function(validation.with_seed(seed), unseen.with_seed(seed))
                    lower = known_contribution+(1-known_weight)*float(lower)
                    upper = known_contribution+(1-known_weight)*float(upper)
                    midpoint = (lower+upper)/2
                    assert np.isfinite([lower, upper]).all() and lower <= upper
                    records.append({**ids, "method": method, "output_type": "metric_interval", "pairs": len(rows),
                        "queries": len(qids), "human_labels": len(oracle.queried), "dcg_estimate": midpoint,
                        "true_dcg": truth, "dcg_absolute_error": abs(midpoint-truth), "lower": lower, "upper": upper,
                        "interval_width": upper-lower, "interval_covers": lower <= truth <= upper,
                        "automatic_label_coverage": None, "query_auto_error": None, "selective_risk": None,
                        "class_0_coverage": None, "class_1_coverage": None})
                ledgers.append(json.dumps({**ids, "method": "metric_estimator_calibration",
                                          "purchases": [{"query_id": q, "doc_id": d} for q, d in oracle.queried]}))
            print(f"Matched comparison {name}: seed {seed} complete", flush=True)
    write_csv(output/"runs.csv", records)
    write_csv(output/"per_query.csv", queries)
    atomic_text(output/"purchases.jsonl", "\n".join(ledgers)+"\n")
    grouped = defaultdict(list)
    for row in records:
        grouped[row["dataset"], row["budget"], row["method"]].append(row)
    fields = ("human_fraction", "dcg_absolute_error", "interval_width", "interval_covers", "automatic_label_coverage",
              "query_auto_error", "selective_risk", "class_0_coverage", "class_1_coverage")
    summary = []
    for (name, budget, method), group in sorted(grouped.items()):
        item = {"dataset": name, "budget": budget, "method": method, "repetitions": len(group)}
        for field in fields:
            values = [r[field] for r in group if r[field] is not None]
            item[field+"_mean"] = float(np.mean(values)) if values else None
        summary.append(item)
    write_csv(output/"summary.csv", summary)
    metadata.update(status="complete", evaluations=len(records), elapsed_seconds=time.perf_counter()-started)
    write_json(output/"experiment.json", metadata)
    atomic_text(output/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file()))
    print(f"Completed {len(records)} matched method evaluations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
