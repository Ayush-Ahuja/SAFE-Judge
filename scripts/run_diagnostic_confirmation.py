"""Frozen-policy confirmation on query-disjoint web and new biomedical topics."""

import argparse
from collections import defaultdict
from dataclasses import replace
import csv
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_error_predictor_ablation import aggregate, complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.safe_judge import SafeJudgeConfig, run_safe_judge
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, write_csv, write_json


def load_scored(manifest_path, score_dir):
    metadata = json.loads((score_dir/"experiment.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Require complete scores matching the frozen manifest")
    with (score_dir/"scores.csv").open(encoding="utf-8", newline="") as stream:
        scores = {pair_key(r): float(r["raw_logit"]) for r in csv.DictReader(stream)}
    manifest = read_manifest(manifest_path)
    if set(scores) != {pair_key(r) for r in manifest}:
        raise ValueError("Incomplete or extra scores")
    rows = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "true_binary": r["true_binary"],
             "raw_logit": scores[pair_key(r)], "prob_relevant": float(expit(scores[pair_key(r)]))} for r in manifest]
    return sorted(rows, key=pair_key), {"manifest_sha256": file_hash(manifest_path),
        "scores_sha256": file_hash(score_dir/"scores.csv"), "model": metadata.get("model"),
        "scoring_config": metadata.get("config")}


def calibration_sample(rows, qids, per_query, seed):
    result = []
    for qid in qids:
        group = sorted([r for r in rows if r["query_id"] == qid], key=pair_key)
        random.Random(f"source-calibration/{seed}/{qid}").shuffle(group)
        if len(group) < per_query:
            raise ValueError("Not enough calibration labels")
        result.extend(group[:per_query])
    return result


def fit_platt(rows):
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-9)
    model.fit(np.asarray([r["raw_logit"] for r in rows])[:, None], [r["true_binary"] for r in rows])
    return model


def probability_diagnostics(rows, probabilities):
    y = np.asarray([r["true_binary"] for r in rows])
    p = np.asarray(probabilities)
    predicted = p >= .5
    ece = 0.
    for i in range(10):
        subset = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        if subset.any():
            ece += subset.mean()*abs(float(p[subset].mean()-y[subset].mean()))
    return {"accuracy": float((predicted == y).mean()), "macro_f1": float(f1_score(y, predicted, average="macro")),
            "quadratic_weighted_kappa_binary": float(cohen_kappa_score(y, predicted, weights="quadratic")),
            "brier": float(brier_score_loss(y, p)), "ece": ece,
            "nll": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1]))}


def execute(args):
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    dl, dl_source = load_scored(ROOT/"data/pilot/dl2019.jsonl", ROOT/"results/baselines/cross_encoder/dl2019_minilm_scores_v1")
    covid, covid_source = load_scored(ROOT/"data/research/trec_covid_5000.jsonl", ROOT/"results/experiments/trec_covid_minilm_v1")
    split = json.loads((ROOT/"data/research/trec_covid_5000.metadata.json").read_text())
    qids = sorted({r["query_id"] for r in dl})
    random.Random(settings["source_split_seed"]).shuffle(qids)
    source_ids, web_test_ids = qids[:settings["source_calibration_queries"]], qids[settings["source_calibration_queries"]:]
    source_labels = calibration_sample(dl, source_ids, settings["source_labels_per_query"], settings["source_split_seed"])
    target_labels = calibration_sample(covid, split["development_query_ids"], settings["target_labels_per_query"], settings["source_split_seed"])
    source_platt, target_platt = fit_platt(source_labels), fit_platt(target_labels)
    cohorts = {"web_ce_heldout": [r for r in dl if r["query_id"] in web_test_ids],
               "biomedical_ce_locked": [r for r in covid if r["query_id"] in split["locked_evaluation_query_ids"]]}
    if any(r["query_id"] in source_ids for r in cohorts["web_ce_heldout"]):
        raise AssertionError("Calibration/test query overlap")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output_dir/"configuration.json")
    sources = [Path(__file__), ROOT/"src/calibration/safe_judge.py", ROOT/"src/evaluation/review_metrics.py",
               ROOT/"scripts/run_error_predictor_ablation.py", ROOT/"src/baselines/lara.py", ROOT/"src/baselines/active_learning.py"]
    metadata = {"status": "running", "settings": settings, "inputs": {"web": dl_source, "biomedical": covid_source},
                "source_calibration_query_ids": source_ids, "source_calibration_labels": len(source_labels),
                "target_calibration_query_ids": split["development_query_ids"], "target_calibration_labels": len(target_labels),
                "test_query_ids": {name: sorted({r['query_id'] for r in rows}) for name, rows in cohorts.items()},
                "calibration_pairs": {"source": [pair_key(r) for r in source_labels], "target": [pair_key(r) for r in target_labels]},
                "platt_parameters": {"source": {"slope": float(source_platt.coef_[0, 0]), "intercept": float(source_platt.intercept_[0])},
                                     "target": {"slope": float(target_platt.coef_[0, 0]), "intercept": float(target_platt.intercept_[0])}},
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "config_sha256": file_hash(args.config),
                "scope": "Same frozen MiniLM checkpoint across domains. Calibration queries disjoint from evaluation topics. Online methods may buy labels on evaluation queries; remaining labels are untouched by the learner.",
                "budget": "Target reviews and external calibration labels reported separately and summed. Raw total-cost control reviews the sum directly on target.",
                "confirmatory_limit": "New biomedical topics for policy evaluation; web benchmark already explored. Pilot-informed test-query adaptation is not zero-shot prediction."}
    write_json(args.output_dir/"experiment.json", metadata)
    records, per_query, logs, calibration_metrics = [], [], [], []
    base = SafeJudgeConfig(**settings["review_config"])
    start = time.perf_counter()
    for cohort, rows in cohorts.items():
        rows = sorted(rows, key=pair_key)
        evaluator = PolicyEvaluator(rows)
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        indices = {pair_key(r): i for i, r in enumerate(rows)}
        raw_p = np.asarray([r["prob_relevant"] for r in rows])
        logits = np.asarray([r["raw_logit"] for r in rows])[:, None]
        maps = {"raw": (raw_p, 0), "source_platt": (source_platt.predict_proba(logits)[:, 1], len(source_labels))}
        if cohort == "biomedical_ce_locked":
            maps["target_platt"] = target_platt.predict_proba(logits)[:, 1], len(target_labels)
        for name, (p, external) in maps.items():
            calibration_metrics.append({"cohort": cohort, "calibration": name, "external_labels": external,
                                        **probability_diagnostics(rows, p)})
        for seed in range(settings["first_seed"], settings["first_seed"]+settings["repetitions"]):
            for fraction in settings["human_budget_fractions"]:
                budget = int(len(rows)*fraction)
                def record(method, human, predicted, external, state=None, errors=None):
                    metrics, qrows = evaluator.evaluate(human, predicted)
                    identifiers = {"dataset": cohort, "seed": seed, "budget": budget, "method": method,
                                   "pilot_labels": 0, "external_calibration_labels": external}
                    metrics.update(total_human_labels=metrics["human_labels"]+external,
                                   automatic_repairs=state.get("automatic_repairs", 0) if state else 0,
                                   sentinel_labels=state.get("sentinel_labels", 0) if state else 0)
                    if errors is not None and (~human).any():
                        metrics["automatic_error_brier"] = float(np.mean((errors[~human]-(predicted[~human] != evaluator.y[~human]))**2))
                    else:
                        metrics["automatic_error_brier"] = None
                    records.append({**identifiers, **metrics})
                    per_query.extend({**identifiers, **q} for q in qrows)
                raw_pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": float(p)} for r, p in zip(rows, raw_p)]
                record("raw_confidence", complete_budget(raw_pool, [], 1-np.maximum(raw_p, 1-raw_p), budget, seed), (raw_p >= .5).astype(int), 0)
                total_control = min(len(rows), budget+len(source_labels))
                record("raw_total_cost_control", complete_budget(raw_pool, [], 1-np.maximum(raw_p, 1-raw_p), total_control, seed), (raw_p >= .5).astype(int), 0)
                for method, config in (("diagnostic_primary", base),
                                       ("diagnostic_no_repair", replace(base, repair_labels=False)),
                                       ("diagnostic_shared_only", replace(base, global_only=True))):
                    human, predicted, error, acquired, state = run_safe_judge(raw_pool, LabelOracle(labels, budget), budget, config, seed)
                    record(method, human, predicted, 0, state, error)
                    logs.append(json.dumps({"cohort": cohort, "seed": seed, "budget": budget, "method": method, "acquisitions": acquired}))
                for preprocessing in ("raw", "upstream_smooth"):
                    for grouping in (1, "topics"):
                        predictions, _, _ = run_lara(raw_pool, LabelOracle(labels, budget), budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
                        human, predicted = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=int)
                        for r in predictions:
                            i = indices[pair_key(r)]
                            human[i], predicted[i] = r["label_source"] == "human", r["predicted_label"]
                        record(f"lara_{grouping}_{preprocessing}", human, predicted, 0)
                for mapping, (p, external) in maps.items():
                    if mapping == "raw":
                        continue
                    pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": float(v)} for r, v in zip(rows, p)]
                    record(mapping+"_confidence", complete_budget(pool, [], 1-np.maximum(p, 1-p), budget, seed), (p >= .5).astype(int), external)
                    human, predicted, error, acquired, state = run_safe_judge(pool, LabelOracle(labels, budget), budget, base, seed)
                    record(mapping+"_diagnostic", human, predicted, external, state, error)
                    logs.append(json.dumps({"cohort": cohort, "seed": seed, "budget": budget, "method": mapping+"_diagnostic", "acquisitions": acquired}))
            print(f"{cohort}: seed {seed} complete", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", per_query)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    write_csv(args.output_dir/"probability_calibration.csv", calibration_metrics)
    atomic_text(args.output_dir/"acquisitions.jsonl", "\n".join(logs)+"\n")
    metadata.update(status="complete", policies=len(records), elapsed_seconds=time.perf_counter()-start)
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} confirmation policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/diagnostic_review_confirmation.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args())
