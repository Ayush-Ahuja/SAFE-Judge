"""Choose target-domain settings on development queries and compare total costs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from decimal import Decimal
import gzip
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import load_cohort
from scripts.run_diagnostic_confirmation import fit_platt
from scripts.run_error_predictor_ablation import complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.safe_judge import SafeJudgeConfig, run_safe_judge
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

EXTRACTED = ROOT / "results/transfer/qwen35_transfer_v1_dataset_full"
OUTPUT = ROOT / "results/experiments/target_domain_tuning_v1"
CONFIG = ROOT / "configs/target_domain_tuning.json"


def scored_rows(dataset):
    rows = load_cohort(EXTRACTED, dataset + "_all")
    return [{**r, "raw_logit": float(r["yes_logit"])-float(r["no_logit"])} for r in rows]


def fit_calibration(training):
    if len({r["true_binary"] for r in training}) == 1:
        return {"kind": "constant", "probability": (sum(r["true_binary"] for r in training)+1)/(len(training)+2)}
    model = fit_platt(training)
    return {"kind": "platt", "slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def apply_calibration(rows, parameters):
    if parameters["kind"] == "constant":
        return np.full(len(rows), parameters["probability"])
    from scipy.special import expit
    return expit(parameters["intercept"] + parameters["slope"]*np.asarray([r["raw_logit"] for r in rows]))


def candidates(settings, frozen):
    result = []
    for view in settings["score_views"]:
        for shared in settings["shared_strengths"]:
            for local in settings["local_strengths"]:
                config = replace(frozen, shared_strength=shared, local_strength=local)
                result.append({"family": "ours", "view": view, "grouping": None, "config": asdict(config)})
            config = replace(frozen, shared_strength=shared, global_only=True)
            result.append({"family": "ours", "view": view, "grouping": None, "config": asdict(config)})
        for learning_rate in settings["lara_learning_rates"]:
            for grouping in (1, "topics"):
                for preprocessing in ("raw", "upstream_smooth"):
                    config = LaraConfig(learning_rate=learning_rate, preprocessing=preprocessing)
                    result.append({"family": "lara", "view": view, "grouping": grouping, "config": asdict(config)})
    for index, candidate in enumerate(result):
        candidate["id"] = f"candidate_{index:02d}"
    return result


def run_policy(rows, probabilities, candidate, budget, seed):
    pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": float(p)}
            for r, p in zip(rows, probabilities)]
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    oracle = LabelOracle(labels, budget)
    if candidate["family"] == "ours":
        human, predicted, _, acquired, _ = run_safe_judge(pool, oracle, budget, SafeJudgeConfig(**candidate["config"]), seed)
    elif candidate["family"] == "lara":
        result, acquired, _ = run_lara(pool, oracle, budget, LaraConfig(**candidate["config"]), seed, candidate["grouping"])
        human = np.asarray([r["label_source"] == "human" for r in result])
        predicted = np.asarray([r["predicted_label"] for r in result])
        if [pair_key(r) for r in result] != [pair_key(r) for r in rows]:
            raise ValueError("LARA output order changed")
    else:
        priority = 1-np.maximum(probabilities, 1-probabilities)
        human = complete_budget(pool, [], priority, budget, seed)
        predicted = np.asarray(probabilities) >= .5
        acquired = []
        for i in np.flatnonzero(human):
            acquired.append({"query_id": rows[i]["query_id"], "doc_id": rows[i]["doc_id"],
                             "observed_label": oracle.query(pair_key(rows[i]))})
    purchases = [pair_key(r) for r in acquired]
    if len(purchases) != len(set(purchases)) or set(purchases) != set(oracle.queried) or len(purchases) != budget:
        raise ValueError("Invalid policy acquisition ledger")
    if {pair_key(r) for r, selected in zip(rows, human) if selected} != set(purchases):
        raise ValueError("Acquisitions differ from human output mask")
    return human, predicted, acquired


def prepare_splits(settings):
    result = []
    for dataset in ("trec_covid", "chatreport"):
        rows = scored_rows(dataset)
        qids = sorted({r["query_id"] for r in rows})
        if dataset == "trec_covid":
            info = json.loads((EXTRACTED / "inputs/trec_covid.metadata.json").read_text())
            development_sets = [("existing_split", sorted(info["sampling"]["development_query_ids"]))]
        else:
            development_sets = []
            for seed in settings["chatreport_split_seeds"]:
                shuffled = qids.copy()
                random.Random(seed).shuffle(shuffled)
                development_sets.append((f"split_{seed}", sorted(shuffled[:settings["chatreport_development_queries"]])))
        for name, development_ids in development_sets:
            dev_set = set(development_ids)
            dev = [r for r in rows if r["query_id"] in dev_set]
            test = [r for r in rows if r["query_id"] not in dev_set]
            if {r["query_id"] for r in test} & dev_set:
                raise ValueError("Development/evaluation query overlap")
            folds = []
            # COVID has five two-query validation folds. ChatReport uses leave-one-development-query-out.
            partitions = np.array_split(np.asarray(development_ids), 5 if dataset == "trec_covid" else len(development_ids))
            for heldout in partitions:
                validation_ids = set(heldout.tolist())
                training = [r for r in dev if r["query_id"] not in validation_ids]
                validation = [r for r in dev if r["query_id"] in validation_ids]
                parameters = fit_calibration(training)
                folds.append({"training_query_ids": sorted({r["query_id"] for r in training}),
                              "validation_query_ids": sorted(validation_ids), "rows": validation,
                              "calibration": parameters,
                              "probabilities": {"raw": [r["prob_relevant"] for r in validation],
                                                "target_platt": apply_calibration(validation, parameters).tolist()}})
            result.append({"id": f"{dataset}_{name}", "dataset": dataset, "development_query_ids": development_ids,
                           "evaluation_query_ids": sorted({r["query_id"] for r in test}),
                           "development_pairs": len(dev), "evaluation_pairs": len(test), "folds": folds})
    return result


def evaluate_candidate(split, candidate, settings):
    destination = OUTPUT / "development" / split["id"] / (candidate["id"] + ".json")
    if destination.exists():
        return json.loads(destination.read_text())
    records = []
    for index, fold in enumerate(split["folds"]):
        rows = fold["rows"]
        evaluator = PolicyEvaluator(rows)
        probabilities = np.asarray(fold["probabilities"][candidate["view"]])
        for fraction in settings["development_review_fractions"]:
            budget = int(Decimal(str(fraction))*len(rows))
            for seed in settings["development_cv_seeds"]:
                human, predicted, _ = run_policy(rows, probabilities, candidate, budget, seed)
                metrics, _ = evaluator.evaluate(human, predicted)
                records.append({"fold": index, "fraction": fraction, "seed": seed,
                                "query_auto_error": metrics["query_mean_auto_error_rate"]})
    result = {"split": split["id"], "candidate": candidate,
              "development_cv_loss": float(np.mean([r["query_auto_error"] for r in records])), "fold_results": records}
    write_json(destination, result)
    return result


def evaluate_split_seed(split, choices, frozen, seed, settings):
    rows = scored_rows(split["dataset"])
    development_ids = set(split["development_query_ids"])
    development = [r for r in rows if r["query_id"] in development_ids]
    test_indices = np.asarray([i for i, r in enumerate(rows) if r["query_id"] not in development_ids])
    test = [rows[i] for i in test_indices]
    calibration = choices["calibration"]
    methods = {"target_tuned_ours": ("development", choices["ours"]),
               "target_tuned_lara": ("development", choices["lara"]),
               "development_frozen_ours": ("development", frozen),
               "development_frozen_lara_raw": ("development", {"family": "lara", "view": "raw", "grouping": 1,
                                                                  "config": asdict(LaraConfig(preprocessing="raw"))}),
               "full_pool_frozen_ours": ("full", frozen),
               "full_pool_confidence": ("full", {"family": "confidence", "view": "raw"})}
    for grouping in (1, "topics"):
        for preprocessing in ("raw", "upstream_smooth"):
            methods[f"full_pool_lara_{grouping}_{preprocessing}"] = ("full", {"family": "lara", "view": "raw", "grouping": grouping,
                                                                                    "config": asdict(LaraConfig(preprocessing=preprocessing))})
    evaluator = PolicyEvaluator(rows)
    results = []
    for fraction in settings["total_human_fractions"]:
        total_budget = int(Decimal(str(fraction))*len(rows))
        if total_budget < len(development):
            raise ValueError("Total budget cannot pay for the declared development set")
        for method, (mode, candidate) in methods.items():
            destination = OUTPUT / "evaluation" / split["id"] / method / f"seed_{seed}_budget_{total_budget}.json.gz"
            if destination.exists():
                with gzip.open(destination, "rt", encoding="utf-8") as stream:
                    results.append(json.load(stream)["record"])
                continue
            current = test if mode == "development" else rows
            budget = total_budget-len(development) if mode == "development" else total_budget
            p = apply_calibration(current, calibration) if candidate["view"] == "target_platt" else np.asarray([r["prob_relevant"] for r in current])
            current_human, current_predicted, acquired = run_policy(current, p, candidate, budget, seed)
            if mode == "development":
                human = np.ones(len(rows), dtype=bool)
                predicted = np.asarray([r["predicted_label"] for r in rows])
                human[test_indices] = current_human
                predicted[test_indices] = current_predicted
                purchases = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "observed_label": r["true_binary"],
                              "stage": "development_label"} for r in development] + acquired
            else:
                human, predicted, purchases = current_human, current_predicted, acquired
            if len({pair_key(r) for r in purchases}) != total_budget:
                raise ValueError("Development and online costs fail the total budget check")
            metrics, query_rows = evaluator.evaluate(human, predicted)
            if metrics["human_labels"] != total_budget:
                raise ValueError("Final human mask violates total cost")
            record = {"split": split["id"], "dataset": split["dataset"], "seed": seed, "method": method,
                      "budget": total_budget, "requested_total_human_fraction": fraction,
                      "development_labels": len(development) if mode == "development" else 0,
                      "online_labels": budget, "score_view": candidate["view"], **metrics}
            result = {"record": record, "candidate": candidate, "per_query": query_rows, "purchases": purchases,
                      "automatic_predicted_labels": np.asarray(predicted, dtype=int).tolist(),
                      "calibration": calibration if candidate["view"] == "target_platt" else None}
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(result, stream, allow_nan=False)
            temporary.replace(destination)
            results.append(record)
    return results


def main():
    settings = json.loads(CONFIG.read_text())
    selected_path = ROOT / "configs/safe_judge.json"
    frozen_config = SafeJudgeConfig(**json.loads(selected_path.read_text())["review_config"])
    frozen = {"family": "ours", "view": "raw", "grouping": None, "config": asdict(frozen_config)}
    design = {"settings": settings, "selected_policy": asdict(frozen_config), "config_sha256": file_hash(CONFIG),
              "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in (Path(__file__),
                  ROOT / "scripts/run_diagnostic_confirmation.py", ROOT / "src/calibration/safe_judge.py",
                  ROOT / "src/baselines/lara.py", ROOT / "src/evaluation/review_metrics.py")},
              "input_predictions_sha256": {dataset: file_hash(EXTRACTED / f"runs/{dataset}/predictions.csv")
                                           for dataset in ("trec_covid", "chatreport")}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity_path = OUTPUT / "design.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != design:
        raise ValueError("Tuning design changed. Choose a new output directory.")
    write_json(identity_path, design)
    splits = prepare_splits(settings)
    write_json(OUTPUT / "splits.json", [{k: v for k, v in split.items() if k != "folds"} for split in splits])
    candidate_list = candidates(settings, frozen_config)
    write_json(OUTPUT / "candidates.json", candidate_list)
    started = time.perf_counter()
    development_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_candidate, split, candidate, settings) for split in splits for candidate in candidate_list]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            development_results.append(result)
            if index % 12 == 0:
                print(f"Development selection {index}/{len(futures)} candidates completed", flush=True)
    choices = {}
    for split in splits:
        best = {}
        for family in ("ours", "lara"):
            eligible = [r for r in development_results if r["split"] == split["id"] and r["candidate"]["family"] == family]
            winner = min(eligible, key=lambda r: (r["development_cv_loss"], r["candidate"]["id"]))
            best[family] = winner["candidate"]
            best[family + "_development_loss"] = winner["development_cv_loss"]
        dev = [r for r in scored_rows(split["dataset"]) if r["query_id"] in set(split["development_query_ids"])]
        best["calibration"] = fit_calibration(dev)
        choices[split["id"]] = best
    # Selection and final calibrators are saved before evaluating the other queries.
    write_json(OUTPUT / "selected_parameters.json", choices)
    write_csv(OUTPUT / "development_scores.csv", [{"split": r["split"], "family": r["candidate"]["family"],
              "candidate": r["candidate"]["id"], "view": r["candidate"]["view"], "cv_loss": r["development_cv_loss"]}
              for r in sorted(development_results, key=lambda r: (r["split"], r["candidate"]["id"]))])
    records = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_split_seed, split, choices[split["id"]], frozen, seed, settings)
                   for split in splits for seed in settings["evaluation_seeds"]]
        for index, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            print(f"Evaluation {index}/{len(futures)} split/seed jobs completed", flush=True)
    records.sort(key=lambda r: (r["split"], r["budget"], r["method"], r["seed"]))
    write_csv(OUTPUT / "runs.csv", records)
    write_json(OUTPUT / "experiment.json", {"status": "complete", "design_sha256": file_hash(identity_path),
        "selected_parameters_sha256": file_hash(OUTPUT / "selected_parameters.json"), "evaluation_rows": len(records),
        "development_candidates": len(development_results), "candidate_policy_evaluations": sum(len(r["fold_results"]) for r in development_results),
        "elapsed_seconds": time.perf_counter()-started,
        "scope": "Exploratory domain-specific tuning after prior dataset inspection. Development and evaluation queries are disjoint. All development labels count once toward each deployment budget. ChatReport repeated splits overlap."})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(OUTPUT).as_posix()}\n" for p in sorted(OUTPUT.rglob("*"))
                if p.is_file() and p.name != "checksums.sha256"))
    print(f"Completed {len(records)} evaluations in {(time.perf_counter()-started)/60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
