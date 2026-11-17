"""Query-disjoint grade densities, error prediction and matched-cost review."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.special import expit, logit
from scipy.stats import skew, kurtosis
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import load_cohort
from scripts.run_diagnostic_confirmation import fit_platt
from scripts.run_error_predictor_ablation import complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.diagnostic_review import ReviewConfig, run_diagnostic_review
from src.calibration.gaussian_grades import GradeDensityModel, relevance_probability, binary_error_probability
from src.evaluation.metrics import classification_metrics
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, read_predictions, write_csv, write_json

EXTRACTED = ROOT / "results/transfer/qwen35_transfer_v1_dataset_full"
OUTPUT = ROOT / "results/experiments/gaussian_grade_v1"
CONFIG = ROOT / "configs/gaussian_grade_experiment.json"


def qwen_rows(cohort):
    if cohort == "dl2019":
        rows, _ = read_predictions(ROOT / "results/stage2/dl2019_qwen35_full_v1/predictions.csv", 2)
        rows.sort(key=pair_key)
    else:
        rows = load_cohort(EXTRACTED, cohort)
    return [{"query_id": r["query_id"], "doc_id": r["doc_id"], "true_relevance": int(r["true_relevance"]),
             "true_binary": int(r["true_binary"]), "prob_relevant": float(r["prob_relevant"]),
             "margin": float(r["yes_logit"])-float(r["no_logit"]),
             "mass_logit": float(logit(np.clip(float(r["answer_probability_mass"]), 1e-8, 1-1e-8))),
             "log_tokens": float(np.log1p(int(r["input_tokens"])))} for r in rows]


def scenarios(settings):
    result = []
    source = qwen_rows("dl2019")
    target = qwen_rows("dl2020_all")
    target_ids = {r["query_id"] for r in target}
    qids = sorted({r["query_id"] for r in source}-target_ids)
    random.Random(settings["dl2019_source_query_seed"]).shuffle(qids)
    selected = set(qids[:settings["source_calibration_queries"]])
    result.append({"id": "qwen_dl2020", "assessor": "Qwen3.5-4B binary", "grades": [0, 1, 2, 3], "threshold": 2,
                   "calibration": [r for r in source if r["query_id"] in selected], "test": target,
                   "feature_names": ["Yes-minus-No logit", "No/Yes vocabulary mass logit", "log(1+input tokens)"], "policy": True})
    covid = qwen_rows("trec_covid_all")
    info = json.loads((EXTRACTED / "inputs/trec_covid.metadata.json").read_text())
    dev_ids = set(info["sampling"]["development_query_ids"])
    result.append({"id": "qwen_covid", "assessor": "Qwen3.5-4B binary, strict COVID target", "grades": [0, 1, 2], "threshold": 2,
                   "calibration": [r for r in covid if r["query_id"] in dev_ids], "test": [r for r in covid if r["query_id"] not in dev_ids],
                   "feature_names": result[0]["feature_names"], "policy": True})
    chat = qwen_rows("chatreport_all")
    for seed in settings["chatreport_split_seeds"]:
        qids = sorted({r["query_id"] for r in chat})
        random.Random(seed).shuffle(qids)
        dev_ids = set(qids[:settings["chatreport_development_queries"]])
        result.append({"id": f"qwen_chat_{seed}", "assessor": "Qwen3.5-4B binary", "grades": [0, 1, 2], "threshold": 1,
                       "calibration": [r for r in chat if r["query_id"] in dev_ids], "test": [r for r in chat if r["query_id"] not in dev_ids],
                       "feature_names": result[0]["feature_names"], "policy": True})
    for name in ("trecdl", "robust04"):
        path = ROOT / f"cache/research/oosterhuis_c1c0c1a/data/{name}.npz"
        with np.load(path) as archive:
            probabilities = archive["pred_label_dist"].astype(float)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gold = archive["true_label_dist"]
            grades = list(range(gold.shape[1]))
            if not np.allclose(gold, np.eye(len(grades))[gold.argmax(axis=1)]):
                raise ValueError("Expected one-hot human grade distributions")
            rows = []
            for index, (p, y, qid) in enumerate(zip(probabilities, gold.argmax(axis=1), archive["query_segments"])):
                relevant = float(p[2:].sum())
                rows.append({"query_id": str(qid), "doc_id": str(index), "true_relevance": int(y), "true_binary": int(y >= 2),
                             "prob_relevant": relevant, "margin": float(logit(np.clip(relevant, 1e-8, 1-1e-8))),
                             "grade_probabilities": p.tolist(), "alr": np.log((p[1:]+1e-8)/(p[0]+1e-8)).tolist()})
        rows.sort(key=pair_key)
        qids = sorted({r["query_id"] for r in rows})
        random.Random(settings["flan_query_split_seed"]).shuffle(qids)
        dev_ids = set(qids[:settings["flan_calibration_queries"]])
        result.append({"id": "flan_"+name, "assessor": "Released Flan-UL2 graded probabilities", "grades": grades, "threshold": 2,
                       "calibration": [r for r in rows if r["query_id"] in dev_ids], "test": [r for r in rows if r["query_id"] not in dev_ids],
                       "feature_names": [f"log(P(grade {g})/P(grade 0))" for g in grades[1:]], "policy": False,
                       "source_sha256": file_hash(path)})
    for study in result:
        if {r["query_id"] for r in study["calibration"]} & {r["query_id"] for r in study["test"]}:
            raise ValueError("Calibration and evaluation queries overlap")
    return result


def features(rows, view):
    if view == "margin":
        return np.asarray([[r["margin"]] for r in rows])
    if "alr" in rows[0]:
        return np.asarray([r["alr"] for r in rows])
    return np.asarray([[r["margin"], r["mass_logit"], r["log_tokens"]] for r in rows])


def calibration_sample(rows, per_query, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    selected = []
    for qid, group in sorted(groups.items()):
        group = sorted(group, key=pair_key)
        random.Random(f"gaussian-grade/{seed}/{qid}").shuffle(group)
        if len(group) < per_query:
            raise ValueError("Not enough calibration pairs per query")
        selected.extend(group[:per_query])
    return sorted(selected, key=pair_key)


def predictive_metrics(study, name, relevant, grade_probabilities=None):
    rows, grades = study["test"], study["grades"]
    y = np.asarray([r["true_binary"] for r in rows])
    raw = np.asarray([r["prob_relevant"] >= .5 for r in rows])
    error = raw != y
    estimated = binary_error_probability(relevant, raw)
    evaluated = [{**r, "prob_relevant": float(p), "predicted_label": int(p >= .5)} for r, p in zip(rows, relevant)]
    metrics = classification_metrics(evaluated)
    metrics.pop("reliability")
    result = {"model": name, **metrics, "raw_error_prediction_brier": float(np.mean((estimated-error)**2)),
              "raw_error_auc": float(roc_auc_score(error, estimated)), "raw_error_average_precision": float(average_precision_score(error, estimated)),
              "grade_macro_f1": None, "grade_quadratic_kappa": None, "grade_nll": None, "grade_brier": None}
    if grade_probabilities is not None:
        predicted = np.asarray(grades)[grade_probabilities.argmax(axis=1)]
        gold = np.asarray([r["true_relevance"] for r in rows])
        onehot = np.eye(len(grades))[gold]
        result.update(grade_macro_f1=float(f1_score(gold, predicted, labels=grades, average="macro", zero_division=0)),
                      grade_quadratic_kappa=float(cohen_kappa_score(gold, predicted, labels=grades, weights="quadratic")),
                      grade_nll=float(-np.log(np.clip(grade_probabilities[np.arange(len(rows)), gold], 1e-12, 1)).mean()),
                      grade_brier=float(((grade_probabilities-onehot)**2).sum(axis=1).mean()))
    return result


def baseline_allocation(pool, labels, method, budget, seed):
    p = np.asarray([r["prob_relevant"] for r in pool])
    oracle = LabelOracle(labels, budget)
    if method == "selected_local":
        cfg = ReviewConfig(**json.loads((ROOT / "configs/diagnostic_review_selected.json").read_text())["review_config"])
        human, predicted, _, acquired, _ = run_diagnostic_review(pool, oracle, budget, cfg, seed)
    elif method.startswith("lara_"):
        grouping = "topics" if method.startswith("lara_topics_") else 1
        preprocessing = "upstream_smooth" if method.endswith("upstream_smooth") else "raw"
        result, acquired, _ = run_lara(pool, oracle, budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
        human = np.asarray([r["label_source"] == "human" for r in result])
        predicted = np.asarray([r["predicted_label"] for r in result])
    else:
        priority = 1-np.maximum(p, 1-p)
        if method == "confidence_queryweighted":
            counts = {q: sum(r["query_id"] == q for r in pool) for q in {r["query_id"] for r in pool}}
            priority /= np.asarray([counts[r["query_id"]] for r in pool])
        human = complete_budget(pool, [], priority, budget, seed)
        predicted = p >= .5
        acquired = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "observed_label": oracle.query(pair_key(r))}
                    for r, selected in zip(pool, human) if selected]
    if len(acquired) != len(set(oracle.queried)) or len(acquired) != budget:
        raise ValueError("Baseline purchase ledger mismatch")
    return human, predicted


def experiment_job(study, seed, settings):
    destination = OUTPUT / "fits" / study["id"] / f"seed_{seed}.json.gz"
    if destination.exists() and destination.with_suffix(".npz").exists():
        with gzip.open(destination, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    calibration = calibration_sample(study["calibration"], settings["labels_per_calibration_query"], seed)
    test, grades = study["test"], study["grades"]
    gold = np.asarray([r["true_relevance"] for r in calibration])
    fitted, parameters, posterior_arrays = {}, {}, {}
    for view in ("margin", "full"):
        train_x, test_x = features(calibration, view), features(test, view)
        for family in settings["grade_model_families"]:
            name = f"grade_{family}_{view}"
            model = GradeDensityModel(family, grades, settings["covariance_shrinkage"], settings["prior_pseudocount"], seed).fit(train_x, gold)
            posterior = model.predict_proba(test_x)
            p = relevance_probability(posterior, grades, study["threshold"])
            fitted[name] = p, posterior
            parameters[name] = model.parameters()
            posterior_arrays[name] = posterior.astype(np.float32)
        for family in settings["binary_gaussian_controls"]:
            name = f"binary_{family}_{view}"
            model = GradeDensityModel(family, [0, 1], settings["covariance_shrinkage"], settings["prior_pseudocount"], seed).fit(train_x, gold >= study["threshold"])
            p = model.predict_proba(test_x)[:, 1]
            fitted[name] = p, None
            parameters[name] = model.parameters()
    prior = GradeDensityModel("prior", grades, prior_pseudocount=settings["prior_pseudocount"]).fit(features(calibration, "margin"), gold)
    posterior = prior.predict_proba(features(test, "margin"))
    fitted["grade_prior"] = relevance_probability(posterior, grades, study["threshold"]), posterior
    platt_rows = [{**r, "raw_logit": r["margin"]} for r in calibration]
    if len({r["true_binary"] for r in platt_rows}) == 1:
        p = np.full(len(test), (sum(r["true_binary"] for r in platt_rows)+.5)/(len(platt_rows)+1))
    else:
        platt = fit_platt(platt_rows)
        p = platt.predict_proba(features(test, "margin"))[:, 1]
    fitted["platt_binary"] = p, None
    raw_p = np.asarray([r["prob_relevant"] for r in test])
    fitted["raw_assessor"] = raw_p, np.asarray([r["grade_probabilities"] for r in test]) if "grade_probabilities" in test[0] else None
    prediction_records = [{"scenario": study["id"], "seed": seed, "calibration_labels": len(calibration),
                           **predictive_metrics(study, name, *values)} for name, values in fitted.items()]
    policy_records, automatic_masks = [], {}
    if study["policy"]:
        evaluator = PolicyEvaluator(test)
        keys = [pair_key(r) for r in test]
        labels = {pair_key(r): int(r["true_binary"]) for r in test}
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in test]
        raw = raw_p >= .5
        sizes = {q: len(ix) for q, ix in evaluator.groups.items()}
        weight = np.asarray([1/sizes[r["query_id"]] for r in test])
        baseline_cache = {}
        baseline_names = ("raw_confidence", "confidence_queryweighted", "selected_local", "lara_1_raw",
                          "lara_1_upstream_smooth", "lara_topics_raw", "lara_topics_upstream_smooth")
        for accounting in settings["accounting"]:
            for fraction in settings["review_fractions"]:
                nominal = int(Decimal(str(fraction))*len(test))
                for baseline in baseline_names:
                    if (baseline, nominal) not in baseline_cache:
                        baseline_cache[baseline, nominal] = baseline_allocation(pool, labels, baseline, nominal, seed)
                    human, predicted = baseline_cache[baseline, nominal]
                    metric, _ = evaluator.evaluate(human, predicted)
                    policy_records.append({"scenario": study["id"], "seed": seed, "accounting": accounting,
                        "nominal_fraction": fraction, "method": baseline, "calibration_labels": 0,
                        "total_human_labels": nominal, **metric})
                for name, (relevant, _) in fitted.items():
                    if name in ("grade_prior", "raw_assessor"):
                        continue
                    budget = nominal if accounting == "matched_review" else nominal-len(calibration)
                    if budget < 0:
                        raise ValueError("Calibration cost exceeds the declared total budget")
                    for mode in ("risk_only", "relabel"):
                        predicted = raw if mode == "risk_only" else relevant >= .5
                        estimated = binary_error_probability(relevant, predicted)
                        human = complete_budget(pool, [], estimated*weight, budget, seed)
                        if int(human.sum()) != budget:
                            raise ValueError("Generative-review budget mismatch")
                        metric, _ = evaluator.evaluate(human, predicted)
                        total = len(calibration)+budget
                        if accounting == "matched_total" and total != nominal:
                            raise ValueError("Calibration-inclusive budget mismatch")
                        method = name+"_"+mode
                        policy_records.append({"scenario": study["id"], "seed": seed, "accounting": accounting,
                            "nominal_fraction": fraction, "method": method, "calibration_labels": len(calibration),
                            "total_human_labels": total, **metric})
                        automatic_masks[f"{accounting}_{fraction}_{method}"] = np.packbits(human).tolist()
    graded_policy_records = []
    grade_y = np.asarray([r["true_relevance"] for r in test])
    query_indices = {q: np.asarray([i for i, r in enumerate(test) if r["query_id"] == q])
                     for q in sorted({r["query_id"] for r in test})}
    grade_pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in test]
    grade_weights = np.asarray([1/len(query_indices[r["query_id"]]) for r in test])
    for name, (_, posterior) in fitted.items():
        if posterior is None:
            continue
        predicted = np.asarray(grades)[posterior.argmax(axis=1)]
        estimated = 1-posterior.max(axis=1)
        cost = 0 if name == "raw_assessor" else len(calibration)
        for accounting in settings["accounting"]:
            for fraction in settings["review_fractions"]:
                nominal = int(Decimal(str(fraction))*len(test))
                budget = nominal if accounting == "matched_review" else nominal-cost
                if budget < 0:
                    raise ValueError("Graded calibration cost exceeds total budget")
                human = complete_budget(grade_pool, [], estimated*grade_weights, budget, seed)
                errors = (~human) & (predicted != grade_y)
                mixed = np.where(human, grade_y, predicted)
                graded_policy_records.append({"scenario": study["id"], "seed": seed, "model": name,
                    "accounting": accounting, "nominal_fraction": fraction, "calibration_labels": cost,
                    "human_labels": budget, "total_human_labels": budget+cost,
                    "coverage": float((~human).mean()), "automatic_errors": int(errors.sum()),
                    "query_auto_error": float(np.mean([errors[ix].mean() for ix in query_indices.values()])),
                    "selective_risk": float(errors.sum()/(~human).sum()) if (~human).any() else None,
                    "mixed_grade_macro_f1": float(f1_score(grade_y, mixed, labels=grades, average="macro", zero_division=0)),
                    "mixed_grade_quadratic_kappa": float(cohen_kappa_score(grade_y, mixed, labels=grades, weights="quadratic"))})
    distribution = []
    for grade in grades:
        subset = [r for r in test if r["true_relevance"] == grade]
        x = np.asarray([r["margin"] for r in subset])
        if len(x):
            distribution.append({"scenario": study["id"], "grade": grade, "test_pairs": len(x),
                "mean_margin": float(x.mean()), "std_margin": float(x.std()),
                "skewness": float(skew(x)) if len(x) >= 3 and x.std() else None,
                "excess_kurtosis": float(kurtosis(x)) if len(x) >= 4 and x.std() else None,
                "raw_binary_error_rate": float(np.mean([(r["prob_relevant"] >= .5) != r["true_binary"] for r in subset]))})
    result = {"scenario": study["id"], "seed": seed, "calibration_pairs": [{"query_id": r["query_id"], "doc_id": r["doc_id"], "grade": r["true_relevance"]} for r in calibration],
              "model_parameters": parameters, "prediction_metrics": prediction_records, "policy_metrics": policy_records,
              "graded_policy_metrics": graded_policy_records,
              "packed_review_masks": automatic_masks, "grade_distribution": distribution}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)
    temporary.replace(destination)
    np.savez_compressed(destination.with_suffix(".npz"), **posterior_arrays)
    return result


def main():
    settings = json.loads(CONFIG.read_text())
    studies = scenarios(settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design = {"settings": settings, "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
              (Path(__file__), ROOT / "src/calibration/gaussian_grades.py", ROOT / "src/calibration/diagnostic_review.py",
               ROOT / "src/baselines/lara.py", ROOT / "src/evaluation/review_metrics.py", ROOT / "scripts/run_diagnostic_confirmation.py")},
              "scenarios": [{k: v for k, v in s.items() if k not in ("calibration", "test")} | {
                  "calibration_pool_sha256": hashlib.sha256(json.dumps(s["calibration"], sort_keys=True).encode()).hexdigest(),
                  "test_pool_sha256": hashlib.sha256(json.dumps(s["test"], sort_keys=True).encode()).hexdigest(),
                  "test_pairs": len(s["test"]), "calibration_queries": sorted({r["query_id"] for r in s["calibration"]}),
                  "test_queries": sorted({r["query_id"] for r in s["test"]})} for s in studies]}
    path = OUTPUT / "design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError("Gaussian experiment design changed; choose a new output directory")
    write_json(path, design)
    predictions, policies, graded_policies, distributions = [], [], [], []
    with ProcessPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(experiment_job, study, seed, settings) for study in studies for seed in settings["seeds"]]
        for index, job in enumerate(as_completed(jobs), 1):
            result = job.result()
            predictions.extend(result["prediction_metrics"])
            policies.extend(result["policy_metrics"])
            graded_policies.extend(result["graded_policy_metrics"])
            if result["seed"] == settings["seeds"][0]:
                distributions.extend(result["grade_distribution"])
            print(f"Gaussian experiment {index}/{len(jobs)} | {result['scenario']} | {result['seed']}", flush=True)
    predictions.sort(key=lambda r: (r["scenario"], r["model"], r["seed"]))
    policies.sort(key=lambda r: (r["scenario"], r["accounting"], r["nominal_fraction"], r["method"], r["seed"]))
    write_csv(OUTPUT / "prediction_metrics.csv", predictions)
    write_csv(OUTPUT / "policy_metrics.csv", policies)
    graded_policies.sort(key=lambda r: (r["scenario"], r["accounting"], r["nominal_fraction"], r["model"], r["seed"]))
    write_csv(OUTPUT / "graded_policy_metrics.csv", graded_policies)
    write_csv(OUTPUT / "grade_distribution.csv", distributions)
    write_json(OUTPUT / "experiment.json", {"status": "complete", "design_sha256": file_hash(path),
               "fit_repetitions": len(studies)*len(settings["seeds"]), "predictive_evaluations": len(predictions),
               "policy_evaluations": len(policies), "graded_policy_evaluations": len(graded_policies), "scope": settings["scope"]})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(OUTPUT).as_posix()}\n" for p in sorted(OUTPUT.rglob("*"))
                if p.is_file() and p.name != "checksums.sha256"))
    print(f"Completed {len(predictions)} predictive and {len(policies)} review evaluations.", flush=True)


if __name__ == "__main__":
    main()
