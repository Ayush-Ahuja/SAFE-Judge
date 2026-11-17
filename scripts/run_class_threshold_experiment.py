"""Matched-total-cost, query-disjoint two-threshold experiment on cached Qwen."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
import gzip
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import load_cohort, DETERMINISTIC_METHODS
from scripts.evaluate_proposal_metrics import DL2020Rankings
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.class_thresholds import confidence_order, select_class_thresholds
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, read_predictions, write_csv, write_json

CONFIG = ROOT / "configs/class_threshold_experiment.json"
OUTPUT = ROOT / "results/experiments/class_threshold_v1"
EXTRACTED = ROOT / "results/transfer/qwen35_transfer_v1_dataset_full"
OLD = ROOT / "results/experiments/qwen_transfer_comparison_v1"
BASELINES = ("raw_confidence", "lara_1_raw", "lara_1_upstream_smooth", "lara_topics_raw",
             "lara_topics_upstream_smooth", "selected_local", "lara_reserved_raw", "lara_reserved_upstream_smooth")


def save_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
    temporary.replace(path)


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def studies(settings):
    source, _ = read_predictions(ROOT / "results/stage2/dl2019_qwen35_full_v1/predictions.csv", 2)
    source = sorted(source, key=pair_key)
    dl20 = load_cohort(EXTRACTED, "dl2020_all")
    allowed = sorted({r["query_id"] for r in source}-{r["query_id"] for r in dl20})
    random.Random(settings["source_query_seed"]).shuffle(allowed)
    source_ids = set(allowed[:settings["calibration_queries"]])
    source = [r for r in source if r["query_id"] in source_ids]
    covid = load_cohort(EXTRACTED, "trec_covid_all")
    info = json.loads((EXTRACTED / "inputs/trec_covid.metadata.json").read_text())
    dev_ids = set(info["sampling"]["development_query_ids"])
    target_calibration = [r for r in covid if r["query_id"] in dev_ids]
    covid_test = load_cohort(EXTRACTED, "trec_covid_locked")
    result = [dict(id="dl2020_source", cohort="dl2020_all", calibration=source, test=dl20, query_disjoint=True),
              dict(id="covid_target", cohort="trec_covid_locked", calibration=target_calibration, test=covid_test, query_disjoint=True),
              dict(id="covid_source", cohort="trec_covid_locked", calibration=source, test=covid_test, query_disjoint=False)]
    for study in result:
        # Numeric query IDs from unrelated collections are not shared queries.
        if study["query_disjoint"] and {r["query_id"] for r in study["calibration"]} & {r["query_id"] for r in study["test"]}:
            raise ValueError("Within-collection query overlap")
        if len({r["query_id"] for r in study["calibration"]}) != settings["calibration_queries"]:
            raise ValueError("Calibration query count changed")
    return result


def sample_calibration(rows, per_query, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    result = []
    for qid, group in sorted(groups.items()):
        group = sorted(group, key=pair_key)
        random.Random(f"class-threshold/{seed}/{qid}").shuffle(group)
        if len(group) < per_query:
            raise ValueError("Calibration query has too few pairs")
        result.extend(group[:per_query])
    return sorted(result, key=pair_key)


def reserved_job(cohort, preprocessing, fraction, settings):
    rows = load_cohort(EXTRACTED, cohort)
    budget = int(Decimal(str(fraction))*len(rows))
    destination = OUTPUT / "reserved_lara" / cohort / preprocessing / f"budget_{budget}.json.gz"
    if destination.exists():
        return str(destination)
    p = np.asarray([r["prob_relevant"] for r in rows])
    reserved = np.zeros(len(rows), dtype=bool)
    for ix in confidence_order(p, 6000):
        reserved[ix[:int(np.ceil(settings["minimum_class_coverage"]*len(ix)))]] = True
    keep = np.flatnonzero(~reserved)
    pool = [{k: rows[i][k] for k in ("query_id", "doc_id", "prob_relevant")} for i in keep]
    labels = {pair_key(rows[i]): int(rows[i]["true_binary"]) for i in keep}
    oracle = LabelOracle(labels, budget)
    started = time.perf_counter()
    scored, purchases, state = run_lara(pool, oracle, budget, LaraConfig(preprocessing=preprocessing), 6000, 1)
    human = np.zeros(len(rows), dtype=bool)
    predicted = (p >= .5).astype(int)
    human[keep] = [r["label_source"] == "human" for r in scored]
    predicted[keep] = [r["predicted_label"] for r in scored]
    if human.sum() != budget or len(purchases) != budget or len(oracle.queried) != budget:
        raise ValueError("Reserved-LARA cost mismatch")
    save_gzip(destination, {"human_packed": np.packbits(human).tolist(), "predicted": predicted.tolist(),
        "purchases": purchases, "reserved_packed": np.packbits(reserved).tolist(), "state": state,
        "elapsed_seconds": time.perf_counter()-started,
        "scope": "Supplementary LARA variant: most confident 5% in each raw class are reserved automatic with raw labels. Remaining rows use original LARA."})
    return str(destination)


def baseline(study, method, seed, budget):
    n = len(study["test"])
    if method.startswith("lara_reserved_"):
        preprocessing = method.removeprefix("lara_reserved_")
        path = OUTPUT / "reserved_lara" / study["cohort"] / preprocessing / f"budget_{budget}.json.gz"
        saved = read_gzip(path)
        human = np.unpackbits(np.asarray(saved["human_packed"], dtype=np.uint8))[:n].astype(bool)
        predicted = np.asarray(saved["predicted"])
    else:
        stored_seed = 6000 if method in DETERMINISTIC_METHODS else 6000+(seed-6000) % 10
        path = OLD / "allocations" / study["cohort"] / method / f"seed_{stored_seed}_budget_{budget}.json.gz"
        saved = read_gzip(path)
        purchased = {pair_key(r) for r in saved["purchases"]}
        human = np.asarray([pair_key(r) in purchased for r in study["test"]])
        predicted = np.asarray(saved["automatic_predicted_labels"])
    if len(saved["purchases"]) != budget or human.sum() != budget:
        raise ValueError("Baseline purchase mismatch")
    return human, predicted, {"allocation_source": path.relative_to(ROOT).as_posix(), "allocation_sha256": file_hash(path)}


class RankingMetrics:
    def __init__(self, rows):
        self.binary = DL2020Rankings(rows)
        self.grades = np.asarray([r["true_relevance"] for r in rows])
        self.gold_graded = self.scores(self.grades)
        self.gold_rank = rankdata(-self.gold_graded, method="average")

    def scores(self, grades):
        # trec_eval's default gain is the qrels grade itself, not 2**grade-1.
        ideal = np.asarray([np.dot(np.sort(grades[ix])[::-1][:10], self.binary.discount[:min(10, len(ix))])
                            for ix in self.binary.query_indices])
        gains = np.r_[grades, 0][self.binary.indices]
        dcg = (gains*self.binary.discount).sum(axis=2)
        return np.divide(dcg, ideal[None, :], out=np.zeros_like(dcg, dtype=float), where=ideal[None, :] > 0).mean(axis=1)

    def evaluate(self, human, predicted, y):
        binary_scores = self.binary.scores(np.where(human, y, predicted))
        mixed_grades = np.where(human, self.grades, predicted*2)
        graded_scores = self.scores(mixed_grades)
        metrics = self.binary.evaluate(np.where(human, y, predicted))
        metrics.update(graded_system_kendall_tau=float(kendalltau(self.gold_graded, graded_scores, variant="b").statistic),
                       graded_system_ndcg10_mae=float(np.abs(graded_scores-self.gold_graded).mean()),
                       graded_maximum_rank_change=float(np.abs(rankdata(-graded_scores, method="average")-self.gold_rank).max()))
        return metrics, {"systems": self.binary.systems, "binary_scores": binary_scores.tolist(),
                         "graded_scores": graded_scores.tolist(), "gold_binary_scores": self.binary.gold_scores.tolist(),
                         "gold_graded_scores": self.gold_graded.tolist()}


def evaluate(study, evaluator, rankings, human, predicted):
    metrics, per_query = evaluator.evaluate(human, predicted)
    y, raw = evaluator.y, evaluator.raw
    automatic = ~human
    wrong = automatic & (predicted != y)
    both = metrics["both_classes_accepted"]
    metrics["worst_class_risk"] = max(metrics[f"class_{c}_selective_risk"] for c in (0, 1)) if both else None
    raw_risks = []
    for c in (0, 1):
        a = int((automatic & (raw == c)).sum())
        e = int((wrong & (raw == c)).sum())
        metrics[f"raw_class_{c}_coverage"] = a/int((raw == c).sum())
        metrics[f"raw_class_{c}_risk"] = e/a if a else None
        raw_risks.append(e/a if a else None)
    metrics["worst_raw_class_risk"] = max(raw_risks) if all(r is not None for r in raw_risks) else None
    metrics["both_classes_5percent_coverage"] = bool(both and min(metrics[f"class_{c}_coverage"] for c in (0, 1)) >= .05)
    counts = []
    for q, ix in evaluator.groups.items():
        counts.append({"query_id": q, **{f"accepted_{c}": int((automatic[ix] & (predicted[ix] == c)).sum()) for c in (0, 1)},
            **{f"errors_{c}": int((wrong[ix] & (predicted[ix] == c)).sum()) for c in (0, 1)}})
    scores = None
    if rankings is not None:
        rank_metrics, scores = rankings.evaluate(human, predicted, y)
        metrics.update(rank_metrics)
    return metrics, counts, scores


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads(CONFIG.read_text())
    source_paths = [Path(__file__), CONFIG, ROOT / "src/calibration/class_thresholds.py", ROOT / "src/baselines/lara.py",
                    ROOT / "src/evaluation/review_metrics.py", ROOT / "scripts/evaluate_proposal_metrics.py",
                    ROOT / "scripts/compare_qwen_transfer.py"]
    input_paths = [ROOT / "results/stage2/dl2019_qwen35_full_v1/predictions.csv",
                   EXTRACTED / "runs/dl2020/predictions.csv", EXTRACTED / "runs/trec_covid/predictions.csv",
                   EXTRACTED / "inputs/trec_covid.metadata.json", OLD / "experiment.json",
                   ROOT / "cache/research/transfer_inputs/dl20_texts.tar.xz"]
    design = {"settings": settings, "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_paths},
              "input_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in input_paths}}
    design_path = OUTPUT / "design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError("Frozen experiment inputs/source changed; use a fresh version")
    write_json(design_path, design)
    all_studies = studies(settings)
    tasks = [(cohort, preprocessing, fraction, settings) for cohort in ("dl2020_all", "trec_covid_locked")
             for preprocessing in ("raw", "upstream_smooth") for fraction in settings["human_fractions"]]
    with ProcessPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(reserved_job, *task) for task in tasks]
        for future in as_completed(pending):
            print("Reserved comparator ready:", future.result(), flush=True)
    records = []
    for study in all_studies:
        evaluator = PolicyEvaluator(study["test"])
        rankings = RankingMetrics(study["test"]) if study["id"] == "dl2020_source" else None
        if rankings:
            write_json(OUTPUT / "ranking_provenance.json", {**rankings.binary.provenance,
                "graded_reference": "Original grades 0/1/2/3; linear gains as trec_eval defaults.",
                "graded_automatic_mapping": "Binary predicted labels map to grades 0 and 2; reviewed pairs retain original human grade.",
                "trec_eval_reference": "https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c"})
        p = np.asarray([r["prob_relevant"] for r in study["test"]])
        raw = (p >= .5).astype(int)
        baseline_cache = {}
        for seed in settings["seeds"]:
            for fraction in settings["human_fractions"]:
                total_budget = int(Decimal(str(fraction))*len(p))
                for method in BASELINES:
                    # Deterministic methods are evaluated once and reused explicitly.
                    effective_seed = 6000 if method in DETERMINISTIC_METHODS or method.startswith("lara_reserved_") else 6000+(seed-6000) % 10
                    cache_key = method, effective_seed, total_budget
                    if cache_key not in baseline_cache:
                        human, predicted, state = baseline(study, method, seed, total_budget)
                        metrics, counts, scores = evaluate(study, evaluator, rankings, human, predicted)
                        baseline_cache[cache_key] = metrics, counts, scores, state
                    metrics, counts, scores, state = baseline_cache[cache_key]
                    record = dict(scenario=study["id"], seed=seed, fraction=fraction, method=method,
                        calibration_labels=0, total_human_labels=total_budget, **metrics)
                    records.append(record)
                    destination = OUTPUT / "allocations" / study["id"] / method / f"seed_{seed}_budget_{total_budget}.json.gz"
                    save_gzip(destination, {"record": record, "query_class_counts": counts, "system_scores": scores,
                        "state": {**state, "effective_allocation_seed": effective_seed}})
            for per_query in settings["labels_per_calibration_query"]:
                calibration = sample_calibration(study["calibration"], per_query, seed)
                cp = np.asarray([r["prob_relevant"] for r in calibration])
                cy = np.asarray([r["true_binary"] for r in calibration])
                c = len(calibration)
                if set(map(pair_key, calibration)) & set(map(pair_key, study["test"])) and study["query_disjoint"]:
                    raise ValueError("Calibration/test pair overlap")
                write_json(OUTPUT / "calibration" / study["id"] / f"seed_{seed}_labels_{c}.json",
                    [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": r["prob_relevant"],
                      "observed_label": r["true_binary"]} for r in calibration])
                for fraction in settings["human_fractions"]:
                    total_budget = int(Decimal(str(fraction))*len(p))
                    review_budget = total_budget-c
                    if review_budget < 0:
                        raise ValueError("Calibration cost exceeds total budget")
                    for objective in (*settings["objectives"], "common_paid"):
                        started = time.perf_counter()
                        if objective == "common_paid":
                            confidence = np.maximum(p, 1-p)
                            tie = np.random.default_rng(seed).permutation(len(p))
                            order = np.lexsort((tie, -confidence))
                            human = np.ones(len(p), dtype=bool)
                            human[order[:len(p)-review_budget]] = False
                            state = {"scope": "Common threshold with the same paid calibration overhead; calibration labels unused."}
                        else:
                            human, state = select_class_thresholds(cp, cy, p, len(p)-review_budget, seed,
                                settings["minimum_class_coverage"], settings["minimum_calibration_tail_support"], objective)
                        metrics, counts, scores = evaluate(study, evaluator, rankings, human, raw)
                        if metrics["human_labels"] != review_budget or c+int(human.sum()) != total_budget:
                            raise ValueError("Total annotation budget mismatch")
                        method = "threshold_"+objective
                        record = {"scenario": study["id"], "seed": seed, "fraction": fraction, "method": method,
                            "calibration_labels": c, "total_human_labels": total_budget, **metrics,
                            "selection_seconds": time.perf_counter()-started,
                            **{k: v for k, v in state.items() if k.startswith("class_") or k == "both_tails_supported"}}
                        records.append(record)
                        destination = OUTPUT / "allocations" / study["id"] / method / f"seed_{seed}_cal_{c}_budget_{total_budget}.json.gz"
                        save_gzip(destination, {"record": record, "human_packed": np.packbits(human).tolist(),
                            "query_class_counts": counts, "system_scores": scores, "state": state})
            print(study["id"], "calibration repetition complete", seed, flush=True)
    write_csv(OUTPUT / "runs.csv", records)
    write_json(OUTPUT / "experiment.json", {"status": "complete", "evaluations": len(records),
        "calibration_samples": len(all_studies)*len(settings["seeds"])*len(settings["labels_per_calibration_query"]),
        "new_reserved_lara_runs": len(tasks), "design_sha256": file_hash(design_path),
        "scope": settings["scope"], "source_hashes_checked": True})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(OUTPUT).as_posix()}\n"
        for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != "checksums.sha256"))
    print("Complete:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
