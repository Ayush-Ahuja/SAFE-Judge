"""Explore class-aware shared/local review on frozen, already inspected pools."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import load_cohort
from scripts.run_class_threshold_experiment import (
    EXTRACTED, OUTPUT as THRESHOLDS, RankingMetrics, baseline, evaluate, read_gzip, save_gzip)
from src.baselines.active_learning import LabelOracle
from src.calibration.class_aware_review import run_class_aware_review
from src.calibration.diagnostic_review import ReviewConfig
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import file_hash, pair_key, write_csv, write_json

DEFAULT_OUTPUT = ROOT / "results/experiments/class_aware_review_v1"


def job(cohort, seed, settings, output):
    rows = load_cohort(EXTRACTED, cohort)
    pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    evaluator = PolicyEvaluator(rows)
    rankings = RankingMetrics(rows) if cohort == "dl2020_all" else None
    cfg = ReviewConfig(**json.loads((ROOT / "configs/diagnostic_review_selected.json").read_text())["review_config"])
    result_paths = []
    for fraction in settings["human_fractions"]:
        budget = int(Decimal(str(fraction))*len(rows))
        for floor in settings["minimum_class_coverages"]:
            for variant in settings["variants"]:
                method = f"{variant}_floor{round(floor*100):02d}"
                destination = output / "allocations" / cohort / method / f"seed_{seed}_budget_{budget}.json.gz"
                if not destination.exists():
                    started = time.perf_counter()
                    oracle = LabelOracle(labels, budget)
                    human, predicted, _, purchases, state = run_class_aware_review(pool, oracle, budget,
                        replace(cfg, global_only=variant == "shared_minimax"), seed, floor,
                        settings["batch_size"], "query_error" if variant == "local_query_error" else "worst_class")
                    purchased = {pair_key(r) for r in purchases}
                    if (len(purchased) != budget or purchased != set(oracle.queried)
                            or purchased != {pair_key(rows[i]) for i in np.flatnonzero(human)}):
                        raise AssertionError("Purchase ledger and final human mask differ")
                    metrics, counts, scores = evaluate({"test": rows}, evaluator, rankings, human, predicted)
                    record = {"cohort": cohort, "seed": seed, "fraction": fraction, "method": method,
                        "minimum_class_coverage": floor, "calibration_labels": 0, "total_human_labels": budget,
                        **metrics, "elapsed_seconds": time.perf_counter()-started}
                    save_gzip(destination, {"record": record, "human_packed": np.packbits(human).tolist(),
                        "predicted": predicted.tolist(), "purchases": purchases, "state": state,
                        "query_class_counts": counts, "system_scores": scores})
                result_paths.append(destination)
    return cohort, seed, result_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/class_aware_review.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    sources = [Path(__file__), args.config, ROOT / "configs/diagnostic_review_selected.json",
        ROOT / "src/calibration/class_aware_review.py", ROOT / "src/calibration/diagnostic_review.py",
        ROOT / "scripts/run_class_threshold_experiment.py", ROOT / "scripts/compare_qwen_transfer.py",
        ROOT / "scripts/evaluate_proposal_metrics.py", ROOT / "src/evaluation/review_metrics.py",
        ROOT / "src/baselines/active_learning.py", ROOT / "src/utils/io.py"]
    inputs = [EXTRACTED / f"runs/{d}/predictions.csv" for d in ("dl2020", "trec_covid", "chatreport")]
    inputs += [EXTRACTED / f"inputs/{d}.metadata.json" for d in ("dl2020", "trec_covid", "chatreport")]
    inputs += [ROOT / "cache/research/transfer_inputs/dl20_texts.tar.xz", THRESHOLDS / "design.json"]
    design = {"settings": settings, "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources},
              "input_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in inputs}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design_path = args.output_dir / "design.json"
    if design_path.exists() and json.loads(design_path.read_text()) != design:
        raise ValueError("Inputs or design changed; use a fresh output version")
    write_json(design_path, design)
    paths = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(job, cohort, seed, settings, args.output_dir)
            for cohort in settings["cohorts"] for seed in settings["seeds"]]
        for future in as_completed(pending):
            cohort, seed, completed = future.result()
            paths.extend(completed)
            print(cohort, seed, "complete", flush=True)
    # Original baselines spend exactly the same total budget directly on the
    # target pool. The saved threshold rule pays its external calibration cost.
    for cohort in settings["cohorts"]:
        rows = load_cohort(EXTRACTED, cohort)
        study = {"cohort": cohort, "test": rows}
        evaluator = PolicyEvaluator(rows)
        rankings = RankingMetrics(rows) if cohort == "dl2020_all" else None
        methods = ["selected_local", "raw_confidence", "lara_1_raw", "lara_1_upstream_smooth",
                   "lara_topics_raw", "lara_topics_upstream_smooth"]
        if cohort != "chatreport_all":
            methods += ["lara_reserved_raw", "lara_reserved_upstream_smooth", "threshold_worst_class_cal200"]
        for seed in settings["seeds"]:
            for fraction in settings["human_fractions"]:
                budget = int(Decimal(str(fraction))*len(rows))
                for method in methods:
                    destination = args.output_dir / "allocations" / cohort / method / f"seed_{seed}_budget_{budget}.json.gz"
                    if method == "threshold_worst_class_cal200":
                        scenario = "dl2020_source" if cohort == "dl2020_all" else "covid_target"
                        source = THRESHOLDS / "allocations" / scenario / "threshold_worst_class" / f"seed_{seed}_cal_200_budget_{budget}.json.gz"
                        saved = read_gzip(source)
                        record = {**saved["record"], "cohort": cohort, "method": method, "minimum_class_coverage": .05}
                        state = {"allocation_source": source.relative_to(ROOT).as_posix(), "allocation_sha256": file_hash(source)}
                        counts, scores = saved["query_class_counts"], saved["system_scores"]
                    else:
                        human, predicted, state = baseline(study, method, seed, budget)
                        metrics, counts, scores = evaluate(study, evaluator, rankings, human, predicted)
                        record = {"cohort": cohort, "seed": seed, "fraction": fraction, "method": method,
                            "minimum_class_coverage": .05 if method.startswith("lara_reserved") else 0.,
                            "calibration_labels": 0, "total_human_labels": budget, **metrics}
                    if record["human_labels"]+record["calibration_labels"] != budget:
                        raise AssertionError("Matched total cost violated")
                    save_gzip(destination, {"record": record, "state": state,
                        "query_class_counts": counts, "system_scores": scores})
                    paths.append(destination)
    paths = sorted(paths)
    records = [read_gzip(p)["record"] for p in paths]
    fields = list(dict.fromkeys(k for r in records for k in r))
    write_csv(args.output_dir / "runs.csv", records, fields)
    write_json(args.output_dir / "experiment.json", {"status": "complete", "evaluations": len(records),
        "new_policy_allocations": len(settings["cohorts"])*len(settings["seeds"])*len(settings["human_fractions"])*len(settings["minimum_class_coverages"])*len(settings["variants"]),
        "design_sha256": file_hash(design_path), "scope": settings["scope"],
        "allocation_sha256": {p.relative_to(args.output_dir).as_posix(): file_hash(p) for p in paths}})
    print("Complete:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
