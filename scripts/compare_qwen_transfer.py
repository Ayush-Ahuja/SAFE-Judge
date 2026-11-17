"""Evaluate the frozen local policy and binary LARA on returned Colab scores."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from decimal import Decimal
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pilot import validate_against_manifest
from scripts.run_error_predictor_ablation import aggregate, complete_budget
from scripts.transfer_colab_support import safe_extract
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.safe_judge import SafeJudgeConfig, run_safe_judge
from src.evaluation.metrics import classification_metrics
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json

FIRST_SEED = 6000
FRACTIONS = (0, .10, .25, .50, .75, .90)
METHODS = ("selected_local", "lara_1_upstream_smooth", "lara_topics_upstream_smooth",
           "lara_1_raw", "lara_topics_raw", "raw_confidence", "random_review")
DETERMINISTIC_METHODS = {"lara_1_raw", "lara_1_upstream_smooth"}
COHORTS = ("dl2020_all", "trec_covid_all", "trec_covid_locked", "chatreport_all")


def validate_input(archive, extracted):
    """Verify archive provenance, complete manifests, scores and SQLite snapshots."""
    archive_hash = file_hash(archive)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Corrupt ZIP member")
        receipt = json.loads(zipped.read("provenance/bundle_hashes.json"))
        checked = 0
        for name, digest in receipt.items():
            exported = "inputs/" + name.removeprefix("data/transfer/") if name.startswith("data/transfer/") else "source/" + name
            if name.startswith("docs/"):
                exported = "provenance/" + name
            if exported not in zipped.namelist():
                continue
            if hashlib.sha256(zipped.read(exported)).hexdigest() != digest:
                raise ValueError(f"Changed exported input/source {exported}")
            checked += 1
        pipeline_files = ("scripts/run_transfer_inference.py", "src/assessor/transfer_model.py",
                          "src/assessor/transfer_prompts.py", "src/assessor/model.py",
                          "src/assessor/prompts.py", "src/utils/io.py")
        pipeline = hashlib.sha256("".join(hashlib.sha256(zipped.read("source/" + name)).hexdigest()
                                            for name in pipeline_files).encode()).hexdigest()
    if (extracted / "archive_identity.json").exists():
        previous = json.loads((extracted / "archive_identity.json").read_text())
        if previous["archive_sha256"] != archive_hash:
            raise ValueError("Choose a fresh extraction directory for a changed archive")
    else:
        safe_extract(archive, extracted)
        write_json(extracted / "archive_identity.json", {"archive_sha256": archive_hash})
    suite = json.loads((extracted / "inputs/suite.json").read_text())
    results = {}
    for dataset, info in suite["datasets"].items():
        manifest_path = extracted / f"inputs/{dataset}.jsonl"
        score_dir = extracted / "runs" / dataset
        experiment = json.loads((score_dir / "experiment.json").read_text())
        progress = json.loads((score_dir / "progress.json").read_text())
        if not progress["complete"] or progress["status"] != "complete" or experiment["limit"] != 0:
            raise ValueError(f"Incomplete or smoke inference {dataset}")
        if not (file_hash(manifest_path) == info["manifest_sha256"] == experiment["manifest_sha256"]):
            raise ValueError(f"Changed manifest {dataset}")
        if experiment["pipeline_sha256"] != pipeline:
            raise ValueError(f"Inference pipeline hash mismatch {dataset}")
        manifest = read_manifest(manifest_path)
        rows, duplicates = read_predictions(score_dir / "predictions.csv", info["relevance_threshold"])
        validate_against_manifest(rows, manifest)
        if duplicates or len(rows) != info["pairs"] or len({r["query_id"] for r in rows}) != info["queries"]:
            raise ValueError(f"Pair/query count mismatch {dataset}")
        with closing(sqlite3.connect(f"{(score_dir / 'checkpoint.sqlite3').as_uri()}?mode=ro", uri=True)) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError(f"Invalid SQLite checkpoint {dataset}")
            stored_experiment = json.loads(connection.execute("SELECT value FROM metadata WHERE id=1").fetchone()[0])
            stored = {pair_key(r): r for r in (json.loads(v[0]) for v in connection.execute("SELECT value FROM predictions"))}
        if stored_experiment != experiment or set(stored) != {pair_key(r) for r in rows}:
            raise ValueError(f"Checkpoint metadata/pair mismatch {dataset}")
        for row in rows:
            saved = stored[pair_key(row)]
            for name in ("prob_relevant", "no_logit", "yes_logit", "predicted_label", "true_relevance", "true_binary"):
                if float(row[name]) != saved[name]:
                    raise ValueError(f"CSV/checkpoint mismatch {dataset} {pair_key(row)} {name}")
            expected = 1 / (1 + np.exp(float(row["no_logit"]) - float(row["yes_logit"])))
            if abs(row["prob_relevant"] - expected) > 2e-6 or row["predicted_label"] != int(row["prob_relevant"] >= .5):
                raise ValueError(f"Answer-logit or label inconsistency {dataset}")
        raw = classification_metrics(rows)
        raw.pop("reliability")
        results[dataset] = {"manifest_sha256": file_hash(manifest_path),
                            "predictions_sha256": file_hash(score_dir / "predictions.csv"),
                            "queries": info["queries"], "pairs": len(rows), "raw_metrics": raw,
                            "query_macro_accuracy": float(np.mean([np.mean([r["predicted_label"] == r["true_binary"]
                                for r in rows if r["query_id"] == q]) for q in sorted({r["query_id"] for r in rows})])),
                            "truncated_pairs": sum(r["truncated"].lower() == "true" for r in rows),
                            "inference_experiment": experiment, "inference_progress": progress}
    assessor_settings = {json.dumps(info["inference_experiment"]["assessor"], sort_keys=True) for info in results.values()}
    revisions = {info["inference_experiment"]["resolved_revision"] for info in results.values()}
    if len(assessor_settings) != 1 or len(revisions) != 1:
        raise ValueError("Model/scoring settings differ across returned datasets")
    return {"archive_sha256": archive_hash, "checked_exported_input_source_files": checked,
            "pipeline_sha256": pipeline, "datasets": results}


def load_cohort(extracted, cohort):
    dataset = "trec_covid" if cohort.startswith("trec_covid") else cohort.removesuffix("_all")
    info = json.loads((extracted / f"inputs/{dataset}.metadata.json").read_text())
    rows, _ = read_predictions(extracted / f"runs/{dataset}/predictions.csv", info["relevance_threshold"])
    if cohort == "trec_covid_locked":
        allowed = set(info["sampling"]["locked_evaluation_query_ids"])
        rows = [r for r in rows if r["query_id"] in allowed]
        if len(rows) != 4000 or len({r["query_id"] for r in rows}) != 40:
            raise ValueError("COVID reserved evaluation cohort changed")
    return sorted(rows, key=pair_key)


def run_job(extracted_string, output_string, cohort, method, seed, policy_config):
    extracted, output = Path(extracted_string), Path(output_string)
    rows = load_cohort(extracted, cohort)
    pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
    keys = [pair_key(r) for r in rows]
    indices = {key: i for i, key in enumerate(keys)}
    labels = {pair_key(r): int(r["true_binary"]) for r in rows}
    raw_p = np.asarray([r["prob_relevant"] for r in rows])
    evaluator = PolicyEvaluator(rows)
    files = []
    for fraction in FRACTIONS:
        budget = int(Decimal(str(fraction)) * len(rows))
        destination = output / "allocations" / cohort / method / f"seed_{seed}_budget_{budget}.json.gz"
        if destination.exists():
            files.append(str(destination))
            continue
        oracle = LabelOracle(labels, budget)
        started = time.perf_counter()
        predicted = (raw_p >= .5).astype(int)
        state = None
        if method == "selected_local":
            human, predicted, _, purchases, state = run_safe_judge(
                pool, oracle, budget, SafeJudgeConfig(**policy_config), seed)
        elif method.startswith("lara_"):
            grouping = "topics" if method.startswith("lara_topics_") else 1
            preprocessing = "upstream_smooth" if method.endswith("upstream_smooth") else "raw"
            scored, purchases, state = run_lara(pool, oracle, budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
            human = np.zeros(len(rows), dtype=bool)
            predicted = np.zeros(len(rows), dtype=int)
            for row in scored:
                i = indices[pair_key(row)]
                human[i] = row["label_source"] == "human"
                predicted[i] = row["predicted_label"]
        else:
            priority = 1 - np.maximum(raw_p, 1-raw_p) if method == "raw_confidence" else np.random.default_rng(seed).random(len(pool))
            human = complete_budget(pool, [], priority, budget, seed)
            purchases = []
            for i in np.flatnonzero(human):
                qid, did = keys[i]
                purchases.append({"query_id": qid, "doc_id": did, "observed_label": oracle.query(keys[i])})
        bought = [pair_key(p) for p in purchases]
        if len(bought) != len(set(bought)) or set(bought) != set(oracle.queried) or len(bought) != budget:
            raise ValueError("Paid-label ledger violates the exact budget")
        if {keys[i] for i in np.flatnonzero(human)} != set(bought):
            raise ValueError("Human output mask differs from the purchase ledger")
        metrics, query_rows = evaluator.evaluate(human, predicted)
        if metrics["human_labels"] != budget:
            raise ValueError("Human cost mismatch")
        identifiers = {"dataset": cohort, "seed": seed, "budget": budget, "method": method,
                       "requested_human_fraction": fraction, "pilot_labels": 0}
        record = {**identifiers, **metrics, "elapsed_seconds": time.perf_counter()-started,
                  "sentinel_labels": state.get("sentinel_labels", 0) if state else 0}
        result = {"record": record, "per_query": [{**identifiers, **q} for q in query_rows],
                  "purchases": purchases, "automatic_predicted_labels": predicted.tolist(),
                  "state": state}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(result, stream, allow_nan=False)
        temporary.replace(destination)
        files.append(str(destination))
    return cohort, method, seed, files


def collect_results(output, extracted, repetitions):
    records, per_query = [], []
    for cohort in COHORTS:
        rows = load_cohort(extracted, cohort)
        write_csv(output / "cohorts" / f"{cohort}.csv", [{"pair_index": i, **{k: r[k] for k in
                  ("query_id", "doc_id", "true_binary", "prob_relevant")}} for i, r in enumerate(rows)])
        for method in METHODS:
            for seed in range(FIRST_SEED, FIRST_SEED + repetitions):
                stored_seed = FIRST_SEED if method in DETERMINISTIC_METHODS else seed
                for fraction in FRACTIONS:
                    budget = int(Decimal(str(fraction)) * len(rows))
                    path = output / "allocations" / cohort / method / f"seed_{stored_seed}_budget_{budget}.json.gz"
                    with gzip.open(path, "rt", encoding="utf-8") as stream:
                        result = json.load(stream)
                    record = {**result["record"], "seed": seed,
                              "deterministic_reused_from_seed": stored_seed if stored_seed != seed else None}
                    records.append(record)
                    per_query.extend({**q, "seed": seed} for q in result["per_query"])
        full, full_queries = PolicyEvaluator(rows).evaluate(np.ones(len(rows), dtype=bool))
        ids = {"dataset": cohort, "seed": FIRST_SEED, "budget": len(rows), "method": "full_human",
               "requested_human_fraction": 1., "pilot_labels": 0}
        records.append({**ids, **full})
        per_query.extend({**ids, **q} for q in full_queries)
    write_csv(output / "runs.csv", records)
    write_csv(output / "per_query.csv", per_query)
    write_csv(output / "summary.csv", aggregate(records, ("dataset", "budget", "method")))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, default=ROOT / "dist/qwen35_transfer_v1_dataset_full_results.zip")
    parser.add_argument("--extracted-dir", type=Path, default=ROOT / "results/transfer/qwen35_transfer_v1_dataset_full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/experiments/qwen_transfer_comparison_v1")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.workers < 1:
        parser.error("Positive repetitions and worker count required")
    # Resolve paths for SQLite's read-only URI and worker processes.
    args.extracted_dir = args.extracted_dir.resolve()
    validation = validate_input(args.input_zip.resolve(), args.extracted_dir)
    print(json.dumps({name: {"pairs": info["pairs"], "queries": info["queries"],
                       "raw_accuracy": info["raw_metrics"]["accuracy"]} for name, info in validation["datasets"].items()}), flush=True)
    if args.validate_only:
        print("Input provenance, CSV/logits and SQLite snapshots passed.")
        return
    config_path = ROOT / "configs/safe_judge.json"
    selected = json.loads(config_path.read_text())
    sources = (Path(__file__), ROOT / "scripts/run_error_predictor_ablation.py", ROOT / "src/baselines/lara.py",
               ROOT / "src/baselines/active_learning.py", ROOT / "src/calibration/safe_judge.py",
               ROOT / "src/evaluation/review_metrics.py", ROOT / "src/utils/io.py")
    identity = {"input_archive_sha256": validation["archive_sha256"], "first_seed": FIRST_SEED,
                "repetitions": args.repetitions, "human_budget_fractions": FRACTIONS,
                "methods": METHODS, "primary_lara": "lara_1_upstream_smooth", "cohorts": COHORTS,
                "selected_policy": selected, "selected_config_sha256": file_hash(config_path),
                "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources},
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}}
    # Round-trip tuples to match a stored JSON identity on resume.
    identity = json.loads(json.dumps(identity))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = args.output_dir / "experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text())["identity"] != identity:
        raise ValueError("Comparison identity changed. Choose a new output directory.")
    metadata = {"status": "running", "identity": identity, "input_validation": validation,
                "scope": "Fixed-pool Qwen policy evaluation with simulated purchases of existing gold labels. All sentinel, calibration and review labels charged. Frozen pre-existing selected policy, no new tuning.",
                "cohort_scope": "DL2020 uses all 54 queries. COVID reserved-40 cohort retains existing exclusions; all-50 cohort is descriptive. ChatReport was used previously in method discovery with another assessor.",
                "deterministic_reuse": "Global binary LARA has no seed-dependent operation when grouping=1. Compute once per budget and reuse the identical allocation for all seeds. These copies are not independent observations.",
                "full_human_endpoint": "Exact evaluator endpoint under the same perfect-oracle assumption; recorded once per cohort.",
                "python": platform.python_version()}
    write_json(experiment_path, metadata)
    jobs = []
    for cohort in COHORTS:
        for method in METHODS:
            seeds = [FIRST_SEED] if method in DETERMINISTIC_METHODS else range(FIRST_SEED, FIRST_SEED+args.repetitions)
            jobs.extend((str(args.extracted_dir), str(args.output_dir.resolve()), cohort, method, seed, selected["review_config"])
                        for seed in seeds)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, *job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            cohort, method, seed, _ = future.result()
            print(f"{number}/{len(jobs)} jobs completed | {cohort} | {method} | seed {seed} | {(time.perf_counter()-started)/60:.1f} minutes", flush=True)
    records = collect_results(args.output_dir, args.extracted_dir, args.repetitions)
    metadata.update(status="complete", method_evaluations=len(records), elapsed_seconds=time.perf_counter()-started)
    write_json(experiment_path, metadata)
    atomic_text(args.output_dir / "checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(args.output_dir).as_posix()}\n"
                for p in sorted(args.output_dir.rglob("*")) if p.is_file() and p.name != "checksums.sha256"))
    print(f"Completed {len(records)} method evaluations.", flush=True)


if __name__ == "__main__":
    main()
