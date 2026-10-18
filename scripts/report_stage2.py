"""Assemble measured pilot results and identify outstanding submission work."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pilot import validate_against_manifest, validate_experiment_manifest
from src.assessor.prompts import PROMPT_VERSION, RUBRICS, SYSTEM
from src.utils.io import atomic_text, file_hash, pair_key, read_manifest, read_predictions, write_json


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def reviewed_failures(path, predictions):
    if not path.exists():
        return {"exported": 0, "reviewed": 0, "categories": {}, "error_types": {}}
    errors = {pair_key(row): row for row in predictions if row["predicted_label"] != row["true_binary"]}
    seen, reviewed = set(), []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        key = pair_key(row)
        if key in seen or key not in errors:
            raise ValueError("Failure review contains duplicate pairs or rows outside the run's errors")
        seen.add(key)
        expected = errors[key]
        if int(row["human_grade"]) != expected["true_relevance"] or int(row["predicted_label"]) != expected["predicted_label"]:
            raise ValueError("Review labels differ from the recorded experiment")
        if row.get("reviewed", "").strip().lower() in ("yes", "true", "1") and all(
                row.get(field, "").strip() for field in ("failure_category", "notes", "reviewer")):
            reviewed.append(row)
    return {"exported": len(rows), "reviewed": len(reviewed),
            "categories": dict(Counter(r["failure_category"] for r in reviewed)),
            "error_types": dict(Counter(r["error_type"] for r in reviewed))}


def table(path, columns):
    if not path.exists():
        return "Pending.\n"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            try:
                value = f"{float(value):.4f}" if column not in ("method", "pairs", "queries") else value
            except ValueError:
                pass
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_report(manifest_path, output_dir, review_path, plan_path=None):
    manifest = read_manifest(manifest_path)
    predictions, _ = read_predictions(output_dir / "predictions.csv", manifest[0]["relevance_threshold"])
    missing = validate_against_manifest(predictions, manifest, allow_partial=True)
    validate_experiment_manifest(output_dir / "predictions.csv", manifest_path)
    experiment = read_json(output_dir / "experiment.json", {})
    summary = read_json(output_dir / "evaluation/summary.json", {})
    if summary.get("predictions_sha256") != file_hash(output_dir / "predictions.csv"):
        raise ValueError("Evaluation is missing or stale; evaluate the current predictions first")
    intervals = read_json(output_dir / "evaluation/confidence_intervals.json", {})
    sessions = read_json(output_dir / "sessions.json", [])
    consistency = read_json(output_dir / "consistency.json", {"status": "not_run"})
    calibration = read_json(output_dir / "evaluation/calibration/split.json", {})
    preparation = read_json(output_dir / "preparation.json", {})
    storage = read_json(output_dir / "storage.json", {})
    reviews = reviewed_failures(review_path, predictions)
    review_metadata = read_json(review_path.with_suffix(".metadata.json"), {})
    if review_path.exists() and (review_metadata.get("manifest_sha256") != file_hash(manifest_path)
            or review_metadata.get("predictions_sha256") != file_hash(output_dir / "predictions.csv")):
        raise ValueError("Failure-review metadata does not match the current experiment")
    plan = read_json(plan_path, {}) if plan_path else {}
    required = ("assessor_gpu_model", "assessor_gpu_hours", "teacher_model", "teacher_route",
                "teacher_gpu_model", "teacher_gpu_hours", "api_provider", "api_spend_cap_usd",
                "manual_review_hours", "additional_annotation_pairs", "indexing_cpu_hours", "indexing_storage_gb")
    pending_plan = [key for key in required if plan.get(key) is None or plan.get(key) == ""]
    pending = []
    if missing:
        pending.append(f"Finish inference for {missing} remaining pairs")
    if not intervals:
        pending.append("Compute query-bootstrap intervals")
    if reviews["reviewed"] < 50:
        pending.append(f"Complete {50 - reviews['reviewed']} additional manual failure reviews")
    if pending_plan:
        pending.append("Complete remaining resource budget: " + ", ".join(pending_plan))
    pending.append("Author review: literature/novelty discussion, limitations, AI disclosure, repository URL and final submission")
    status = {"complete_inference": missing == 0, "expected_pairs": len(manifest), "scored_pairs": len(predictions),
              "queries": len({r['query_id'] for r in manifest}), "manual_analysis": reviews,
              "pending_resource_fields": pending_plan, "pending_submission_work": pending,
              "manifest_sha256": file_hash(manifest_path), "predictions_sha256": file_hash(output_dir / "predictions.csv")}
    write_json(output_dir / "submission_status.json", status)
    latest = next((s for s in reversed(sessions) if s.get("gpu")), {})
    total_session_hours = sum(s.get("session_seconds", 0) for s in sessions) / 3600
    inference_seconds = sum(s.get("inference_seconds", 0) for s in sessions)
    inference_pairs = sum(s.get("new_predictions", 0) for s in sessions if s.get("inference_seconds"))
    rate = inference_pairs / inference_seconds if inference_seconds else None
    lines = ["# SAFE-Judge: Stage 2 pilot", "", "## Scope", "",
             "Question: does a frozen compact relevance assessor's confidence distinguish correct and incorrect judgments?",
             "The pilot uses Qwen3.5-4B with binary judgments. This is a change from the proposed 3B graded assessor.",
             f"Scored {len(predictions)} of {len(manifest)} judged pairs across {status['queries']} queries. "
             + ("The complete frozen pool was evaluated." if not missing else "This run is incomplete; its metrics are exploratory."),
             "No model training, teacher-label generation, cross-domain validation or conformal guarantee is claimed.", "",
             "## Data and scoring", "",
             f"Dataset: {experiment.get('dataset')}; grade cutoff: {experiment.get('relevance_threshold')}.",
             f"Manifest SHA256: `{status['manifest_sha256']}`.",
             f"Model revision: `{experiment.get('resolved_revision')}`; prompt: `{PROMPT_VERSION}`.",
             f"Scoring configuration: `{json.dumps(experiment.get('assessor', {}), sort_keys=True)}`.",
             "The chat template disables thinking. Only document text is truncated; query, rubric and answer boundary are preserved.",
             "P(relevant) is the float32 softmax over canonical No/Yes logits; ties at 0.5 predict relevant.",
             "This is a conditional two-token score. The saved full-vocabulary Yes/No probability mass diagnoses answer-format adherence.",
             f"Answer probability mass: `{json.dumps(summary.get('answer_probability_mass', {}), sort_keys=True)}`; truncated pairs: {summary.get('truncated_pairs', 'not recorded')}.",
             f"System instruction: {SYSTEM}", f"Rubric: {RUBRICS[manifest[0]['relevance_threshold']]}", "",
             "## Results", "",
             table(output_dir / "evaluation/baselines.csv", ["method", "pairs", "accuracy", "macro_f1", "brier_score", "ece_probability_bins_10"]),
             "Constant-label baselines describe class imbalance; they do not replace the proposed retrieval/LLM baselines.",
             "Per-query results: `evaluation/per_query.csv`. Raw threshold sweeps: `evaluation/risk_coverage.csv`.",
             "Curves show descriptive coverage, selective risk and auto-error, including predicted-class groups. Thresholds were not validated for deployment.",
             "![Risk and coverage](evaluation/risk_coverage.png)", "![Confusion and reliability](evaluation/diagnostics.png)", "",
             "## Query-level uncertainty", ""]
    if intervals:
        lines.append(f"95% percentile intervals from {intervals['samples']} whole-query bootstrap resamples; seed {intervals['seed']}.")
        lines += ["| Metric | Lower | Upper |", "| --- | --- | --- |"]
        for name, interval in intervals["intervals"].items():
            lines.append(f"| {name} | {interval['lower']:.4f} | {interval['upper']:.4f} |")
    lines += ["", "## Calibration and batch consistency", ""]
    if calibration:
        lines += [f"Temperature {calibration['temperature']:.4f} was fitted using {len(calibration['calibration_query_ids'])} calibration queries; "
                  f"evaluation uses {len(calibration['test_query_ids'])} disjoint queries.",
                  table(output_dir / "evaluation/calibration/metrics.csv", ["method", "pairs", "accuracy", "macro_f1", "brier_score", "ece_probability_bins_10"]),
                  "A positive temperature preserves the mathematical ordering of absolute logit differences and the classification boundary. Finite-precision ties can change. It is not CRC."]
    lines += [f"Batch consistency: `{json.dumps(consistency, sort_keys=True)}`.", "",
              "## Efficiency and resource use", "",
              f"GPU: {latest.get('gpu', 'not recorded')}; GPU memory bytes: {latest.get('gpu_memory_bytes', 'not recorded')}.",
              f"Inference process hours across recorded sessions: {total_session_hours:.4f}; scoring seconds: {inference_seconds:.2f}.",
              f"Measured scoring throughput: {rate if rate else 'not recorded'} pairs/s.",
              f"Peak allocated GPU bytes: {max((s.get('peak_allocated_bytes', 0) for s in sessions), default=0)}.",
              f"Preparation: `{json.dumps(preparation, sort_keys=True)}`.",
              f"Storage: `{json.dumps(storage, sort_keys=True)}`.",
              "Process hours exclude notebook installation and corpus preparation. See `preparation.json` and `environment.json` for setup evidence; "
              "Colab allocation limits and idle time are not scheduler accounting.",
              "Batch selection uses throughput and memory only. CUDA OOM retries reduce batch size without changing text length or dropping pairs.",
              "", "### Remaining resource plan", "", "Values below are proposed requirements, not secured allocations or measured future costs.",
              "| Resource | Plan |", "| --- | --- |"]
    for key in required:
        lines.append(f"| {key} | {plan.get(key) if plan.get(key) is not None else 'Pending'} |")
    if rate:
        lines.append(f"One further 9,260-pair run at this observed rate would take about {9260 / rate / 3600:.3f} inference hours, excluding model loading and preparation.")
    lines += ["", "## Manual failure analysis", "",
              f"Exported {reviews['exported']} failures; {reviews['reviewed']} rows have all manual review fields completed.",
              f"Reviewed category counts: `{json.dumps(reviews['categories'], sort_keys=True)}`.",
              f"Reviewed error types: `{json.dumps(reviews['error_types'], sort_keys=True)}`.",
              "Sampling uses shuffled query/error/confidence strata for diagnosis. Category frequencies describe the reviewed sample, not population prevalence.",
              "Add representative examples and explanations after personally reading the original and model-visible passages.", "",
              "## Statistical plan and remaining experiments", "",
              "For held-out comparisons, freeze prompts and policy parameters before testing. Fit calibrators on query-disjoint calibration data.",
              "For future system-ranking comparisons, bootstrap the same queries for each method, recompute system means/rankings in every replicate, "
              "and report paired differences in Kendall tau with 10,000 query resamples. Use matched human-label budgets including calibration labels.",
              "The Oosterhuis confidence-interval baseline and the proposed selective-labeling policy have different outputs. Reproduction and downstream ranking tests remain future work.",
              "The proposed CRC rule uses (n * empirical_loss + 1)/(n + 1) <= alpha for bounded loss under stated exchangeability assumptions. "
              "It controls expected loss, not success in 95% of random splits. Ten calibration queries at alpha=0.05 give a vacuous vanilla correction. "
              "Auto-error and selective risk differ; domain shift requires separate validation.",
              "Teacher generation, adapter training, recent LLM baselines, substantive method ablations, DL2020 and shifted-domain experiments remain Stage 3 work.", "",
              "## Limitations and AI use", "",
              "This pilot uses a single collection, a fixed binary rubric and quantized inference. Truncation, answer-format mass, grade ambiguity and GPU arithmetic can affect results. "
              "Public benchmark contamination cannot be excluded. No training or generalization guarantee follows from the full-pool diagnostics.",
              "AI assistance was used to develop code, checks and this report scaffold. Numerical results come from recorded predictions. "
              "Human review and the author's final interpretation must be completed separately.", "", "## Outstanding submission work", ""]
    lines.extend("- " + item for item in pending)
    atomic_text(output_dir / "stage2_report.md", "\n".join(lines) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--resource-plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.manifest, args.output_dir, args.review, args.resource_plan), indent=2))


if __name__ == "__main__":
    main()
