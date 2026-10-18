"""Evaluate cached predictions on CPU; duplicates and incomplete runs require explicit opt-in."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import classification_metrics, per_query_table, risk_coverage_table, query_bootstrap, temperature_calibration
from src.utils.io import file_hash, pair_key, read_manifest, read_predictions, write_csv, write_json


def validate_against_manifest(rows, manifest, allow_partial=False):
    expected = {pair_key(row): row for row in manifest}
    for row in rows:
        original = expected.get(pair_key(row))
        if original is None or original["true_relevance"] != row["true_relevance"]:
            raise ValueError(f"Prediction missing from manifest or gold label differs: {pair_key(row)}")
    missing = len(expected) - len(rows)
    if missing and not allow_partial:
        raise ValueError(f"Incomplete run: {missing} missing pairs. Use --allow-partial only for diagnostics.")
    return missing


def plot_curves(curves, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [("all", "Combined"), ("predicted_nonrelevant", "Predicted non-relevant"),
              ("predicted_relevant", "Predicted relevant")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, (name, title) in zip(axes, groups):
        points = sorted((row for row in curves if row["group"] == name), key=lambda r: r["coverage"] or 0)
        axis.plot([r["coverage"] if r["coverage"] is not None else float("nan") for r in points],
                  [r["selective_risk"] if r["selective_risk"] is not None else float("nan") for r in points])
        axis.set(title=title, xlabel="Coverage" if name == "all" else "Coverage within predicted class", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Selective risk (error among accepted pairs)")
    fig.tight_layout()
    fig.savefig(directory / "risk_coverage.png", dpi=160)
    plt.close(fig)


def plot_diagnostics(summary, reliability, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    matrix = [[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]]
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i][j]), ha="center", va="center")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                xlabel="Predicted", ylabel="Gold", title="Confusion matrix")
    buckets = [r for r in reliability if r["pairs"]]
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].plot([r["mean_probability_relevant"] for r in buckets],
                 [r["observed_relevance_rate"] for r in buckets], "o-")
    axes[1].set(xlabel="Mean P(relevant)", ylabel="Observed relevance", title="Reliability", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(directory / "diagnostics.png", dpi=160)
    plt.close(fig)


def validate_experiment_manifest(predictions_path, manifest_path):
    experiment_path = predictions_path.parent / "experiment.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment["manifest_sha256"] != file_hash(manifest_path):
            raise ValueError("Experiment was scored against a different manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relevance-threshold", type=int, choices=(1, 2))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--deduplicate-last", action="store_true", help="Legacy diagnostics only; never silently deduplicate")
    parser.add_argument("--plots", action="store_true", help="Requires matplotlib; JSON/CSV need only Python")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--calibration-queries", type=int, default=0, help="Query-disjoint temperature-scaling experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 0 or args.calibration_queries < 0:
        parser.error("Bootstrap and calibration counts must be nonnegative")
    manifest = read_manifest(args.manifest) if args.manifest else None
    threshold = args.relevance_threshold or (manifest[0]["relevance_threshold"] if manifest else 2)
    if manifest and threshold != manifest[0]["relevance_threshold"]:
        parser.error("Relevance threshold differs from manifest")
    rows, duplicates = read_predictions(args.predictions, threshold, args.deduplicate_last)
    missing = validate_against_manifest(rows, manifest, args.allow_partial) if manifest else None
    if manifest:
        validate_experiment_manifest(args.predictions, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = classification_metrics(rows)
    reliability = summary.pop("reliability")
    per_query = per_query_table(rows)
    summary.update({"status": "complete_pilot" if manifest and missing == 0 and not args.deduplicate_last else "exploratory",
                    "relevance_threshold": threshold, "missing_manifest_pairs": missing,
                    "duplicate_extra_rows_removed": duplicates, "deduplication_requested": args.deduplicate_last,
                    "manifest_sha256": file_hash(args.manifest) if manifest else None,
                    "predictions_sha256": file_hash(args.predictions),
                    "query_macro_accuracy": sum(r["accuracy"] for r in per_query) / len(per_query),
                    "interpretation": "Raw confidence sweep; no CRC guarantee or held-out threshold claim."})
    if all(r.get("answer_probability_mass") not in (None, "") for r in rows):
        mass = [float(r["answer_probability_mass"]) for r in rows]
        import math
        if any(not math.isfinite(value) or not 0 <= value <= 1.000001 for value in mass):
            raise ValueError("Invalid saved Yes/No answer probability mass")
        summary["answer_probability_mass"] = {"mean": sum(mass) / len(mass), "minimum": min(mass),
                                               "pairs_below_0_5": sum(value < .5 for value in mass)}
    summary["truncated_pairs"] = sum(str(r.get("truncated", "")).lower() in ("true", "1") for r in rows)
    curves = risk_coverage_table(rows)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "per_query.csv", per_query)
    write_csv(args.output_dir / "risk_coverage.csv", curves)
    write_csv(args.output_dir / "reliability.csv", reliability)
    baselines = [{"method": "Qwen3.5-4B", **summary}]
    for label, name in ((0, "always_nonrelevant"), (1, "always_relevant")):
        metrics = classification_metrics([{**r, "predicted_label": label, "prob_relevant": float(label), "confidence": 1.0} for r in rows])
        metrics.pop("reliability")
        baselines.append({"method": name, **metrics})
    fields = ["method", "pairs", "queries", "accuracy", "macro_f1", "brier_score", "ece_probability_bins_10", "tp", "tn", "fp", "fn"]
    write_csv(args.output_dir / "baselines.csv", baselines, fields)
    if args.bootstrap_samples:
        write_json(args.output_dir / "confidence_intervals.json", query_bootstrap(rows, args.bootstrap_samples, args.seed))
    if args.calibration_queries:
        if missing != 0 or args.deduplicate_last:
            parser.error("Calibration requires a complete manifest-verified run")
        raw, calibrated, split = temperature_calibration(rows, args.calibration_queries, args.seed)
        directory = args.output_dir / "calibration"
        write_json(directory / "split.json", split)
        table = []
        for name, subset in (("raw", raw), ("temperature_scaled", calibrated)):
            metrics = classification_metrics(subset)
            bins = metrics.pop("reliability")
            table.append({"method": name, **metrics})
            write_csv(directory / f"{name}_risk_coverage.csv", risk_coverage_table(subset))
            write_csv(directory / f"{name}_reliability.csv", bins)
            if args.plots:
                plot_curves(risk_coverage_table(subset), directory)
                # Distinct names preserve both method plots.
                (directory / "risk_coverage.png").replace(directory / f"{name}_risk_coverage.png")
                plot_diagnostics(metrics, bins, directory)
                (directory / "diagnostics.png").replace(directory / f"{name}_diagnostics.png")
        write_csv(directory / "metrics.csv", table, fields)
        write_csv(directory / "predictions.csv", calibrated)
    if args.plots:
        plot_curves(curves, args.output_dir)
        plot_diagnostics(summary, reliability, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
