"""Validate the completed Stage 2 run and rebuild report tables and individual plots.

This CPU-only step uses the frozen manifest, predictions and completed manual review.
The report narrative and future resource plan are maintained in the LaTeX source.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pilot import validate_against_manifest, validate_experiment_manifest
from scripts.report_stage2 import reviewed_failures
from src.evaluation.metrics import classification_metrics
from src.utils.io import atomic_text, file_hash, read_manifest, read_predictions, write_json

BLUE, ORANGE, GREEN, INK = "#2563A6", "#C55A24", "#237A57", "#253346"
REPORT_PREDICTIONS_SHA256 = "c321f8374496f5c48deaf51c21e8ba55f29ffa446a7ed2e0ad49a1bb9b86dc65"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def tex(value):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                     "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                     "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value))


def document_summary(artifact_root, review_path=None, run_dir=None):
    manifest_path = artifact_root / "data/pilot/dl2019.jsonl"
    manifest = read_manifest(manifest_path)
    candidates = ([run_dir / "experiment.json"] if run_dir is not None else
                  list((artifact_root / "results/stage2").glob("*/experiment.json")))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError("Specify --run-dir when the artifact root does not contain exactly one completed pilot run")
    run = candidates[0].parent
    if file_hash(run / "predictions.csv") != REPORT_PREDICTIONS_SHA256:
        raise ValueError("The report narrative describes the completed DL2019 pilot. Update its text before using another prediction set.")
    predictions, duplicates = read_predictions(run / "predictions.csv")
    validate_against_manifest(predictions, manifest)
    validate_experiment_manifest(run / "predictions.csv", manifest_path)
    summary = read_json(run / "evaluation/summary.json")
    if summary["predictions_sha256"] != file_hash(run / "predictions.csv"):
        raise ValueError("Saved evaluation does not match predictions")
    metrics = classification_metrics(predictions)
    for key in ("accuracy", "macro_f1", "brier_score", "ece_probability_bins_10", "tp", "tn", "fp", "fn"):
        if not math.isclose(summary[key], metrics[key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Saved metric disagrees with recomputation: {key}")
    if duplicates or len(manifest) != 9260 or len({r["query_id"] for r in manifest}) != 43:
        raise ValueError("The Stage 2 report requires all 9,260 unique DL2019 pairs and 43 queries")
    original_review = run / "failure_review.csv"
    review = review_path if review_path is not None else original_review
    review_metadata = read_json(original_review.with_suffix(".metadata.json"))
    if review_metadata["manifest_sha256"] != file_hash(manifest_path) or review_metadata["predictions_sha256"] != file_hash(run / "predictions.csv"):
        raise ValueError("Failure-review metadata does not match this experiment")
    original_rows = {(r["query_id"], r["doc_id"]): r for r in read_csv(original_review)}
    review_rows = read_csv(review)
    if {(r["query_id"], r["doc_id"]) for r in review_rows} != original_rows.keys():
        raise ValueError("The completed review must retain the originally selected 50 pairs")
    for row in review_rows:
        original = original_rows[row["query_id"], row["doc_id"]]
        for field in ("query_text", "document_text", "model_document_text", "error_type"):
            if row[field] != original[field]:
                # Excel can reopen UTF-8 passage text as Windows-1252 before
                # exporting. Accept only that exact reversible encoding change.
                try:
                    restored = row[field].encode("cp1252").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    restored = None
                if restored != original[field]:
                    raise ValueError(f"Review changed original {field} for pair {(row['query_id'], row['doc_id'])}")
    reviews = reviewed_failures(review, predictions)
    if review.resolve() != original_review.resolve():
        write_json(review.with_suffix(".metadata.json"), {**review_metadata,
            "manual_reviews_completed": reviews["reviewed"], "review_sha256": file_hash(review),
            "category_counts": reviews["categories"], "error_type_counts": reviews["error_types"],
            "reviewed_queries": len({r["query_id"] for r in review_rows if r.get("reviewed", "").strip().lower() in ("yes", "true", "1")}),
            "annotation": "Author-supplied personal review; official evaluation labels retained."})
    return run, manifest, predictions, summary, reviews, review


def make_figures(run, predictions, summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.edgecolor": "#617080",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": .18, "grid.linewidth": .6, "legend.frameon": False,
        "lines.linewidth": 2, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
    destination.mkdir(parents=True, exist_ok=True)

    def axis(width=5.6, height=2.85):
        return plt.subplots(figsize=(width, height), layout="constrained")

    def save(fig, name):
        labels = []
        for ax in fig.axes:
            # Keep rotated labels shorter than the available plot height. PDF
            # renderers can otherwise crop glyphs even when Agg's tight box fits.
            for label, width in ((ax.xaxis.label, 46), (ax.yaxis.label, 26)):
                label.set_text("\n".join(textwrap.fill(line, width=width,
                    break_long_words=False, break_on_hyphens=False)
                    for line in label.get_text().splitlines()))
                label.set_clip_on(False)
                labels.append(label)
            ax.xaxis.labelpad = 8
            ax.yaxis.labelpad = 8
        fig.canvas.draw()
        for suffix in ("pdf", "png"):
            fig.savefig(destination / f"{name}.{suffix}", bbox_inches="tight",
                        bbox_extra_artists=labels, pad_inches=.18, facecolor="white")
        plt.close(fig)

    def percent(ax, *, x=False):
        if x:
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xlim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    curves = read_csv(run / "evaluation/risk_coverage.csv")
    overall = sorted((r for r in curves if r["group"] == "all" and r["selective_risk"]), key=lambda r: float(r["coverage"]))
    fig, ax = axis()
    ax.plot([float(r["coverage"]) for r in overall], [float(r["selective_risk"]) for r in overall], color=BLUE)
    ax.axhline(1 - summary["accuracy"], color="#7B8794", linestyle="--", linewidth=1.2, label=f"Full coverage error ({1 - summary['accuracy']:.2%})")
    ax.set(xlabel="Coverage: fraction of all pairs accepted", ylabel="Selective risk: errors among accepted pairs", ylim=(0, .25))
    percent(ax, x=True)
    ax.legend(loc="upper left")
    save(fig, "risk_coverage")

    fig, ax = axis()
    for group, label, color, style in (("predicted_nonrelevant", "Predicted non-relevant", BLUE, "-"),
                                      ("predicted_relevant", "Predicted relevant", ORANGE, "--")):
        subset = sorted((r for r in curves if r["group"] == group and r["selective_risk"]), key=lambda r: float(r["coverage"]))
        ax.plot([float(r["coverage"]) for r in subset], [float(r["selective_risk"]) for r in subset], color=color, linestyle=style, label=label)
    ax.set(xlabel="Coverage within the predicted class", ylabel="Selective risk within accepted class", ylim=(0, .45))
    percent(ax, x=True)
    ax.legend(loc="upper left")
    save(fig, "class_risk_coverage")

    fig, ax = axis()
    ax.plot([float(r["coverage"]) for r in overall], [float(r["auto_error_rate"]) for r in overall], color=GREEN)
    ax.axhline(.05, color="#7B8794", linestyle="--", linewidth=1.2, label="Proposed 5% target (reference only)")
    ax.set(xlabel="Coverage: fraction of all pairs accepted", ylabel="Auto-error rate:\naccepted errors / all pairs", ylim=(0, .23))
    percent(ax, x=True)
    ax.legend(loc="upper left")
    save(fig, "auto_error_coverage")

    reliability = read_csv(run / "evaluation/reliability.csv")
    fig, ax = axis(5.2, 2.95)
    ax.plot([0, 1], [0, 1], "--", color="#7B8794", linewidth=1.2, label="Perfect calibration")
    nonempty = [r for r in reliability if int(r["pairs"])]
    ax.plot([float(r["mean_probability_relevant"]) for r in nonempty], [float(r["observed_relevance_rate"]) for r in nonempty], "o-", color=BLUE, label="Qwen3.5-4B")
    ax.set(xlabel="Mean predicted P(relevant) in each bin", ylabel="Observed relevant fraction in each bin", xlim=(0, 1), ylim=(0, 1))
    percent(ax, x=True)
    ax.legend(loc="upper left")
    save(fig, "reliability")

    fig, ax = axis(5.2, 2.95)
    ax.plot([0, 1], [0, 1], "--", color="#7B8794", linewidth=1.2, label="Perfect calibration")
    for method, label, color, style, marker in (("raw", "Raw", BLUE, "-", "o"),
                                              ("temperature_scaled", "Temperature scaled", ORANGE, "--", "s")):
        bins = [r for r in read_csv(run / f"evaluation/calibration/{method}_reliability.csv") if int(r["pairs"])]
        ax.plot([float(r["mean_probability_relevant"]) for r in bins], [float(r["observed_relevance_rate"]) for r in bins], color=color, linestyle=style, marker=marker, markersize=4, label=label)
    ax.set(xlabel="Mean predicted P(relevant) in each bin", ylabel="Observed relevant fraction in each bin", xlim=(0, 1), ylim=(0, 1))
    percent(ax, x=True)
    ax.legend(loc="upper left")
    save(fig, "calibration_reliability")

    fig, ax = axis(5.2, 3.7)
    matrix = np.array([[summary["tn"], summary["fp"]], [summary["fn"], summary["tp"]]])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}\n({matrix[i, j] / matrix[i].sum():.1%} of row)", ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else INK, fontsize=11)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-relevant", "Relevant"], yticklabels=["Non-relevant", "Relevant"], xlabel="Predicted binary label", ylabel="Official human binary label")
    ax.grid(False)
    save(fig, "confusion_matrix")

    fig, ax = axis()
    bins = np.linspace(.5, 1, 21)
    for correct, label, color, hatch in ((True, "Correct", BLUE, None), (False, "Incorrect", ORANGE, "//")):
        values = [r["confidence"] for r in predictions if (r["predicted_label"] == r["true_binary"]) == correct]
        ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), histtype="stepfilled", alpha=.3, edgecolor=color, color=color, hatch=hatch, linewidth=1.3, label=f"{label} (n={len(values):,})")
    ax.set(xlabel="Binary confidence:\nmax[P(relevant), 1 - P(relevant)]", ylabel="Fraction within correctness group", xlim=(.5, 1))
    percent(ax)
    ax.legend(loc="upper left")
    save(fig, "confidence_distribution")

    queries = sorted(read_csv(run / "evaluation/per_query.csv"), key=lambda r: float(r["accuracy"]))
    fig, ax = axis(6.2, 7.1)
    y = np.arange(len(queries))
    ax.hlines(y, 0, [float(r["accuracy"]) for r in queries], color="#D6E2EE", linewidth=1)
    ax.scatter([float(r["accuracy"]) for r in queries], y, color=BLUE, s=22, zorder=3)
    ax.axvline(summary["query_macro_accuracy"], color=ORANGE, linestyle="--", linewidth=1.2, label=f"Query mean ({summary['query_macro_accuracy']:.1%})")
    ax.set(yticks=y, yticklabels=[r["query_id"] for r in queries], xlabel="Accuracy within query", ylabel="TREC DL 2019 query ID", xlim=(0, 1), ylim=(-1, len(queries)))
    ax.tick_params(axis="y", labelsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "query_accuracy")

    benchmark_record = read_json(run / "batch_benchmark.json")
    benchmark = [r for r in benchmark_record["measurements"] if r["status"] == "complete"]
    fig, ax = axis()
    positions = np.arange(len(benchmark))
    colors = [GREEN if r["batch_size"] == benchmark_record["selected_batch_size"] else BLUE for r in benchmark]
    bars = ax.bar(positions, [r["pairs_per_second"] for r in benchmark], color=colors, width=.6)
    ax.bar_label(bars, labels=[f"{r['pairs_per_second']:.2f}" for r in benchmark], padding=4, fontsize=9)
    ax.set(xticks=positions, xticklabels=[str(r["batch_size"]) for r in benchmark], xlabel="Batch size (32 long-input benchmark pairs)", ylabel="Scoring throughput\n(pairs / second)", ylim=(0, 4.3))
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    save(fig, "batch_throughput")


def make_tables(run, manifest, predictions, summary, reviews, path, review_path=None):
    intervals = read_json(run / "evaluation/confidence_intervals.json")["intervals"]
    sessions = read_json(run / "sessions.json")
    session = sessions[-1]
    experiment = read_json(run / "experiment.json")
    preparation = read_json(run / "preparation.json")
    storage = read_json(run / "storage.json")
    consistency = read_json(run / "consistency.json")
    split = read_json(run / "evaluation/calibration/split.json")
    grades = Counter(r["true_relevance"] for r in manifest)
    errors_by_grade = Counter(r["true_relevance"] for r in predictions if r["predicted_label"] != r["true_binary"])
    curves = read_csv(run / "evaluation/risk_coverage.csv")
    p = lambda value: f"{100 * value:.2f}"
    f = lambda value: f"{value:.4f}"
    commands = {"PilotAccuracy": p(summary["accuracy"]), "PilotFone": f(summary["macro_f1"]),
        "PilotBrier": f(summary["brier_score"]), "PilotECE": f(summary["ece_probability_bins_10"]),
        "PilotQueryAccuracy": p(summary["query_macro_accuracy"]),
        "PilotScoreMinutes": f"{session['inference_seconds'] / 60:.2f}",
        "PilotProcessMinutes": f"{session['session_seconds'] / 60:.2f}",
        "PilotPrepMinutes": f"{preparation['seconds'] / 60:.2f}",
        "PilotThroughput": f"{session['pairs_per_second']:.2f}",
        "PilotPeakAllocated": f"{session['peak_allocated_bytes'] / 2**30:.2f}",
        "PilotPeakReserved": f"{session['peak_reserved_bytes'] / 2**30:.2f}",
        "PilotGPUVRAM": f"{session['gpu_memory_bytes'] / 2**30:.2f}",
        "PilotTemperature": f(split["temperature"]), "PilotReviewed": str(reviews["reviewed"]),
        "PilotManifestHash": experiment["manifest_sha256"], "PilotPredictionHash": file_hash(run / "predictions.csv"),
        "PilotModelRevision": experiment["resolved_revision"], "PilotPipelineHash": experiment["pipeline_sha256"],
        "PilotMassMean": f(summary["answer_probability_mass"]["mean"]),
        "PilotMassMin": f(summary["answer_probability_mass"]["minimum"]),
        "PilotRepeatMean": f"{consistency['mean_absolute_probability_difference']:.6f}",
        "PilotRepeatMax": f"{consistency['max_absolute_probability_difference']:.6f}",
        "PilotRepeatSeconds": f"{consistency['seconds']:.2f}",
        "PilotHFCache": f"{storage['huggingface_cache_bytes'] / 10**9:.2f}",
        "PilotCorpusCache": f"{storage['ir_datasets_cache_bytes'] / 10**9:.2f}"}
    lines = ["% Generated from validated pilot artifacts; rerun scripts/build_stage2_report.py."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in commands.items()]

    def rows_command(name, rows):
        lines.append(rf"\newcommand{{\{name}}}{{%")
        lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
        lines.append("}")

    baselines = read_csv(run / "evaluation/baselines.csv")
    labels = {"Qwen3.5-4B": "Qwen3.5-4B", "always_nonrelevant": "Always non-relevant", "always_relevant": "Always relevant"}
    rows_command("PilotBaselineRows", [[labels[r["method"]], p(float(r["accuracy"])), f(float(r["macro_f1"])), f(float(r["brier_score"])), f(float(r["ece_probability_bins_10"]))] for r in baselines])
    metric_labels = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "brier_score": "Brier score", "ece_probability_bins_10": "ECE (10 probability bins)", "query_macro_accuracy": "Query-mean accuracy", "accuracy_minus_always_nonrelevant": "Accuracy gain over always non-relevant"}
    rows_command("PilotIntervalRows", [[metric_labels[name], f(values["lower"]), f(values["upper"])] for name, values in intervals.items()])
    rows_command("PilotGradeRows", [[grade, f"{grades[grade]:,}", p(grades[grade] / len(manifest)), f"{errors_by_grade[grade]:,}", p(errors_by_grade[grade] / grades[grade])] for grade in range(4)])
    thresholds = [.5, .7, .8, .9, .95, .99]
    rows_command("PilotThresholdRows", [[f"{float(r['threshold']):.2f}", f"{int(r['accepted']):,}", p(float(r["coverage"])), p(float(r["selective_risk"])), p(float(r["auto_error_rate"]))] for threshold in thresholds for r in curves if r["group"] == "all" and math.isclose(float(r["threshold"]), threshold) and r["selective_risk"]])
    rows_command("PilotCalibrationRows", [["Raw" if r["method"] == "raw" else "Temperature scaled", p(float(r["accuracy"])), f(float(r["macro_f1"])), f(float(r["brier_score"])), f(float(r["ece_probability_bins_10"]))] for r in read_csv(run / "evaluation/calibration/metrics.csv")])

    if reviews["reviewed"] < 50:
        lines += [r"\newcommand{\PilotManualAnalysis}{%",
            rf"The diagnostic sheet contains {reviews['exported']} distinct errors; \textbf{{{reviews['reviewed']} complete personal reviews are recorded}}. "
            r"The required analysis of at least 50 failures is pending. Category frequencies and example interpretations will be added after I complete the review sheet. No automatic categories are presented as manual findings.", "}"]
    else:
        completed = [r for r in read_csv(review_path or run / "failure_review.csv") if r.get("reviewed", "").strip().lower() in ("yes", "true", "1") and all(r.get(k, "").strip() for k in ("failure_category", "notes", "reviewer"))]
        categories = sorted(reviews["categories"].items(), key=lambda item: -item[1])
        display_names = {"judgment_disagreement": "Judgment disagreement", "incomplete_answer": "Incomplete answer",
                         "subtle_relevance": "Subtle relevance", "keyword_distraction": "Keyword distraction",
                         "entity_mismatch": "Entity mismatch", "domain_vocabulary": "Domain vocabulary"}
        lines += [r"\newcommand{\PilotManualAnalysis}{%",
            rf"I reviewed {reviews['reviewed']} unique errors across {len({r['query_id'] for r in completed})} queries: "
            rf"{reviews['error_types'].get('false_positive', 0)} false positives and {reviews['error_types'].get('false_negative', 0)} false negatives.",
            r"\begin{table}[H]\centering\small",
            r"\caption{Primary categories in the 50 personally reviewed disagreements. FP/FN use the official binary labels.}\label{tab:manual}",
            r"\begin{tabular}{@{}lrrrr@{}}\toprule Category & Count & Share (\%) & FP & FN \\\midrule"]
        for key, count in categories:
            selected = [r for r in completed if r["failure_category"] == key]
            lines.append(tex(display_names.get(key, key.replace("_", " "))) + f" & {count} & {100 * count / len(completed):.0f} & "
                         + str(sum(r["error_type"] == "false_positive" for r in selected)) + " & "
                         + str(sum(r["error_type"] == "false_negative" for r in selected)) + r" \\")
        high_confidence = sum(float(r["confidence"]) >= .9 for r in completed)
        lines += [r"\bottomrule\end{tabular}\end{table}",
            f"The sample contains {high_confidence} high-confidence disagreements (confidence at least 0.90). "
            "Category shares describe this diagnostic sample, not the prevalence of causes in all 1,908 errors.", "}"]
    atomic_text(path, "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT, help="Repository or extracted artifact root")
    parser.add_argument("--run-dir", type=Path, help="Run directory; required if multiple runs are present")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--review", type=Path, help="Completed review CSV; defaults to reports/failure_review.csv when present")
    args = parser.parse_args()
    review_source = args.review
    if review_source is None and (args.report_dir / "failure_review.csv").exists():
        review_source = args.report_dir / "failure_review.csv"
    run, manifest, predictions, summary, reviews, review_source = document_summary(args.artifacts, review_source, args.run_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    make_figures(run, predictions, summary, args.report_dir / "figures/stage2")
    make_tables(run, manifest, predictions, summary, reviews, args.report_dir / "stage2_results.tex", review_source)
    print(f"Validated {len(predictions):,} predictions; {reviews['reviewed']}/50 personal reviews complete.")
    print(f"Review source: {review_source}")
    print(f"Wrote 9 separate figures in PDF/PNG and generated tables under {args.report_dir}.")


if __name__ == "__main__":
    main()
