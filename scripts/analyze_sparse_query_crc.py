"""Plot allocation results and bootstrap held-out topics with policies fixed."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration.sparse_query import evaluation_table, group_pool
from src.utils.io import atomic_text, file_hash, read_predictions, write_csv, write_json


def execute(args):
    metadata = json.loads((args.results/"experiment.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or file_hash(args.predictions) != metadata["predictions_sha256"]:
        raise ValueError("Require completed results and the identical prediction file")
    with (args.results/"runs.csv").open(encoding="utf-8", newline="") as stream:
        runs = list(csv.DictReader(stream))
    with (args.results/"summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    rows, _ = read_predictions(args.predictions, metadata["relevance_threshold"])
    groups = group_pool([r for r in rows if r["query_id"] in metadata["test_query_ids"]])
    grid = np.linspace(.5, 1., metadata["settings"]["grid_steps"]+1)
    tables = [evaluation_table(group, grid) for group in groups.values()]
    query_sizes = np.asarray([len(group) for group in groups.values()])
    error_counts = np.asarray([[r["automatic_errors"] for r in table] for table in tables])
    accept_counts = np.asarray([[r["automatic_labels"] for r in table] for table in tables])
    rng = np.random.default_rng(20261113)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    conditions = defaultdict(list)
    fields = ("sampling", "method", "alpha", "calibration_queries", "budget_condition")
    for row in runs:
        conditions[tuple(row[f] for f in fields)].append(row)
    intervals, query_diagnostics = [], []
    for key, records in sorted(conditions.items()):
        indices = np.asarray([int(r["threshold_index"]) for r in records])
        # Average policy randomness BEFORE bootstrapping; seeds are not topics.
        errors = error_counts[:, indices].mean(axis=1)
        accepted = accept_counts[:, indices].mean(axis=1)
        losses = errors/query_sizes
        for i, qid in enumerate(groups):
            query_diagnostics.append({**dict(zip(fields, key)), "query_id": qid,
                                      "pairs": int(query_sizes[i]), "mean_automatic_errors": float(errors[i]),
                                      "mean_auto_error_rate": float(losses[i]),
                                      "mean_coverage": float(accepted[i]/query_sizes[i])})
        estimates = {"query_mean_auto_error_rate": (float(losses.mean()), losses[draws].mean(axis=1)),
                     "coverage": (float(accepted.sum()/query_sizes.sum()),
                                  accepted[draws].sum(axis=1)/query_sizes[draws].sum(axis=1))}
        for metric, (point, samples) in estimates.items():
            intervals.append({**dict(zip(fields, key)), "metric": metric, "estimate": point,
                              "conditional_query_bootstrap_low": float(np.quantile(samples, .025)),
                              "conditional_query_bootstrap_high": float(np.quantile(samples, .975)),
                              "test_queries": len(groups), "bootstrap_samples": len(draws)})
    args.output_dir.mkdir(parents=True, exist_ok=args.overwrite)
    for name in ("summary.csv", "paired_comparisons.csv", "experiment.json"):
        shutil.copyfile(args.results/name, args.output_dir/name)
    companion = None
    if args.target_pool_results is not None:
        companion = json.loads((args.target_pool_results/"experiment.json").read_text(encoding="utf-8"))
        if companion["status"] != "complete" or companion["predictions_sha256"] != metadata["predictions_sha256"]:
            raise ValueError("Target-pool results must be complete and use identical predictions")
        for name in ("summary.csv", "experiment.json"):
            shutil.copyfile(args.target_pool_results/name, args.output_dir/("target_pool_"+name))
    write_csv(args.output_dir/"query_bootstrap.csv", intervals)
    write_csv(args.output_dir/"per_query_diagnostics.csv", query_diagnostics)
    write_json(args.output_dir/"analysis.json", {
        "source_results": str(args.results), "script_sha256": file_hash(Path(__file__)),
        "runs_sha256": file_hash(args.results/"runs.csv"), "bootstrap_seed": 20261113,
        "target_pool_results": str(args.target_pool_results) if companion else None,
        "target_pool_runs_sha256": file_hash(args.target_pool_results/"runs.csv") if companion else None,
        "bootstrap_samples": len(draws), "units": "Held-out queries, resampled as complete blocks",
        "interpretation": "Conditional on fitted policies and this calibration pool; excludes calibration-population uncertainty",
        "shaded_plot_bands": "10th-90th percentiles of acquisition repetitions, NOT confidence intervals"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    costs, cost_axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True, sharey=True)
    for ri, alpha in enumerate(metadata["settings"]["alphas"]):
        for budget, color in zip(metadata["settings"]["label_budgets"], ("#0072B2", "#D55E00", "#009E73")):
            points = sorted([r for r in summary if r["sampling"] == "sparse_uniform" and r["method"] == "crc"
                             and float(r["alpha"]) == alpha and int(r["budget_condition"]) == budget],
                            key=lambda r: int(r["calibration_queries"]))
            x = [int(r["calibration_queries"]) for r in points]
            for ci, metric in enumerate(("coverage", "query_mean_auto_error_rate")):
                axis = axes[ri, ci]
                axis.plot(x, [100*float(r[metric+"_mean"]) for r in points], "o-", color=color, label=f"CRC, G={budget}")
                axis.fill_between(x, [100*float(r[metric+"_p10"]) for r in points],
                                  [100*float(r[metric+"_p90"]) for r in points], color=color, alpha=.12)
            cost_axes[ri].plot(x, [float(r["total_human_labels_mean"]) for r in points], "o-", color=color, label=f"G={budget}")
        naive = sorted([r for r in summary if r["sampling"] == "sparse_uniform" and r["method"] == "empirical"
                        and float(r["alpha"]) == alpha and r["budget_condition"] == "600"],
                       key=lambda r: int(r["calibration_queries"]))
        for ci, metric in enumerate(("coverage", "query_mean_auto_error_rate")):
            axes[ri, ci].plot([int(r["calibration_queries"]) for r in naive],
                              [100*float(r[metric+"_mean"]) for r in naive], "--", color="#666666", label="Empirical, G=600")
            axes[ri, ci].set_title(f"Target auto-error {alpha:.0%}: " + ("coverage" if ci == 0 else "query-mean auto-error"))
            axes[ri, ci].set_ylabel("Percent")
            axes[ri, ci].grid(alpha=.25)
        axes[ri, 1].axhline(100*alpha, color="black", linestyle=":", label="Target")
        cost_axes[ri].axhline(metadata["test_pairs"], color="black", linestyle=":", label="All test pairs human-labelled")
        cost_axes[ri].set(title=f"Target auto-error {alpha:.0%}", xlabel="Calibration queries")
        cost_axes[ri].grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Calibration queries")
    cost_axes[0].set_ylabel("Calibration labels + test escalations")
    cost_axes[0].legend(fontsize=8)
    fig.suptitle("Fixed label budgets; 100 acquisition repetitions; one fixed 13-query test set\nShading: acquisition P10-P90, not confidence intervals", fontsize=11)
    fig.tight_layout()
    costs.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir/f"allocation_tradeoff.{extension}", dpi=170)
        costs.savefig(args.output_dir/f"human_cost.{extension}", dpi=170)
    plt.close(fig)
    plt.close(costs)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file() and p.name != "checksums.sha256")
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=ROOT/"results/stage2/dl2019_qwen35_full_v1/predictions.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-pool-results", type=Path, help="Include a completed same-query companion summary")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate analysis files in an existing output directory")
    execute(parser.parse_args())
