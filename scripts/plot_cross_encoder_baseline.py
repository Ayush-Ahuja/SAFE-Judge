"""Plot calibration and total-budget results from a completed neural baseline."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures/cross_encoder")
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "experiment.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("Plots require a complete evaluation")
    groups = defaultdict(list)
    paths = sorted(args.run_dir.glob("seed_*/budget_*/summary.json"))
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        groups[row["budget"]].append(row)
    if not groups:
        raise ValueError("No feasible budgets to plot")
    points = []
    for budget, rows in sorted(groups.items()):
        if {r["seed"] for r in rows} != set(metadata["seeds"]) or len(rows) != len(metadata["seeds"]):
            raise ValueError("Incomplete seed coverage")
        point = {"cost": rows[0]["total_human_fraction_of_test"], "coverage": rows[0]["coverage"]}
        for name in ("selective_risk", "auto_error_rate"):
            values = [r[name] for r in rows if r[name] is not None]
            point[name] = statistics.mean(values) if values else None
            point[name + "_min"] = min(values) if values else None
            point[name + "_max"] = max(values) if values else None
        for label in ("0", "1"):
            values = [r["predicted_class_metrics"][label]["selective_risk"] for r in rows
                      if r["predicted_class_metrics"][label]["selective_risk"] is not None]
            point["class_" + label] = statistics.mean(values) if values else None
        points.append(point)
    fraction = metadata["calibration_labels"] / metadata["test_pairs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "svg.fonttype": "none"})

    def style(axis, title, xlabel, ylabel):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.25)

    def draw(axis, kind):
        if kind == "reliability":
            axis.plot([0, 1], [0, 1], "--", color="#88939D", linewidth=1)
            for name, color in (("raw", "#A15B29"), ("temperature_scaled", "#2369A5")):
                with (args.run_dir / f"{name}_reliability.csv").open() as stream:
                    bins = [r for r in csv.DictReader(stream) if int(r["pairs"]) > 0]
                axis.plot([float(r["mean_probability_relevant"]) for r in bins],
                          [float(r["observed_relevance_rate"]) for r in bins], "o-",
                          color=color, label=name.replace("_", " "))
            style(axis, "Held-out probability calibration", "Mean P(relevant)", "Observed relevant fraction")
            axis.set(xlim=(0, 1), ylim=(0, 1))
        elif kind in ("risk_coverage", "auto_error_budget"):
            risk = kind == "risk_coverage"
            metric, xname = ("selective_risk", "coverage") if risk else ("auto_error_rate", "cost")
            selected = sorted((p for p in points if p[metric] is not None), key=lambda p: p[xname])
            axis.plot([p[xname] for p in selected], [p[metric] for p in selected], "o-", color="#2369A5",
                      label="Temperature-scaled MiniLM")
            axis.fill_between([p[xname] for p in selected], [p[metric + "_min"] for p in selected],
                              [p[metric + "_max"] for p in selected], color="#2369A5", alpha=.15)
            if risk:
                style(axis, "Selective risk versus coverage", "Automatic test coverage", "Error among automatic labels")
                axis.set(xlim=(-.02, 1.02), ylim=(0, max(.3, max(p[metric+"_max"] for p in selected)*1.15)))
            else:
                axis.axvspan(0, fraction, color="#E3E7EB", alpha=.7, label="Below calibration cost")
                axis.axhline(.05, linestyle="--", color="#88939D", linewidth=1, label="5% reference")
                style(axis, "Auto-error versus total human cost", "(Calibration + escalation labels) / test pairs",
                      "Wrong automatic labels / all test pairs")
                axis.set(xlim=(0, max(p["cost"] for p in points)*1.03), ylim=(0, max(.15, max(p[metric+"_max"] for p in selected)*1.15)))
        else:
            for name, label, color in (("class_0", "Predicted non-relevant", "#2369A5"),
                                        ("class_1", "Predicted relevant", "#C15B19")):
                selected = sorted((p for p in points if p[name] is not None), key=lambda p: p["coverage"])
                axis.plot([p["coverage"] for p in selected], [p[name] for p in selected], "o-", color=color, label=label)
            style(axis, "Selective risk by predicted class", "Overall automatic test coverage", "Error within automatic predicted class")
            axis.set(xlim=(-.02, 1.02), ylim=(0, 1))
        axis.legend(frameon=False, fontsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("reliability", "risk_coverage", "auto_error_budget", "class_risk")
    note = (f"One held-out query split; {len(metadata['seeds'])} seeds resolve score ties only. Shading: seed range, not confidence intervals.\n"
            "Segments connect evaluated budgets; undefined risks and infeasible budgets have no performance points.")
    outputs = []

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = args.output_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=180, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.7))
    for axis, kind in zip(axes.flat, kinds):
        draw(axis, kind)
    fig.suptitle("Calibrated MiniLM cross-encoder | TREC DL2019", fontsize=17, fontweight="bold", y=.975)
    fig.text(.5, .025, note, ha="center", va="bottom", color="#52606D", fontsize=9)
    fig.subplots_adjust(left=.08, right=.975, bottom=.135, top=.9, hspace=.43, wspace=.29)
    save(fig, "overview")
    for kind in kinds:
        fig, axis = plt.subplots(figsize=(8.5, 5.7))
        draw(axis, kind)
        fig.text(.5, .02, note, ha="center", va="bottom", color="#52606D", fontsize=8)
        fig.subplots_adjust(left=.13, right=.97, bottom=.23, top=.89)
        save(fig, kind)
    write_json(args.output_dir / "plot_metadata.json", {
        "plotter_sha256": file_hash(Path(__file__)),
        "inputs_sha256": {str(p.resolve()): file_hash(p) for p in [
            args.run_dir / "experiment.json", *paths, args.run_dir / "raw_reliability.csv",
            args.run_dir / "temperature_scaled_reliability.csv"]},
        "outputs_sha256": {p.name: file_hash(p) for p in outputs},
        "uncertainty": "seed tie-break range, not confidence intervals"})
    print(f"Saved {len(outputs)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
