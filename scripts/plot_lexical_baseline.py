"""Plot completed lexical baseline results without rerunning the experiments."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import file_hash, write_csv, write_json


def load_run(path, label):
    metadata = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError(f"Refusing incomplete experiment: {path}")
    with (path / "budget_curve.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["budget"])].append(row)
    if set(groups) != set(metadata["budgets"]):
        raise ValueError("Budget curve does not match completed experiment")
    points = []
    for budget, group in sorted(groups.items()):
        if len(group) != len(metadata["seeds"]) or {int(r["seed"]) for r in group} != set(metadata["seeds"]):
            raise ValueError("Every budget must have exactly one result per configured seed")
        point = {"setting": label, "budget": budget, "seeds": len(group)}
        for field in ("human_fraction", "coverage", "selective_risk", "auto_error_rate", "mixed_label_accuracy"):
            values = [float(r[field]) for r in group if r[field] != ""]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f"Invalid {field}")
            if len(values) not in (0, len(group)):
                raise ValueError("Inconsistent undefined metric across seeds")
            point[field] = statistics.mean(values) if values else None
            point[field + "_min"] = min(values) if values else None
            point[field + "_max"] = max(values) if values else None
        points.append(point)
    half = [p for p in points if p["coverage"] == .5]
    if len(half) != 1:
        raise ValueError("The class-composition plot requires a 50% coverage result")
    class_values = []
    summary_paths = []
    for seed in metadata["seeds"]:
        summary_path = path / f"seed_{seed}" / f"budget_{half[0]['budget']:06d}" / "summary.json"
        summary_paths.append(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        class_values.append([summary["predicted_class_metrics"][str(c)]["accepted"] /
                             summary["automatic_labels"] for c in (0, 1)])
    composition = [statistics.mean(v[c] for v in class_values) for c in (0, 1)]
    return {"path": path, "label": label, "points": points, "composition": composition,
            "metadata": metadata, "summary_paths": summary_paths}


COLORS = ("#2369A5", "#C15B19")


def style_axis(axis, xlabel, ylabel, title):
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="both", color="#E3E7EC", linewidth=.7)
    axis.set_axisbelow(True)


def draw_curve(axis, runs, metric, coverage=False):
    for run, color in zip(runs, COLORS):
        # Zero labels give an untrained 0.5 prior; not a trained-model result.
        points = [p for p in run["points"] if p["budget"] > 0 and p[metric] is not None]
        xfield = "coverage" if coverage else "human_fraction"
        points.sort(key=lambda p: p[xfield])
        x = [p[xfield] for p in points]
        axis.plot(x, [p[metric] for p in points], "o-", color=color, linewidth=2,
                  markersize=4.5, label=run["label"])
        axis.fill_between(x, [p[metric + "_min"] for p in points],
                          [p[metric + "_max"] for p in points], color=color, alpha=.16)
    axis.legend(frameon=False, fontsize=9)


def panel(axis, kind, runs):
    if kind == "risk_coverage":
        draw_curve(axis, runs, "selective_risk", coverage=True)
        style_axis(axis, "Automatic coverage", "Error among automatic labels", "Risk versus coverage")
        axis.set(xlim=(.45, 1.01), ylim=(0, .4))
    elif kind == "auto_error_budget":
        draw_curve(axis, runs, "auto_error_rate")
        style_axis(axis, "Human-label budget / pool size", "Wrong automatic labels / all pairs", "Auto-error versus human budget")
        axis.axhline(.05, linestyle="--", color="#586574", linewidth=1)
        axis.text(.98, .055, "5% target (reference)", ha="right", color="#586574", fontsize=8.5)
        axis.set(xlim=(-.02, 1.02), ylim=(0, .38))
    elif kind == "mixed_accuracy_budget":
        draw_curve(axis, runs, "mixed_label_accuracy")
        style_axis(axis, "Human-label budget / pool size", "Mixed human + automatic accuracy", "Combined label accuracy")
        axis.set(xlim=(-.02, 1.02), ylim=(.6, 1.025))
    elif kind == "class_composition":
        positions = list(range(len(runs)))
        negative = [r["composition"][0] for r in runs]
        positive = [r["composition"][1] for r in runs]
        axis.bar(positions, negative, width=.5, color="#2369A5", label="Predicted non-relevant")
        axis.bar(positions, positive, width=.5, bottom=negative, color="#76AD83", label="Predicted relevant")
        for x, neg, pos in zip(positions, negative, positive):
            axis.text(x, neg / 2, f"{neg:.0%}\nnon-relevant", ha="center", va="center", color="white", fontsize=11)
            axis.text(x, 1.025, f"{pos:.0%} relevant", ha="center", fontsize=9, color="#405060")
        axis.set(xticks=positions, xticklabels=[r["label"] for r in runs], ylim=(0, 1.16),
                 ylabel="Share of automatic predictions", title="Predicted classes at 50% coverage")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E3E7EC", linewidth=.7)
        axis.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, default=ROOT / "results/baselines/lexical/dl2019_full_v2")
    parser.add_argument("--test-run", type=Path, default=ROOT / "results/baselines/lexical/dl2019_shared_test_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures/lexical")
    args = parser.parse_args()
    runs = [load_run(args.full_run, "Full pool (43 queries)"),
            load_run(args.test_run, "Shared test (33 queries)")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "svg.fonttype": "none"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("risk_coverage", "auto_error_budget", "mixed_accuracy_budget", "class_composition")
    seed_counts = "/".join(str(len(r["metadata"]["seeds"])) for r in runs)
    note = (f"Means over {seed_counts} seeds (full/test); shading: seed min-max, not confidence intervals.\n"
            "Segments join tested budgets; intermediate budgets were not evaluated.\n"
            "Zero-budget untrained reference excluded. Full pool and test subset overlap.")
    artifacts = []

    def save(fig, name):
        for extension in ("png", "svg"):
            target = args.output_dir / f"{name}.{extension}"
            fig.savefig(target, dpi=180, facecolor="white")
            artifacts.append(target)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.7))
    for axis, kind in zip(axes.flat, kinds):
        panel(axis, kind, runs)
    fig.suptitle("Classical lexical active learning | TREC DL2019", fontsize=17, fontweight="bold", y=.975)
    fig.text(.5, .025, note, ha="center", va="bottom", color="#52606D", fontsize=9)
    fig.subplots_adjust(left=.08, right=.975, bottom=.135, top=.9, hspace=.43, wspace=.29)
    save(fig, "overview")
    for kind in kinds:
        fig, axis = plt.subplots(figsize=(8.5, 5.7))
        panel(axis, kind, runs)
        fig.text(.5, .02, note, ha="center", va="bottom", color="#52606D", fontsize=8.5)
        fig.subplots_adjust(left=.13, right=.97, bottom=.23, top=.89)
        save(fig, kind)
    write_csv(args.output_dir / "plot_data.csv", [p for r in runs for p in r["points"]])
    sources = [p for r in runs for p in [r["path"] / "budget_curve.csv", r["path"] / "experiment.json", *r["summary_paths"]]]
    write_json(args.output_dir / "plot_metadata.json", {
        "plotter_sha256": file_hash(Path(__file__)),
        "inputs_sha256": {str(p.resolve()): file_hash(p) for p in sources},
        "outputs_sha256": {p.name: file_hash(p) for p in artifacts},
        "matplotlib": matplotlib.__version__, "seeds": [r["metadata"]["seeds"] for r in runs],
        "uncertainty": "range across seeds, not a confidence interval",
        "excluded": "untrained zero-budget endpoint; undefined selective risk at full human budget",
    })
    print(f"Saved {len(artifacts)} figures and plot data to {args.output_dir}")


if __name__ == "__main__":
    main()
