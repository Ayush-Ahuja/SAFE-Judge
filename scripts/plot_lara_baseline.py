"""Plot completed binary LARA budget sweeps and calibration diagnostics."""

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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures/lara")
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "experiment.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("A complete LARA execution is required")
    with (args.run_dir / "budget_curve.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    groups = [str(g) for g in metadata["groupings"]]
    by_point = defaultdict(list)
    for record in records:
        by_point[(record["grouping"], int(record["budget"]))].append(record)
    for rows in by_point.values():
        if len(rows) != len(metadata["seeds"]) or {int(r["seed"]) for r in rows} != set(metadata["seeds"]):
            raise ValueError("Incomplete seed coverage")
    points = defaultdict(list)
    for (group, budget), rows in sorted(by_point.items()):
        point = {"budget": budget, "human_fraction": float(rows[0]["human_fraction"]),
                 "coverage": float(rows[0]["coverage"])}
        for field in ("selective_risk", "auto_error_rate"):
            values = [float(r[field]) for r in rows if r[field] != ""]
            point[field] = statistics.mean(values) if values else None
            point[field + "_min"] = min(values) if values else None
            point[field + "_max"] = max(values) if values else None
        points[group].append(point)
    half = {group: next((p for p in points[group] if p["coverage"] == .5), None) for group in groups}
    if any(p is None for p in half.values()):
        raise ValueError("The diagnostics require an evaluated 50%-coverage budget")
    summaries, mapping, inputs = {}, {}, [args.run_dir / "experiment.json", args.run_dir / "budget_curve.csv"]
    for group in groups:
        paths = [args.run_dir / f"groups_{group}" / f"seed_{seed}" / f"budget_{half[group]['budget']:06d}"
                 for seed in metadata["seeds"]]
        summaries[group] = [json.loads((p / "summary.json").read_text(encoding="utf-8")) for p in paths]
        prediction_path = paths[0] / "predictions.csv"
        with prediction_path.open(encoding="utf-8", newline="") as stream:
            mapping[group] = sorted((float(r["llm_probability"]), float(r["prob_relevant"])) for r in csv.DictReader(stream))
        inputs.extend([p / "summary.json" for p in paths])
        inputs.append(prediction_path)
    colors = ("#2369A5", "#C15B19", "#3E8052", "#8D61A3")
    labels = {g: "LARA(topics)" if g == "topics" else f"LARA({g})" for g in groups}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})

    def draw(axis, kind):
        if kind in ("risk_coverage", "auto_error_budget"):
            field, xname = ("selective_risk", "coverage") if kind == "risk_coverage" else ("auto_error_rate", "human_fraction")
            for group, color in zip(groups, colors):
                selected = sorted((p for p in points[group] if p[field] is not None), key=lambda p: p[xname])
                x = [p[xname] for p in selected]
                axis.plot(x, [p[field] for p in selected], "o-", color=color, label=labels[group], markersize=4)
                axis.fill_between(x, [p[field + "_min"] for p in selected], [p[field + "_max"] for p in selected], color=color, alpha=.14)
            axis.set(title="Selective risk versus coverage" if kind == "risk_coverage" else "Auto-error versus human budget",
                     xlabel="Automatic coverage" if kind == "risk_coverage" else "Total human judgments / pool size",
                     ylabel="Error among automatic labels" if kind == "risk_coverage" else "Wrong automatic labels / all pairs",
                     xlim=(-.02, 1.02), ylim=(0, .3))
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            if kind == "auto_error_budget":
                axis.axhline(.05, linestyle="--", linewidth=1, color="#7D8994")
                axis.text(.98, .055, "5% reference", ha="right", fontsize=8, color="#626F7D")
        elif kind == "class_risk":
            x = np.arange(len(groups))
            width = .34
            for label, color, offset, title in (("0", "#2369A5", -width/2, "Predicted non-relevant"),
                                                 ("1", "#C15B19", width/2, "Predicted relevant")):
                for i, group in enumerate(groups):
                    values = [s["predicted_class_metrics"][label]["selective_risk"] for s in summaries[group]
                              if s["predicted_class_metrics"][label]["selective_risk"] is not None]
                    if values:
                        mean = statistics.mean(values)
                        axis.bar(x[i]+offset, mean, width=width, color=color, label=title if i == 0 else None)
                        axis.text(x[i]+offset, mean+.012, f"{mean:.1%}", ha="center", fontsize=8)
                    else:
                        axis.text(x[i]+offset, .02, "undefined", rotation=90, ha="center", fontsize=8)
            axis.set(title="Predicted-class risk at 50% coverage", ylabel="Error within automatic predicted class",
                     xticks=x, xticklabels=[labels[g] for g in groups], ylim=(0, .5))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            axis.plot([0, 1], [0, 1], "--", color="#8C98A3", linewidth=1)
            for group, color in zip(groups, colors):
                pairs = mapping[group]
                axis.plot([p[0] for p in pairs], [p[1] for p in pairs], color=color, label=labels[group])
            axis.set(title="Learned mapping after 50% human labels", xlabel="Original LLM P(relevant)",
                     ylabel="LARA P(relevant)", xlim=(0, 1), ylim=(0, 1))
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.25, axis="y")
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kinds = ("risk_coverage", "auto_error_budget", "class_risk", "calibration_mapping")
    assessor_name = (metadata.get("assessor") or {}).get("model_name", "external LLM")
    note = (f"Input: {assessor_name}; preprocessing: {metadata['config']['preprocessing']}.\n"
            "Means and ranges over grouping seeds, not confidence intervals. Mapping panel uses the first seed.\n"
            "Lines join tested budgets; undefined risks are omitted. No formal risk guarantee.")
    outputs = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = args.output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.7))
    for axis, kind in zip(axes.flat, kinds):
        draw(axis, kind)
    fig.suptitle(f"Binary LARA | {metadata['dataset'].replace('_', ' ').upper()}", fontsize=17, fontweight="bold", y=.975)
    fig.text(.5, .025, note, ha="center", va="bottom", fontsize=8.5, color="#52606D")
    fig.subplots_adjust(left=.08, right=.975, bottom=.15, top=.9, hspace=.43, wspace=.29)
    save(fig, "overview")
    for kind in kinds:
        fig, axis = plt.subplots(figsize=(8.5, 5.7))
        draw(axis, kind)
        fig.text(.5, .02, note, ha="center", va="bottom", fontsize=8, color="#52606D")
        fig.subplots_adjust(left=.13, right=.97, bottom=.25, top=.89)
        save(fig, kind)
    write_json(args.output_dir / "plot_metadata.json", {"plotter_sha256": file_hash(Path(__file__)),
        "inputs_sha256": {str(p.resolve()): file_hash(p) for p in inputs},
        "outputs_sha256": {p.name: file_hash(p) for p in outputs},
        "uncertainty": "ranges across grouping seeds, not confidence intervals"})
    print(f"Saved {len(outputs)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
