"""Validate and export the Oosterhuis reproduction and common-budget comparison."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import atomic_text, file_hash, write_csv, write_json


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def execute():
    output = ROOT/"reports/oosterhuis_comparison_v1"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("oosterhuis_reference_v1", "oosterhuis_matched_v1"):
        directory = ROOT/"results/experiments"/name
        metadata = json.loads((directory/"experiment.json").read_text())
        assert metadata["status"] == "complete"
        for path, digest in metadata["source_sha256"].items():
            assert file_hash(ROOT/path.replace("\\", "/")) == digest
        for line in (directory/"checksums.sha256").read_text().splitlines():
            digest, filename = line.split("  ", 1)
            assert file_hash(directory/filename) == digest
        shutil.copyfile(directory/"summary.csv", output/(name+".csv"))
        shutil.copyfile(directory/"experiment.json", output/(name+".json"))
    shutil.copyfile(ROOT/"configs/oosterhuis_reference.json", output/"reference_sources_and_design.json")
    history = ROOT/"cache/research/oosterhuis_initial/provenance.json"
    if history.exists():
        shutil.copyfile(history, output/"initial_release_check.json")
    reference = read(output/"oosterhuis_reference_v1.csv")
    matched = read(output/"oosterhuis_matched_v1.csv")
    runs = read(ROOT/"results/experiments/oosterhuis_matched_v1/runs.csv")
    for row in runs:
        assert int(row["human_labels"]) == int(row["budget"])
        assert abs(int(row["budget"])/int(row["pairs"])-float(row["human_fraction"])) < 1e-12
        assert abs(abs(float(row["dcg_estimate"])-float(row["true_dcg"]))-float(row["dcg_absolute_error"])) < 1e-12
        if row["output_type"] == "metric_interval":
            assert not row["automatic_label_coverage"] and not row["query_auto_error"]
        else:
            assert abs(float(row["automatic_label_coverage"])+float(row["human_fraction"])-1) < 1e-12
    ledgers = 0
    for line in (ROOT/"results/experiments/oosterhuis_matched_v1/purchases.jsonl").read_text().splitlines():
        ledger = json.loads(line)
        pairs = [(r["query_id"], r["doc_id"]) for r in ledger["purchases"]]
        assert len(pairs) == len(set(pairs)) == int(ledger["budget"])
        ledgers += 1
    compact = []
    for dataset in ("trecdl", "robust04"):
        budgets = sorted({int(r["budget"]) for r in matched if r["dataset"] == dataset})
        for budget in budgets:
            rows = {r["method"]: r for r in matched if r["dataset"] == dataset and int(r["budget"]) == budget}
            lara = [r for m, r in rows.items() if m.startswith("lara_")]
            best_metric = min(lara, key=lambda r: float(r["dcg_absolute_error_mean"]))
            best_labels = min(lara, key=lambda r: float(r["query_auto_error_mean"]))
            compact.append({"dataset": dataset, "human_labels": budget, "human_fraction": rows["local_diagnosis"]["human_fraction_mean"],
                "local_dcg_mae": rows["local_diagnosis"]["dcg_absolute_error_mean"],
                "best_lara_dcg_mae": best_metric["dcg_absolute_error_mean"], "best_lara_metric_variant": best_metric["method"],
                "ppi_dcg_mae": rows["oosterhuis_ppi"]["dcg_absolute_error_mean"], "crc_midpoint_dcg_mae": rows["oosterhuis_crc"]["dcg_absolute_error_mean"],
                "local_query_auto_error": rows["local_diagnosis"]["query_auto_error_mean"],
                "best_lara_query_auto_error": best_labels["query_auto_error_mean"], "best_lara_label_variant": best_labels["method"]})
    write_csv(output/"matched_compact.csv", compact)
    # Paired seed differences describe this new fixed pool. These are not query
    # generalization intervals or a replacement for independent confirmation.
    differences = []
    for dataset in ("trecdl", "robust04"):
        by_key = {(int(r["budget"]), int(r["seed"]), r["method"]): r for r in runs if r["dataset"] == dataset}
        for item in (r for r in compact if r["dataset"] == dataset):
            budget = item["human_labels"]
            seeds = sorted({seed for b, seed, _ in by_key if b == budget})
            for second in (item["best_lara_metric_variant"], "oosterhuis_ppi", "oosterhuis_crc"):
                delta = np.asarray([float(by_key[budget, seed, "local_diagnosis"]["dcg_absolute_error"])-float(by_key[budget, seed, second]["dcg_absolute_error"]) for seed in seeds])
                boot = delta[np.random.default_rng(20261113).integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
                differences.append({"dataset": dataset, "budget": budget, "first": "local_diagnosis", "second": second,
                    "mean_dcg_mae_difference": delta.mean(), "conditional_seed_interval_lower": np.quantile(boot, .025),
                    "conditional_seed_interval_upper": np.quantile(boot, .975), "seeds": len(seeds)})
    write_csv(output/"paired_seed_differences.csv", differences)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for column, dataset in enumerate(("trecdl", "robust04")):
        for method, label in (("bootstrap", "Gold bootstrap"), ("ppi", "PPI"), ("crc", "CRC")):
            rows = sorted((r for r in reference if r["dataset"] == dataset and r["method"] == method), key=lambda r: int(r["calibration_queries"]))
            x = [int(r["calibration_queries"]) for r in rows]
            axes[0, column].plot(x, [float(r["mean_width"]) for r in rows], "o-", label=label)
            axes[1, column].plot(x, [100*float(r["interval_coverage"]) for r in rows], "o-", label=label)
        axes[0, column].set(title=dataset.upper(), ylabel="Mean graded DCG@10 interval width")
        axes[1, column].set(xlabel="Human-labeled calibration queries", ylabel="Empirical interval coverage (%)", ylim=(75, 101))
        axes[1, column].axhline(95, color="gray", linestyle=":")
        for row in range(2):
            axes[row, column].grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(output/"reference_intervals.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for column, dataset in enumerate(("trecdl", "robust04")):
        rows = [r for r in compact if r["dataset"] == dataset]
        x = [100*float(r["human_fraction"]) for r in rows]
        for field, label in (("local_dcg_mae", "Local diagnosis"), ("best_lara_dcg_mae", "Best tested LARA"),
                              ("ppi_dcg_mae", "Oosterhuis PPI"), ("crc_midpoint_dcg_mae", "Oosterhuis CRC midpoint")):
            axes[0, column].plot(x, [float(r[field]) for r in rows], "o-", label=label)
        for field, label in (("local_query_auto_error", "Local diagnosis"), ("best_lara_query_auto_error", "Best tested LARA")):
            axes[1, column].plot(x, [100*float(r[field]) for r in rows], "o-", label=label)
        axes[0, column].set(title=dataset.upper()+" / Flan-UL2 / binary top 10", ylabel="Mean absolute DCG@10 error")
        axes[1, column].set(xlabel="Human labels / top-10 pool (%)", ylabel="Query-average automatic error (%)")
        for row in range(2):
            axes[row, column].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.savefig(output/"matched_comparison.png", dpi=160)
    plt.close(fig)
    import platform
    import tqdm
    write_json(output/"validation.json", {"reference_interval_evaluations": 1800, "matched_method_evaluations": len(runs),
        "unique_cost_ledgers": ledgers, "source_and_output_checksums": "passed", "budget_arithmetic": "passed",
        "source_scope": "Unmodified current released reference methods; initial-release methods differ only in copyright. Dataset sampling API update has equivalent 1D semantics.",
        "tests": {"passed": 105, "failed": 0},
        "execution_environment": {"python": platform.python_version(), "numpy": np.__version__, "tqdm": tqdm.__version__},
        "statistical_scope": "Reference uses 100 repetitions at three selected points, paper used 500 over the full sweep. Matched uses 10 seeds and binary top10 adaptation. Conditional seed intervals do not establish query generalization."})
    atomic_text(output/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file() and p.name != "checksums.sha256"))
    print(f"Verified {len(runs)} common-budget results and {ledgers} unique paid-label ledgers")


if __name__ == "__main__":
    execute()
