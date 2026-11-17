"""Check returned-Qwen policy allocations and report matched-cost comparisons."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import COHORTS, METHODS, load_cohort
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

TITLES = {"dl2020_all": "DL2020, 54 queries", "trec_covid_all": "COVID, all 50 queries",
          "trec_covid_locked": "COVID, reserved 40 queries", "chatreport_all": "ChatReport, 11 queries"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_allocations(input_dir, extracted):
    metadata = json.loads((input_dir / "experiment.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Comparison has not completed")
    for relative, digest in metadata["identity"]["source_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f"Changed comparison source {relative}")
    checked_files = 0
    for line in (input_dir / "checksums.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if file_hash(input_dir / relative) != digest:
            raise ValueError(f"Changed comparison output {relative}")
        checked_files += 1
    checked_ledgers = 0
    for cohort in COHORTS:
        rows = load_cohort(extracted, cohort)
        evaluator = PolicyEvaluator(rows)
        indices = {pair_key(r): i for i, r in enumerate(rows)}
        for path in sorted((input_dir / "allocations" / cohort).rglob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                allocation = json.load(stream)
            record = allocation["record"]
            human = np.zeros(len(rows), dtype=bool)
            seen = set()
            for purchase in allocation["purchases"]:
                key = pair_key(purchase)
                if key in seen or key not in indices:
                    raise ValueError("Duplicate or out-of-pool purchase")
                seen.add(key)
                i = indices[key]
                human[i] = True
                if purchase["observed_label"] != rows[i]["true_binary"]:
                    raise ValueError("Purchase label mismatch")
            if len(seen) != record["budget"]:
                raise ValueError("Purchase count mismatch")
            metrics, query_rows = evaluator.evaluate(human, allocation["automatic_predicted_labels"])
            for field, expected in metrics.items():
                actual = record[field]
                if expected is None:
                    if actual is not None:
                        raise ValueError(f"Undefined metric changed {field}")
                elif actual != expected and abs(actual-expected) > 1e-12:
                    raise ValueError(f"Metric recomputation failed {field}")
            stored_queries = {r["query_id"]: r for r in allocation["per_query"]}
            for query in query_rows:
                for name, value in query.items():
                    if stored_queries[query["query_id"]][name] != value:
                        raise ValueError("Per-query metric mismatch")
            checked_ledgers += 1
    return metadata, {"checksummed_files": checked_files, "stored_allocation_ledgers": checked_ledgers,
                      "source_hashes": "passed", "exact_budgets": "passed",
                      "gold_purchase_labels": "passed", "metric_recomputation": "passed"}


def comparison_tables(runs, per_query, primary_lara):
    groups = defaultdict(list)
    by_seed = {}
    for row in runs:
        key = row["dataset"], int(row["budget"]), row["method"]
        groups[key].append(row)
        by_seed[*key, int(row["seed"])] = row
    averages = {}
    fields = ("coverage", "query_mean_auto_error_rate", "auto_error_rate", "selective_risk",
              "automatic_errors", "mixed_accuracy", "tail20_query_auto_error", "worst_query_auto_error")
    for key, rows in groups.items():
        averages[key] = {name: float(np.mean([float(r[name]) for r in rows if r[name] != ""]))
                         if any(r[name] != "" for r in rows) else None for name in fields}
    compact, differences = [], []
    query_losses = defaultdict(dict)
    for row in per_query:
        key = row["dataset"], int(row["budget"]), row["method"]
        query_losses[key].setdefault(row["query_id"], []).append(float(row["auto_error_rate"]))
    for cohort in COHORTS:
        budgets = sorted({budget for dataset, budget, method in averages if dataset == cohort and method == "selected_local"})
        for budget in budgets:
            ours = averages[cohort, budget, "selected_local"]
            variants = [method for method in METHODS if method.startswith("lara_")]
            best = min(variants, key=lambda method: averages[cohort, budget, method]["query_mean_auto_error_rate"])
            for comparator in variants + ["raw_confidence", "random_review"]:
                seeds = sorted(int(r["seed"]) for r in groups[cohort, budget, "selected_local"])
                delta = np.asarray([float(by_seed[cohort, budget, "selected_local", seed]["query_mean_auto_error_rate"])
                                    - float(by_seed[cohort, budget, comparator, seed]["query_mean_auto_error_rate"]) for seed in seeds])
                rng = np.random.default_rng(20261114)
                boot = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
                ours_queries = query_losses[cohort, budget, "selected_local"]
                other_queries = query_losses[cohort, budget, comparator]
                query_delta = np.asarray([np.mean(ours_queries[q])-np.mean(other_queries[q]) for q in sorted(ours_queries)])
                query_boot = query_delta[rng.integers(0, len(query_delta), size=(20000, len(query_delta)))].mean(axis=1)
                differences.append({"dataset": cohort, "budget": budget, "coverage": ours["coverage"],
                    "comparator": comparator, "seeds": len(seeds), "mean_difference": float(delta.mean()),
                    "conditional_seed_interval_lower": float(np.quantile(boot, .025)),
                    "conditional_seed_interval_upper": float(np.quantile(boot, .975)),
                    "fixed_allocation_query_interval_lower": float(np.quantile(query_boot, .025)),
                    "fixed_allocation_query_interval_upper": float(np.quantile(query_boot, .975)),
                    "seed_wins": int((delta < -1e-12).sum()), "seed_ties": int((abs(delta) <= 1e-12).sum()),
                    "seed_losses": int((delta > 1e-12).sum())})
            row = {"dataset": cohort, "human_labels": budget, "coverage": ours["coverage"],
                   "best_tested_lara_variant": best}
            for prefix, method in (("ours", "selected_local"), ("primary_lara", primary_lara),
                                   ("best_lara", best), ("confidence", "raw_confidence"), ("random", "random_review")):
                row.update({f"{prefix}_{name}": value for name, value in averages[cohort, budget, method].items()})
            row["ours_minus_best_lara_query_error"] = row["ours_query_mean_auto_error_rate"] - row["best_lara_query_mean_auto_error_rate"]
            compact.append(row)
    return compact, differences, averages


def plot_comparison(compact, output):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, layout="constrained")
    series = (("ours", "Our frozen policy", "#D55E00", "-"),
              ("primary_lara", "LARA global, normalized", "#0072B2", "--"),
              ("best_lara", "Best tested LARA per budget", "#222222", ":"),
              ("confidence", "Raw confidence", "#009E73", "-."),
              ("random", "Random review", "#999999", "-"))
    for column, cohort in enumerate(COHORTS):
        rows = sorted((r for r in compact if r["dataset"] == cohort), key=lambda r: r["coverage"])
        for prefix, label, color, linestyle in series:
            for axis_row, field in ((0, "query_mean_auto_error_rate"), (1, "selective_risk")):
                points = [(r["coverage"], r[f"{prefix}_{field}"]) for r in rows]
                if axis_row == 0:
                    points = [(0., 0.)] + points
                axes[axis_row, column].plot([100*x for x, _ in points], [100*y for _, y in points],
                                          marker="o", markersize=3, label=label, color=color, linestyle=linestyle)
                axes[axis_row, column].grid(alpha=.2)
        axes[0, column].set_title(TITLES[cohort])
        axes[1, column].set_xlabel("Automatic coverage (%)")
        for axis_row in range(2):
            axes[axis_row, column].set_xlim(0, 100)
            axes[axis_row, column].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Query-average automatic error (%)")
    axes[1, 0].set_ylabel("Error among automatic labels (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=9)
    fig.savefig(output / "risk_coverage_comparison.png", dpi=180)
    plt.close(fig)


def make_report(output, metadata, compact, differences, validation):
    lines = ["# Qwen transfer comparison with LARA", "", "14 November 2026.", "",
        "This comparison uses the returned Colab Qwen3.5-4B probabilities. The frozen local policy is the existing development-selected `local_unguarded` variant in `configs/diagnostic_review_selected.json`. Each policy starts fresh at each budget. Ten seeds cover 6000–6009. All sentinel and calibration labels count toward the common budget. Existing human judgments simulate paid expert responses.", "",
        "The four LARA variants combine global or per-query acquisition with raw or upstream-normalized probabilities. The fixed primary comparator is global LARA with upstream normalization. The best-tested LARA column selects the lowest mean query-average automatic error among all four variants separately for each budget. This retrospective selection uses evaluation labels. All four variants remain in the saved tables.", "",
        "## Input checks and raw assessor results", "",
        "All 17,046 unique pairs passed manifest/source hash checks, complete-pair validation, answer-logit checks and CSV/SQLite consistency. These checks verify the returned artifacts and their recorded inference provenance.", "",
        "| Dataset | Queries | Pairs | Accuracy | Macro-F1 | Brier | ECE | Truncated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("dl2020", "trec_covid", "chatreport"):
        info = metadata["input_validation"]["datasets"][dataset]
        m = info["raw_metrics"]
        lines.append(f"| {dataset} | {info['queries']} | {info['pairs']:,} | {100*m['accuracy']:.2f}% | {m['macro_f1']:.4f} | {m['brier_score']:.4f} | {m['ece_probability_bins_10']:.4f} | {info['truncated_pairs']:,} |")
    lines += ["", "COVID contains the frozen 100-pairs-per-query sample. Its strict binary target counts only grade 2 as positive. DL counts grades 2–3 as positive. ChatReport counts partial and full relevance as positive and uses question-plus-passage inputs. Dataset-specific instructions were used. Cross-domain comparisons therefore also reflect changes in the task rubric. The CUDA run recorded PyTorch 2.14.1+cu130. The earlier DL2019 run used PyTorch 2.11.0+cu130.", "",
        "The reserved COVID cohort retains the prior 40 evaluation topics and 4,000 pairs. The full 50-query cohort also includes 10 development topics. The two COVID cohorts overlap and allocate their review budgets independently. ChatReport was used in earlier method discovery with other assessor outputs. These distinctions limit claims about independent validation.", "",
        "## Matched-budget policy results", "",
        "Automatic coverage is the number of unreviewed pairs divided by the total pool size. Query-average automatic error averages each query's remaining automatic-error count divided by its full number of pairs. Pair-average automatic error divides all remaining automatic errors by all pairs. Selective risk divides those errors by the number of automatic labels. The full-human endpoint has zero automatic errors under the perfect-oracle assumption.", ""]
    for cohort in COHORTS:
        lines += [f"### {TITLES[cohort]}", "",
                  "| Human labels | Auto coverage | Our query error | Primary LARA query error | Best LARA query error | Our pair error | Best LARA pair error | Our selective risk | Best LARA selective risk |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in (r for r in compact if r["dataset"] == cohort):
            fields = ("ours_query_mean_auto_error_rate", "primary_lara_query_mean_auto_error_rate", "best_lara_query_mean_auto_error_rate",
                      "ours_auto_error_rate", "best_lara_auto_error_rate", "ours_selective_risk", "best_lara_selective_risk")
            cells = " | ".join(f"{100*r[field]:.3f}%" for field in fields)
            lines.append(f"| {r['human_labels']:,} | {100*r['coverage']:.2f}% | {cells} |")
        selected = next(r for r in compact if r["dataset"] == cohort and abs(r["coverage"]-.5) < .001)
        diff = selected["ours_minus_best_lara_query_error"]
        direction = "lower" if diff < 0 else "higher" if diff > 0 else "equal"
        lines += ["", f"At approximately 50% automatic coverage, our mean query-average automatic error is {abs(100*diff):.3f} percentage points {direction} than the best tested LARA variant, `{selected['best_tested_lara_variant']}`. Our remaining automatic-error count averages {selected['ours_automatic_errors']:.1f}, compared with {selected['best_lara_automatic_errors']:.1f} for that LARA variant.", ""]
    lines += ["![Policy comparison](risk_coverage_comparison.png)", "", "## Paired differences and limitations", "",
              "Differences below equal our query-average automatic error minus the comparator's error. Negative values favor our policy. Conditional seed-bootstrap intervals describe review-selection variation on the fixed input pool. Global LARA is deterministic in the existing implementation and its allocation is reused across seeds. These repeated rows add no independent information. A direct two-seed check on the returned ChatReport scores confirmed identical global-LARA acquisitions and outputs for both preprocessing variants.", "",
              "The CSV also includes query-bootstrap intervals formed by resampling per-query losses after averaging seeds, with acquired allocations held fixed. Adaptive fitting is held fixed in these intervals. They do not represent a complete rerun of the learning procedure on a new query sample. Pointwise intervals have no correction for comparing several budgets or variants.", "",
              "| Cohort at 50% auto coverage | Comparator | Difference, pp | Conditional seed 95% interval, pp | Fixed-allocation query 95% interval, pp | Seed wins / ties / losses |",
              "|---|---|---:|---|---|---|"]
    for cohort in COHORTS:
        r = next(r for r in compact if r["dataset"] == cohort and abs(r["coverage"]-.5) < .001)
        for method in (metadata["identity"]["primary_lara"], r["best_tested_lara_variant"]):
            if method == metadata["identity"]["primary_lara"] and method == r["best_tested_lara_variant"]:
                label = "Primary and best tested LARA"
            else:
                label = method
            d = next(d for d in differences if d["dataset"] == cohort and d["budget"] == r["human_labels"] and d["comparator"] == method)
            lines.append(f"| {TITLES[cohort]} | {label} | {100*d['mean_difference']:.3f} | [{100*d['conditional_seed_interval_lower']:.3f}, {100*d['conditional_seed_interval_upper']:.3f}] | [{100*d['fixed_allocation_query_interval_lower']:.3f}, {100*d['fixed_allocation_query_interval_upper']:.3f}] | {d['seed_wins']} / {d['seed_ties']} / {d['seed_losses']} |")
            if method == r["best_tested_lara_variant"] and method == metadata["identity"]["primary_lara"]:
                break
    lines += ["", "The current comparison measures relevance-label quality and cost. Retrieval-system rankings are absent from the returned archive. Ranking fidelity and comparisons with Oosterhuis require retrieval runs and their own common metric target. Remaining automatic errors are measured using the held-out labels available to the evaluator. No release certificate was requested or computed in these policy runs.", "",
              "The selected policy and all hyperparameters were retained from the prior configuration. No setting was selected using these new policy outcomes. Raw confidence and random review are included as controls. Domain-specific prompts, the 512-token context limit, the strict COVID positive class and the assumption of perfectly correct human labels affect interpretation.", "",
              "## Artifacts and reproduction", "",
              f"The runner produced {metadata['method_evaluations']:,} result rows, including the full-human references. Verification checked {validation['stored_allocation_ledgers']:,} distinct stored allocation ledgers and recomputed their metrics. Reused deterministic LARA rows are identified in `runs.csv`.", "",
              "The report directory contains `comparison.csv`, `paired_differences.csv`, `all_methods_summary.csv`, `input_validation.json` and the plot. The experiment directory retains all run-level results, per-query errors, purchase ledgers, automatic predictions, policy state, configuration identity and checksums.", "", "```powershell",
              "python scripts/compare_qwen_transfer.py --workers 4", "python scripts/analyze_qwen_transfer.py", "```", "",
              "The comparison runner resumes completed allocations when its source, settings and input identity match. Use a new output directory when those inputs change."]
    atomic_text(output / "README.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/experiments/qwen_transfer_comparison_v1")
    parser.add_argument("--extracted-dir", type=Path, default=ROOT / "results/transfer/qwen35_transfer_v1_dataset_full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/qwen_transfer_comparison_v1")
    args = parser.parse_args()
    metadata, validation = validate_allocations(args.input_dir, args.extracted_dir)
    runs = read_csv(args.input_dir / "runs.csv")
    per_query = read_csv(args.input_dir / "per_query.csv")
    if len(runs) != metadata["method_evaluations"]:
        raise ValueError("Run count mismatch")
    compact, differences, _ = comparison_tables(runs, per_query, metadata["identity"]["primary_lara"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison.csv", compact)
    write_csv(args.output_dir / "paired_differences.csv", differences)
    atomic_text(args.output_dir / "all_methods_summary.csv", (args.input_dir / "summary.csv").read_text(encoding="utf-8"))
    write_json(args.output_dir / "input_validation.json", metadata["input_validation"])
    write_json(args.output_dir / "validation.json", {**validation, "result_rows": len(runs),
               "analysis_source_sha256": file_hash(Path(__file__)),
               "statistical_scope": "Paired seed resampling on fixed pools and conditional query resampling with allocations held fixed."})
    plot_comparison(compact, args.output_dir)
    make_report(args.output_dir, metadata, compact, differences, validation)
    atomic_text(args.output_dir / "checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(args.output_dir.iterdir())
                if p.is_file() and p.name != "checksums.sha256"))
    print(json.dumps(validation), flush=True)
    for r in compact:
        if abs(r["coverage"]-.5) < .001:
            print(r["dataset"], "ours", r["ours_query_mean_auto_error_rate"], "best LARA", r["best_lara_query_mean_auto_error_rate"],
                  r["best_tested_lara_variant"], flush=True)
    print(args.output_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
