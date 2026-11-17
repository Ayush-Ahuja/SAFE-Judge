"""Report all predeclared class-aware allocations and conditional uncertainty."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_class_threshold_experiment import read_gzip
from src.utils.io import atomic_text, file_hash, write_csv, write_json

INPUT = ROOT / "results/experiments/class_aware_review_v1"
OUTPUT = ROOT / "reports/class_aware_review_v1"
PRIMARY = "local_minimax_floor05"
FIELDS = ("worst_class_risk", "class_0_selective_risk", "class_1_selective_risk",
    "class_0_coverage", "class_1_coverage", "raw_class_0_coverage", "raw_class_1_coverage",
    "raw_class_0_risk", "raw_class_1_risk", "worst_raw_class_risk", "coverage", "selective_risk",
    "query_mean_auto_error_rate", "tail20_query_auto_error", "system_ranking_kendall_tau",
    "system_binary_ndcg10_mae", "graded_system_kendall_tau", "graded_system_ndcg10_mae")


def bootstrap_risks(saved, weights):
    counts = [s["query_class_counts"] for s in saved]
    numerator, denominator = [], []
    for c in (0, 1):
        e = np.array([[q[f"errors_{c}"] for q in run] for run in counts])
        a = np.array([[q[f"accepted_{c}"] for q in run] for run in counts])
        numerator.append(weights @ e.T)
        denominator.append(weights @ a.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        worst = np.maximum(numerator[0]/denominator[0], numerator[1]/denominator[1])
    # A draw must define both class risks in every retained repetition.
    return worst.mean(axis=1)


def main():
    experiment = json.loads((INPUT / "experiment.json").read_text())
    if experiment["status"] != "complete":
        raise ValueError("Experiment incomplete")
    settings = json.loads((INPUT / "design.json").read_text())["settings"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for relative, digest in experiment["allocation_sha256"].items():
        path = INPUT / relative
        if file_hash(path) != digest:
            raise ValueError("Allocation changed")
        loaded = read_gzip(path)
        saved = {k: loaded[k] for k in ("record", "query_class_counts", "system_scores")}
        r = saved["record"]
        groups[r["cohort"], r["fraction"], r["method"]].append(saved)
    summary = []
    for (cohort, fraction, method), saved in sorted(groups.items()):
        saved.sort(key=lambda s: s["record"]["seed"])
        rows = [s["record"] for s in saved]
        row = dict(cohort=cohort, fraction=fraction, method=method, repetitions=len(rows))
        for field in FIELDS:
            vals = [r[field] for r in rows if r.get(field) is not None]
            row[field] = float(np.mean(vals)) if vals else None
            row[field+"_defined"] = len(vals)
        row["both_classes_risk_at_most20"] = sum(r["worst_class_risk"] is not None and r["worst_class_risk"] <= .2 for r in rows)
        row["both_classes_5percent"] = sum(r["both_classes_5percent_coverage"] for r in rows)
        summary.append(row)
    write_csv(OUTPUT / "summary.csv", summary)
    differences = []
    comparators = ["local_query_error_floor05", "shared_minimax_floor05", "local_minimax_floor20",
        "raw_confidence", "lara_1_raw", "lara_1_upstream_smooth", "lara_topics_raw",
        "lara_topics_upstream_smooth", "lara_reserved_raw", "threshold_worst_class_cal200"]
    for cohort in settings["cohorts"]:
        target = groups[cohort, .5, PRIMARY]
        qids = [q["query_id"] for q in target[0]["query_class_counts"]]
        rng = np.random.default_rng(settings["bootstrap_seed"])
        weights = rng.multinomial(len(qids), np.full(len(qids), 1/len(qids)), size=settings["bootstrap_draws"])
        a = bootstrap_risks(target, weights)
        for comparator in comparators:
            other = groups.get((cohort, .5, comparator))
            if other is None:
                continue
            if any([q["query_id"] for q in s["query_class_counts"]] != qids for s in target+other):
                raise ValueError("Query bootstrap alignment differs")
            b = bootstrap_risks(other, weights)
            diff = a-b
            finite = diff[np.isfinite(diff)]
            differences.append(dict(cohort=cohort, method=PRIMARY, comparator=comparator,
                defined_draws=len(finite), draws=len(diff),
                low=float(np.quantile(finite, .025)) if len(finite) else None,
                high=float(np.quantile(finite, .975)) if len(finite) else None))
    write_csv(OUTPUT / "paired_query_intervals.csv", differences)
    scores = []
    for (cohort, fraction, method), saved in sorted(groups.items()):
        if cohort != "dl2020_all":
            continue
        for s in saved:
            v = s["system_scores"]
            mixed_rank = rankdata(-np.array(v["binary_scores"]), method="average")
            gold_rank = rankdata(-np.array(v["gold_binary_scores"]), method="average")
            for i, system in enumerate(v["systems"]):
                scores.append(dict(cohort=cohort, fraction=fraction, method=method, seed=s["record"]["seed"],
                    system=system, binary_score=v["binary_scores"][i], gold_binary_score=v["gold_binary_scores"][i],
                    rank=float(mixed_rank[i]), gold_rank=float(gold_rank[i]), rank_change=float(mixed_rank[i]-gold_rank[i])))
    write_csv(OUTPUT / "official_run_scores.csv", scores)
    retained = [PRIMARY, "local_minimax_floor20", "local_query_error_floor05", "shared_minimax_floor05",
        "local_query_error_floor20", "shared_minimax_floor20", "selected_local", "raw_confidence",
        "lara_1_raw", "lara_1_upstream_smooth", "lara_topics_raw", "lara_topics_upstream_smooth",
        "lara_reserved_raw", "lara_reserved_upstream_smooth", "threshold_worst_class_cal200"]
    lines = ["# Class-aware shared/local review", "", "16 November 2026.", "",
        "This is an exploratory extension of the existing local review policy, developed after inspecting the earlier DL2020, COVID and ChatReport results. It tests whether explicit worst-class allocation improves the risk/coverage tradeoff. It does not establish an independent replication, a formal guarantee or publication novelty.", "",
        "## Method and cost", "",
        "The frozen assessor, raw labels, six class/confidence bins, shared/local prior strengths and paid per-query/class sentinels are retained. Local updates use only purchased target labels. After each batch of up to 32 purchases, the policy sorts remaining pairs by estimated error within each raw predicted class. It enumerates every feasible pair of final acceptance counts and minimizes the larger estimated mean class error. It then purchases the highest estimated-error pairs in the proposed review complement. Purchased pairs stay human-labelled. There are no label flips, additional model calls, external calibration labels or tuning searches.", "",
        "Both 5% and 20% minimum automatic coverage per original predicted class are retained. These are operational settings, not numbers specified by the instructor. The primary setting is 5%. The local query-error ablation uses the same floors, batch schedule and model but minimizes query-weighted error. The shared minimax ablation removes query-local updates. This separates the objective change from the class floor and the local evidence.", "",
        "Every sentinel and review label counts once toward the exact budget. Human responses are simulated by existing gold judgments and assumed correct. Earlier method-development costs are treated as sunk costs; this is deployment-budget accounting. Each of ten seeds starts fresh at 25%, 50% and 75% human cost. Original raw confidence, local policy and all four LARA variants are reused from saved allocations, with their source hashes retained. Global LARA is deterministic; repeated rows are not independent evidence. Reserved LARA is the earlier modified baseline with a 5% raw-class reservation. The separate-threshold comparator spends 200 labels on external calibration queries and the remainder of the same budget on evaluation review, so its evaluation-pool coverage differs.", "",
        "## Half-total-budget results", "",
        "Error is measured among automatic labels within each method's final predicted class. An empty accepted class has undefined risk. Means of defined repetitions are accompanied by their counts; those incomplete means should not be treated as a complete two-class comparison. Raw-positive coverage is the proportion of original assessor-positive cases left automatic, including any relabelled by LARA. Final-class coverage and original-class errors are also retained in summary.csv. The new policies preserve all raw predictions; LARA may correct them. The threshold comparator here uses the same ten seeds as the new study, whereas the earlier threshold report used twenty.", "",
        "| Cohort | Method | Worst-class error | Negative error | Positive error | Raw-positive coverage | Auto coverage | Defined repetitions |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    def pct(x):
        return "undefined" if x is None else f"{x*100:.2f}%"
    for r in summary:
        if r["fraction"] == .5 and r["method"] in retained:
            lines.append(f"| {r['cohort']} | {r['method']} | {pct(r['worst_class_risk'])} | {pct(r['class_0_selective_risk'])} | {pct(r['class_1_selective_risk'])} | {pct(r['raw_class_1_coverage'])} | {pct(r['coverage'])} | {r['worst_class_risk_defined']}/{r['repetitions']} |")
    lines += ["", "## Interpretation", "",
        "At half total cost, primary local minimax has 10.74% worst-class error on DL2020 with 5.04% positive automation, compared with 17.70% for the same-floor query-error ablation and 27.54% for shared-only minimax. Its 13.03-point mean advantage over reserved raw LARA has a conditional query interval excluding zero. This supports further study of query-local selection beyond a raw-confidence prefix, subject to the exploratory limitations below.", "",
        "On the current COVID sample, primary local minimax has 13.81% worst-class error with 5.01% positive automation. The same-floor query-error ablation is identical at this budget: there is no separate demonstrated benefit of changing the objective there. Its difference from reserved raw LARA is uncertain, with the query interval including zero. ChatReport provides no clear local-over-shared gain.", "",
        "Increasing the class floor to 20% raises worst-class error to 34.19% on DL2020 and 25.83% on COVID. Thus the strong primary numbers do not establish reliable automation of a large positive fraction. DL2020 binary ranking tau is 0.9889, but its nDCG score MAE of 0.0140 is worse than raw LARA's 0.0030. The policy is promising evidence for a class-risk/coverage tradeoff, not universal superiority or a completed paper claim."]
    lines += ["", "## Paired query uncertainty", "",
        "Intervals are primary local-minimax minus comparator worst-class error at half total cost, in percentage points. Queries are resampled jointly and the risk difference is averaged across the ten seeds. Purchased allocations and fitted models stay fixed. These conditional percentile intervals omit development/search uncertainty, re-fitting, repeated acquisition and multiplicity; they are exploratory. A draw must define both classes in every repetition.", "",
        "| Cohort | Comparator | Difference interval, pp | Defined draws |", "|---|---|---:|---:|"]
    for r in differences:
        interval = "undefined" if r["low"] is None else f"[{r['low']*100:.2f}, {r['high']*100:.2f}]"
        lines.append(f"| {r['cohort']} | {r['comparator']} | {interval} | {r['defined_draws']}/{r['draws']} |")
    lines += ["", "## Official DL2020 run rankings", "",
        "All 59 submitted passage runs retain their original top-ten positions over 54 topics. Primary binary nDCG@10 uses grade>=2; graded diagnostics use original human 0/1/2/3 linear gains and automatic labels mapped to 0/2. The 1,315 unjudged top-ten positions receive zero gain under both mixed and reference qrels, as in the existing ranking protocol. Agreement is measured against available judgments. `official_run_scores.csv` retains per-system scores and rank changes at every budget and seed.", "",
        "| Method at half total cost | Binary Kendall tau-b | Binary nDCG@10 MAE | Graded Kendall tau-b |", "|---|---:|---:|---:|"]
    for r in summary:
        if r["cohort"] == "dl2020_all" and r["fraction"] == .5 and r["method"] in retained:
            lines.append(f"| {r['method']} | {r['system_ranking_kendall_tau']:.4f} | {r['system_binary_ndcg10_mae']:.4f} | {r['graded_system_kendall_tau']:.4f} |")
    lines += ["", "## Limits and next evidence", "",
        "The planner minimizes working model estimates, not actual unknown error. Adaptively purchased labels need not give unbiased estimates for arbitrary accepted sets. No Bayesian posterior mean is used as a frequentist certificate. The frozen prior clips raw error at 1%, inherited from the original model; fine probability distinctions above 99% confidence can therefore be lost.", "",
        "All benchmarks were previously inspected. DL2020 is policy-development evidence in this study. COVID still uses the cached strict grade>=2 target, with substantial input truncation; the proposal's grade>=1 inference remains pending. Its 100-pairs-per-topic sample does not support faithful official-run ranking evaluation. ChatReport was involved in earlier discovery. More seeds on these pools cannot create untouched confirmation.", "",
        "A publication-oriented follow-up should freeze the retained policy and cost/coverage targets, then test new topics or another untouched collection. Compare risk at the same class-coverage requirements, report the cost-risk-coverage frontier rather than a single winning setting, and complete proposal-aligned COVID inference and official-run pools. If the class objective needs more human review than the allowed budget, report that failure rather than claiming guaranteed safety.", "",
        "![Worst-class error versus positive coverage](risk_positive_coverage.png)", "",
        "Each curve joins the declared 25%, 50% and 75% total human-budget settings. Undefined two-class risks are omitted. Settings at different positions on a curve do not have equal cost.", "", "## Reproduction", "",
        "```powershell", ".venv-lexical/Scripts/python.exe scripts/run_class_aware_review.py --workers 2",
        ".venv-lexical/Scripts/python.exe scripts/analyze_class_aware_review.py", "```", ""]
    atomic_text(OUTPUT / "README.md", "\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    plotted = [PRIMARY, "local_minimax_floor20", "local_query_error_floor05", "shared_minimax_floor05",
               "raw_confidence", "lara_1_upstream_smooth"]
    for ax, cohort in zip(axes, settings["cohorts"]):
        for method in plotted:
            rows = sorted([r for r in summary if r["cohort"] == cohort and r["method"] == method], key=lambda r: r["fraction"])
            ax.plot([r["raw_class_1_coverage"]*100 for r in rows],
                    [r["worst_class_risk"]*100 if r["worst_class_risk"] is not None else np.nan for r in rows], "o-", label=method)
        ax.set(title=cohort, xlabel="Original predicted-positive coverage (%)", ylabel="Worst-class error (%)")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6)
    fig.savefig(OUTPUT / "risk_positive_coverage.png", dpi=160)
    plt.close(fig)
    write_json(OUTPUT / "validation.json", {"allocations_hash_checked": len(experiment["allocation_sha256"]),
        "new_policy_allocations": experiment["new_policy_allocations"], "experiment_sha256": file_hash(INPUT / "experiment.json"),
        "analysis_sha256": file_hash(Path(__file__)), "bootstrap": "Paired query resampling; allocations and fits fixed"})
    print("Report:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
