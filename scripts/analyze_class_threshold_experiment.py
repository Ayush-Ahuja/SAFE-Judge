"""Validate and report the fixed-budget class-threshold experiment."""

from collections import defaultdict
import csv
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
from scripts.run_class_threshold_experiment import (OUTPUT as INPUT, studies, read_gzip, baseline,
    sample_calibration, RankingMetrics)
from src.calibration.class_thresholds import select_class_thresholds, confidence_order
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

OUTPUT = ROOT / "reports/class_threshold_v1"


def allocation_path(record):
    suffix = f"seed_{record['seed']}_"
    if record["calibration_labels"]:
        suffix += f"cal_{record['calibration_labels']}_"
    suffix += f"budget_{record['total_human_labels']}.json.gz"
    return INPUT / "allocations" / record["scenario"] / record["method"] / suffix


def summary(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["scenario"], r["method"], r["calibration_labels"], r["fraction"]].append(r)
    fields = ("coverage", "worst_class_risk", "class_0_selective_risk", "class_1_selective_risk",
        "class_0_coverage", "class_1_coverage", "selective_risk", "query_mean_auto_error_rate",
        "worst_raw_class_risk", "system_ranking_kendall_tau", "system_binary_ndcg10_mae",
        "graded_system_kendall_tau", "graded_system_ndcg10_mae", "graded_maximum_rank_change",
        "class_0_threshold", "class_1_threshold")
    result = []
    for (scenario, method, cal, fraction), rows in sorted(groups.items()):
        item = dict(scenario=scenario, method=method, calibration_labels=cal, fraction=fraction,
            repetitions=len(rows), both_classes_defined=sum(r["both_classes_accepted"] for r in rows),
            both_classes_5percent_coverage=sum(r["both_classes_5percent_coverage"] for r in rows),
            both_calibration_tails_supported=sum(r.get("both_tails_supported", False) for r in rows),
            total_human_labels=rows[0]["total_human_labels"])
        for field in fields:
            values = [r[field] for r in rows if r.get(field) is not None]
            item[field+"_mean"] = float(np.mean(values)) if values else None
            item[field+"_std"] = float(np.std(values)) if values else None
            item[field+"_defined_repetitions"] = len(values)
        result.append(item)
    return result


def oracle_diagnostic(study, settings):
    """Gold-only attainable frontier; never used by the deployed selector."""
    p = np.asarray([r["prob_relevant"] for r in study["test"]])
    y = np.asarray([r["true_binary"] for r in study["test"]])
    raw = p >= .5
    result = []
    cost = settings["primary_labels_per_query"]*settings["calibration_queries"]
    for seed in settings["seeds"]:
        orders = confidence_order(p, seed)
        floor = [int(np.ceil(settings["minimum_class_coverage"]*len(ix))) for ix in orders]
        prefixes = [np.r_[0, np.cumsum((raw[ix] != y[ix]).astype(int))] for ix in orders]
        for fraction in settings["human_fractions"]:
            budget = int(fraction*len(p))
            automatic = len(p)-(budget-cost)
            k0 = np.arange(max(floor[0], automatic-len(orders[1])), min(len(orders[0]), automatic-floor[1])+1)
            k1 = automatic-k0
            r0, r1 = prefixes[0][k0]/k0, prefixes[1][k1]/k1
            worst = np.maximum(r0, r1)
            total = (prefixes[0][k0]+prefixes[1][k1])/automatic
            best = int(np.lexsort((k0, total, worst))[0])
            result.append(dict(scenario=study["id"], seed=seed, fraction=fraction,
                oracle_worst_class_risk=float(worst[best]), oracle_class_0_risk=float(r0[best]),
                oracle_class_1_risk=float(r1[best]), class_0_accepted=int(k0[best]), class_1_accepted=int(k1[best]),
                scope="Diagnostic uses evaluation gold to find the attainable minimum within the two-threshold family. Not a deployable result or selected method."))
    return result


def paired_query_intervals(study, grouped, settings):
    chosen = [r for r in grouped if r["scenario"] == study["id"] and r["fraction"] == .5 and
              (r["calibration_labels"] == 0 or r["calibration_labels"] == 200)]
    data = defaultdict(list)
    for r in chosen:
        data[r["method"]].append(r)
    qids = sorted({r["query_id"] for r in study["test"]})
    query_weights = np.random.default_rng(settings["bootstrap_seed"]).multinomial(
        len(qids), np.full(len(qids), 1/len(qids)), size=settings["bootstrap_draws"])
    boot = {}
    for method, rows in data.items():
        samples = []
        for row in sorted(rows, key=lambda r: r["seed"]):
            counts = {r["query_id"]: r for r in read_gzip(allocation_path(row))["query_class_counts"]}
            matrix = np.asarray([[counts[q][name] for name in ("accepted_0", "accepted_1", "errors_0", "errors_1")] for q in qids])
            sampled = query_weights @ matrix
            denominator = sampled[:, :2]
            risks = np.divide(sampled[:, 2:], denominator, out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)
            samples.append(np.max(risks, axis=1))
        boot[method] = np.mean(samples, axis=0)
    result = []
    for other in ("raw_confidence", "threshold_common_paid", "lara_1_raw", "lara_1_upstream_smooth", "lara_reserved_raw", "lara_reserved_upstream_smooth"):
        delta = boot["threshold_worst_class"]-boot[other]
        valid = delta[np.isfinite(delta)]
        result.append(dict(scenario=study["id"], comparator=other, valid_bootstrap_draws=len(valid),
            difference_lower_95=float(np.quantile(valid, .025)) if len(valid) else None,
            difference_upper_95=float(np.quantile(valid, .975)) if len(valid) else None,
            probability_difference_below_zero=float(np.mean(valid < 0)) if len(valid) else None,
            scope="Paired evaluation-query bootstrap of the mean worst-class risk across 20 fixed calibration samples. Allocations are held fixed; no refitting or adaptive reacquisition. Undefined class risks are not replaced with zero. Exploratory, unadjusted intervals."))
    return result


def validate(all_studies, records, settings):
    study_map = {s["id"]: s for s in all_studies}
    evaluators = {s["id"]: PolicyEvaluator(s["test"]) for s in all_studies}
    calibrated = {}
    reconstructed = checked = 0
    for record in records:
        study = study_map[record["scenario"]]
        evaluator = evaluators[study["id"]]
        saved = read_gzip(allocation_path(record))
        if record != saved["record"]:
            raise ValueError("Allocation record changed")
        c = record["calibration_labels"]
        if c:
            key = study["id"], record["seed"], c
            if key not in calibrated:
                cal = sample_calibration(study["calibration"], c//settings["calibration_queries"], record["seed"])
                path = INPUT / "calibration" / study["id"] / f"seed_{record['seed']}_labels_{c}.json"
                stored = json.loads(path.read_text())
                if [pair_key(r) for r in cal] != [pair_key(r) for r in stored]:
                    raise ValueError("Calibration sample does not reproduce")
                if [r["true_binary"] for r in cal] != [r["observed_label"] for r in stored]:
                    raise ValueError("Paid calibration labels changed")
                calibrated[key] = cal
            cal = calibrated[key]
            human = np.unpackbits(np.asarray(saved["human_packed"], dtype=np.uint8))[:len(study["test"])].astype(bool)
            predicted = evaluator.raw
            if record["method"] != "threshold_common_paid":
                rebuilt, state = select_class_thresholds([r["prob_relevant"] for r in cal], [r["true_binary"] for r in cal],
                    [r["prob_relevant"] for r in study["test"]], record["automatic_labels"], record["seed"],
                    settings["minimum_class_coverage"], settings["minimum_calibration_tail_support"],
                    record["method"].removeprefix("threshold_"))
                if not np.array_equal(human, rebuilt) or state != saved["state"]:
                    raise ValueError("Threshold selection does not reproduce")
                reconstructed += 1
        else:
            human, predicted, state = baseline(study, record["method"], record["seed"], record["total_human_labels"])
            if saved["state"]["allocation_sha256"] != state["allocation_sha256"]:
                raise ValueError("Baseline source allocation changed")
        metrics, _ = evaluator.evaluate(human, predicted)
        for name in ("human_labels", "automatic_errors", "class_0_accepted", "class_1_accepted", "class_0_errors", "class_1_errors"):
            if metrics[name] != record[name]:
                raise ValueError("Class metric differs from reconstructed allocation")
        if int(human.sum())+c != record["total_human_labels"]:
            raise ValueError("Calibration-inclusive human budget mismatch")
        checked += 1
    return {"allocations_and_costs_recomputed": checked, "two_threshold_selections_reconstructed": reconstructed,
            "calibration_samples_checked": len(calibrated), "test_labels_enter_selector": False}


def display(value, percent=False):
    return "undefined" if value is None else (f"{100*value:.2f}%" if percent else f"{value:.4f}")


def make_report(summaries, intervals, oracle, metadata, settings):
    lines = ["# Separate positive/negative confidence thresholds", "", "16 November 2026.", "",
        "This experiment directly tests the instructor's request: can separate confidence cutoffs reduce the larger accepted-label error rate for positive and negative predictions, compared with raw confidence and LARA at the same total human-label cost? It uses frozen cached Qwen3.5-4B scores. It is a new experiment, distinct from the earlier shared/local adaptive review policy and Gaussian model study.", "",
        "## Result", "",
        "The primary two-threshold rule gives a small DL2020 improvement over raw confidence, but does not establish the requested advantage over LARA. At half total human-label cost, worst accepted-class error is 38.08% for thresholds versus 40.32% for raw confidence, 34.06% for global upstream-normalized LARA, and 23.77% for the supplemental raw LARA with reserved classes. Original global raw LARA accepts no positives, making its two-class risk undefined. Its ranking fidelity remains stronger than the learned thresholds.", "",
        "The DL2020 threshold-minus-raw difference is -2.24 percentage points; its paired conditional query-bootstrap interval is [-4.14, -0.17] points. This exploratory interval holds the fitted policies fixed and is not an untouched confirmation. On COVID, primary target-calibrated thresholds give 27.19% versus raw confidence's 27.13%, with a difference interval [-2.76, 3.68] points. Thus the web-domain improvement does not transfer reliably under this protocol.", "",
        "Increasing calibration from 200 to 500 labels yields 36.25% worst-class error on DL2020 and 25.76% on COVID at half total cost. These are sensitivity settings, not a new selected primary model. No consistent superiority to the retained LARA variants is established.", "",
        "A gold-only search over this threshold family, with the same required 5% automatic coverage in each raw class, bottoms out near 23.67% on DL2020 and 23.16% on COVID, averaged over boundary-tie seeds. High-confidence positive errors constrain this family in these cached scores. This diagnosis is conditional on confidence-prefix acceptance, the specified class floor and fixed boundary tie rules; it does not rule out other policies or a different assessor.", "",
        "## Method and frozen protocol", "",
        "Let c=max(p,1-p) and a=1[p>=0.5]. Accept a negative prediction when c>=t0 and a positive prediction when c>=t1. Boundary ties are resolved by a recorded seeded order to meet the exact budget. All remaining pairs are reviewed by a human. Automatic labels are unchanged.", "",
        "For each candidate threshold in class a, n_a paid calibration judgments fall in the accepted confidence tail and e_a are wrong. The working tail-risk estimate is (e_a+0.5)/(n_a+1), with risk set to 1 when fewer than five calibration tail observations exist. This is Jeffreys smoothing, not an upper confidence bound or guarantee.", "",
        "For budget G and calibration cost C, exactly G-C evaluation pairs are reviewed. All feasible within-class acceptance counts are enumerated using unlabeled evaluation scores. The primary rule minimizes the maximum of the two estimated class risks. Ties use estimated overall error, larger minimum calibration support, more balanced class coverage, then negative acceptance count. A total-error objective is an ablation. Common-paid confidence spends the same C and reviews by ordinary confidence, isolating the effect of allocation from calibration overhead.", "",
        "At least 5% of each raw predicted class must remain automatic for the learned thresholds. This prevents an empty class from appearing safe. The 5% floor is a declared operational diagnostic, not part of the instructor's quoted instruction. Raw confidence and original LARA are reported without altering their class coverage. Supplementary reserved LARA protects the most confident 5% of each raw class before acquisition, keeps their raw labels, and runs the original global LARA on all other rows. This is explicitly a modified baseline, not the authors' original algorithm.", "",
        "DL2020 uses ten source DL2019 queries, excluding any shared DL2020 query IDs. COVID-target uses the existing ten COVID development queries and evaluates the other forty. COVID-source transfers the same source queries to those forty COVID evaluation queries. Uniform per-query calibration samples contain 100, 200 or 500 labels. The primary setting uses 200. All calibration labels, including source-domain labels, are charged to G. Twenty calibration sampling/tie seeds are used. Original adaptive baselines reuse their ten saved allocation seeds (global LARA is deterministic); repeated rows are not twenty independent datasets.", "",
        "COVID uses the cached strict grade>=2 relevance definition. The proposal-aligned grade>=1 prompt has not been inferred. All benchmark outcomes were previously inspected; this is exploratory evidence, not a fresh blind confirmation.", "",
        "## Primary half-total-budget results", "",
        "Error here is the fraction of accepted labels that are wrong within each predicted class. Worst-class risk is defined only when both classes are accepted. It differs from earlier query-average error divided by all query pairs. Values are means across the twenty repetitions; undefined runs are counted explicitly.", "",
        "| Evaluation/calibration | Method | Negative risk | Positive risk | Worst-class risk | Positive coverage | Overall coverage | Both classes defined |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    methods = ("threshold_worst_class", "threshold_total_error", "threshold_common_paid", "raw_confidence",
        "lara_1_raw", "lara_1_upstream_smooth", "lara_topics_raw", "lara_topics_upstream_smooth",
        "lara_reserved_raw", "lara_reserved_upstream_smooth", "selected_local")
    for scenario in ("dl2020_source", "covid_target", "covid_source"):
        for method in methods:
            c = 200 if method.startswith("threshold_") else 0
            row = next(r for r in summaries if r["scenario"] == scenario and r["method"] == method and r["fraction"] == .5 and r["calibration_labels"] == c)
            fields = ("class_0_selective_risk", "class_1_selective_risk", "worst_class_risk", "class_1_coverage", "coverage")
            cells = " | ".join(display(row[f+"_mean"], True) for f in fields)
            lines.append(f"| {scenario} | {method} | {cells} | {row['both_classes_defined']}/20 |")
    lines += ["", "A baseline with no accepted positive labels has undefined positive risk. It cannot be declared better or worse on a two-defined-class objective. The reserved-LARA rows provide an additional comparison that retains both original prediction types. Final predicted-class and original raw-class risks are both retained in the CSV.", "",
        "## All budgets and calibration-cost sensitivity", "",
        "| Scenario | Calibration labels | Total human fraction | Threshold worst-class risk | Automatic coverage | Supported calibration tails, repetitions |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["method"] == "threshold_worst_class":
            lines.append(f"| {r['scenario']} | {r['calibration_labels']} | {r['fraction']:.0%} | {display(r['worst_class_risk_mean'], True)} | {display(r['coverage_mean'], True)} | {r['both_calibration_tails_supported']}/20 |")
    lines += ["", "## Paired query uncertainty at half total budget", "",
        "Intervals describe threshold worst-class risk minus the comparator. Negative values favor thresholds. Queries are jointly resampled across methods; fitted rules and purchased allocations remain fixed. Each bootstrap draw averages the worst-class risk across the twenty calibration samples. These exploratory percentile intervals do not account for model development, source-query selection, adaptive reacquisition after resampling, or multiplicity.", "",
        "| Scenario | Comparator | Difference interval, percentage points | Defined draws |", "|---|---|---:|---:|"]
    for r in intervals:
        bounds = "undefined" if r["difference_lower_95"] is None else f"[{100*r['difference_lower_95']:.2f}, {100*r['difference_upper_95']:.2f}]"
        lines.append(f"| {r['scenario']} | {r['comparator']} | {bounds} | {r['valid_bootstrap_draws']}/{settings['bootstrap_draws']} |")
    lines += ["", "## Official DL2020 run ranking", "",
        "All 59 submitted passage runs retain their original top-ten positions across all 54 queries. Binary nDCG@10 uses grade>=2 in both reference and mixed qrels. The graded diagnostic uses the original 0/1/2/3 human grades as linear gains, matching trec_eval's default. Reviewed pairs retain their true grade; automatic binary predictions map to grades 0 and 2. This fixed mapping is not a graded LLM prediction. Unjudged positions remain unjudged and contribute zero gain in both qrels sets.", "",
        "| Method at half total budget | Binary Kendall tau-b | Binary nDCG@10 MAE | Graded Kendall tau-b | Graded nDCG@10 MAE |",
        "|---|---:|---:|---:|---:|"]
    for method in methods:
        c = 200 if method.startswith("threshold_") else 0
        r = next(r for r in summaries if r["scenario"] == "dl2020_source" and r["method"] == method and r["fraction"] == .5 and r["calibration_labels"] == c)
        fields = ("system_ranking_kendall_tau", "system_binary_ndcg10_mae", "graded_system_kendall_tau", "graded_system_ndcg10_mae")
        lines.append(f"| {method} | "+" | ".join(display(r[f+"_mean"]) for f in fields)+" |")
    lines += ["", "Original judgments have 1,315 unjudged top-ten positions among the 31,860 run/query positions. The binary and graded references therefore measure agreement under available judgments, not complete relevance knowledge. Graded gains follow the [trec_eval implementation](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c). Run-count provenance and missing positions are retained in `ranking_provenance.json`; per-system scores/rank changes appear in `official_run_scores.csv`.", "",
        "## Gold-only threshold-family diagnostic", "",
        "The following table finds the best achievable pair of cutoffs using evaluation gold, at the same primary calibration overhead and 5% class floor. It is an unattainable oracle comparison, not the learned method. It separates limitations of the confidence-threshold family from calibration-transfer failures. It never changes the selected rule.", "",
        "| Scenario | Total human fraction | Oracle minimum worst-class risk, mean over tie seeds |", "|---|---:|---:|"]
    for scenario in ("dl2020_source", "covid_target", "covid_source"):
        for fraction in settings["human_fractions"]:
            values = [r["oracle_worst_class_risk"] for r in oracle if r["scenario"] == scenario and r["fraction"] == fraction]
            lines.append(f"| {scenario} | {fraction:.0%} | {display(float(np.mean(values)), True)} |")
    lines += ["", "![Worst-class risk across total budgets](worst_class_budget.png)", "",
        "![Held-out class-specific confidence tails](class_tail_risk.png)", "",
        "The tail-risk plot is a held-out descriptive diagnostic using evaluation gold, with seed 6000 for tied confidence scores. It shows accepted-label error as progressively more predictions are accepted within each raw class. The 5% line marks the operational coverage floor. No point on these curves is used to choose the learned thresholds.", "",
        "## Reproduction and limits", "",
        f"The study contains {metadata['evaluations']:,} policy evaluations, {metadata['calibration_samples']} paid calibration samples and {metadata['new_reserved_lara_runs']} new reserved-class LARA runs. The original LARA/local allocation ledgers are reused and hash checked. All threshold selections receive evaluation probabilities only. Post-selection evaluation and the separately marked oracle use evaluation gold.", "",
        "The learned rule provides an empirical selective policy, not CRC or a finite-sample risk guarantee. Separate thresholds are established selective-classification ingredients; this experiment does not establish novelty. A successful held-out result here would still require confirmation on untouched topics before a publication claim. COVID's current pool supports label-risk evaluation, but its 100-pairs-per-query sample does not support a faithful official-run ranking comparison.", "",
        "```powershell", "python scripts/run_class_threshold_experiment.py", "python scripts/analyze_class_threshold_experiment.py", "```"]
    atomic_text(OUTPUT / "README.md", "\n".join(lines)+"\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((INPUT / "experiment.json").read_text())
    design = json.loads((INPUT / "design.json").read_text())
    settings = design["settings"]
    if metadata["status"] != "complete":
        raise ValueError("Experiment incomplete")
    for group in ("source_sha256", "input_sha256"):
        for name, digest in design[group].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Changed frozen source/input: {name}")
    for line in (INPUT / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if file_hash(INPUT / name) != digest:
            raise ValueError(f"Changed result: {name}")
    paths = sorted((INPUT / "allocations").rglob("*.json.gz"))
    records = [read_gzip(path)["record"] for path in paths]
    if len(records) != metadata["evaluations"]:
        raise ValueError("Evaluation row count mismatch")
    all_studies = studies(settings)
    validation = validate(all_studies, records, settings)
    summaries = summary(records)
    write_csv(OUTPUT / "summary.csv", summaries)
    intervals = [r for study in all_studies for r in paired_query_intervals(study, records, settings)]
    write_csv(OUTPUT / "paired_query_intervals.csv", intervals)
    oracle = [r for study in all_studies for r in oracle_diagnostic(study, settings)]
    write_csv(OUTPUT / "oracle_threshold_diagnostic.csv", oracle)
    primary = [r for r in records if r["scenario"] == "dl2020_source" and r["fraction"] == .5 and r["calibration_labels"] in (0, 200)]
    run_scores = []
    for r in primary:
        score = read_gzip(allocation_path(r))["system_scores"]
        reference_rank = rankdata(-np.asarray(score["gold_graded_scores"]), method="average")
        rank = rankdata(-np.asarray(score["graded_scores"]), method="average")
        for i, system in enumerate(score["systems"]):
            run_scores.append(dict(method=r["method"], seed=r["seed"], system=system,
                binary_gold_ndcg10=score["gold_binary_scores"][i], binary_mixed_ndcg10=score["binary_scores"][i],
                graded_gold_ndcg10=score["gold_graded_scores"][i], graded_mixed_ndcg10=score["graded_scores"][i],
                graded_gold_rank=reference_rank[i], graded_mixed_rank=rank[i], graded_rank_change=rank[i]-reference_rank[i]))
    write_csv(OUTPUT / "official_run_scores.csv", run_scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, scenario in zip(axes, ("dl2020_source", "covid_target")):
        for method in ("threshold_worst_class", "raw_confidence", "lara_1_upstream_smooth", "lara_reserved_raw"):
            c = 200 if method == "threshold_worst_class" else 0
            rows = sorted([r for r in summaries if r["scenario"] == scenario and r["method"] == method and r["calibration_labels"] == c], key=lambda r:r["fraction"])
            axis.plot([r["fraction"]*100 for r in rows], [100*r["worst_class_risk_mean"] if r["worst_class_risk_mean"] is not None else np.nan for r in rows], marker="o", label=method)
        axis.set(title=scenario, xlabel="Total human budget (% of evaluation pool)", ylabel="Worst accepted-class error (%)")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
    fig.savefig(OUTPUT / "worst_class_budget.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, scenario in zip(axes, ("dl2020_source", "covid_target")):
        study = next(s for s in all_studies if s["id"] == scenario)
        p = np.asarray([r["prob_relevant"] for r in study["test"]])
        y = np.asarray([r["true_binary"] for r in study["test"]])
        raw = p >= .5
        for c, ix in enumerate(confidence_order(p, 6000)):
            count = np.arange(1, len(ix)+1)
            error = np.cumsum(raw[ix] != y[ix])/count
            keep = count >= max(1, int(.01*len(ix)))
            axis.plot(100*count[keep]/len(ix), 100*error[keep], label=f"Predicted {'positive' if c else 'negative'}")
        axis.axvline(5, color="gray", linestyle="--", linewidth=1, label="5% class-coverage floor")
        axis.set(title=scenario, xlabel="Automatic coverage within predicted class (%)", ylabel="Accepted-label error (%)", ylim=(0, 70))
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
    fig.savefig(OUTPUT / "class_tail_risk.png", dpi=180)
    plt.close(fig)
    make_report(summaries, intervals, oracle, metadata, settings)
    for name in ("design.json", "ranking_provenance.json"):
        atomic_text(OUTPUT / name, (INPUT / name).read_text(encoding="utf-8"))
    write_json(OUTPUT / "validation.json", {**validation, "analysis_sha256": file_hash(Path(__file__))})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "checksums.sha256"))
    print(json.dumps(validation), flush=True)
    for r in summaries:
        if r["fraction"] == .5 and r["calibration_labels"] in (0, 200) and r["method"] in ("threshold_worst_class", "raw_confidence", "lara_1_raw", "lara_1_upstream_smooth", "lara_reserved_raw"):
            print(r["scenario"], r["method"], "worst", r["worst_class_risk_mean"], "positive coverage", r["class_1_coverage_mean"], "binary tau", r["system_ranking_kendall_tau_mean"], flush=True)
    print(OUTPUT / "README.md", flush=True)


if __name__ == "__main__":
    main()
