"""Summarize completed proposal metrics, hypothesis evidence and remaining inputs."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import atomic_text, file_hash, write_json

OUTPUT = ROOT / "reports/proposal_metrics_v1"


def read(name):
    with (OUTPUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def value(row, field, digits=3, percent=False):
    item = row.get(field)
    if item in (None, ""):
        return "undefined"
    return f"{float(item)*(100 if percent else 1):.{digits}f}" + ("%" if percent else "")


def main():
    metadata = json.loads((OUTPUT / "experiment.json").read_text())
    audit_dir = ROOT / "results/experiments/qwen_proposal_audit_v1"
    audit_meta = json.loads((audit_dir / "experiment.json").read_text())
    if metadata["status"] != "complete" or audit_meta["status"] != "complete":
        raise ValueError("Complete metric and audit studies first")
    for line in (audit_dir / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if file_hash(audit_dir / name) != digest:
            raise ValueError("Audit output changed")
    for name in ("runs.csv", "summary.csv", "experiment.json"):
        atomic_text(OUTPUT / ("audit_" + name), (audit_dir / name).read_text(encoding="utf-8"))
    raw = read("raw_assessor_metrics.csv")
    policy = read("policy_summary.csv")
    with (audit_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        audits = list(csv.DictReader(stream))
    methods = (("selected_local", "Our policy", "#D55E00"), ("lara_1_raw", "Global raw LARA", "#0072B2"),
               ("raw_confidence", "Raw confidence", "#009E73"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method, label, color in methods:
        rows = sorted((r for r in policy if r["dataset"] == "dl2020_all" and r["method"] == method), key=lambda r: float(r["coverage_mean"]))
        x = [100*float(r["coverage_mean"]) for r in rows]
        axes[0].plot(x, [100*float(r["query_mean_auto_error_rate_mean"]) for r in rows], "o-", color=color, label=label)
        axes[1].plot(x, [float(r["system_ranking_kendall_tau_mean"]) for r in rows], "o-", color=color, label=label)
    axes[0].axhline(5, color="gray", linestyle=":", label="5% error target")
    axes[1].axhline(.9, color="gray", linestyle=":", label="0.90 ranking target")
    axes[0].set(xlabel="Automatic coverage (%)", ylabel="Query-average automatic error (%)", title="DL2020 label quality", ylim=(0, None))
    axes[1].set(xlabel="Automatic coverage (%)", ylabel="Kendall tau-b", title="59 submitted passage runs", ylim=(.75, 1.005))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(OUTPUT / "dl2020_label_and_ranking.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for axis, cohort, title in zip(axes, ("dl2020_all", "trec_covid_locked", "chatreport_all"),
                                   ("DL2020", "COVID reserved 40", "ChatReport")):
        for method, label, color in methods:
            rows = sorted((r for r in policy if r["dataset"] == cohort and r["method"] == method), key=lambda r: float(r["coverage_mean"]))
            axis.plot([100*float(r["coverage_mean"]) for r in rows], [100*float(r["class_1_coverage_mean"]) for r in rows], "o-", color=color, label=label)
        axis.axhline(5, color="gray", linestyle=":")
        axis.set(title=title, xlabel="Overall automatic coverage (%)", ylim=(0, 102), xlim=(0, 100))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Predicted-positive automatic coverage (%)")
    axes[0].legend(fontsize=8)
    fig.savefig(OUTPUT / "predicted_positive_coverage.png", dpi=180)
    plt.close(fig)
    lines = ["# Project evidence and proposal evaluation", "", "15 November 2026.", "",
        "The implemented review policy, matched-budget baselines, DL2020 improvement, component experiments and documented transfer failures form a substantive empirical course project. The final claim should describe those measured results. The proposal's full formal-reliability hypothesis remains unestablished.", "",
        "A supported empirical claim is that the retained shared/local review policy reduces automatic-label error on DL2020 at the tested budgets while preserving binary retrieval-system ordering. The local-update ablation supports its mechanism. COVID and ChatReport establish limits to the advantage. This claim is narrower than universal superiority across domains or an economical formal certificate.", "",
        "## Raw assessor metrics", "",
        "| Cohort | Pairs | Macro-F1 | Binary quadratic weighted kappa | Brier | ECE | Tie-averaged confidence AURC |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in raw:
        lines.append(f"| {r['dataset']} | {int(r['pairs']):,} | {value(r, 'macro_f1', 4)} | {value(r, 'quadratic_weighted_kappa_binary', 4)} | {value(r, 'brier_score', 4)} | {value(r, 'ece_probability_bins_10', 4)} | {value(r, 'confidence_aurc_tie_averaged', 4)} |")
    lines += ["", "These are binary relevance metrics. Binary quadratic weighted kappa coincides with ordinary binary kappa. Multi-grade agreement requires graded predictions. Raw confidence AURC averages prefix selective risk and analytically averages over uniform ordering within equal-confidence groups. Lower AURC is better.", "",
              "`predicted_class_calibration.csv` and `predicted_class_reliability.csv` report confidence correctness calibration within each predicted class. Policy-level mixed-label macro-F1 and kappa include purchased human judgments. Their Brier/ECE counterparts are not synthesized from human corrections. `policy_partial_aurc.csv` separately reports trapezoidal area over measured adaptive budget points at approximately 10–100% coverage. This sampled partial area differs from the raw scorer's exact discrete AURC.", "",
              "## Label quality and system fidelity at 50% automatic coverage", "",
              "| Cohort | Method | Query-average automatic error | Mixed-label macro-F1 | Mixed-label kappa | Kendall tau-b | Spearman rho | Binary nDCG@10 MAE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cohort in ("dl2020_all", "trec_covid_locked", "chatreport_all"):
        for method, label, _ in methods:
            r = next(r for r in policy if r["dataset"] == cohort and r["method"] == method and abs(float(r["coverage_mean"])-.5) < .001)
            lines.append(f"| {cohort} | {label} | {value(r, 'query_mean_auto_error_rate_mean', 3, True)} | {value(r, 'mixed_macro_f1_mean', 4)} | {value(r, 'mixed_quadratic_weighted_kappa_binary_mean', 4)} | {value(r, 'system_ranking_kendall_tau_mean', 4)} | {value(r, 'system_ranking_spearman_rho_mean', 4)} | {value(r, 'system_binary_ndcg10_mae_mean', 5)} |")
    ranking = metadata["ranking_provenance"]
    dl_half = {r["method"]: r for r in policy if r["dataset"] == "dl2020_all" and abs(float(r["coverage_mean"])-.5) < .001}
    raw_full = next(r for r in policy if r["dataset"] == "dl2020_all" and r["method"] == "raw_confidence" and float(r["coverage_mean"]) == 1.)
    lines += ["", f"At 50% coverage, our mean absolute binary nDCG@10 score error is {value(dl_half['selected_local'], 'system_binary_ndcg10_mae_mean', 5)}, compared with {value(dl_half['lara_1_raw'], 'system_binary_ndcg10_mae_mean', 5)} for global raw LARA. LARA is more accurate on this aggregate score-estimation metric. Full automation already attains tau={value(raw_full, 'system_ranking_kendall_tau_mean', 4)} under this binary evaluation, above the original 0.90 target. Ranking correlation should therefore be reported as a continuous result alongside score error and label quality.", "",
              "Ranking evaluation uses all 59 submitted DL2020 passage systems in the released workbench data, retaining original top-ten positions for all 54 queries. The official track overview reports 59 passage runs in Table 3. Both mixed and reference qrels use binary relevance grade>=2. This is a binary adaptation of the original graded benchmark.", "",
              f"The release contains {ranking['unjudged_top10_positions']:,} unjudged positions out of {ranking['ranked_positions']:,}, corresponding to {ranking['unique_unjudged_top10_pairs']:,} unique pairs. These remain unjudged and contribute zero gain under both qrels sets. Missingness is recorded per run. The supplementary `high_judgment_system_kendall_tau_mean` field restricts the calculation to {ranking['systems_at_least_95percent_judged_top10']} systems with at least 95% judged top-ten positions.", "",
              "The evaluation measures agreement under the available judgments. It does not claim relevance knowledge for unjudged documents. Source and method references are the [NIST track overview](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.DL.pdf), [workbench release](https://www.cs.unh.edu/~dietz/autograder/data-dl20.tar.xz), and [trec_eval nDCG implementation](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c).", "",
              "![DL2020 label quality and ranking fidelity](dl2020_label_and_ranking.png)", "",
              "## Class coverage and numerical hypothesis checks", "",
              "The proposal's H1 requires automatic error <=5%, selective risk <=20% within both predicted classes, coverage >=50%, system-ranking tau>=0.90, and a favorable same-cost comparison with LARA. It specifies 20 topic splits and calibrated thresholds. The current results use adaptive paid review and ten allocation seeds on fixed query cohorts. Numerical satisfaction on those cohorts is recorded separately from a statistical guarantee.", "",
              "| Cohort at 50% overall coverage | Method | Predicted-negative automatic coverage | Predicted-positive automatic coverage | Both class risks defined, runs | Both class risks <=20%, runs |",
              "|---|---|---:|---:|---:|---:|"]
    for cohort in ("dl2020_all", "trec_covid_locked", "chatreport_all"):
        for method, label, _ in methods:
            r = next(r for r in policy if r["dataset"] == cohort and r["method"] == method and abs(float(r["coverage_mean"])-.5) < .001)
            lines.append(f"| {cohort} | {label} | {value(r, 'class_0_coverage_mean', 3, True)} | {value(r, 'class_1_coverage_mean', 3, True)} | {r['both_classes_accepted_count']}/{r['seeds']} | {r['empirical_both_class_risk_target_count']}/{r['seeds']} |")
    lines += ["", "The small predicted-positive coverage is material to interpreting the web gain. Our policy assigns almost all predicted-positive labels to humans at 50% overall coverage. Global raw LARA assigns all of its remaining predicted-positive cases to humans. An empty accepted class has undefined selective risk. It is not counted as passing the two-class target.", "",
              "The later mathematical note also asks for at least 5% automatic coverage within each predicted class. That numerical floor is an additional operational criterion, rather than an explicit number in the original H1. `policy_metrics.csv` and `policy_summary.csv` retain both the proposal's numerical checks and the additional floor.", "",
              "![Coverage of predicted-positive cases](predicted_positive_coverage.png)", "",
              "## Existing quality controller on new Qwen outputs", "",
              "The existing progressive controller was evaluated on three cohorts, three allocation methods and ten seeds, giving 90 runs. It requests 5% query-average automatic error, 20% per-predicted-class selective risk, the later 5% class coverage floor, and 95% fixed-pool audit confidence across four looks. Fresh audit labels and work labels are charged once. This controller addresses a different statistical target from the proposal's expected future-query CRC.", "",
              "| Cohort | Allocation | Certified releases | Mean final automatic coverage | Mean total human labels |",
              "|---|---|---:|---:|---:|"]
    for r in audits:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['releases']}/{r['runs']} | {100*float(r['mean_coverage']):.2f}% | {float(r['mean_human_labels']):,.0f} |")
    lines += ["", "All runs fell back to full human review. The absence of observed violations comes from that fallback. This records the cost limitation of the implemented controller and does not establish economical certified automation or refute a separate CRC theorem.", "",
              "## Proposal status and remaining work", "",
              "| Proposal item | Current evidence | Status |",
              "|---|---|---|",
              "| Shared/local review versus LARA at matched cost | DL2019 discovery, DL2020 transfer, COVID and climate counterexamples | Evaluated with a limited empirical claim |",
              "| Macro-F1, weighted kappa, Brier, ECE, class calibration, risk–coverage, annotation cost | Cached Qwen metrics and adaptive policy tables | Completed for current binary targets |",
              "| Ranking Kendall/Spearman and nDCG@10 error | All 59 released DL2020 passage runs | Completed with binary gains and stated unjudged convention |",
              "| Query-specific mechanism | Component removal and earlier shuffled-query controls | Evidence available |",
              "| Target-domain calibration and tuning | Development-only selection, matched-cost tests, small-split instability | Evaluated, mixed/negative results retained |",
              "| Practical certificate | Existing controller tested in 90 new Qwen runs | Full-human fallback in all runs |",
              "| COVID positive label matches proposal | Current scores use grade>=2, proposal requires grade>=1 | Corrected inference notebook prepared, GPU run pending |",
              "| Official COVID system-ranking fidelity | Current 5,000-pair sample does not cover the full judged run pools | Requires suitable run inputs and matching inference pool |",
              "| 20 topic splits and untouched ten-topic COVID audit | Existing 10/40 split and inspected outcomes | Original protocol has not been executed |",
              "| Prompt paraphrases, wording, batching and self-consistency | Single saved prompt/score per pair for new cohorts | Additional inference needed |",
              "| Graded predictions and QLoRA-adapted 3B model | Current assessor is frozen binary Qwen3.5-4B | Explicit scope deviation |",
              "| Assessor/retriever family bias and student-reviewed failures | Earlier studies and review artifacts exist | Additional reliable family metadata and student review needed |", "",
              "The original H1 has not been established as written. The proposal explicitly permits a backoff study of calibration under shift with clear negative results. The current evidence supports that empirical direction together with a web-domain allocation improvement. A fresh held-out dataset or new topics would be needed for an untouched final confirmation after the exploratory analysis already performed.", "",
              "The next GPU artifact is [the proposal-aligned COVID notebook](../../notebooks/qwen_covid_proposal_colab.ipynb), with [the companion upload ZIP](../../dist/qwen_covid_proposal_colab.zip). It changes the prompt and target so partially relevant documents count positive, using the same 5,000 sampled pairs. Its syntax, manifest and bundled source hashes were validated locally. Actual inference still needs to run on a CUDA GPU.", "",
              "The final write-up should preserve the original proposal and explain how the implemented scope evolved. Report H1 as unestablished and the backoff evidence as measured results. The algorithm's combination remains a candidate contribution, with the established ingredients and Oosterhuis overlap discussed in the existing literature review.", "",
              "## Reproduction", "", "```powershell", "python scripts/evaluate_proposal_metrics.py",
              "python scripts/audit_qwen_proposal.py", "python scripts/report_proposal_progress.py",
              "python scripts/build_proposal_covid_colab.py", "```"]
    atomic_text(OUTPUT / "README.md", "\n".join(lines)+"\n")
    write_json(OUTPUT / "report_validation.json", {"metric_rows": metadata["policy_rows"], "audit_runs": audit_meta["runs"],
        "ranking_systems": 59, "ranking_queries": 54, "unjudged_top10_positions": ranking["unjudged_top10_positions"],
        "report_source_sha256": file_hash(Path(__file__)),
        "remaining_gpu_run": "qwen35_covid_proposal_v1, grades 1 and 2 positive"})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(OUTPUT.iterdir())
                if p.is_file() and p.name != "checksums.sha256"))
    print(OUTPUT / "README.md", flush=True)


if __name__ == "__main__":
    main()
