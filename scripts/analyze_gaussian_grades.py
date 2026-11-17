"""Inspect grade distributions and summarize the Gaussian relevance experiment."""

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
from scipy.stats import norm, probplot
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_gaussian_grade_experiment import OUTPUT as INPUT, CONFIG, scenarios, features, predictive_metrics
from scripts.run_error_predictor_ablation import complete_budget
from src.calibration.gaussian_grades import GradeDensityModel, relevance_probability
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

OUTPUT = ROOT / "reports/gaussian_grade_v1"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def grouped_summary(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, records in sorted(groups.items()):
        item = dict(zip(keys, key))
        item["repetitions"] = len(records)
        for name in fields:
            values = [float(r[name]) for r in records if r.get(name) not in (None, "")]
            item[name+"_mean"] = float(np.mean(values)) if values else None
            item[name+"_std"] = float(np.std(values)) if values else None
        result.append(item)
    return result


def restore_gaussian(parameters):
    model = GradeDensityModel(parameters["kind"], parameters["grades"])
    model.center = np.asarray(parameters["feature_center"])
    model.scale = np.asarray(parameters["feature_scale"])
    model.priors = np.asarray(parameters["class_priors"])
    model.means = np.asarray(parameters["standardized_class_means"])
    model.covariances = np.asarray(parameters["standardized_covariances"])
    model.mixtures = [None]*len(model.grades)
    return model


def verify_study(studies, settings):
    checked_fits = checked_masks = checked_graded = 0
    for study in studies:
        test = study["test"]
        cal_map = {pair_key(r): r for r in study["calibration"]}
        evaluator = PolicyEvaluator(test)
        for seed in settings["seeds"]:
            path = INPUT / "fits" / study["id"] / f"seed_{seed}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            cal_keys = [pair_key(r) for r in saved["calibration_pairs"]]
            if len(cal_keys) != len(set(cal_keys)) or not set(cal_keys) <= set(cal_map):
                raise ValueError("Invalid paid calibration sample")
            if set(cal_keys) & {pair_key(r) for r in test}:
                raise ValueError("Calibration/evaluation pair overlap")
            for row in saved["calibration_pairs"]:
                if row["grade"] != cal_map[pair_key(row)]["true_relevance"]:
                    raise ValueError("Calibration grade changed")
            params = saved["model_parameters"]["grade_qda_margin"]
            expected_counts = [sum(r["grade"] == grade for r in saved["calibration_pairs"]) for grade in study["grades"]]
            if params["calibration_counts"] != expected_counts:
                raise ValueError("Gaussian class counts differ from paid labels")
            primary = restore_gaussian(params)
            posterior = primary.predict_proba(features(test, "margin"))
            p = relevance_probability(posterior, study["grades"], study["threshold"])
            metrics = predictive_metrics(study, "grade_qda_margin", p, posterior)
            stored = next(r for r in saved["prediction_metrics"] if r["model"] == "grade_qda_margin")
            for name in ("brier_score", "raw_error_prediction_brier", "grade_nll", "grade_brier", "grade_quadratic_kappa"):
                if abs(metrics[name]-stored[name]) > 1e-10:
                    raise ValueError("Stored Gaussian posterior/metric changed")
            for r in saved["policy_metrics"]:
                if r["total_human_labels"] != r["calibration_labels"]+r["human_labels"]:
                    raise ValueError("Review cost arithmetic failed")
                if r["accounting"] == "matched_total" and r["total_human_labels"] != int(float(r["nominal_fraction"])*len(test)):
                    raise ValueError("Total-cost matching failed")
                if r["method"].endswith("risk_only"):
                    key = f"{r['accounting']}_{r['nominal_fraction']}_{r['method']}"
                    human = np.unpackbits(np.asarray(saved["packed_review_masks"][key], dtype=np.uint8))[:len(test)].astype(bool)
                    actual, _ = evaluator.evaluate(human)
                    if actual["human_labels"] != r["human_labels"] or actual["automatic_errors"] != r["automatic_errors"]:
                        raise ValueError("Review mask does not reproduce errors/cost")
                    if abs(actual["query_mean_auto_error_rate"]-r["query_mean_auto_error_rate"]) > 1e-12:
                        raise ValueError("Query-average review error mismatch")
                    checked_masks += 1
            grade_gold = np.asarray([r["true_relevance"] for r in test])
            grade_weights = np.asarray([1/len(evaluator.groups[r["query_id"]]) for r in test])
            grade_pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in test]
            verified_posteriors = {}
            for r in saved["graded_policy_metrics"]:
                name = r["model"]
                if name not in ("grade_qda_margin", "grade_qda_full", "raw_assessor"):
                    continue
                if name not in verified_posteriors:
                    if name == "raw_assessor":
                        verified_posteriors[name] = np.asarray([x["grade_probabilities"] for x in test])
                    else:
                        restored = restore_gaussian(saved["model_parameters"][name])
                        view = "margin" if name.endswith("margin") else "full"
                        verified_posteriors[name] = restored.predict_proba(features(test, view))
                posterior = verified_posteriors[name]
                predicted = np.asarray(study["grades"])[posterior.argmax(axis=1)]
                human = complete_budget(grade_pool, [], (1-posterior.max(axis=1))*grade_weights,
                                        r["human_labels"], seed)
                errors = (~human) & (predicted != grade_gold)
                query_error = float(np.mean([errors[ix].mean() for ix in evaluator.groups.values()]))
                mixed = np.where(human, grade_gold, predicted)
                kappa = cohen_kappa_score(grade_gold, mixed, labels=study["grades"], weights="quadratic")
                if int(errors.sum()) != r["automatic_errors"] or abs(query_error-r["query_auto_error"]) > 1e-12:
                    raise ValueError("Reconstructed graded-review errors differ")
                if abs(kappa-r["mixed_grade_quadratic_kappa"]) > 1e-12 or int(human.sum()) != r["human_labels"]:
                    raise ValueError("Graded-review kappa or purchased labels differ")
                expected_cost = 0 if name == "raw_assessor" else len(cal_keys)
                if r["calibration_labels"] != expected_cost or r["total_human_labels"] != expected_cost+int(human.sum()):
                    raise ValueError("Graded-review calibration cost differs")
                checked_graded += 1
            checked_fits += 1
    return {"calibration_fits": checked_fits, "risk_only_review_masks_recomputed": checked_masks,
            "primary_gaussian_posteriors_recomputed": checked_fits,
            "primary_and_full_qda_graded_reviews_recomputed": checked_graded,
            "paid_calibration_labels_and_costs": "passed", "query_disjointness": "passed"}


def distribution_plots(studies, settings):
    selected = [next(s for s in studies if s["id"] == name) for name in
                ("qwen_dl2020", "qwen_covid", "qwen_chat_20261114")]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), layout="constrained")
    qqfig, qqaxes = plt.subplots(3, 4, figsize=(14, 9), layout="constrained")
    descriptions = []
    for row_index, study in enumerate(selected):
        path = INPUT / "fits" / study["id"] / f"seed_{settings['seeds'][0]}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            fitted = json.load(stream)
        parameters = fitted["model_parameters"]["grade_qda_margin"]
        center, scale = parameters["feature_center"][0], parameters["feature_scale"][0]
        for column in range(4):
            axis, qqaxis = axes[row_index, column], qqaxes[row_index, column]
            if column >= len(study["grades"]):
                axis.set_visible(False)
                qqaxis.set_visible(False)
                continue
            grade = study["grades"][column]
            group = [r for r in study["test"] if r["true_relevance"] == grade]
            x = np.asarray([r["margin"] for r in group])
            error = np.asarray([(r["prob_relevant"] >= .5) != r["true_binary"] for r in group])
            mu = parameters["standardized_class_means"][column][0]*scale+center
            sigma = float(np.sqrt(parameters["standardized_covariances"][column][0][0])*scale)
            bins = np.linspace(min(x.min(), mu-3*sigma), max(x.max(), mu+3*sigma), 33)
            axis.hist([x[~error], x[error]], bins=bins, stacked=True, density=True,
                      color=["#0072B2", "#D55E00"], alpha=.65, label=["Correct binary judgment", "Wrong binary judgment"])
            grid = np.linspace(bins[0], bins[-1], 300)
            axis.plot(grid, norm.pdf(grid, mu, sigma), color="black", linestyle="--", label="Calibration-fitted Gaussian")
            axis.axvline(0, color="gray", linewidth=.7)
            axis.set_title(f"{study['id'].replace('qwen_', '')}, grade {grade}\nn={len(x)}, raw error={100*error.mean():.1f}%", fontsize=9)
            axis.set_xlabel("Qwen Yes-minus-No logit")
            if column == 0:
                axis.set_ylabel("Conditional density")
            theoretical, ordered = probplot((x-mu)/sigma, dist="norm", fit=False)
            qqaxis.scatter(theoretical, ordered, s=3, alpha=.35, color="#0072B2")
            lo = min(np.min(theoretical), np.min(ordered))
            hi = max(np.max(theoretical), np.max(ordered))
            qqaxis.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            qqaxis.set(title=f"{study['id'].replace('qwen_', '')}, grade {grade}",
                       xlabel="Standard normal quantile", ylabel="Calibration-standardized score")
            descriptions.append({"scenario": study["id"], "grade": grade, "calibration_pairs_in_grade": parameters["calibration_counts"][column],
                                 "fitted_gaussian_mean": mu, "fitted_gaussian_std": sigma,
                                 "test_pairs": len(x), "test_mean": float(x.mean()), "test_std": float(x.std()),
                                 "test_binary_error_rate": float(error.mean())})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    fig.savefig(OUTPUT / "qwen_grade_distributions.png", dpi=170)
    qqfig.savefig(OUTPUT / "qwen_gaussian_qq.png", dpi=170)
    plt.close(fig)
    plt.close(qqfig)
    write_csv(OUTPUT / "gaussian_fit_by_grade.csv", descriptions)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads(CONFIG.read_text())
    metadata = json.loads((INPUT / "experiment.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Wait for all Gaussian experiments to finish")
    design = json.loads((INPUT / "design.json").read_text())
    for name, digest in design["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Changed Gaussian experiment source {name}")
    for line in (INPUT / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if file_hash(INPUT / name) != expected:
            raise ValueError(f"Changed Gaussian artifact {name}")
    studies = scenarios(settings)
    validation = verify_study(studies, settings)
    point_rows = read_csv(INPUT / "prediction_metrics.csv")
    policy_rows = read_csv(INPUT / "policy_metrics.csv")
    graded_rows = read_csv(INPUT / "graded_policy_metrics.csv")
    for rows in (point_rows, policy_rows, graded_rows):
        for row in rows:
            row["group"] = "qwen_chatreport" if row["scenario"].startswith("qwen_chat_") else row["scenario"]
    point_fields = ("macro_f1", "brier_score", "ece_probability_bins_10", "raw_error_auc", "raw_error_average_precision",
                    "raw_error_prediction_brier", "grade_macro_f1", "grade_quadratic_kappa", "grade_nll", "grade_brier")
    point_summary = grouped_summary(point_rows, ("group", "model"), point_fields)
    policy_summary = grouped_summary(policy_rows, ("group", "accounting", "nominal_fraction", "method"),
                     ("coverage", "query_mean_auto_error_rate", "auto_error_rate", "selective_risk", "total_human_labels",
                      "class_0_coverage", "class_1_coverage"))
    graded_summary = grouped_summary(graded_rows, ("group", "accounting", "nominal_fraction", "model"),
                     ("coverage", "query_auto_error", "selective_risk", "mixed_grade_macro_f1", "mixed_grade_quadratic_kappa", "total_human_labels"))
    write_csv(OUTPUT / "prediction_summary.csv", point_summary)
    write_csv(OUTPUT / "binary_review_summary.csv", policy_summary)
    write_csv(OUTPUT / "graded_review_summary.csv", graded_summary)
    for name in ("grade_distribution.csv", "design.json"):
        atomic_text(OUTPUT / name, (INPUT / name).read_text(encoding="utf-8"))
    distribution_plots(studies, settings)
    lines = ["# Grade-conditional Gaussian relevance models", "", "15 November 2026.", "",
        "The experiment fits a statistical distribution of judge-score features for each original human relevance grade. It tests grade prediction, probability calibration, prediction of the original judge's binary errors, and human-review selection. The primary predeclared model is regularized grade-QDA on the one-dimensional score margin. Other models are retained as comparisons.", "",
        "## Findings", "",
        "Grade-specific score distributions overlap substantially. On Qwen DL2020, the held-out mean score rises from -2.37 for grade 0 to -0.56, 0.57 and 0.90 for grades 1, 2 and 3, while their standard deviations are 1.48, 1.78, 1.71 and 2.29. The binary score contains limited information for separating grades 2 and 3. These held-out moments describe the data; they were not used for fitting.", "",
        "For binary Qwen review at half total label cost, the primary grade-QDA model does not beat LARA or the existing local policy on DL2020, COVID or ChatReport. Shared-covariance LDA is stronger on DL2020: its error-prediction AUC rises from raw 0.769 to 0.850, but its review error of 1.669% remains above LARA's 1.086% and the local policy's 0.539%. Retaining grades helps within LDA: the corresponding binary-LDA control has 1.926% review error.", "",
        "For genuine graded review on the released Flan outputs, the primary QDA reduces query-average remaining grade errors at half total cost from 6.949% to 3.569% on TREC-DL and from 14.404% to 3.627% on Robust04. Full-vector QDA gives 4.853% and 2.901%, respectively. These comparisons use raw confidence-based graded review as the reference, not LARA or Oosterhuis's full method. Multiclass logistic regression is competitive, so these gains do not isolate Gaussian density modeling as the cause.", "",
        "Unreviewed grade prediction also has tradeoffs: full-vector QDA increases Flan TREC-DL weighted kappa from 0.525 to 0.550 while reducing macro-F1 from 0.417 to 0.377. On Robust04 its unreviewed kappa falls from 0.244 to 0.150. Improved selective review therefore does not imply uniformly improved grade prediction.", "",
        "The reported query-average error divides remaining automatic mistakes by every evaluation pair in each query, including reviewed pairs. It is distinct from selective risk, which divides errors by automatically labeled pairs only; both appear in the CSVs. These are exploratory results on previously inspected datasets, with no significance, novelty or risk-guarantee claim.", "",
        "## Model", "",
        r"For each grade $g$, Gaussian models use $x\mid Y=g\sim\mathcal{N}(\mu_g,\Sigma_g)$ and $P(Y=g\mid x)=\pi_g\mathcal{N}(x;\mu_g,\Sigma_g)/\sum_h\pi_h\mathcal{N}(x;\mu_h,\Sigma_h)$.", "",
        "LDA shares a covariance matrix across grades. QDA estimates a separate covariance per grade, shrunk 25% toward the pooled within-grade covariance. A small variance ridge stabilizes nearly constant features. Priors use a 0.5 pseudocount per known grade. A grade absent from the paid sample uses the pooled Gaussian and its smoothed prior, with the absence recorded. The mixture alternative fits two Gaussians within a grade when at least ten calibration examples are available, and one component for smaller observed grades.", "",
        r"For the binary project target, $p_{\mathrm{rel}}(x)=\sum_{g\ge t}P(Y=g\mid x)$. The estimated error of a fixed original label $a$ is $1-p_{\mathrm{rel}}$ for $a=1$ and $p_{\mathrm{rel}}$ for $a=0$. Risk-only review keeps that original label. The relabeling variant uses $\mathbf{1}[p_{\mathrm{rel}}\ge 0.5]$ and reviews according to its estimated error. Query-balanced priority divides estimated error by the query's pool size.", "",
        "Direct binary-Gaussian controls collapse the training grades before fitting. They test whether retaining separate grade distributions adds value. Multiclass logistic regression uses the same features and paid grade labels. Platt scaling supplies a binary calibration baseline. Standard LDA/QDA formulas are documented in [scikit-learn's discriminant-analysis guide](https://scikit-learn.org/stable/modules/lda_qda.html). These are established statistical models.", "",
        "## Data, features and label use", "",
        "Qwen supplies binary No/Yes logits and graded human labels. The scalar feature is the Yes-minus-No logit. The additional-feature variant also uses the logit of total No/Yes vocabulary probability mass and log(1+input tokens). Its graded predictions are learned from those features and paid grades. The cached Qwen model itself produced binary judgments.", "",
        "Released Flan-UL2 outputs provide genuine probabilities for every grade on TREC-DL and Robust04. Their multivariate features are additive log-ratios log(P(grade g)/P(grade 0)). This avoids fitting a full Gaussian directly to probability coordinates constrained to sum to one.", "",
        "DL2020 uses 200 paid samples from ten DL2019 queries. COVID uses 200 samples from its existing ten development queries and evaluates the reserved 40 queries. COVID retains the existing strict grade>=2 binary target. ChatReport uses three fixed 3-development/8-evaluation query splits, with 60 paid labels per split. Each Flan study samples 200 labels from ten queries and evaluates the remaining queries in the released top-100 pool. Every within-query sample is uniform and label-independent. Calibration seeds are 6000–6009. Feature scaling and density fitting use only paid calibration rows.", "",
        "Each purchased judgment supplies its original grade. Binary baselines use the corresponding binary label. The extra annotation time for choosing a grade is unmodeled. Query sets are disjoint within each fit, while all benchmark outcomes have appeared in earlier exploratory work. Repeated ChatReport splits overlap. No new LLM inference was performed.", "",
        "## Distribution inspection", "",
        "The histograms show held-out score distributions within each human grade, split by whether the original Qwen binary judgment is correct. Dashed curves use Gaussians fitted to the first paid calibration sample. The Q–Q plots compare held-out margins, standardized with those fitted parameters, to standard-normal quantiles. Departures can reflect both non-Gaussian shape and calibration/evaluation shift. No independent-observation normality p-values are claimed for query-clustered pairs.", "",
        "![Qwen grade distributions](qwen_grade_distributions.png)", "",
        "![Gaussian Q-Q diagnostics](qwen_gaussian_qq.png)", "",
        "`grade_distribution.csv` records grade counts, raw binary error rates, means, standard deviations, skewness and excess kurtosis. `gaussian_fit_by_grade.csv` compares fitted and held-out marginal moments. Fit JSON files retain class counts, priors, means and covariance matrices.", "",
        "## Predictive results", "",
        "The table below shows raw scores, the primary QDA model, shared-covariance LDA, the additional-feature QDA, multiclass logistic regression and binary Platt calibration. Grade metrics for a binary Qwen raw output are undefined. Multiclass Brier score sums squared probability errors over grades. Grade NLL clips probabilities at 1e-12.", "",
        "| Data | Model | Binary Brier | Binary error-prediction AUC | Grade macro-F1 | Grade weighted kappa | Grade NLL |",
        "|---|---|---:|---:|---:|---:|---:|"]
    show_models = ("raw_assessor", "grade_qda_margin", "grade_lda_margin", "grade_qda_full", "grade_logistic_full", "platt_binary")
    for group in ("qwen_dl2020", "qwen_covid", "qwen_chatreport", "flan_trecdl", "flan_robust04"):
        for model in show_models:
            r = next(r for r in point_summary if r["group"] == group and r["model"] == model)
            names = ("brier_score", "raw_error_auc", "grade_macro_f1", "grade_quadratic_kappa", "grade_nll")
            cells = " | ".join("undefined" if r[name+"_mean"] is None else f"{r[name+'_mean']:.4f}" for name in names)
            lines.append(f"| {group} | {model} | {cells} |")
    lines += ["", "## Binary review at equal total label cost", "",
              "All methods below receive total budget equal to half the evaluation pool size. Fitted models first pay for their 200 or 60 calibration labels, then use the remainder for review. The unfitted baselines can spend the entire budget on evaluation-pool review. Automatic coverage therefore differs. Error averages the remaining automatic-error fraction within each query. Lower error is better.", "",
              "| Data | Method | Query-average automatic error | Automatic coverage | Total labels |",
              "|---|---|---:|---:|---:|"]
    show_review = ("grade_qda_margin_risk_only", "grade_qda_margin_relabel", "grade_lda_margin_risk_only",
                   "binary_qda_margin_risk_only", "platt_binary_risk_only", "selected_local", "lara_1_raw", "raw_confidence")
    for group in ("qwen_dl2020", "qwen_covid", "qwen_chatreport"):
        for method in show_review:
            r = next(r for r in policy_summary if r["group"] == group and r["accounting"] == "matched_total" and r["nominal_fraction"] == "0.5" and r["method"] == method)
            lines.append(f"| {group} | {method} | {100*r['query_mean_auto_error_rate_mean']:.3f}% | {100*r['coverage_mean']:.2f}% | {r['total_human_labels_mean']:.0f} |")
    lines += ["", "The CSV includes every model, both review modes and all three budgets. Its matched-review rows hold evaluation-pool review constant and add the calibration cost separately. These rows isolate the score model's allocation effect before accounting for that investment. The study does not select a deployment model from evaluation gold.", "",
              "## Graded selective review", "",
              "Graded review predicts the posterior's most likely grade and orders review by one minus its posterior probability, divided by query size. Errors count any wrong grade. Mixed-grade macro-F1 and weighted kappa replace purchased labels with the true grade. Raw Flan-UL2 is the graded zero-calibration-cost reference. The binary LARA implementation is compared only on the project's binary target.", "",
              "| Graded data | Model | Query-average grade error at half total budget | Mixed-grade weighted kappa | Automatic coverage |",
              "|---|---|---:|---:|---:|"]
    for group in ("flan_trecdl", "flan_robust04"):
        for model in ("raw_assessor", "grade_qda_margin", "grade_qda_full", "grade_lda_full", "grade_gmm2_full", "grade_logistic_full"):
            r = next(r for r in graded_summary if r["group"] == group and r["accounting"] == "matched_total" and r["nominal_fraction"] == "0.5" and r["model"] == model)
            lines.append(f"| {group} | {model} | {100*r['query_auto_error_mean']:.3f}% | {r['mixed_grade_quadratic_kappa_mean']:.4f} | {100*r['coverage_mean']:.2f}% |")
    lines += ["", "## Interpretation and reproducibility", "",
              "A reduction in probability Brier score and a better review order are separate outcomes. A generative model can be well calibrated on average while misordering rare errors. Gaussian assumptions are approximations to judge-score distributions, and sparse grade samples make class variance estimates sensitive. Mixtures and logistic controls help assess whether Gaussian shape or the available score information limits the result.", "",
              "The primary model and all alternatives were declared before these fits. The tables preserve unsuccessful models. Comparing the best observed alternative would be post-hoc model selection requiring another validation experiment. Density modeling itself supplies an estimated error probability, with no finite-sample risk guarantee claimed.", "",
              f"The experiment contains {metadata['fit_repetitions']} paid calibration repetitions, {metadata['predictive_evaluations']:,} predictive evaluations, {metadata['policy_evaluations']:,} binary-review evaluations and {metadata['graded_policy_evaluations']:,} graded-review evaluations. These repetitions reuse fixed datasets.", "",
              "```powershell", "python scripts/run_gaussian_grade_experiment.py", "python scripts/analyze_gaussian_grades.py", "```"]
    atomic_text(OUTPUT / "README.md", "\n".join(lines)+"\n")
    write_json(OUTPUT / "validation.json", {**validation, "analysis_source_sha256": file_hash(Path(__file__)),
        "scope": "Fitted Gaussian parameters and primary predictive metrics recomputed; all binary risk-only review masks checked; primary/full QDA and raw graded review reconstructed, including errors, kappa and calibration costs."})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(OUTPUT.iterdir())
                if p.is_file() and p.name != "checksums.sha256"))
    print(json.dumps(validation), flush=True)
    print(OUTPUT / "README.md", flush=True)


if __name__ == "__main__":
    main()
