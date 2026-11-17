"""Validate target-domain tuning and compare both annotation-cost conventions."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_target_domain import OUTPUT as INPUT, CONFIG, scored_rows, prepare_splits, apply_calibration, run_policy
from src.baselines.lara import LaraConfig
from src.evaluation.metrics import classification_metrics
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json

REPORT = ROOT / "reports/target_domain_tuning_v1"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def heldout_half_job(split, selected, frozen, seed):
    rows = [r for r in scored_rows(split["dataset"]) if r["query_id"] in set(split["evaluation_query_ids"])]
    budget = len(rows)//2
    methods = {"frozen_ours": frozen, "tuned_ours": selected["ours"], "tuned_lara": selected["lara"],
               "frozen_lara_raw": {"family": "lara", "view": "raw", "grouping": 1,
                                   "config": asdict(LaraConfig(preprocessing="raw"))}}
    results = []
    for name, candidate in methods.items():
        destination = REPORT / "heldout_half_allocations" / split["id"] / f"seed_{seed}_{name}.json.gz"
        if destination.exists():
            with gzip.open(destination, "rt", encoding="utf-8") as stream:
                results.append(json.load(stream)["record"])
            continue
        p = apply_calibration(rows, selected["calibration"]) if candidate["view"] == "target_platt" else np.asarray([r["prob_relevant"] for r in rows])
        human, predicted, acquired = run_policy(rows, p, candidate, budget, seed)
        metrics, _ = PolicyEvaluator(rows).evaluate(human, predicted)
        record = {"split": split["id"], "dataset": split["dataset"], "method": name, "seed": seed,
                  "development_labels": split["development_pairs"], "total_labels": split["development_pairs"]+budget,
                  "score_view": candidate["view"], **metrics}
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump({"record": record, "purchases": acquired, "automatic_predicted_labels": np.asarray(predicted, dtype=int).tolist()}, stream)
        temporary.replace(destination)
        results.append(record)
    return results


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((INPUT / "experiment.json").read_text())
    design = json.loads((INPUT / "design.json").read_text())
    settings = json.loads(CONFIG.read_text())
    if metadata["status"] != "complete" or file_hash(INPUT / "design.json") != metadata["design_sha256"]:
        raise ValueError("Incomplete or changed tuning design")
    for name, digest in design["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Changed tuning source {name}")
    for line in (INPUT / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if file_hash(INPUT / name) != expected:
            raise ValueError(f"Changed tuning artifact {name}")
    choices = json.loads((INPUT / "selected_parameters.json").read_text())
    splits = prepare_splits(settings)
    development_evaluations = 0
    for split in splits:
        dev = set(split["development_query_ids"])
        if dev & set(split["evaluation_query_ids"]):
            raise ValueError("Query split leakage")
        for fold in split["folds"]:
            train, val = set(fold["training_query_ids"]), set(fold["validation_query_ids"])
            if train & val or not (train | val) <= dev:
                raise ValueError("Development calibration fold leakage")
        saved = [json.loads(path.read_text()) for path in (INPUT / "development" / split["id"]).glob("*.json")]
        for record in saved:
            if abs(np.mean([r["query_auto_error"] for r in record["fold_results"]])-record["development_cv_loss"]) > 1e-12:
                raise ValueError("Invalid selection loss")
            development_evaluations += len(record["fold_results"])
        for family in ("ours", "lara"):
            winner = min((r for r in saved if r["candidate"]["family"] == family), key=lambda r: (r["development_cv_loss"], r["candidate"]["id"]))
            if winner["candidate"] != choices[split["id"]][family]:
                raise ValueError("Selected parameter differs from development-only winner")
    run_rows = read_csv(INPUT / "runs.csv")
    checked = 0
    for split in splits:
        rows = scored_rows(split["dataset"])
        index = {pair_key(r): i for i, r in enumerate(rows)}
        evaluator = PolicyEvaluator(rows)
        for path in sorted((INPUT / "evaluation" / split["id"]).rglob("*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            seen = set()
            human = np.zeros(len(rows), dtype=bool)
            for purchase in saved["purchases"]:
                key = pair_key(purchase)
                if key in seen or key not in index:
                    raise ValueError("Duplicate or out-of-pool purchase")
                seen.add(key)
                human[index[key]] = True
                if purchase["observed_label"] != rows[index[key]]["true_binary"]:
                    raise ValueError("Purchase label mismatch")
            r = saved["record"]
            if len(seen) != r["budget"] or r["development_labels"] + r["online_labels"] != r["budget"]:
                raise ValueError("Total-cost accounting mismatch")
            metrics, _ = evaluator.evaluate(human, saved["automatic_predicted_labels"])
            for name in ("query_mean_auto_error_rate", "auto_error_rate", "coverage", "automatic_errors", "human_labels"):
                if abs(metrics[name]-r[name]) > 1e-12:
                    raise ValueError("Evaluation metric recomputation failed")
            checked += 1
    if checked != len(run_rows) or checked != metadata["evaluation_rows"]:
        raise ValueError("Evaluation count mismatch")
    calibration_rows = []
    for split in splits:
        test = [r for r in scored_rows(split["dataset"]) if r["query_id"] in set(split["evaluation_query_ids"])]
        for view in ("raw", "target_platt"):
            p = apply_calibration(test, choices[split["id"]]["calibration"]) if view == "target_platt" else [r["prob_relevant"] for r in test]
            evaluated = [{**r, "prob_relevant": float(value), "predicted_label": int(value >= .5)} for r, value in zip(test, p)]
            m = classification_metrics(evaluated)
            m.pop("reliability")
            calibration_rows.append({"split": split["id"], "dataset": split["dataset"], "view": view, **m})
    write_csv(REPORT / "probability_calibration.csv", calibration_rows)
    frozen = {"family": "ours", "view": "raw", "grouping": None, "config": design["selected_policy"]}
    half_rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(heldout_half_job, split, choices[split["id"]], frozen, seed)
                   for split in splits for seed in settings["evaluation_seeds"]]
        for future in as_completed(futures):
            half_rows.extend(future.result())
    half_rows.sort(key=lambda r: (r["split"], r["method"], r["seed"]))
    checked_half = 0
    for split in splits:
        rows = [r for r in scored_rows(split["dataset"]) if r["query_id"] in set(split["evaluation_query_ids"])]
        index = {pair_key(r): i for i, r in enumerate(rows)}
        evaluator = PolicyEvaluator(rows)
        for path in (REPORT / "heldout_half_allocations" / split["id"]).glob("*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                saved = json.load(stream)
            purchases = saved["purchases"]
            if len({pair_key(r) for r in purchases}) != len(rows)//2:
                raise ValueError("Held-out review ledger mismatch")
            human = np.zeros(len(rows), dtype=bool)
            for purchase in purchases:
                i = index[pair_key(purchase)]
                if purchase["observed_label"] != rows[i]["true_binary"]:
                    raise ValueError("Held-out purchase label mismatch")
                human[i] = True
            recalculated, _ = evaluator.evaluate(human, saved["automatic_predicted_labels"])
            record = saved["record"]
            if recalculated["automatic_errors"] != record["automatic_errors"] or abs(recalculated["query_mean_auto_error_rate"]-record["query_mean_auto_error_rate"]) > 1e-12:
                raise ValueError("Held-out metric recomputation mismatch")
            if record["total_labels"] != split["development_pairs"] + len(rows)//2:
                raise ValueError("Held-out development cost mismatch")
            checked_half += 1
    if checked_half != len(half_rows):
        raise ValueError("Held-out allocation count mismatch")
    write_csv(REPORT / "heldout_half_runs.csv", half_rows)
    groups = defaultdict(list)
    for row in run_rows:
        groups[row["dataset"], float(row["requested_total_human_fraction"]), row["method"]].append(row)
    summary = []
    for (dataset, fraction, method), records in sorted(groups.items()):
        summary.append({"dataset": dataset, "total_human_fraction": fraction, "method": method,
                        "split_seed_rows": len(records), "coverage": float(np.mean([float(r["coverage"]) for r in records])),
                        "query_auto_error": float(np.mean([float(r["query_mean_auto_error_rate"]) for r in records])),
                        "automatic_errors": float(np.mean([float(r["automatic_errors"]) for r in records]))})
    write_csv(REPORT / "total_cost_summary.csv", summary)
    half_summary = []
    for dataset, method in sorted({(r["dataset"], r["method"]) for r in half_rows}):
        subset = [r for r in half_rows if r["dataset"] == dataset and r["method"] == method]
        half_summary.append({"dataset": dataset, "method": method, "split_seed_rows": len(subset),
                             "query_auto_error": float(np.mean([r["query_mean_auto_error_rate"] for r in subset])),
                             "automatic_errors": float(np.mean([r["automatic_errors"] for r in subset])),
                             "development_labels": subset[0]["development_labels"], "online_labels": subset[0]["human_labels"],
                             "total_labels": subset[0]["total_labels"]})
    write_csv(REPORT / "heldout_half_summary.csv", half_summary)
    write_json(REPORT / "selected_parameters.json", choices)
    lines = ["# Target-domain tuning on COVID and ChatReport", "", "14 November 2026.", "",
        "The earlier policy selected settings using DL2019/Qwen and ChatReport/Phi-3, then kept them fixed for the returned Qwen transfer scores. Its online updates already used purchased target labels. This experiment additionally selects hyperparameters and an optional Platt probability calibrator using target development queries.", "",
        "## Protocol", "",
        "COVID retains the existing 10 development queries with 1,000 labels and 40 evaluation queries with 4,000 pairs. ChatReport uses three predetermined 3-development/8-evaluation query splits, with 180 development labels and 480 evaluation pairs per split. The ChatReport splits overlap. All development labels are purchased once per simulated deployment and reused for every candidate comparison.", "",
        "Each family receives 24 candidates. Our grid varies shared prior strength over 2, 10 and 50, local strength over 1, 5 and 20, and includes shared-only variants. LARA varies global/per-query acquisition, raw/upstream-normalized preprocessing and learning rate over 0.003, 0.01 and 0.03. Both families can use raw scores or a Platt mapping fitted on development logits. Other settings retain the prior implementation defaults.", "",
        "The selection objective averages query-average automatic error across 10%, 25% and 50% review on development validation folds, with three allocation seeds. COVID uses five two-query validation folds. ChatReport leaves out one development query at a time. Each Platt model is fitted using only the training development queries of its fold. After selection, the final calibrator is refitted on all development labels. Selected settings were saved before evaluating the other queries.", "",
        "This search used 6,048 development policy evaluations and produced 1,200 total-cost evaluation rows. Ten allocation seeds are used for evaluation. The search is limited to this declared grid. Existing benchmark outcomes had already been inspected, so this is exploratory adaptation evidence.", "",
        "## Selected settings", "",
        "| Dataset/split | Our score view | Shared strength | Local strength | Shared-only | LARA score view | LARA grouping | LARA preprocessing | LARA learning rate |",
        "|---|---|---:|---:|---|---|---|---|---:|"]
    for name, choice in choices.items():
        ours, lara = choice["ours"], choice["lara"]
        lines.append(f"| {name} | {ours['view']} | {ours['config']['shared_strength']} | {ours['config']['local_strength']} | {ours['config']['global_only']} | {lara['view']} | {lara['grouping']} | {lara['config']['preprocessing']} | {lara['config']['learning_rate']} |")
    lines += ["", "## Same review budget on held-out queries", "",
              "These rows review half of the evaluation pairs. All four methods are charged for the same development-label investment. COVID therefore costs 1,000 development + 2,000 evaluation labels, for 3,000 labels total. ChatReport costs 180 + 240 = 420 labels per split. Error is averaged over evaluation queries, with every evaluation pair in its query denominator.", "",
              "| Dataset | Frozen ours | Tuned ours | Frozen global raw LARA | Tuned LARA |",
              "|---|---:|---:|---:|---:|"]
    for dataset in ("trec_covid", "chatreport"):
        values = {r["method"]: r["query_auto_error"] for r in half_summary if r["dataset"] == dataset}
        cells = " | ".join(f"{100*values[m]:.3f}%" for m in ("frozen_ours", "tuned_ours", "frozen_lara_raw", "tuned_lara"))
        lines.append(f"| {dataset} | {cells} |")
    lines += ["", "ChatReport values average the three split means and ten seeds. They describe overlapping repeated splits. The 50% held-out-review point was evaluated with the same frozen selections as the total-cost sweep, without repeating parameter selection for that point.", "",
              "## Same total human budget on the full collection", "",
              "Here the budget includes all development labels. Those pairs become final human labels. The remaining budget reviews other queries. Full-pool controls can spend their entire budget anywhere in the collection. Automatic coverage of the full collection is equal at each budget. This accounts for the cost of gathering labels for tuning.", "",
              "| Dataset | Total human fraction | Frozen ours, free allocation | Frozen ours, development reserved | Tuned ours, development reserved | Tuned LARA, development reserved | Best tested full-pool LARA |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("trec_covid", "chatreport"):
        for fraction in settings["total_human_fractions"]:
            values = {r["method"]: r["query_auto_error"] for r in summary if r["dataset"] == dataset and r["total_human_fraction"] == fraction}
            best = min(v for m, v in values.items() if m.startswith("full_pool_lara_"))
            columns = [values[m] for m in ("full_pool_frozen_ours", "development_frozen_ours", "target_tuned_ours", "target_tuned_lara")] + [best]
            cells = " | ".join(f"{100*v:.3f}%" for v in columns)
            lines.append(f"| {dataset} | {100*fraction:.0f}% | {cells} |")
    lines += ["", "The two development-reserved versions of our policy isolate the effect of the selected settings at the same development and online costs. Comparing them with the free-allocation control measures whether this particular development-label investment is repaid on these collection sizes. Best-tested full-pool LARA chooses the lowest evaluation mean among the four retained variants and is a retrospective reference.", "",
              "## Interpretation and limits", "",
              "Hyperparameter tuning chooses settings such as prior strengths. Probability calibration fits a mapping from Qwen logits to relevance probabilities. Online adaptation updates the review policy as it purchases labels. These three operations consume information in different ways and are all explicitly represented in the experiment.", "",
              "Target tuning can improve a development objective without improving every evaluation budget. Three ChatReport development queries give limited information for choosing among 24 candidates. The selected local strength varies by split, while all three selections retain raw scores for our policy. The probability-calibration CSV separately reports raw and Platt Brier/ECE/accuracy on the other queries.", "",
              "This experiment labels every pair on the chosen development queries. That is a design choice, not a minimum mathematical requirement. Smaller development samples or reuse across later deployments would have different cost accounting and require an additional experiment. These outcomes assess this grid, split design and annotation investment.", "",
              "The original fixed-policy comparison remains available in `reports/qwen_transfer_comparison_v1`. The tuning results are an additional adaptation experiment. They do not replace earlier comparisons or establish a risk certificate.", "",
              "## Artifacts", "",
              "The report retains `total_cost_summary.csv`, `heldout_half_summary.csv`, `heldout_half_runs.csv`, `selected_parameters.json` and `probability_calibration.csv`. The experiment directory retains all candidate validation losses, selections, 1,200 allocation ledgers and source/input identities. The report additionally retains 160 allocations at half review of held-out queries.", "",
              "```powershell", "python scripts/tune_target_domain.py", "python scripts/analyze_target_domain_tuning.py", "```"]
    atomic_text(REPORT / "README.md", "\n".join(lines)+"\n")
    write_json(REPORT / "validation.json", {"evaluation_ledgers_recomputed": checked, "development_policy_evaluations": development_evaluations,
               "development_evaluation_query_disjointness": "passed", "within_development_calibration_disjointness": "passed",
               "selection_matches_development_objective": "passed", "total_cost_and_original_metric_recomputation": "passed",
               "additional_heldout_half_rows": len(half_rows), "additional_heldout_half_ledgers_recomputed": checked_half,
               "analysis_source_sha256": file_hash(Path(__file__))})
    atomic_text(REPORT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(REPORT).as_posix()}\n" for p in sorted(REPORT.rglob("*"))
                if p.is_file() and p.name != "checksums.sha256"))
    print(json.dumps(half_summary, indent=2), flush=True)
    print(REPORT / "README.md", flush=True)


if __name__ == "__main__":
    main()
