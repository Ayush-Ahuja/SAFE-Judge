"""Complete cached-score metrics and numerical proposal-target diagnostics."""

from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import COHORTS, FIRST_SEED, DETERMINISTIC_METHODS, load_cohort
from src.evaluation.metrics import classification_metrics
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, read_predictions, write_csv, write_json

EXTRACTED = ROOT / "results/transfer/qwen35_transfer_v1_dataset_full"
COMPARISON = ROOT / "results/experiments/qwen_transfer_comparison_v1"
OUTPUT = ROOT / "reports/proposal_metrics_v1"


def confidence_aurc(probabilities, labels):
    """Exact expected discrete AURC under uniform ordering within score ties."""
    probabilities = np.asarray(probabilities)
    confidence = np.maximum(probabilities, 1-probabilities)
    wrong = (probabilities >= .5) != np.asarray(labels)
    order = np.argsort(-confidence, kind="stable")
    confidence, wrong = confidence[order], wrong[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(confidence) != 0)+1, len(order)]
    total, previous_errors = 0., 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        errors = int(wrong[left:right].sum())
        positions = np.arange(1, right-left+1)
        total += float(((previous_errors + errors*positions/(right-left))/(left+positions)).sum())
        previous_errors += errors
    return total / len(order)


def classification(y, predicted):
    return {"macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "quadratic_weighted_kappa_binary": float(cohen_kappa_score(y, predicted, labels=[0, 1], weights="quadratic"))}


def raw_metrics(name, rows):
    p = np.asarray([r["prob_relevant"] for r in rows])
    y = np.asarray([r["true_binary"] for r in rows])
    predicted = p >= .5
    metric = classification_metrics(rows)
    metric.pop("reliability")
    metric.update(classification(y, predicted))
    metric.update(dataset=name, confidence_aurc_tie_averaged=confidence_aurc(p, y))
    class_metrics, bins = [], []
    for label in (0, 1):
        subset = predicted == label
        c = np.maximum(p[subset], 1-p[subset])
        correct = predicted[subset] == y[subset]
        bin_id = np.minimum(((c-.5)*20).astype(int), 9)
        ece = 0.
        for b in range(10):
            mask = bin_id == b
            count = int(mask.sum())
            avg_c = float(c[mask].mean()) if count else None
            accuracy = float(correct[mask].mean()) if count else None
            if count:
                ece += count/len(c)*abs(avg_c-accuracy)
            bins.append({"dataset": name, "predicted_class": label, "lower": .5+b*.05,
                         "upper": .5+(b+1)*.05, "pairs": count,
                         "mean_confidence": avg_c, "observed_correctness": accuracy})
        class_metrics.append({"dataset": name, "predicted_class": label, "pairs": int(subset.sum()),
            "confidence_ece_10_bins": ece if len(c) else None,
            "error_rate": float((~correct).mean()) if len(c) else None,
            "brier_score": float(((p[subset]-y[subset])**2).mean()) if len(c) else None})
    return metric, class_metrics, bins


class DL2020Rankings:
    """Original top-ten positions from the released TREC-DL workbench archive."""
    def __init__(self, rows):
        archive_path = ROOT / "cache/research/transfer_inputs/dl20_texts.tar.xz"
        rankings = defaultdict(lambda: defaultdict(dict))
        with tarfile.open(archive_path) as archive:
            with gzip.GzipFile(fileobj=archive.extractfile("data/dl20/trecDL2020-qrels-runs-with-text.jsonl.gz")) as stream:
                for line in stream:
                    qid, passages = json.loads(line)
                    for passage in passages:
                        did = str(passage["paragraph_id"])
                        for entry in passage["paragraph_data"].get("rankings", []):
                            rank = int(entry["rank"])
                            if rank <= 10:
                                previous = rankings[entry["method"]][qid].get(rank)
                                value = (did, float(entry["score"]))
                                if previous is not None and previous != value:
                                    raise ValueError("Conflicting released ranking position")
                                rankings[entry["method"]][qid][rank] = value
        self.systems = sorted(rankings)
        self.queries = sorted({r["query_id"] for r in rows})
        if len(self.systems) != 59 or len(self.queries) != 54:
            raise ValueError("Unexpected DL2020 system/query count")
        index = {pair_key(r): i for i, r in enumerate(rows)}
        self.query_indices = [np.asarray([i for i, r in enumerate(rows) if r["query_id"] == q]) for q in self.queries]
        tensor, exported, missing = [], [], set()
        judged_counts = []
        for system in self.systems:
            if set(rankings[system]) != set(self.queries):
                raise ValueError("Run misses evaluation queries")
            by_query = []
            for qid in self.queries:
                entries = rankings[system][qid]
                if set(entries) != set(range(1, 11)) or len({d for d, _ in entries.values()}) != 10:
                    raise ValueError("Run does not have ten distinct judged top positions")
                current = []
                for rank in range(1, 11):
                    did, score = entries[rank]
                    judged = (qid, did) in index
                    current.append(index.get((qid, did), len(rows)))
                    if not judged:
                        missing.add((qid, did))
                    exported.append({"system": system, "query_id": qid, "doc_id": did,
                                     "rank": rank, "retrieval_score": score, "judged_in_manifest": judged})
                by_query.append(current)
            tensor.append(by_query)
            judged_counts.append(sum(item != len(rows) for group in by_query for item in group))
        self.indices = np.asarray(tensor)
        self.discount = 1/np.log2(np.arange(2, 12))
        self.gold = np.asarray([r["true_binary"] for r in rows])
        self.gold_scores = self.scores(self.gold)
        self.high_judgment = np.asarray(judged_counts) >= .95*54*10
        write_csv(OUTPUT / "dl2020_top10_rankings.csv", exported)
        write_csv(OUTPUT / "dl2020_gold_system_scores.csv", [{"system": system, "binary_ndcg10": float(score)}
                  for system, score in zip(self.systems, self.gold_scores)])
        write_csv(OUTPUT / "dl2020_judged_top10_coverage.csv", [{"system": system, "judged_positions": count,
                  "total_positions": 540, "judged_fraction": count/540} for system, count in zip(self.systems, judged_counts)])
        self.provenance = {"systems": 59, "queries": 54, "ranked_positions": len(exported),
            "unjudged_top10_positions": 59*54*10-sum(judged_counts), "unique_unjudged_top10_pairs": len(missing),
            "systems_at_least_95percent_judged_top10": int(self.high_judgment.sum()),
            "unjudged_convention": "Unjudged positions remain unjudged and receive zero gain under both reference and mixed qrels.",
            "source_archive_sha256": file_hash(archive_path),
            "source_url": "https://www.cs.unh.edu/~dietz/autograder/data-dl20.tar.xz",
            "official_count_reference": "https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.DL.pdf, Table 3",
            "metric": "Binary nDCG@10, grade>=2 positive. Both reference and mixed qrels use the binary target. Original submitted top-ten rank positions retained."}

    def scores(self, labels):
        labels = np.asarray(labels)
        relevant = np.asarray([int(labels[ix].sum()) for ix in self.query_indices])
        ideal = np.asarray([self.discount[:min(10, count)].sum() for count in relevant])
        # The extra zero represents an unjudged retrieval position in either qrels set.
        gains = np.r_[labels, 0][self.indices]
        dcg = (gains*self.discount).sum(axis=2)
        per_query = np.divide(dcg, ideal[None, :], out=np.zeros_like(dcg, dtype=float), where=ideal[None, :] > 0)
        return per_query.mean(axis=1)

    def evaluate(self, mixed):
        scores = self.scores(mixed)
        tau = float(kendalltau(self.gold_scores, scores, variant="b").statistic)
        rho = float(spearmanr(self.gold_scores, scores).statistic)
        return {"system_ranking_kendall_tau": tau if np.isfinite(tau) else None,
                "system_ranking_spearman_rho": rho if np.isfinite(rho) else None,
                "high_judgment_system_kendall_tau": float(kendalltau(self.gold_scores[self.high_judgment], scores[self.high_judgment], variant="b").statistic),
                "system_binary_ndcg10_mae": float(np.abs(scores-self.gold_scores).mean())}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert confidence_aurc([.8, .8], [1, 0]) == .5
    assert confidence_aurc([.9, .9], [1, 1]) == 0.
    assert confidence_aurc([.9, .9], [0, 0]) == 1.
    rows_by_cohort = {name: load_cohort(EXTRACTED, name) for name in COHORTS}
    dl19, _ = read_predictions(ROOT / "results/stage2/dl2019_qwen35_full_v1/predictions.csv", 2)
    raw, classes, reliability = [], [], []
    for name, rows in {"dl2019_discovery": dl19, **rows_by_cohort}.items():
        m, c, bins = raw_metrics(name, rows)
        raw.append(m)
        classes.extend(c)
        reliability.extend(bins)
    write_csv(OUTPUT / "raw_assessor_metrics.csv", raw)
    write_csv(OUTPUT / "predicted_class_calibration.csv", classes)
    write_csv(OUTPUT / "predicted_class_reliability.csv", reliability)
    rankings = DL2020Rankings(rows_by_cohort["dl2020_all"])
    if not np.isclose(rankings.evaluate(rankings.gold)["system_ranking_kendall_tau"], 1., atol=1e-12):
        raise ValueError("Gold self-ranking correlation is not one")
    evaluators = {name: PolicyEvaluator(rows) for name, rows in rows_by_cohort.items()}
    with (COMPARISON / "runs.csv").open(encoding="utf-8", newline="") as stream:
        originals = list(csv.DictReader(stream))
    results, cache = [], {}
    for original in originals:
        cohort, method = original["dataset"], original["method"]
        seed, budget = int(original["seed"]), int(original["budget"])
        stored_seed = FIRST_SEED if method in DETERMINISTIC_METHODS else seed
        key = cohort, method, stored_seed, budget
        if key not in cache:
            rows = rows_by_cohort[cohort]
            y = np.asarray([r["true_binary"] for r in rows])
            if method == "full_human":
                human = np.ones(len(rows), dtype=bool)
                predicted = np.asarray([r["predicted_label"] for r in rows])
            else:
                path = COMPARISON / "allocations" / cohort / method / f"seed_{stored_seed}_budget_{budget}.json.gz"
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    saved = json.load(stream)
                known = {pair_key(r) for r in saved["purchases"]}
                if len(known) != budget:
                    raise ValueError("Purchase ledger budget changed")
                human = np.asarray([pair_key(r) in known for r in rows])
                predicted = np.asarray(saved["automatic_predicted_labels"])
            metrics, _ = evaluators[cohort].evaluate(human, predicted)
            if abs(metrics["query_mean_auto_error_rate"]-float(original["query_mean_auto_error_rate"])) > 1e-12:
                raise ValueError("Original comparison metric changed")
            mixed = np.where(human, y, predicted)
            metrics.update({"mixed_"+name: value for name, value in classification(y, mixed).items()})
            metrics.update({"system_ranking_kendall_tau": None, "system_ranking_spearman_rho": None,
                            "high_judgment_system_kendall_tau": None, "system_binary_ndcg10_mae": None})
            if cohort == "dl2020_all":
                metrics.update(rankings.evaluate(mixed))
            both = metrics["both_classes_accepted"]
            class_ok = both and all(metrics[f"class_{c}_selective_risk"] <= .20 for c in (0, 1))
            metrics["worst_class_risk_if_both_defined"] = max(metrics[f"class_{c}_selective_risk"] for c in (0, 1)) if both else None
            metrics["empirical_error_target"] = metrics["query_mean_auto_error_rate"] <= .05
            metrics["empirical_both_class_risk_target"] = bool(class_ok)
            metrics["empirical_coverage_target"] = metrics["coverage"] >= .5
            metrics["empirical_ranking_target"] = metrics["system_ranking_kendall_tau"] >= .9 if cohort == "dl2020_all" else None
            metrics["empirical_joint_error_class_coverage"] = bool(metrics["empirical_error_target"] and class_ok and metrics["empirical_coverage_target"])
            metrics["empirical_joint_with_ranking"] = bool(metrics["empirical_joint_error_class_coverage"] and metrics["empirical_ranking_target"]) if cohort == "dl2020_all" else None
            metrics["minimum_5percent_predicted_class_coverage"] = bool(both and min(metrics[f"class_{c}_coverage"] for c in (0, 1)) >= .05)
            cache[key] = metrics
        results.append({"dataset": cohort, "method": method, "seed": seed, "budget": budget, **cache[key]})
    write_csv(OUTPUT / "policy_metrics.csv", results)
    groups = defaultdict(list)
    for r in results:
        groups[r["dataset"], r["budget"], r["method"]].append(r)
    fields = ("coverage", "query_mean_auto_error_rate", "auto_error_rate", "selective_risk",
              "mixed_macro_f1", "mixed_quadratic_weighted_kappa_binary", "class_0_coverage", "class_1_coverage",
              "class_0_selective_risk", "class_1_selective_risk", "worst_class_risk_if_both_defined",
              "system_ranking_kendall_tau", "system_ranking_spearman_rho", "high_judgment_system_kendall_tau", "system_binary_ndcg10_mae")
    flags = ("empirical_error_target", "empirical_both_class_risk_target", "empirical_coverage_target",
             "empirical_ranking_target", "empirical_joint_error_class_coverage", "empirical_joint_with_ranking",
             "minimum_5percent_predicted_class_coverage", "both_classes_accepted")
    summary = []
    for (cohort, budget, method), records in sorted(groups.items()):
        r = {"dataset": cohort, "budget": budget, "method": method, "seeds": len(records)}
        for name in fields:
            values = [x[name] for x in records if x[name] is not None]
            r[name+"_mean"] = float(np.mean(values)) if values else None
            r[name+"_defined_runs"] = len(values)
        for name in flags:
            values = [x[name] for x in records if x[name] is not None]
            r[name+"_count"] = sum(values) if values else None
        summary.append(r)
    write_csv(OUTPUT / "policy_summary.csv", summary)
    areas = []
    for cohort, method in sorted({(r["dataset"], r["method"]) for r in summary if r["method"] != "full_human"}):
        curve = sorted((r for r in summary if r["dataset"] == cohort and r["method"] == method), key=lambda r: r["coverage_mean"])
        x = np.asarray([r["coverage_mean"] for r in curve])
        y = np.asarray([r["selective_risk_mean"] for r in curve])
        area = float(np.trapezoid(y, x))
        areas.append({"dataset": cohort, "method": method, "coverage_lower": float(x.min()), "coverage_upper": float(x.max()),
                      "sampled_partial_aurc": area, "range_normalized_partial_aurc": area/(x.max()-x.min()),
                      "points": len(curve), "definition": "Trapezoidal area on measured mean budget points. Excludes zero coverage."})
    write_csv(OUTPUT / "policy_partial_aurc.csv", areas)
    write_json(OUTPUT / "experiment.json", {"status": "complete", "raw_cohorts": len(raw), "policy_rows": len(results),
        "ranking_provenance": rankings.provenance, "source_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
            (Path(__file__), ROOT / "src/evaluation/review_metrics.py", ROOT / "src/evaluation/metrics.py")},
        "original_comparison_metadata_sha256": file_hash(COMPARISON / "experiment.json"),
        "proposal_sha256": file_hash(ROOT / "reports/proposal.tex"),
        "scope": "Numerical target diagnostics on cached binary Qwen policy runs. Ten allocation seeds on fixed query pools. No new 20-query-split study or formal guarantee inferred.",
        "deviations": ["Assessor is frozen Qwen3.5-4B, not the proposed QLoRA-adapted 3B model.",
                       "Current COVID scoring defines only grade 2 positive, while proposal defines grades 1 and 2 positive.",
                       "COVID is a 100-pairs-per-query sample, with previously inspected outcomes and no fresh ten-topic blind audit.",
                       "Ranking evaluation uses all 59 released DL2020 top-ten positions with binary gains. COVID rankings and graded system fidelity remain pending.",
                       "Five-percent per-predicted-class coverage is an additional diagnostic from the later mathematical note, not an explicit numerical H1 requirement."]})
    atomic_text(OUTPUT / "checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(OUTPUT.iterdir())
                if p.is_file() and p.name != "checksums.sha256"))
    for r in summary:
        if r["method"] in ("selected_local", "lara_1_raw", "raw_confidence") and abs(r["coverage_mean"]-.5) < .001:
            print(r["dataset"], r["method"], "F1", r["mixed_macro_f1_mean"], "tau", r["system_ranking_kendall_tau_mean"],
                  "positive coverage", r["class_1_coverage_mean"], "joint", r["empirical_joint_with_ranking_count"], flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
