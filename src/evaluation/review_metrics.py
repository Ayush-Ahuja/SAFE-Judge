"""Allocation diagnostics, including query tails and cached retrieval scorers."""

import math

import numpy as np
from scipy.stats import kendalltau


class PolicyEvaluator:
    def __init__(self, rows):
        self.rows = rows
        self.y = np.asarray([r["true_binary"] for r in rows])
        self.raw = np.asarray([int(r["prob_relevant"] >= .5) for r in rows])
        self.groups = {q: np.asarray([i for i, r in enumerate(rows) if r["query_id"] == q])
                       for q in sorted({r["query_id"] for r in rows})}
        self.rankings = {}
        if all("ranking_scores" in r for r in rows):
            for system in rows[0]["ranking_scores"]:
                self.rankings[system] = {q: np.asarray(sorted(indices, key=lambda i:
                    (rows[i]["ranking_scores"][system], rows[i]["doc_id"]), reverse=True)[:10])
                    for q, indices in self.groups.items()}
        self.gold_systems = self.system_scores(self.y) if self.rankings else None

    def system_scores(self, labels):
        discount = 1/np.log2(np.arange(2, 12))
        values = []
        for system, rankings in self.rankings.items():
            scores = []
            for q, indices in rankings.items():
                ideal = discount[:min(10, int(labels[self.groups[q]].sum()))].sum()
                dcg = float((labels[indices]*discount[:len(indices)]).sum())
                scores.append(dcg/ideal if ideal else 0.)
            values.append(float(np.mean(scores)))
        return values

    def evaluate(self, human, predictions=None):
        human = np.asarray(human, dtype=bool)
        predicted = self.raw if predictions is None else np.asarray(predictions, dtype=int)
        if human.shape != self.y.shape or predicted.shape != self.y.shape or not np.isin(predicted, [0, 1]).all():
            raise ValueError("Allocation shape/labels invalid")
        accepted = ~human
        error = accepted & (predicted != self.y)
        n, a, e = len(human), int(accepted.sum()), int(error.sum())
        query_rows = [{"query_id": q, "pairs": len(ix), "coverage": float(accepted[ix].mean()),
                       "auto_error_rate": float(error[ix].mean()), "automatic_errors": int(error[ix].sum())}
                      for q, ix in self.groups.items()]
        losses = np.asarray([r["auto_error_rate"] for r in query_rows])
        tail = max(1, math.ceil(.2*len(losses)))
        metrics = {"pairs": n, "queries": len(self.groups), "human_labels": int(human.sum()),
                   "automatic_labels": a, "automatic_errors": e, "coverage": a/n,
                   "auto_error_rate": e/n, "query_mean_auto_error_rate": float(losses.mean()),
                   "tail20_query_auto_error": float(np.sort(losses)[-tail:].mean()),
                   "worst_query_auto_error": float(losses.max()), "selective_risk": e/a if a else None,
                   "mixed_accuracy": 1-e/n}
        risks, coverages = [], []
        for label in (0, 1):
            group = predicted == label
            ca, ce = int((accepted & group).sum()), int((error & group).sum())
            risk = ce/ca if ca else None
            coverage = ca/int(group.sum()) if group.any() else None
            metrics.update({f"class_{label}_accepted": ca, f"class_{label}_errors": ce,
                            f"class_{label}_coverage": coverage, f"class_{label}_selective_risk": risk})
            if risk is not None:
                risks.append(risk)
            if coverage is not None:
                coverages.append(coverage)
        metrics["both_classes_accepted"] = bool(metrics["class_0_accepted"] and metrics["class_1_accepted"])
        metrics["worst_defined_class_selective_risk"] = max(risks) if risks else None
        metrics["minimum_defined_class_coverage"] = min(coverages) if coverages else None
        metrics["system_ranking_kendall_tau"] = None
        if self.rankings:
            mixed = np.where(human, self.y, predicted)
            value = float(kendalltau(self.gold_systems, self.system_scores(mixed), variant="b").statistic)
            metrics["system_ranking_kendall_tau"] = value if np.isfinite(value) else None
        return metrics, query_rows
