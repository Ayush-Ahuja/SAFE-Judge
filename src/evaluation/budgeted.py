"""Metrics for an explicit human/automatic allocation, not a threshold sweep."""

from collections import defaultdict

from src.evaluation.metrics import classification_metrics


def allocation_metrics(rows):
    if not rows:
        raise ValueError("Allocation metrics require a nonempty pool")
    automatic = [r for r in rows if r["label_source"] == "automatic"]
    human = [r for r in rows if r["label_source"] == "human"]
    if len(automatic) + len(human) != len(rows):
        raise ValueError("Unknown label source")
    if any(r["final_label"] != r["true_binary"] for r in human):
        raise ValueError("Simulated human labels must equal the official binary labels")
    if any(r["final_label"] != r["predicted_label"] for r in automatic):
        raise ValueError("Automatic labels must equal model predictions")
    errors = sum(r["predicted_label"] != r["true_binary"] for r in automatic)
    classes = {}
    for label in (0, 1):
        subset = [r for r in automatic if r["predicted_label"] == label]
        count_errors = sum(r["predicted_label"] != r["true_binary"] for r in subset)
        classes[str(label)] = {"accepted": len(subset), "errors": count_errors,
                               "selective_risk": count_errors / len(subset) if subset else None}
    risks = [group["selective_risk"] for group in classes.values() if group["accepted"]]
    result = {"pairs": len(rows), "queries": len({r["query_id"] for r in rows}),
              "human_labels": len(human), "automatic_labels": len(automatic),
              "human_fraction": len(human) / len(rows), "coverage": len(automatic) / len(rows),
              "automatic_errors": errors, "auto_error_rate": errors / len(rows),
              "selective_risk": errors / len(automatic) if automatic else None,
              "worst_observed_class_selective_risk": max(risks) if risks else None,
              "both_predicted_classes_accepted": all(c["accepted"] for c in classes.values()),
              "predicted_class_metrics": classes, "mixed_label_accuracy": 1 - errors / len(rows)}
    metrics = classification_metrics(automatic) if automatic else None
    if metrics:
        metrics.pop("reliability")
    result["automatic_classification"] = metrics
    # The model has trained on acquired labels: all-pool model accuracy is diagnostic only.
    diagnostic = classification_metrics(rows)
    diagnostic.pop("reliability")
    result["all_pool_model_diagnostic"] = diagnostic
    return result


def allocation_per_query(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    fields = ("pairs", "human_labels", "automatic_labels", "coverage", "automatic_errors",
              "auto_error_rate", "selective_risk", "mixed_label_accuracy")
    return [{"query_id": qid, **{k: v for k, v in allocation_metrics(group).items() if k in fields}}
            for qid, group in sorted(groups.items())]


def evaluate_runs(rows, run_paths, cutoff=10):
    """Binary nDCG@10 on identical topics/qrels for gold and mixed labels.

    Unjudged documents receive zero gain; scores sort descending with descending
    doc-ID tie breaks (trec_eval convention). IDCG uses all judged relevant docs.
    """
    import math
    from scipy.stats import kendalltau

    labels = defaultdict(dict)
    for row in rows:
        labels[row["query_id"]][row["doc_id"]] = (row["true_binary"], row["final_label"])
    system_rows, query_rows = [], []
    names = set()
    for path in run_paths:
        if path.name in names:
            raise ValueError("Run filenames must be unique")
        names.add(path.name)
        rankings = defaultdict(dict)
        tags = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 6:
                    raise ValueError(f"Invalid TREC run row {path}:{line_number}")
                qid, _, did, rank, score, tag = parts
                value = float(score)
                if not math.isfinite(value) or int(rank) < 1:
                    raise ValueError(f"Invalid TREC rank/score {path}:{line_number}")
                tags.add(tag)
                if qid not in labels:
                    continue
                if did in rankings[qid]:
                    raise ValueError(f"Duplicate TREC run pair {qid, did}")
                rankings[qid][did] = value
        if len(tags) != 1:
            raise ValueError(f"Expected one nonempty system in run {path}")
        values = []
        for qid, qrels in sorted(labels.items()):
            ranked = sorted(rankings[qid], key=lambda d: (rankings[qid][d], d), reverse=True)[:cutoff]
            current = {"system": path.name, "query_id": qid}
            for index, name in ((0, "gold_ndcg_at_10"), (1, "mixed_ndcg_at_10")):
                dcg = sum(qrels.get(did, (0, 0))[index] / math.log2(position + 2)
                          for position, did in enumerate(ranked))
                ideal_count = min(cutoff, sum(pair[index] for pair in qrels.values()))
                ideal = sum(1 / math.log2(position + 2) for position in range(ideal_count))
                current[name] = dcg / ideal if ideal else 0.0
            values.append(current)
        query_rows.extend(values)
        system_rows.append({"system": path.name, **{
            key: sum(r[key] for r in values) / len(values)
            for key in ("gold_ndcg_at_10", "mixed_ndcg_at_10")}})
    tau = None
    if len(system_rows) >= 2:
        value = float(kendalltau([r["gold_ndcg_at_10"] for r in system_rows],
                                 [r["mixed_ndcg_at_10"] for r in system_rows], variant="b").statistic)
        tau = value if math.isfinite(value) else None
    return system_rows, query_rows, {"metric": "binary nDCG@10", "systems": len(system_rows),
                                    "queries": len(labels), "kendall_tau_b": tau,
                                    "unjudged_gain": 0, "missing_query_score": 0,
                                    "tie_break": "descending score, then descending document ID"}
