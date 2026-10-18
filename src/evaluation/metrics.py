"""Binary pilot metrics. No claim of calibration or a conformal guarantee."""

from collections import defaultdict


def selection_metrics(rows, threshold=0.5):
    accepted = [r for r in rows if r["confidence"] >= threshold]
    errors = sum(r["predicted_label"] != r["true_binary"] for r in accepted)
    return {"pairs": len(rows), "accepted": len(accepted), "errors": errors,
            "coverage": len(accepted) / len(rows) if rows else None,
            "selective_risk": errors / len(accepted) if accepted else None,
            "auto_error_rate": errors / len(rows) if rows else None}


def classification_metrics(rows, bins=10):
    n = len(rows)
    if not n:
        raise ValueError("Metrics require at least one prediction")
    tp = sum(r["true_binary"] == 1 and r["predicted_label"] == 1 for r in rows)
    tn = sum(r["true_binary"] == 0 and r["predicted_label"] == 0 for r in rows)
    fp, fn = sum(r["true_binary"] == 0 for r in rows) - tn, sum(r["true_binary"] == 1 for r in rows) - tp
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    ece, reliability = 0.0, []
    for i in range(bins):
        subset = [r for r in rows if min(int(r["prob_relevant"] * bins), bins - 1) == i]
        predicted = sum(r["prob_relevant"] for r in subset) / len(subset) if subset else None
        observed = sum(r["true_binary"] for r in subset) / len(subset) if subset else None
        if subset:
            ece += len(subset) / n * abs(predicted - observed)
        reliability.append({"lower": i / bins, "upper": (i + 1) / bins, "pairs": len(subset),
                            "mean_probability_relevant": predicted, "observed_relevance_rate": observed})
    return {"pairs": n, "queries": len({r["query_id"] for r in rows}), "accuracy": (tp + tn) / n,
            "macro_f1": (f1_pos + f1_neg) / 2, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "brier_score": sum((r["prob_relevant"] - r["true_binary"]) ** 2 for r in rows) / n,
            "ece_probability_bins_10": ece, "reliability": reliability}


def risk_coverage_table(rows):
    output = []
    for index in range(101):
        threshold = round(0.5 + index * 0.005, 3)
        for label in ("all", "predicted_nonrelevant", "predicted_relevant"):
            subset = rows if label == "all" else [r for r in rows if r["predicted_label"] == int(label == "predicted_relevant")]
            point = selection_metrics(subset, threshold)
            output.append({"group": label, "threshold": threshold, **point,
                           "accepted_share_of_all_pairs": point["accepted"] / len(rows)})
    return output


def per_query_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    return [{"query_id": query, **{k: v for k, v in classification_metrics(group).items() if k != "reliability"}}
            for query, group in sorted(groups.items())]


def query_bootstrap(rows, samples=10000, seed=42):
    """Percentile intervals from resampling whole queries, not individual pairs."""
    import random
    from collections import Counter
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    if not groups or samples < 1:
        raise ValueError("Bootstrap requires predictions and a positive sample count")
    vectors = []
    for _, group in sorted(groups.items()):
        m = classification_metrics(group)
        vector = [len(group), m["tp"], m["tn"], m["fp"], m["fn"], m["brier_score"] * len(group), m["accuracy"]]
        for bucket in m["reliability"]:
            count = bucket["pairs"]
            vector.extend([count, (bucket["mean_probability_relevant"] or 0) * count,
                           (bucket["observed_relevance_rate"] or 0) * count])
        vectors.append(vector)
    rng = random.Random(seed)
    estimates = defaultdict(list)
    for _ in range(samples):
        weights = Counter(rng.choices(range(len(vectors)), k=len(vectors)))
        values = [sum(vectors[index][column] * weight for index, weight in weights.items())
                  for column in range(len(vectors[0]))]
        n, tp, tn, fp, fn, brier, query_accuracy = values[:7]
        f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        ece = sum(abs(values[i + 1] - values[i + 2]) for i in range(7, len(values), 3)) / n
        current = {"accuracy": (tp + tn) / n, "macro_f1": (f1_positive + f1_negative) / 2,
                   "brier_score": brier / n, "ece_probability_bins_10": ece,
                   "query_macro_accuracy": query_accuracy / len(vectors),
                   "accuracy_minus_always_nonrelevant": (tp + tn - (tn + fp)) / n}
        for name, value in current.items():
            estimates[name].append(value)
    def percentile(values, fraction):
        values = sorted(values)
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"method": "query-cluster percentile bootstrap", "queries": len(vectors), "samples": samples,
            "seed": seed, "confidence_level": 0.95,
            "intervals": {name: {"lower": percentile(values, .025), "upper": percentile(values, .975)}
                          for name, values in estimates.items()}}


def temperature_calibration(rows, calibration_queries=10, seed=42):
    """Fit one positive temperature on query-disjoint calibration topics."""
    import math
    import random
    topics = sorted({r["query_id"] for r in rows})
    if not 0 < calibration_queries < len(topics):
        raise ValueError("Temperature calibration needs nonempty disjoint calibration and test queries")
    calibration_ids = set(random.Random(seed).sample(topics, calibration_queries))
    calibration = [r for r in rows if r["query_id"] in calibration_ids]
    test = [r for r in rows if r["query_id"] not in calibration_ids]
    def difference(row):
        value = float(row["yes_logit"]) - float(row["no_logit"])
        if not math.isfinite(value):
            raise ValueError("Temperature scaling requires finite saved Yes/No logits")
        return value
    training = [(difference(r), r["true_binary"]) for r in calibration]
    candidates = [2 ** (index / 10) for index in range(-30, 31)]
    def loss(temperature):
        return sum(max(z / temperature, 0) + math.log1p(math.exp(-abs(z / temperature)))
                   - y * z / temperature for z, y in training) / len(training)
    temperature = min(candidates, key=loss)
    calibrated = []
    for row in test:
        value = difference(row) / temperature
        exp = math.exp(-abs(value))
        probability = 1 / (1 + exp) if value >= 0 else exp / (1 + exp)
        calibrated.append({**row, "prob_relevant": probability, "confidence": max(probability, 1 - probability),
                           "predicted_label": int(probability >= .5)})
    return test, calibrated, {"method": "temperature scaling of conditional Yes/No logit difference",
                              "temperature": temperature, "candidate_temperatures": candidates,
                              "calibration_query_ids": sorted(calibration_ids),
                              "test_query_ids": sorted(set(topics) - calibration_ids), "seed": seed,
                              "calibration_pairs": len(calibration), "test_pairs": len(test),
                              "calibration_nll_before": loss(1), "calibration_nll_after": loss(temperature),
                              "interpretation": "Temperature fitted on calibration queries only; positive scaling preserves logit ordering and labels except rounding ties. No conformal guarantee."}
