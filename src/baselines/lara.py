"""Binary LARA with pinned-author semantics and exact annotation budgets.

Independent implementation of the calibrator and acquisition equations. Source
provenance and upstream endpoint differences are documented in docs/lara.md.
"""

from dataclasses import asdict, dataclass
import math
import random

import numpy as np


@dataclass(frozen=True)
class LaraConfig:
    learning_rate: float = .01
    initial_strength: float = 200.0
    transition_scale: float = 1000.0
    blend_multiplier: float = 1.2
    preprocessing: str = "upstream_smooth"
    smoothness: float = 5.0

    def __post_init__(self):
        for name in ("learning_rate", "initial_strength", "transition_scale", "blend_multiplier", "smoothness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.preprocessing not in ("raw", "upstream_smooth"):
            raise ValueError("Preprocessing must be raw or upstream_smooth")
        # Beta(1,1) has the largest variance encountered after binary updates.
        if self.blend_multiplier < 1/12 or self.blend_multiplier * (-math.expm1(-self.initial_strength/self.transition_scale)) > 1:
            raise ValueError("Settings must keep the calibration mixture within [0, 1]")


def prepare_probabilities(values, config):
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or not len(probabilities) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("LLM probabilities must be a nonempty finite vector in [0, 1]")
    state = {"method": config.preprocessing, "smoothness": config.smoothness, "constant_fallback": False}
    if config.preprocessing == "raw":
        return probabilities.copy(), state
    center = float(probabilities.mean())
    smoothed = 1 / (1 + np.exp(-config.smoothness * (probabilities-center)))
    lower, upper = float(smoothed.min()), float(smoothed.max())
    state.update(center=center, sigmoid_min=lower, sigmoid_max=upper)
    if upper == lower:
        # Upstream divides by zero here; identity retains valid constant input.
        state["constant_fallback"] = True
        return probabilities.copy(), state
    return (smoothed-lower)/(upper-lower), state


class BinaryCalibrator:
    """Online logistic mapping blended with the initial relevance probability."""

    def __init__(self, config):
        self.config = config
        self.intercept, self.slope = 0.0, 0.0
        self.positive_prior, self.negative_prior = 1.0, 1.0
        self.updates = 0

    @staticmethod
    def sigmoid(values):
        # Keep ordinary floating-point arithmetic identical to the reference;
        # underflow/overflow at extremes gives the correct limiting probability.
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(-values))

    def predict(self, probabilities):
        values = np.asarray(probabilities, dtype=np.float64)
        positive, negative = self.positive_prior, self.negative_prior
        total = positive + negative
        variance = positive * negative / (total * total * (total + 1))
        blend = (self.config.blend_multiplier - variance) * (1 - np.exp(-self.config.initial_strength/self.config.transition_scale))
        logistic = self.sigmoid(self.intercept + self.slope * values)
        result = (1-blend)*values + blend*logistic
        if not np.isfinite(result).all() or np.any(result < -1e-14) or np.any(result > 1+1e-14):
            raise ArithmeticError("Invalid calibrated probability")
        return np.clip(result, 0, 1)

    def observe(self, probability, label):
        if type(label) is not int or label not in (0, 1):
            raise ValueError("A purchased binary label is required")
        error = float(self.sigmoid(self.intercept + self.slope * probability)) - label
        self.intercept -= self.config.learning_rate * error
        self.slope -= self.config.learning_rate * error * probability
        self.positive_prior += label
        self.negative_prior += 1-label
        self.updates += 1

    def state(self):
        return {"intercept": self.intercept, "slope": self.slope,
                "positive_prior": self.positive_prior, "negative_prior": self.negative_prior,
                "updates": self.updates, "initial_strength_is_fixed": True}


def topic_groups(pool, grouping, seed):
    topics = list(dict.fromkeys(r["query_id"] for r in pool))
    count = len(topics) if grouping == "topics" else grouping
    if type(count) is not int or count < 1:
        raise ValueError("Groups must be a positive integer or topics")
    count = min(count, len(topics))
    shuffled = topics.copy()
    random.Random(seed).shuffle(shuffled)
    indices = {qid: [] for qid in topics}
    for i, row in enumerate(pool):
        indices[row["query_id"]].append(i)
    groups, members, sizes = [[] for _ in range(count)], [[] for _ in range(count)], [0]*count
    for topic in shuffled:
        group = min(range(count), key=lambda i: sizes[i])
        groups[group].append(topic)
        members[group].extend(indices[topic])
        sizes[group] += len(indices[topic])
    # The reference filters a dataframe, preserving its original row order.
    return groups, [np.asarray(sorted(group), dtype=int) for group in members]


def group_quotas(capacities, budget):
    quotas = [min(size, budget//len(capacities)) for size in capacities]
    remaining = budget - sum(quotas)
    while remaining:
        progressed = False
        for i, size in enumerate(capacities):
            if quotas[i] < size and remaining:
                quotas[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("Group capacities cannot satisfy the budget")
    return quotas


def run_lara(pool, oracle, budget, config=None, seed=42, grouping=1):
    config = config or LaraConfig()
    if not pool or any("true_binary" in r or "true_relevance" in r or "annotation" in r for r in pool):
        raise ValueError("LARA requires a nonempty pool without gold fields")
    keys = [(r["query_id"], r["doc_id"]) for r in pool]
    if keys != sorted(keys) or len(set(keys)) != len(keys) or any(
            not isinstance(value, str) or not value or any(c.isspace() for c in value) for key in keys for value in key):
        raise ValueError("Pool IDs must be unique, sorted and contain no whitespace")
    if type(budget) is not int or not 0 <= budget <= len(pool) or oracle.budget != budget or oracle.queried:
        raise ValueError("A fresh oracle and exact budget in [0, pool size] are required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    original = np.asarray([r["prob_relevant"] for r in pool], dtype=np.float64)
    values, normalization = prepare_probabilities(original, config)
    topics, members = topic_groups(pool, grouping, seed)
    quotas = group_quotas([len(indices) for indices in members], budget)
    remaining = quotas.copy()
    model = BinaryCalibrator(config)
    observed = np.zeros(len(pool), dtype=bool)
    labels, acquisitions = {}, []

    def acquire(index, group, probability, stage):
        label = oracle.query(keys[index])
        labels[index] = label
        observed[index] = True
        acquisitions.append({"query_id": keys[index][0], "doc_id": keys[index][1],
                             "acquisition_rank": len(acquisitions)+1, "group": group, "stage": stage,
                             "llm_probability": float(original[index]), "input_probability": float(values[index]),
                             "prob_relevant_before_query": float(probability), "observed_label": label})
        return label

    if budget == len(pool):
        # As in upstream's full-human endpoint, calibrator fitting is unnecessary.
        initial = model.predict(values)
        group_for_index = {int(i): g for g, indices in enumerate(members) for i in indices}
        for index in range(len(pool)):
            acquire(index, group_for_index[index], initial[index], "full_human")
    else:
        while len(acquisitions) < budget:
            for group, indices in enumerate(members):
                if remaining[group] == 0:
                    continue
                available = indices[~observed[indices]]
                probabilities = model.predict(values[available])
                position = int(np.argmin(np.abs(probabilities-.5)))
                index = int(available[position])
                label = acquire(index, group, probabilities[position], "active_calibration")
                model.observe(float(values[index]), label)
                remaining[group] -= 1
    calibrated = model.predict(values)
    ranks = {pair_key: rank+1 for rank, pair_key in enumerate(oracle.queried)}
    predictions = [{"query_id": keys[i][0], "doc_id": keys[i][1],
                    "llm_probability": float(original[i]), "input_probability": float(values[i]),
                    "prob_relevant": float(p), "predicted_label": int(p >= .5), "confidence": float(max(p, 1-p)),
                    "label_source": "human" if observed[i] else "automatic",
                    "acquisition_rank": ranks.get(keys[i], ""),
                    "final_label": labels[i] if observed[i] else int(p >= .5)}
                   for i, p in enumerate(calibrated)]
    return predictions, acquisitions, {"config": asdict(config), "seed": seed, "requested_grouping": grouping,
                                        "groups": topics, "group_quotas": quotas, "budget": budget,
                                        "preprocessor": normalization, "calibrator": model.state(),
                                        "full_human_shortcut": budget == len(pool),
                                        "tie_rule": "first minimum in canonical pair order"}
