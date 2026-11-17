"""Class-constrained allocation using purchased shared/local error evidence.

The minimax objective concerns model-estimated accepted-class error. It is not
a certificate. Raw relevance predictions are retained throughout this study.
"""

from dataclasses import asdict
import math
import random

import numpy as np

from src.calibration.safe_judge import SafeJudgeModel, SafeJudgeConfig


def acceptance_plan(raw, error, reviewed, automatic_count, floors, tie,
                    objective="worst_class", weights=None):
    """Optimize two within-class prefixes of estimated error, without gold.

    For minimax, sorted prefixes suffice: for any fixed pair of class counts,
    they minimize both class error sums. Enumerate every feasible count split.
    The query-error ablation instead sorts and sums query-weighted errors.
    """
    raw, error, reviewed = np.asarray(raw), np.asarray(error), np.asarray(reviewed, dtype=bool)
    tie, floors = np.asarray(tie), np.asarray(floors)
    if (raw.ndim != 1 or error.shape != raw.shape or reviewed.shape != raw.shape
            or tie.shape != raw.shape or not np.isin(raw, [0, 1]).all()
            or not np.isfinite(error).all() or np.any((error < 0) | (error > 1))
            or floors.shape != (2,) or np.any(floors < 1) or np.any(floors != floors.astype(int))
            or type(automatic_count) is not int):
        raise ValueError("Invalid class allocation inputs")
    if objective not in ("worst_class", "query_error"):
        raise ValueError("Unknown allocation objective")
    weights = np.ones(len(raw)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != raw.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Finite positive allocation weights required")
    value = error if objective == "worst_class" else error*weights
    orders = []
    for c in (0, 1):
        ix = np.flatnonzero(~reviewed & (raw == c))
        orders.append(ix[np.lexsort((tie[ix], value[ix]))])
    low = max(int(floors[0]), automatic_count-len(orders[1]))
    high = min(len(orders[0]), automatic_count-int(floors[1]))
    if low > high or automatic_count > int((~reviewed).sum()):
        raise ValueError("Budget cannot retain the requested class coverage")
    k = [np.arange(low, high+1)]
    k.append(automatic_count-k[0])
    sums = [np.r_[0., np.cumsum(error[ix])][k[c]] for c, ix in enumerate(orders)]
    risk = [sums[c]/k[c] for c in (0, 1)]
    worst = np.maximum(*risk)
    total = sums[0]+sums[1]
    weighted = sum(np.r_[0., np.cumsum((error*weights)[ix])][k[c]] for c, ix in enumerate(orders))
    primary = worst if objective == "worst_class" else weighted
    balance = np.abs(k[0]/max(1, int((raw == 0).sum()))-k[1]/max(1, int((raw == 1).sum())))
    selected = int(np.lexsort((k[0], balance, total, primary))[0])
    accepted = np.zeros(len(raw), dtype=bool)
    for c, ix in enumerate(orders):
        accepted[ix[:k[c][selected]]] = True
    return accepted, {"estimated_worst_class_risk": float(worst[selected]),
        "estimated_automatic_errors": float(total[selected]),
        "planned_class_counts": [int(ks[selected]) for ks in k],
        "feasible_count_splits": len(k[0])}


def run_class_aware_review(pool, oracle, budget, config=None, seed=0,
                           minimum_class_coverage=.05, batch_size=32,
                           objective="worst_class"):
    config = config or SafeJudgeConfig(gate_local=False, repair_labels=False, explore_every=0)
    if config.repair_labels:
        raise ValueError("This experiment preserves raw classes; repairs must be disabled")
    model = SafeJudgeModel(pool, config)
    if (type(budget) is not int or not 0 <= budget <= len(pool)
            or oracle.budget != budget or oracle.queried
            or not 0 < minimum_class_coverage <= 1
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Fresh exact-budget oracle, positive batch and class coverage required")
    sizes = np.bincount(model.raw, minlength=2)
    if np.any(sizes == 0):
        raise ValueError("Both raw predicted classes required")
    floors = np.maximum(1, np.ceil(sizes*minimum_class_coverage).astype(int))
    if len(pool)-budget < int(floors.sum()):
        raise ValueError("Budget cannot retain the requested class coverage")
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    tie = np.empty(len(pool), dtype=int)
    tie[order] = np.arange(len(pool))
    purchases, plans = [], []
    remaining = sizes.copy()

    def purchase(i, stage):
        label = oracle.query(model.keys[i])
        model.observe(i, label)
        remaining[model.raw[i]] -= 1
        purchases.append({"query_id": model.keys[i][0], "doc_id": model.keys[i][1],
            "rank": len(purchases)+1, "stage": stage, "observed_label": label})

    # Same confident per-query/class sentinels as the frozen policy, subject to
    # the declared coverage floor. These are paid final human labels.
    sentinels = []
    for q in range(model.q_count):
        for c in (0, 1):
            available = np.flatnonzero((model.query == q) & (model.raw == c))
            confident = available[model.confidence[available] >= .9]
            if len(available):
                sentinels.append(int(rng.choice(list(confident if len(confident) else available))))
    rng.shuffle(sentinels)
    for i in sentinels:
        if len(purchases) == budget:
            break
        if remaining[model.raw[i]] > floors[model.raw[i]]:
            purchase(i, "confident_sentinel")
    while len(purchases) < budget:
        _, error, _, _ = model.predict()
        accepted, plan = acceptance_plan(model.raw, error, model.reviewed, len(pool)-budget,
            floors, tie, objective, model.weight)
        plans.append({"paid_labels": len(purchases), **plan})
        candidates = np.flatnonzero(~model.reviewed & ~accepted)
        priority = error if objective == "worst_class" else error*model.weight
        candidates = candidates[np.lexsort((tie[candidates], -priority[candidates]))]
        for i in candidates[:min(batch_size, budget-len(purchases))]:
            purchase(int(i), "class_allocation")
    predicted, error, _, _ = model.predict()
    actual = np.bincount(model.raw[~model.reviewed], minlength=2)
    if np.any(actual < floors) or len(purchases) != budget or len(oracle.queried) != budget:
        raise AssertionError("Class floor or paid-label budget violated")
    state = {"config": asdict(config), "objective": objective, "batch_size": batch_size,
        "minimum_class_coverage": minimum_class_coverage, "class_floors": floors.tolist(),
        "sentinel_labels": sum(p["stage"] == "confident_sentinel" for p in purchases),
        "plans": plans, "final_class_counts": actual.tolist(),
        "guarantee": "Working error estimates only; no risk guarantee"}
    return model.reviewed.copy(), predicted, error, purchases, state
