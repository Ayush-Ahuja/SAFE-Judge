"""Two confidence thresholds selected from paid calibration labels.

Selection receives evaluation probabilities but never evaluation labels. The
fixed-budget decision is which confidence prefix to accept in each raw class.
Tail risks are smoothed descriptive estimates, not confidence bounds.
"""

import math
import numpy as np


def confidence_order(probabilities, seed=0):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite nonempty probability vector required")
    tie = np.random.default_rng(seed).permutation(len(p))
    raw = p >= .5
    confidence = np.maximum(p, 1-p)
    orders = [np.flatnonzero(raw == c) for c in (0, 1)]
    return [ix[np.lexsort((tie[ix], -confidence[ix]))] for ix in orders]


def tail_risk(calibration_p, calibration_y, thresholds, label, min_support=5):
    p, y = np.asarray(calibration_p), np.asarray(calibration_y)
    confidence = np.maximum(p, 1-p)
    ix = np.flatnonzero((p >= .5) == label)
    ix = ix[np.argsort(confidence[ix], kind="stable")]
    position = np.searchsorted(confidence[ix], thresholds, side="left")
    counts = len(ix)-position
    errors = ((p[ix] >= .5) != y[ix]).astype(int)
    suffix = np.r_[np.cumsum(errors[::-1])[::-1], 0]
    wrong = suffix[position]
    estimate = (wrong+.5)/(counts+1.)
    # Unsupported tails cannot appear risk-free merely by having no labels.
    estimate = np.where(counts >= min_support, estimate, 1.)
    return estimate, counts, wrong


def select_class_thresholds(calibration_p, calibration_y, evaluation_p, automatic_count,
                            seed=0, minimum_class_coverage=.05, min_support=5,
                            objective="worst_class"):
    p, y = np.asarray(calibration_p, dtype=float), np.asarray(calibration_y, dtype=int)
    if p.ndim != 1 or y.shape != p.shape or not len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Paid calibration probabilities and binary labels required")
    confidence_order(p)
    if objective not in ("worst_class", "total_error") or not 0 < minimum_class_coverage <= 1:
        raise ValueError("Invalid objective or class coverage floor")
    if type(min_support) is not int or min_support < 1:
        raise ValueError("Positive minimum calibration tail support required")
    target = np.asarray(evaluation_p, dtype=float)
    orders = confidence_order(target, seed)
    if type(automatic_count) is not int or not 0 <= automatic_count <= len(target):
        raise ValueError("Automatic count outside pool")
    if any(not len(ix) for ix in orders):
        raise ValueError("Both predicted classes must be present")
    floor = [max(1, math.ceil(minimum_class_coverage*len(ix))) for ix in orders]
    lower = max(floor[0], automatic_count-len(orders[1]))
    upper = min(len(orders[0]), automatic_count-floor[1])
    if lower > upper:
        raise ValueError("Budget cannot retain the required automatic class coverage")
    k0 = np.arange(lower, upper+1)
    k1 = automatic_count-k0
    confidence = np.maximum(target, 1-target)
    thresholds = [confidence[orders[c]][ks-1] for c, ks in enumerate((k0, k1))]
    estimates = [tail_risk(p, y, thresholds[c], c, min_support) for c in (0, 1)]
    worst = np.maximum(estimates[0][0], estimates[1][0])
    total = (k0*estimates[0][0]+k1*estimates[1][0])/automatic_count
    balance = np.abs(k0/len(orders[0])-k1/len(orders[1]))
    support = np.minimum(estimates[0][1], estimates[1][1])
    first, second = (worst, total) if objective == "worst_class" else (total, worst)
    chosen = int(np.lexsort((k0, balance, -support, second, first))[0])
    counts = [int(k0[chosen]), int(k1[chosen])]
    accepted = np.zeros(len(target), dtype=bool)
    for ix, count in zip(orders, counts):
        accepted[ix[:count]] = True
    state = {"objective": objective, "automatic_count": automatic_count,
             "minimum_class_coverage": minimum_class_coverage, "minimum_calibration_tail_support": min_support,
             "estimated_worst_class_risk": float(worst[chosen]),
             "estimated_total_selective_risk": float(total[chosen]),
             "calibration_labels": len(p), "candidate_allocations": len(k0),
             "both_tails_supported": bool(support[chosen] >= min_support),
             "tie_rule": "Seeded label-independent ordering within equal-confidence boundary groups",
             "guarantee": "Smoothed calibration-tail estimates; no risk guarantee"}
    for c in (0, 1):
        threshold = float(thresholds[c][chosen])
        equal = (target >= .5) == c
        equal &= confidence == threshold
        state[f"class_{c}_threshold"] = threshold
        state[f"class_{c}_accepted"] = counts[c]
        state[f"class_{c}_calibration_tail_count"] = int(estimates[c][1][chosen])
        state[f"class_{c}_calibration_tail_errors"] = int(estimates[c][2][chosen])
        state[f"class_{c}_estimated_risk"] = float(estimates[c][0][chosen])
        state[f"class_{c}_boundary_accepted"] = int((equal & accepted).sum())
        state[f"class_{c}_boundary_rejected"] = int((equal & ~accepted).sum())
    return ~accepted, state
