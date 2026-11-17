"""Pilot-frozen candidate audit; query-weighted extension of acceptance sampling."""

from collections import Counter
import math
import random

import numpy as np

from src.calibration.audit_release import equal_quotas, upper_error_total
from src.utils.io import pair_key


def candidate_release(pool, scores, pilot, purchase, audit_limit, seed, audit_fraction=.7,
                      alpha=.05, class_risk=.2, delta=.05, minimum_class_coverage=.05):
    if not 0 < audit_fraction < 1 or audit_limit < 0:
        raise ValueError("Invalid audit fraction/limit")
    known = set(pilot)
    sizes = Counter(r["query_id"] for r in pool)
    predicted = np.asarray([int(r["prob_relevant"] >= .5) for r in pool])
    weight = np.asarray([1/(len(sizes)*sizes[r["query_id"]]) for r in pool])
    candidates, capacities, class_sizes = [], [], []
    for c, fraction in ((0, .25), (1, .15)):
        class_size = int((predicted == c).sum())
        indices = [i for i, r in enumerate(pool) if predicted[i] == c and pair_key(r) not in known]
        random.Random(f"candidate-ties/{seed}/{c}").shuffle(indices)
        indices.sort(key=lambda i: scores[i])
        count = max(math.ceil(fraction*len(indices)), min(70, len(indices)))
        candidate = indices[:count]
        minimum = math.ceil(minimum_class_coverage*class_size)
        capacities.append(max(0, min(math.floor(audit_fraction*len(candidate)), len(candidate)-minimum)))
        candidates.append(candidate)
        class_sizes.append(class_size)
    quotas = equal_quotas(capacities, min(audit_limit, sum(capacities)))
    accepted, counts, bounds = [], [], []
    for c, (candidate, quota) in enumerate(zip(candidates, quotas)):
        sample = random.Random(f"candidate-audit/{seed}/{c}").sample(candidate, quota)
        sampled = set(sample)
        e = sum(int(predicted[i] != purchase(pair_key(pool[i]))) for i in sample)
        remaining = [i for i in candidate if i not in sampled]
        upper = upper_error_total(len(candidate), quota, e, delta/2)-e
        q_upper = float(np.sort(weight[remaining])[::-1][:upper].sum())
        counts.append({"class": c, "candidate_size": len(candidate), "audit_labels": quota,
                       "audit_errors": e, "remaining_pairs": len(remaining), "remaining_error_upper": upper,
                       "query_error_upper": q_upper})
        accepted.extend(remaining)
        bounds.append(upper/len(remaining) if remaining else None)
    query_upper = sum(c["query_error_upper"] for c in counts)
    error_upper = sum(c["remaining_error_upper"] for c in counts)
    passed = query_upper <= alpha and all(bounds[c] is not None and bounds[c] <= class_risk
                and counts[c]["remaining_pairs"] >= math.ceil(minimum_class_coverage*class_sizes[c]) for c in (0, 1))
    human = np.ones(len(pool), dtype=bool)
    if passed:
        human[accepted] = False
    return human, {"audit_labels": sum(quotas), "query_error_upper_if_released": query_upper,
                   "error_count_upper_if_released": error_upper,
                   "class_0_risk_upper_if_released": bounds[0], "class_1_risk_upper_if_released": bounds[1],
                   "released": passed, "candidate_0_size": len(candidates[0]), "candidate_1_size": len(candidates[1]),
                   "class_0_audit_labels": quotas[0], "class_1_audit_labels": quotas[1],
                   "delta": delta}, counts
