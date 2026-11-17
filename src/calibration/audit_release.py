"""Finite-pool release with simultaneous cell bounds and query-weighted safety.

Classical hypergeometric inversion supplies the statistical component. Within
each accepted cell, assigning its upper error count to the smallest queries
gives a conservative bound on equal-query auto-error, without query iid claims.
"""

from collections import Counter
from functools import lru_cache
import math
import random

import numpy as np
from scipy.stats import hypergeom

from src.utils.io import pair_key


@lru_cache(maxsize=65536)
def upper_error_total(population, sample_size, errors, delta):
    """Invert the exact lower tail, returning an upper total number of defects."""
    if (any(type(v) is not int for v in (population, sample_size, errors))
            or not 0 <= errors <= sample_size <= population or not 0 < delta < 1):
        raise ValueError("Invalid finite-population audit arguments")
    if sample_size == 0:
        return population
    if sample_size == population:
        return errors
    low, high = errors, population-sample_size+errors
    while low < high:
        mid = (low+high+1)//2
        if hypergeom.cdf(errors, population, mid, sample_size) >= delta:
            low = mid
        else:
            high = mid-1
    return low


def equal_quotas(capacities, budget):
    if type(budget) is not int or not 0 <= budget <= sum(capacities) or any(c < 0 for c in capacities):
        raise ValueError("Infeasible audit budget")
    quotas = [0]*len(capacities)
    remaining = budget
    while remaining:
        for i, capacity in enumerate(capacities):
            if quotas[i] < capacity and remaining:
                quotas[i] += 1
                remaining -= 1
    return quotas


def build_audit(pool, scores, pilot_keys, purchase, budget, bins_per_class, seed, delta=.05):
    """Freeze all strata/quotas before reading any audit labels.

    Scores may depend on separate pilot labels. Only `purchase` supplies audit
    labels. The same audit supports any later union of the certified cells.
    """
    scores = np.asarray(scores, dtype=float)
    keys = [pair_key(r) for r in pool]
    known = set(pilot_keys)
    if (len(set(keys)) != len(keys) or not known <= set(keys) or scores.shape != (len(pool),)
            or not np.isfinite(scores).all() or type(bins_per_class) is not int or bins_per_class < 1
            or not 0 < delta < 1):
        raise ValueError("Invalid pool, pilot, scores or audit parameters")
    predicted = np.asarray([int(r["prob_relevant"] >= .5) for r in pool])
    query_sizes = Counter(str(r["query_id"]) for r in pool)
    q_count = len(query_sizes)
    weights = np.asarray([1/(q_count*query_sizes[str(r["query_id"])]) for r in pool])
    strata = []
    for label in (0, 1):
        indices = [i for i in range(len(pool)) if predicted[i] == label and keys[i] not in known]
        random.Random(f"audit-strata-ties/{seed}/{label}").shuffle(indices)
        indices.sort(key=lambda i: scores[i])
        if indices:
            for rank, cell in enumerate(np.array_split(indices, min(bins_per_class, len(indices)))):
                strata.append({"predicted_class": label, "rank": rank, "indices": [int(i) for i in cell]})
    quotas = equal_quotas([len(s["indices"]) for s in strata], budget)
    acquired = set(known)
    logs, cells = [], []
    for h, (stratum, quota) in enumerate(zip(strata, quotas)):
        indices = stratum["indices"]
        selected = random.Random(f"independent-audit/{seed}/{h}").sample(indices, quota)
        errors = 0
        for i in selected:
            label = purchase(keys[i])
            if label not in (0, 1):
                raise ValueError("Audit purchase must return a binary label")
            wrong = int(predicted[i] != label)
            errors += wrong
            acquired.add(keys[i])
            logs.append({"query_id": keys[i][0], "doc_id": keys[i][1], "cell": h,
                         "observed_label": int(label), "model_error": wrong})
        remaining = [i for i in indices if keys[i] not in acquired]
        upper = upper_error_total(len(indices), quota, errors, delta/len(strata))-errors
        # The true error count in the unreviewed cell is <= upper on its audit event.
        # No assumptions about locations/correlation of these errors are needed.
        weight_upper = float(np.sort(weights[remaining])[::-1][:upper].sum())
        cells.append({**stratum, "audit_labels": quota, "observed_errors": errors,
                      "remaining_indices": remaining, "remaining_size": len(remaining),
                      "remaining_error_upper": upper, "query_weighted_error_upper": weight_upper})
    return {"cells": cells, "acquired_keys": acquired, "audit_log": logs, "delta": delta,
            "pairs": len(pool), "query_sizes": dict(query_sizes),
            "class_sizes": {label: int((predicted == label).sum()) for label in (0, 1)}}


def select_release(pool, audit, alpha=.05, class_risk=.20, risk_unit="query_balanced", minimum_class_coverage=.05):
    """Maximize coverage over two class prefixes, using only audit certificates."""
    if (risk_unit not in ("pooled", "query_balanced") or not 0 < alpha < 1
            or not 0 < class_risk < 1 or not 0 <= minimum_class_coverage < 1):
        raise ValueError("Invalid release constraints")
    cells = audit["cells"]
    by_class = {label: sorted([c for c in cells if c["predicted_class"] == label], key=lambda c: c["rank"])
                for label in (0, 1)}
    best = None
    for k0 in range(len(by_class[0])+1):
        for k1 in range(len(by_class[1])+1):
            chosen = by_class[0][:k0]+by_class[1][:k1]
            a = sum(c["remaining_size"] for c in chosen)
            upper = sum(c["remaining_error_upper"] for c in chosen)
            query_bound = sum(c["query_weighted_error_upper"] for c in chosen)
            pooled_bound = upper/audit["pairs"]
            if (query_bound if risk_unit == "query_balanced" else pooled_bound) > alpha:
                continue
            class_bounds, valid = {}, True
            for label in (0, 1):
                subset = [c for c in chosen if c["predicted_class"] == label]
                ca = sum(c["remaining_size"] for c in subset)
                ce = sum(c["remaining_error_upper"] for c in subset)
                total = audit["class_sizes"][label]
                class_bounds[label] = ce/ca if ca else None
                if ((total and ca/total < minimum_class_coverage)
                        or (ca and ce/ca > class_risk)):
                    valid = False
            if not valid:
                continue
            objective = a, -query_bound, -upper, -k0, -k1
            if best is None or objective > best[0]:
                best = objective, chosen, upper, query_bound, pooled_bound, class_bounds, (k0, k1)
    human = np.ones(len(pool), dtype=bool)
    if best is None:
        upper, query_bound, pooled_bound, class_bounds, prefixes = 0, 0., 0., {0: None, 1: None}, (0, 0)
    else:
        _, chosen, upper, query_bound, pooled_bound, class_bounds, prefixes = best
        for cell in chosen:
            human[cell["remaining_indices"]] = False
    certificate = {"risk_unit": risk_unit, "alpha": alpha, "class_risk_target": class_risk,
                   "delta": audit["delta"], "certified_error_count_upper": upper,
                   "certified_pooled_auto_error_upper": pooled_bound,
                   "certified_query_mean_auto_error_upper": query_bound,
                   "certified_class_0_selective_risk_upper": class_bounds[0],
                   "certified_class_1_selective_risk_upper": class_bounds[1],
                   "accepted_class_0_bins": prefixes[0], "accepted_class_1_bins": prefixes[1],
                   "full_human_fallback": best is None}
    return human, certificate
