"""Progressive review with fresh quality checks and a finite-pool stopping rule."""

from collections import Counter
import math
import random

import numpy as np

from src.calibration.audit_release import equal_quotas, upper_error_total
from src.utils.io import pair_key


class CachedReviewOracle:
    """A fresh local work budget backed by the already-paid global label cache."""
    def __init__(self, budget, acquire):
        self.budget, self.acquire, self.queried = budget, acquire, []
        self.seen = set()

    def query(self, key):
        if key in self.seen or len(self.queried) >= self.budget:
            raise ValueError("Duplicate or over-budget local review")
        self.seen.add(key)
        self.queried.append(key)
        return self.acquire(key)


def progressive_review(pool, oracle, work_policy, seed, checkpoints=(.1, .25, .4, .5),
                       audit_fraction=.05, alpha=.05, class_risk=.2, delta=.05,
                       minimum_class_coverage=.05, feasibility_gate=True):
    """At each look: learn from work labels, freeze, audit fresh unknown labels.

    Each look spends delta/T, divided over the two predicted classes. Fresh
    uniform audits conditional on the full past permit adaptive stopping.
    Work policies may use any previously purchased labels via the cache.
    """
    keys = [pair_key(r) for r in pool]
    if (not pool or len(set(keys)) != len(keys) or not checkpoints
            or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or any(not 0 <= f <= 1 for f in checkpoints) or not 0 < audit_fraction < 1
            or not 0 < alpha < 1 or not 0 < class_risk < 1 or not 0 < delta < 1
            or not 0 <= minimum_class_coverage < 1 or oracle.queried or oracle.budget != len(pool)):
        raise ValueError("Invalid progressive-review design or global oracle")
    index = {key: i for i, key in enumerate(keys)}
    q_sizes = Counter(q for q, _ in keys)
    weights = np.asarray([1/(len(q_sizes)*q_sizes[q]) for q, _ in keys])
    cache, ledger, looks = {}, [], []
    per_class_delta = delta/(2*len(checkpoints))

    def acquire(key, step, reason):
        if key not in index:
            raise ValueError("Unknown pair")
        if key not in cache:
            cache[key] = oracle.query(key)
            ledger.append({"query_id": key[0], "doc_id": key[1], "step": step,
                           "reason": reason, "observed_label": cache[key], "paid_rank": len(cache)})
        return cache[key]

    predicted = np.asarray([int(r["prob_relevant"] >= .5) for r in pool])
    for step, fraction in enumerate(checkpoints):
        work_budget = int(len(pool)*fraction)
        local = CachedReviewOracle(work_budget, lambda key: acquire(key, step, "work"))
        work_human, predicted = work_policy(local, work_budget)
        predicted = np.asarray(predicted, dtype=int)
        if (len(local.queried) != work_budget or int(np.asarray(work_human).sum()) != work_budget
                or predicted.shape != (len(pool),) or not np.isin(predicted, [0, 1]).all()):
            raise AssertionError("Work-policy allocation/budget is invalid")
        unknown = np.asarray([key not in cache for key in keys])
        groups = [np.flatnonzero(unknown & (predicted == c)) for c in (0, 1)]
        class_sizes = [int((predicted == c).sum()) for c in (0, 1)]
        capacities = [max(0, min(len(g)//2, len(g)-math.ceil(minimum_class_coverage*class_sizes[c])))
                      for c, g in enumerate(groups)]
        audit_budget = min(math.ceil(audit_fraction*len(pool)), sum(capacities))
        quotas = equal_quotas(capacities, audit_budget)
        optimistic, query_optimistic = [], 0.
        for c, (group, n) in enumerate(zip(groups, quotas)):
            upper = upper_error_total(len(group), n, 0, per_class_delta)
            remaining = len(group)-n
            optimistic.append(upper/remaining if remaining else 1.)
            # Even removing the highest-weight audited pairs cannot beat this
            # lower bound on the worst-case query-weight certificate.
            ordered = np.sort(weights[group])[::-1]
            query_optimistic += float(ordered[n:n+upper].sum())
        feasible = (all(class_sizes) and all(len(g)-n >= math.ceil(minimum_class_coverage*class_sizes[c])
                    for c, (g, n) in enumerate(zip(groups, quotas)))
                    and max(optimistic) <= class_risk and query_optimistic <= alpha)
        info = {"step": step, "work_fraction": fraction, "work_budget": work_budget,
                "paid_before_audit": len(cache), "planned_audit_labels": audit_budget,
                "optimistically_feasible": bool(feasible), "look_delta": delta/len(checkpoints)}
        if feasibility_gate and not feasible:
            looks.append({**info, "audit_labels": 0, "status": "skipped_infeasible"})
            continue
        class_bounds, query_bound, total_upper = [], 0., 0
        before = len(cache)
        for c, (group, n) in enumerate(zip(groups, quotas)):
            chosen = random.Random(f"fresh-quality-check/{seed}/{step}/{c}").sample(list(group), n)
            errors = 0
            for i in chosen:
                errors += int(predicted[i] != acquire(keys[i], step, "quality_check"))
            chosen = set(chosen)
            remaining = [int(i) for i in group if i not in chosen]
            upper = upper_error_total(len(group), n, errors, per_class_delta)-errors
            total_upper += upper
            query_bound += float(np.sort(weights[remaining])[::-1][:upper].sum())
            class_bounds.append(upper/len(remaining) if remaining else None)
        human = np.asarray([key in cache for key in keys])
        passed = (query_bound <= alpha and all(class_bounds[c] is not None and class_bounds[c] <= class_risk
                  and int(((~human) & (predicted == c)).sum()) >= math.ceil(minimum_class_coverage*class_sizes[c])
                  and class_sizes[c] > 0 for c in (0, 1)))
        certificate = {"query_auto_error_upper": query_bound, "error_count_upper": total_upper,
                       "class_0_selective_risk_upper": class_bounds[0], "class_1_selective_risk_upper": class_bounds[1]}
        looks.append({**info, "audit_labels": len(cache)-before, "status": "released" if passed else "continued",
                      **certificate})
        if passed:
            return human, predicted, {"released": True, "stop_step": step, "target_human_labels": len(cache),
                "delta": delta, "checkpoints": len(checkpoints), **certificate}, looks, ledger
    for key in keys:
        acquire(key, len(checkpoints), "full_review_fallback")
    return np.ones(len(pool), dtype=bool), predicted, {"released": False, "stop_step": len(checkpoints),
        "target_human_labels": len(cache), "delta": delta, "checkpoints": len(checkpoints),
        "query_auto_error_upper": 0., "error_count_upper": 0,
        "class_0_selective_risk_upper": None, "class_1_selective_risk_upper": None}, looks, ledger
