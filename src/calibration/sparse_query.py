"""Query-level CRC using uniform, equally sized within-query label samples.

Selection receives only sampled labels. Held-out gold is used separately for
evaluation. The loss is automatic mistakes / ALL pairs, not selective risk.
"""

from collections import defaultdict
import math
import random

import numpy as np

from src.utils.io import pair_key


def group_pool(rows):
    groups, seen = defaultdict(list), set()
    for row in rows:
        key = pair_key(row)
        probability = float(row["prob_relevant"])
        if key in seen or not all(key) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid/duplicate pair or probability")
        seen.add(key)
        groups[key[0]].append(row)
    if not groups:
        raise ValueError("Empty pool")
    return {qid: sorted(group, key=pair_key) for qid, group in sorted(groups.items())}


def sampling_plan(pool, query_ids, budget, seed):
    """Exactly G/n labels per query, uniformly without replacement; no gold read.

    A per-query random priority couples different budgets and query counts.
    Divisibility avoids unequal sample sizes and their exchangeability caveat.
    """
    groups = group_pool(pool)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("Query IDs must be nonempty and unique")
    n = len(query_ids)
    if type(budget) is not int or budget < n or budget % n:
        raise ValueError("Budget must be a positive multiple of the query count")
    k = budget // n
    result = {}
    for qid in query_ids:
        if qid not in groups or len(groups[qid]) < k:
            raise ValueError("Insufficient pairs for uniform sampling without replacement")
        keys = [pair_key(row) for row in groups[qid]]
        random.Random(f"sparse-query/{seed}/{qid}").shuffle(keys)
        result[qid] = keys[:k]
    return result


def validate_grid(grid):
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or grid[0] != 0.5 or grid[-1] != 1 or np.any(np.diff(grid) <= 0)):
        raise ValueError("Grid must increase strictly from 0.5 to 1")
    return grid


def _counts(rows, grid):
    confidence = np.asarray([max(r["prob_relevant"], 1-r["prob_relevant"]) for r in rows])
    errors = np.asarray([int(r["prob_relevant"] >= .5) != r["true_binary"] for r in rows], dtype=bool)
    # Strict > makes the loss right-continuous, including ties at confidence=1.
    accepted = len(rows) - np.searchsorted(np.sort(confidence), grid, side="right")
    wrong = errors.sum() - np.searchsorted(np.sort(confidence[errors]), grid, side="right")
    return accepted, wrong


def query_loss_curves(pool, plan, observed_labels, grid):
    """One row per query, denominator = sampled pairs for that query."""
    grid = validate_grid(grid)
    model_rows = {pair_key(row): row for row in pool}
    curves = []
    for qid, keys in plan.items():
        if not keys or len(set(keys)) != len(keys) or any(key[0] != qid for key in keys):
            raise ValueError("Invalid within-query sample")
        samples = []
        for key in keys:
            label = observed_labels[key]  # Only purchased labels are accessible here.
            if label not in (0, 1):
                raise ValueError("Labels must be binary")
            samples.append({"prob_relevant": model_rows[key]["prob_relevant"], "true_binary": label})
        curves.append(_counts(samples, grid)[1] / len(samples))
    if not curves:
        raise ValueError("Empty calibration sample")
    return np.asarray(curves)


def select_threshold(losses, grid, alpha, method="crc"):
    grid = validate_grid(grid)
    losses = np.asarray(losses, dtype=float)
    if (losses.ndim != 2 or not len(losses) or losses.shape[1] != len(grid)
            or not np.isfinite(losses).all() or np.any((losses < 0) | (losses > 1))
            or np.any(np.diff(losses, axis=1) > 1e-12) or np.any(losses[:, -1] != 0)):
        raise ValueError("Expected bounded nonincreasing query losses ending at zero")
    if not math.isfinite(alpha) or not 0 < alpha < 1 or method not in ("crc", "empirical"):
        raise ValueError("Invalid alpha/method")
    n = len(losses)
    empirical = losses.mean(axis=0)
    criterion = (losses.sum(axis=0) + 1) / (n + 1) if method == "crc" else empirical
    valid = np.flatnonzero(criterion <= alpha)
    index = int(valid[0]) if len(valid) else len(grid)-1
    return {"threshold_index": index, "threshold": float(grid[index]),
            "calibration_sample_loss": float(empirical[index]),
            "selection_criterion": float(criterion[index]),
            "full_escalation_fallback": not bool(len(valid)), "calibration_queries": n}


def evaluation_table(rows, grid):
    """Gold-only diagnostics for every fixed grid threshold; never selects one."""
    grid = validate_grid(grid)
    groups = group_pool(rows)
    count = len(rows)
    accepted, errors = _counts(rows, grid)
    query_losses, query_coverages = [], []
    for group in groups.values():
        a, e = _counts(group, grid)
        query_losses.append(e / len(group))
        query_coverages.append(a / len(group))
    query_mean = np.mean(query_losses, axis=0)
    query_coverage = np.mean(query_coverages, axis=0)
    classes = {}
    for label in (0, 1):
        subset = [r for r in rows if int(r["prob_relevant"] >= .5) == label]
        a, e = _counts(subset, grid)
        classes[label] = len(subset), a, e
    table = []
    for i, threshold in enumerate(grid):
        a, e = int(accepted[i]), int(errors[i])
        result = {"threshold": float(threshold), "test_pairs": count, "test_queries": len(groups),
                  "coverage": a/count, "query_mean_coverage": float(query_coverage[i]),
                  "automatic_labels": a, "automatic_errors": e,
                  "auto_error_rate": e/count, "query_mean_auto_error_rate": float(query_mean[i]),
                  "selective_risk": e/a if a else None, "test_human_labels": count-a}
        for label, (total, class_a, class_e) in classes.items():
            ca, ce = int(class_a[i]), int(class_e[i])
            result.update({f"class_{label}_pairs": total, f"class_{label}_accepted": ca,
                           f"class_{label}_errors": ce, f"class_{label}_coverage": ca/total if total else None,
                           f"class_{label}_selective_risk": ce/ca if ca else None})
        table.append(result)
    return table
