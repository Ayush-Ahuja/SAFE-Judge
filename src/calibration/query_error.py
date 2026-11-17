"""Pilot-label error predictors; query effects change deferral, not relevance labels."""

from collections import defaultdict
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.io import pair_key


def pilot_sample(pool, per_query, seed):
    if type(per_query) is not int or per_query < 1:
        raise ValueError("Positive pilot labels/query required")
    groups = defaultdict(list)
    for row in pool:
        groups[str(row["query_id"])].append(pair_key(row))
    selected = []
    for qid, keys in sorted(groups.items()):
        keys = sorted(keys)
        if len(set(keys)) != len(keys) or len(keys) < per_query:
            raise ValueError("Duplicate pairs or inadequate pilot capacity")
        random.Random(f"error-pilot/{seed}/{qid}").shuffle(keys)
        selected.extend(keys[:per_query])
    return selected


def error_features(pool, method, query_scale=2.):
    if method not in ("global", "query", "disagreement", "query_disagreement"):
        raise ValueError("Unknown error-score method")
    p = np.asarray([r["prob_relevant"] for r in pool])
    c = np.maximum(p, 1-p)
    label = (p >= .5).astype(float)
    columns = [c, c*c, label, c*label]
    if "disagreement" in method:
        ce = np.asarray([r["auxiliary_logit"] for r in pool])
        disagree = ((ce >= 0) != label).astype(float)
        columns.extend([np.clip(ce, -20, 20)/5, disagree, disagree*c, np.clip(ce, -20, 20)/5*label])
    x = np.column_stack(columns)
    if "query" in method:
        ids = sorted({str(r["query_id"]) for r in pool})
        mapping = {q: i for i, q in enumerate(ids)}
        onehot = np.zeros((len(pool), len(ids)))
        onehot[np.arange(len(pool)), [mapping[str(r["query_id"])] for r in pool]] = query_scale
        x = np.column_stack([x, onehot])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite model feature")
    return x


def fit_error_scores(pool, observed, method, query_scale=2.):
    """Read only purchased labels; all score features are label-free."""
    x = error_features(pool, method, query_scale)
    indices, errors = [], []
    for i, row in enumerate(pool):
        key = pair_key(row)
        if key in observed:
            if observed[key] not in (0, 1):
                raise ValueError("Binary labels required")
            indices.append(i)
            errors.append(int(int(row["prob_relevant"] >= .5) != observed[key]))
    if not indices or len(indices) != len(observed):
        raise ValueError("Empty or unknown pilot labels")
    if len(set(errors)) == 1:
        return np.full(len(pool), (sum(errors)+1)/(len(errors)+2))
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=500, tol=1e-7)
    model.fit(x[indices], errors)
    return model.predict_proba(x)[:, 1]
