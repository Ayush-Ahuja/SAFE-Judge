"""Ranking-aware review using the SAFE-Judge update and purchase loop.

The optional utility scores determine which unreviewed pairs receive priority.
"""

import random
from dataclasses import asdict
import numpy as np
from src.calibration.safe_judge import SafeJudgeModel, SafeJudgeConfig


def run_utility_review(pool, oracle, budget, config=None, seed=0, utility=None):
    config = config or SafeJudgeConfig()
    model = SafeJudgeModel(pool, config)
    if utility is not None:
        utility = np.asarray(utility, dtype=float)
        if utility.shape != (len(pool),) or not np.isfinite(utility).all() or np.any(utility < 0) or not utility.any():
            raise ValueError("Review utility must be finite, nonnegative and nonzero")
        model.weight = utility.copy()
    if type(budget) is not int or not 0 <= budget <= len(pool) or oracle.budget != budget or oracle.queried:
        raise ValueError("Fresh oracle with the exact review budget required")
    rng = random.Random(seed)
    tie_order = list(range(len(pool)))
    rng.shuffle(tie_order)
    tie = np.empty(len(pool), dtype=int)
    tie[tie_order] = np.arange(len(pool))
    acquisitions = []

    def purchase(i, stage):
        label = oracle.query(model.keys[i])
        model.observe(i, label)
        acquisitions.append({"query_id": model.keys[i][0], "doc_id": model.keys[i][1],
                             "rank": len(acquisitions)+1, "stage": stage, "observed_label": label,
                             "raw_label": int(model.raw[i]), "raw_confidence": float(model.confidence[i])})

    # A supervisor spot-checks a confident decision of each type on every query.
    # All sentinel labels are paid and remain final human labels.
    sentinels = []
    for q in range(model.q_count):
        for label in (0, 1):
            available = np.flatnonzero((model.query == q) & (model.raw == label))
            confident = available[model.confidence[available] >= .9]
            if len(available):
                sentinels.append(rng.choice(list(confident if len(confident) else available)))
    rng.shuffle(sentinels)
    for i in sentinels[:budget]:
        purchase(int(i), "confident_sentinel")
    while len(acquisitions) < budget:
        _, _, priority, _ = model.predict()
        if config.explore_every and (len(acquisitions)+1) % config.explore_every == 0:
            available = np.flatnonzero(~model.reviewed)
            observed_per_q = model.n.sum(axis=1)
            best_count = observed_per_q[model.query[available]].min()
            candidates = available[observed_per_q[model.query[available]] == best_count]
            confident = candidates[model.confidence[candidates] >= .9]
            i = int(rng.choice(list(confident if len(confident) else candidates)))
            stage = "diagnostic_exploration"
        else:
            priority = priority.copy()
            priority[model.reviewed] = -np.inf
            candidates = np.flatnonzero(priority == priority.max())
            i = int(candidates[np.argmin(tie[candidates])])
            stage = "estimated_error"
        purchase(i, stage)
    predicted, error, _, gate = model.predict()
    state = {"config": asdict(config), "query_ids": sorted({q for q, _ in model.keys}),
             "observed_counts": model.n.tolist(), "observed_raw_errors": model.e.tolist(),
             "local_gate": gate.tolist(), "sentinel_labels": sum(a["stage"] == "confident_sentinel" for a in acquisitions),
             "exploration_labels": sum(a["stage"] == "diagnostic_exploration" for a in acquisitions),
             "automatic_repairs": int(((predicted != model.raw) & ~model.reviewed).sum()),
             "guarantee": "Working Bayesian acquisition model; no finite-sample release guarantee"}
    return model.reviewed.copy(), predicted, error, acquisitions, state
