"""SAFE-Judge review allocation and the reported ablation options.

The frozen paper settings are in configs/safe_judge.json. The configuration
class also supports the gating, repair and exploration variants used in studies.
"""

from dataclasses import asdict, dataclass
import math
import random

import numpy as np
from scipy.special import betaln, expit, logit

from src.utils.io import pair_key


@dataclass(frozen=True)
class SafeJudgeConfig:
    shared_strength: float = 10.
    local_strength: float = 5.
    local_prior_probability: float = .1
    global_only: bool = False
    gate_local: bool = True
    repair_labels: bool = True
    repair_threshold: float = .75
    query_weighted: bool = True
    explore_every: int = 10
    information_bonus: float = 0.

    def __post_init__(self):
        if (not math.isfinite(self.shared_strength) or not math.isfinite(self.local_strength)
                or self.shared_strength <= 0 or self.local_strength <= 0
                or not 0 < self.local_prior_probability < 1 or not .5 < self.repair_threshold < 1
                or type(self.explore_every) is not int or self.explore_every < 0
                or not math.isfinite(self.information_bonus) or self.information_bonus < 0):
            raise ValueError("Invalid SAFE-Judge settings")


class SafeJudgeModel:
    def __init__(self, pool, config):
        self.pool, self.config = pool, config
        self.keys = [pair_key(r) for r in pool]
        if (not pool or len(set(self.keys)) != len(self.keys)
                or any("true_binary" in r or "true_relevance" in r for r in pool)):
            raise ValueError("Require a nonempty, unique pool without gold fields")
        p = np.asarray([r["prob_relevant"] for r in pool], dtype=float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid assessor probabilities")
        self.raw = (p >= .5).astype(int)
        self.confidence = np.maximum(p, 1-p)
        self.prior_error = np.clip(1-self.confidence, .01, .49)
        ids = sorted({q for q, _ in self.keys})
        mapping = {q: i for i, q in enumerate(ids)}
        self.query = np.asarray([mapping[q] for q, _ in self.keys])
        self.bin = self.raw*3+np.searchsorted([.8, .95], self.confidence, side="right")
        self.q_count, self.b_count = len(ids), 6
        self.query_size = np.bincount(self.query, minlength=len(ids))
        self.weight = 1/self.query_size[self.query] if config.query_weighted else np.ones(len(pool))
        self.base = np.full(6, .1)
        for b in range(6):
            subset = self.bin == b
            if subset.any():
                self.base[b] = self.prior_error[subset].mean()
        self.n = np.zeros((len(ids), 6), dtype=int)
        self.e = np.zeros_like(self.n)
        self.reviewed = np.zeros(len(pool), dtype=bool)
        self.initial_size = np.bincount(self.query*6+self.bin, minlength=len(ids)*6).reshape(len(ids), 6)

    def predict(self):
        cfg = self.config
        total_n, total_e = self.n.sum(axis=0), self.e.sum(axis=0)
        if cfg.global_only:
            mean = (cfg.shared_strength*self.base+total_e)/(cfg.shared_strength+total_n)
            cell_mean = np.broadcast_to(mean, self.n.shape)
            gate = np.zeros_like(self.n, dtype=float)
            a = np.broadcast_to(cfg.shared_strength*self.base+total_e, self.n.shape)
            b = np.broadcast_to(cfg.shared_strength*(1-self.base)+total_n-total_e, self.n.shape)
        else:
            # Leave this query out of its shared prior to avoid counting its
            # observations once globally and a second time in the local update.
            prior = (cfg.shared_strength*self.base+total_e-self.e)/(cfg.shared_strength+total_n-self.n)
            prior = np.clip(prior, 1e-6, 1-1e-6)
            pa, pb = cfg.local_strength*prior, cfg.local_strength*(1-prior)
            a, b = pa+self.e, pb+self.n-self.e
            local = a/(a+b)
            if cfg.gate_local:
                log_bayes_factor = (betaln(a, b)-betaln(pa, pb)
                                    - self.e*np.log(prior)-(self.n-self.e)*np.log1p(-prior))
                gate = expit(logit(cfg.local_prior_probability)+log_bayes_factor)
            else:
                gate = np.ones_like(prior)
            cell_mean = (1-gate)*prior+gate*local
        offsets = logit(np.clip(cell_mean, 1e-6, 1-1e-6))-logit(self.base)[None, :]
        error = expit(logit(self.prior_error)+offsets[self.query, self.bin])
        flip = (error >= cfg.repair_threshold) if cfg.repair_labels else np.zeros(len(error), dtype=bool)
        predicted = np.where(flip, 1-self.raw, self.raw)
        remaining_error = np.where(flip, 1-error, error)
        variance = a*b/((a+b)**2*(a+b+1))
        remaining = self.initial_size-self.n
        info = np.minimum(np.maximum(remaining-1, 0), 100)*variance/(a+b+1)
        priority = self.weight*(remaining_error+cfg.information_bonus*info[self.query, self.bin])
        return predicted, remaining_error, priority, gate

    def observe(self, index, label):
        if self.reviewed[index] or type(label) is not int or label not in (0, 1):
            raise ValueError("A new purchased binary label is required")
        q, b = self.query[index], self.bin[index]
        self.n[q, b] += 1
        self.e[q, b] += int(self.raw[index] != label)
        self.reviewed[index] = True


def run_safe_judge(pool, oracle, budget, config=None, seed=0):
    config = config or SafeJudgeConfig()
    model = SafeJudgeModel(pool, config)
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
