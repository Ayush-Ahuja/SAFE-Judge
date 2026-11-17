"""Binary nDCG@10 fidelity and label-free utility for public retrieval runs."""

import numpy as np
from scipy.stats import kendalltau, rankdata

from src.utils.io import pair_key


class RetrievalFidelity:
    def __init__(self, rows, run_paths):
        self.rows = rows
        self.keys = {pair_key(r): i for i, r in enumerate(rows)}
        self.groups = {q: np.asarray([i for i, r in enumerate(rows) if str(r["query_id"]) == q])
                       for q in sorted({str(r["query_id"]) for r in rows})}
        self.rankings = {}
        for path in sorted(run_paths):
            groups = {}
            seen = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                qid, _, did, rank, score, tag = line.split()
                if (qid, did) in seen:
                    raise ValueError("Duplicate retrieval pair")
                seen.add((qid, did))
                if qid in self.groups:
                    groups.setdefault(qid, []).append((float(score), did))
            if set(groups) != set(self.groups):
                raise ValueError("A retrieval run does not cover every evaluation query")
            self.rankings[path.stem] = {q: sorted(items, reverse=True)[:10] for q, items in groups.items()}
        self.system_names = sorted(self.rankings)
        self.gold = np.asarray([r["true_binary"] for r in rows], dtype=int)
        self.gold_scores = self.scores(self.gold)
        self.held_out = [s for s in self.system_names if s.startswith("splade_") and not s.endswith("_original")]

    def scores(self, labels):
        discount = 1/np.log2(np.arange(2, 12))
        scores = []
        for system in self.system_names:
            values = []
            for q, ranking in self.rankings[system].items():
                ideal = discount[:min(10, int(labels[self.groups[q]].sum()))].sum()
                gains = [labels[self.keys[(q, did)]] if (q, did) in self.keys else 0 for _, did in ranking]
                values.append(float(np.dot(gains, discount[:len(gains)])/ideal) if ideal else 0.)
            scores.append(float(np.mean(values)))
        return np.asarray(scores)

    def review_utility(self):
        """Rank disagreement uses IDs/positions only; no gold gain or IDCG."""
        design = [s for s in self.system_names if s not in self.held_out]
        matrix = np.zeros((len(design), len(self.rows)))
        for si, system in enumerate(design):
            for q, ranking in self.rankings[system].items():
                for rank, (_, did) in enumerate(ranking, 1):
                    if (q, did) in self.keys:
                        matrix[si, self.keys[(q, did)]] = 1/np.log2(rank+1)
        utility = matrix.max(axis=0)-matrix.min(axis=0)
        return utility, design

    def evaluate(self, human, predictions):
        mixed = np.where(human, self.gold, predictions)
        scores = self.scores(mixed)
        tau = float(kendalltau(self.gold_scores, scores, variant="b").statistic)
        gold_rank = rankdata(-self.gold_scores, method="average")
        mixed_rank = rankdata(-scores, method="average")
        held = [self.system_names.index(s) for s in self.held_out]
        return {"public_system_kendall_tau": tau if np.isfinite(tau) else None,
                "public_system_ndcg_mae": float(np.abs(scores-self.gold_scores).mean()),
                "held_out_system_ndcg_mae": float(np.abs(scores[held]-self.gold_scores[held]).mean()) if held else None,
                "maximum_system_rank_drop": float(np.maximum(mixed_rank-gold_rank, 0).max())}, scores
