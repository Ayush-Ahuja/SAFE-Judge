"""Proposal's lexical SAL(hybrid) variant, not an exact LARA reproduction.

Features and scaling use unlabeled pool text. Only the oracle can supply labels
to the learner. Gold labels for unqueried pairs are never used by this module.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ("bm25", "query_term_overlap", "log1p_document_length")
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text):
    """Unicode alphanumeric tokens, casefolded; no stemming or stopword removal."""
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class ActiveLearningConfig:
    seed_size: int = 50
    batch_size: int = 50
    regularization_c: float = 1.0
    max_iter: int = 2000
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    def __post_init__(self):
        for name in ("seed_size", "batch_size", "max_iter"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("regularization_c", "bm25_k1", "bm25_b"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.regularization_c <= 0 or self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ValueError("Require C > 0, k1 > 0 and 0 <= b <= 1")


def validate_pool(pool):
    if not pool:
        raise ValueError("The annotation pool is empty")
    keys, queries, documents = set(), {}, {}
    for row in pool:
        for name in ("query_id", "doc_id", "query_text", "doc_text"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Invalid pool field {name}")
        key = (row["query_id"], row["doc_id"])
        if any(any(character.isspace() for character in identifier) for identifier in key):
            raise ValueError("Query/document IDs must not contain whitespace (TREC format)")
        if key in keys:
            raise ValueError(f"Duplicate pool pair {key}")
        keys.add(key)
        for mapping, identifier, text in ((queries, row["query_id"], row["query_text"]),
                                           (documents, row["doc_id"], row["doc_text"])):
            if identifier in mapping and mapping[identifier] != text:
                raise ValueError(f"Conflicting text for ID {identifier}")
            mapping[identifier] = text
    return documents


def lexical_features(pool, config, corpus_stats=None):
    """BM25 counts unique documents, never query-document repetitions.

    Default statistics describe the selected judged pool, not the full corpus.
    Optional externally computed statistics must use this exact tokenizer.
    """
    documents = validate_pool(pool)
    term_counts = {did: Counter(tokenize(text)) for did, text in documents.items()}
    if corpus_stats is None:
        frequencies = Counter()
        for counts in term_counts.values():
            frequencies.update(counts.keys())
        corpus_stats = {
            "document_count": len(documents),
            "average_document_length": sum(sum(c.values()) for c in term_counts.values()) / len(documents),
            "document_frequencies": dict(sorted(frequencies.items())),
            "tokenizer": "unicode_alphanumeric_casefold_v1",
            "scope": "unique documents in selected annotation pool",
        }
    else:
        corpus_stats = dict(corpus_stats)
    n = corpus_stats.get("document_count")
    avgdl = corpus_stats.get("average_document_length")
    frequencies = corpus_stats.get("document_frequencies")
    if type(n) is not int or n < 1 or not isinstance(avgdl, (float, int)) or not math.isfinite(avgdl) or avgdl < 0:
        raise ValueError("Invalid BM25 document count/average length")
    if corpus_stats.get("tokenizer") != "unicode_alphanumeric_casefold_v1" or not isinstance(frequencies, dict):
        raise ValueError("BM25 statistics require matching tokenizer and document_frequencies")
    if any(not isinstance(term, str) or type(df) is not int or not 0 <= df <= n
           for term, df in frequencies.items()):
        raise ValueError("BM25 document frequencies must lie in [0, document_count]")
    features = []
    for row in pool:
        terms = set(tokenize(row["query_text"]))
        counts = term_counts[row["doc_id"]]
        length = sum(counts.values())
        score = 0.0
        for term in sorted(terms):
            tf = counts[term]
            if not tf:
                continue
            # External statistics must actually cover documents being scored.
            if not frequencies.get(term, 0):
                raise ValueError(f"BM25 statistics omit a matching document term: {term}")
            df = frequencies[term]
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            normalization = 1 - config.bm25_b + config.bm25_b * length / (avgdl or 1.0)
            score += idf * tf * (config.bm25_k1 + 1) / (tf + config.bm25_k1 * normalization)
        overlap = len(terms.intersection(counts)) / len(terms) if terms else 0.0
        features.append([score, overlap, math.log1p(length)])
    return np.asarray(features, dtype=np.float64), corpus_stats


class LabelOracle:
    """Simulation boundary: disclose exactly one binary label per acquired pair."""

    def __init__(self, labels, budget):
        self._labels = labels
        self.budget = budget
        self.queried = []
        self._seen = set()

    def query(self, key):
        if key in self._seen or len(self.queried) >= self.budget:
            raise ValueError("Duplicate label query or exhausted annotation budget")
        value = self._labels[key]
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Oracle labels must be binary integers")
        self._seen.add(key)
        self.queried.append(key)
        return value


def fit_probabilities(features, observed, config, seed):
    """Use a smoothed constant before both classes have been observed."""
    if not observed:
        return np.full(len(features), 0.5), {"kind": "untrained", "probability": 0.5}
    indices = list(observed)
    labels = [observed[i] for i in indices]
    if len(set(labels)) == 1:
        probability = (sum(labels) + 1) / (len(labels) + 2)
        return np.full(len(features), probability), {"kind": "single_class_prior", "probability": probability}
    model = LogisticRegression(C=config.regularization_c, solver="lbfgs", penalty="l2",
                               max_iter=config.max_iter, tol=1e-8, random_state=seed)
    # Do not quietly publish probabilities from a failed optimization.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features[indices], labels)
    return model.predict_proba(features)[:, 1], {
        "kind": "logistic_regression", "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]), "iterations": int(model.n_iter_[0]),
    }


def run_active_learning(pool, oracle, budget, config=None, seed=42, features=None):
    """Budgeted transductive SAL: random seed, uncertainty batches, final refit.

    `pool` must contain text/IDs only. This is annotation of an available pool,
    not a claim of zero-label prediction on unseen queries. The oracle's budget
    includes the random seed and every subsequent acquisition.
    """
    config = config or ActiveLearningConfig()
    if any("true_binary" in r or "true_relevance" in r for r in pool):
        raise ValueError("Strip gold labels before passing the pool to the learner")
    validate_pool(pool)
    keys = [(r["query_id"], r["doc_id"]) for r in pool]
    if keys != sorted(keys):
        raise ValueError("Sort the pool by (query_id, doc_id) for reproducible acquisition")
    if type(budget) is not int or not 0 <= budget <= len(pool) or oracle.budget != budget:
        raise ValueError("Budget must be an integer in [0, pool size] and match the oracle")
    if oracle.queried:
        raise ValueError("A fresh oracle is required for each independent experiment")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if features is None:
        features, _ = lexical_features(pool, config)
    features = np.asarray(features, dtype=np.float64)
    if features.shape != (len(pool), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("Expected a finite, aligned three-column feature matrix")
    scaler = StandardScaler().fit(features)  # unlabeled transductive pool only
    scaled = scaler.transform(features)
    rng = np.random.default_rng(seed)
    # Fixed random priorities break uncertainty ties without depending on input order.
    random_order = rng.permutation(len(pool))
    priority = np.empty(len(pool), dtype=int)
    priority[random_order] = np.arange(len(pool))
    observed, acquisitions = {}, []

    def acquire(indices, stage, round_number, probabilities):
        for index in indices:
            index = int(index)
            value = oracle.query(keys[index])
            observed[index] = value
            acquisitions.append({"query_id": keys[index][0], "doc_id": keys[index][1],
                                 "acquisition_rank": len(acquisitions) + 1, "round": round_number,
                                 "stage": stage, "prob_relevant_before_query": float(probabilities[index]),
                                 "observed_label": value})

    acquire(random_order[:min(config.seed_size, budget)], "random_seed", 0, np.full(len(pool), .5))
    probabilities, state = fit_probabilities(scaled, observed, config, seed)
    round_number = 0
    while len(observed) < budget:
        remaining = np.asarray([i for i in range(len(pool)) if i not in observed])
        # With a single observed class, a constant prior cannot rank uncertainty.
        uncertainty = np.abs(probabilities[remaining] - .5)
        order = np.lexsort((priority[remaining], uncertainty))
        count = min(config.batch_size, budget - len(observed))
        round_number += 1
        acquire(remaining[order[:count]], "uncertainty", round_number, probabilities)
        probabilities, state = fit_probabilities(scaled, observed, config, seed)
    rows = []
    rank_by_key = {(a["query_id"], a["doc_id"]): a["acquisition_rank"] for a in acquisitions}
    for index, row in enumerate(pool):
        p = float(probabilities[index])
        prediction = int(p >= .5)
        human = index in observed
        rows.append({"query_id": keys[index][0], "doc_id": keys[index][1],
                     "prob_relevant": p, "predicted_label": prediction, "confidence": max(p, 1-p),
                     "label_source": "human" if human else "automatic",
                     "acquisition_rank": rank_by_key.get(keys[index], ""),
                     "final_label": observed[index] if human else prediction})
    return rows, acquisitions, {"config": asdict(config), "seed": seed, "budget": budget,
                                "feature_names": list(FEATURE_NAMES),
                                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                                "model": state, "rounds": round_number}
