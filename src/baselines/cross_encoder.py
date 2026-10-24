"""Frozen MiniLM scoring and calibration/allocation primitives.

Torch and Transformers are imported only by the scorer; cached-score evaluation
and its tests run with the lightweight CPU dependencies already used by CI.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    max_length: int = 512
    batch_size: int = 32
    cpu_threads: int = 4
    temperature_min: float = .01
    temperature_max: float = 1000.0
    calibration_weighting: str = "pair"

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("A model identifier is required")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model revision must be a full immutable commit SHA")
        for name in ("max_length", "batch_size", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 3 <= self.max_length <= 512:
            raise ValueError("MiniLM requires max_length in [3, 512]")
        bounds = (self.temperature_min, self.temperature_max)
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in bounds):
            raise ValueError("Temperature bounds must be finite")
        if not 0 < bounds[0] <= 1 <= bounds[1] or bounds[0] == bounds[1]:
            raise ValueError("Temperature bounds must be positive, distinct, and contain 1")
        if self.calibration_weighting not in ("pair", "query"):
            raise ValueError("Calibration weighting must be pair or query")


def load_config(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    names = set(CrossEncoderConfig.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) - names - {"budget_fractions", "seeds", "include_endpoints"}:
        raise ValueError("Unknown cross-encoder configuration keys")
    return CrossEncoderConfig(**{k: v for k, v in data.items() if k in names}), data


def text_hash(row):
    return hashlib.sha256(json.dumps([row["query_text"], row["doc_text"]],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def text_pool(manifest):
    pool = []
    queries, documents, seen = {}, {}, set()
    for row in manifest:
        item = {k: row[k] for k in ("query_id", "doc_id", "query_text", "doc_text")}
        for name, value in item.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Nonempty strings required for {name}")
        key = item["query_id"], item["doc_id"]
        if key in seen or any(any(c.isspace() for c in identifier) for identifier in key):
            raise ValueError("Duplicate pair or whitespace in query/document ID")
        seen.add(key)
        for mapping, identifier, content in ((queries, key[0], item["query_text"]),
                                               (documents, key[1], item["doc_text"])):
            if identifier in mapping and mapping[identifier] != content:
                raise ValueError(f"Conflicting text for {identifier}")
            mapping[identifier] = content
        pool.append(item)
    if not pool:
        raise ValueError("Empty scoring pool")
    return sorted(pool, key=lambda r: (r["query_id"], r["doc_id"]))


def split_calibration_test(manifest, split=None, calibration_manifest=None):
    """Explicit query-disjoint split, including separate source/target collections."""
    external = calibration_manifest is not None
    source = calibration_manifest if external else manifest
    if split is None:
        if not external:
            raise ValueError("A query split or a separate calibration manifest is required")
        calibration, test = source, manifest
    else:
        groups = []
        for name in ("calibration_query_ids", "test_query_ids", "audit_query_ids"):
            values = split.get(name, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
                raise ValueError(f"Invalid {name}")
            groups.append(set(values))
        if not groups[0] or not groups[1] or any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError("Calibration/test groups must be nonempty and disjoint from audit topics")
        if not groups[0] <= {r["query_id"] for r in source} or not groups[1] <= {r["query_id"] for r in manifest}:
            raise ValueError("Split includes missing calibration/test queries")
        calibration = [r for r in source if r["query_id"] in groups[0]]
        test = [r for r in manifest if r["query_id"] in groups[1]]
    if not calibration or not test or {r["query_id"] for r in calibration} & {r["query_id"] for r in test}:
        raise ValueError("Calibration and test query IDs must be nonempty and disjoint")
    text_pool(calibration)
    text_pool(test)
    return sorted(calibration, key=lambda r: (r["query_id"], r["doc_id"])), sorted(test, key=lambda r: (r["query_id"], r["doc_id"]))


def valid_logits(values):
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError("Expected nonempty finite scalar logits")
    return logits


def binary_nll(logits, labels, temperature=1.0, weights=None):
    logits = valid_logits(logits)
    labels = np.asarray(labels)
    if labels.shape != logits.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary and aligned with logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    # This form avoids cancellation for confidently correct predictions.
    losses = np.logaddexp(0, -(2 * labels.astype(np.float64) - 1) * logits / temperature)
    return float(np.average(losses, weights=weights))


def fit_temperature(logits, labels, config=None, query_ids=None):
    """Convex binary NLL minimization in inverse temperature, with T > 0.

    brentq solves the monotone derivative. Boundary optima are returned explicitly;
    all-zero scores select T=1. Calibration labels are the only supervision.
    """
    config = config or CrossEncoderConfig()
    logits = valid_logits(logits)
    labels = np.asarray(labels, dtype=np.float64)
    before = binary_nll(logits, labels)
    weights = np.ones(len(logits), dtype=np.float64)
    if config.calibration_weighting == "query":
        if query_ids is None or len(query_ids) != len(logits):
            raise ValueError("Query weighting requires aligned calibration query IDs")
        from collections import Counter
        counts = Counter(query_ids)
        weights = np.asarray([1 / counts[q] for q in query_ids])
    weights /= weights.sum()

    def derivative(beta):
        return float(np.sum(weights * logits * (expit(beta * logits) - labels)))

    lower, upper = 1 / config.temperature_max, 1 / config.temperature_min
    if not np.any(logits):
        beta = 1.0
    elif derivative(lower) >= 0:
        beta = lower
    elif derivative(upper) <= 0:
        beta = upper
    else:
        beta = brentq(derivative, lower, upper, xtol=1e-12)
    temperature = float(1 / beta)
    nll_before = binary_nll(logits, labels, weights=weights)
    nll_after = binary_nll(logits, labels, temperature, weights)
    if nll_after > nll_before + 1e-10:
        raise RuntimeError("Temperature optimization increased calibration NLL")
    return {"temperature": temperature, "temperature_min": config.temperature_min,
            "temperature_max": config.temperature_max, "weighting": config.calibration_weighting,
            "optimizer": "brentq of convex inverse-temperature NLL derivative",
            "at_boundary": beta in (lower, upper), "calibration_pairs": len(logits),
            "calibration_both_classes": len(set(labels)) == 2,
            "calibration_nll_before": nll_before, "calibration_nll_after": nll_after,
            "calibration_pair_nll_before": before,
            "calibration_pair_nll_after": binary_nll(logits, labels, temperature)}


def prediction_rows(text_rows, logits, temperature):
    logits = valid_logits(logits)
    if len(logits) != len(text_rows) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Aligned logits and positive finite temperature required")
    probabilities = expit(logits / temperature)
    return [{"query_id": r["query_id"], "doc_id": r["doc_id"], "raw_logit": float(z),
             "prob_relevant": float(p), "predicted_label": int(z >= 0),
             "confidence": float(max(p, 1-p))} for r, z, p in zip(text_rows, logits, probabilities)]


def allocate_by_budget(predictions, test_human_budget, seed):
    """Rank by |raw logit| to avoid sigmoid saturation changing uncertainty ties."""
    if type(test_human_budget) is not int or not 0 <= test_human_budget <= len(predictions):
        raise ValueError("Test human budget must lie in [0, test pairs]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    keys = [(r["query_id"], r["doc_id"]) for r in predictions]
    if not keys or keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("Predictions must have unique sorted pair IDs")
    logits = valid_logits([r["raw_logit"] for r in predictions])
    priorities = np.random.default_rng(seed).permutation(len(predictions))
    order = np.lexsort((priorities, np.abs(logits)))
    return {int(i): rank + 1 for rank, i in enumerate(order[:test_human_budget])}


class MiniLMCrossEncoder:
    """Model-card Transformers path; returns raw scalar logits without sigmoid."""

    def __init__(self, config, device="cpu", cache_dir=None, local_files_only=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested device is unavailable; select cpu or an available cuda GPU")
        torch.set_num_threads(config.cpu_threads)
        self.torch, self.config, self.device = torch, config, device
        common = {"revision": config.revision, "cache_dir": cache_dir,
                  "local_files_only": local_files_only, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, **common)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, use_safetensors=True, torch_dtype=torch.float32, **common).to(device).eval()
        if self.model.config.num_labels != 1 or self.model.config._commit_hash != config.revision:
            raise ValueError("Expected the pinned single-score MS MARCO cross-encoder")
        self.provenance = {"model_name": config.model_name, "resolved_revision": self.model.config._commit_hash,
                           "device": device, "dtype": "float32", "tokenizer_fast": self.tokenizer.is_fast,
                           "truncation": "longest_first", "max_length": config.max_length,
                           "activation": "identity; raw model logits", "torch": torch.__version__}

    def score_batch(self, rows):
        if any("true_binary" in r or "true_relevance" in r for r in rows):
            raise ValueError("The model scorer accepts text and IDs only")
        queries = [r["query_text"].strip() for r in rows]
        documents = [r["doc_text"].strip() for r in rows]
        original = self.tokenizer(queries, documents, truncation=False, padding=False,
                                  return_length=True, verbose=False)["length"]
        encoded = self.tokenizer(queries, documents, padding=True, truncation="longest_first",
                                 max_length=self.config.max_length, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        with self.torch.inference_mode():
            values = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
        if tuple(values.shape) != (len(rows), 1):
            raise ValueError("Expected exactly one raw logit per input pair")
        logits = valid_logits(values[:, 0].detach().float().cpu().numpy())
        return [{"query_id": r["query_id"], "doc_id": r["doc_id"], "raw_logit": float(z),
                 "input_tokens": int(length), "original_tokens": int(full),
                 "truncated": bool(full > length), "text_sha256": text_hash(r)}
                for r, z, length, full in zip(rows, logits, lengths, original)]
