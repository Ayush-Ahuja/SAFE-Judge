"""Validated adapters for the frozen pilot and the public DIRAS climate test."""

import csv
import hashlib
import json
from pathlib import Path
import re

from src.utils.io import file_hash, pair_key, read_manifest, read_predictions


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_datasets(root, partial_positive=True):
    pred_path = root/"results/stage2/dl2019_qwen35_full_v1/predictions.csv"
    ce_path = root/"results/baselines/cross_encoder/dl2019_minilm_scores_v1/scores.csv"
    source, _ = read_predictions(pred_path)
    auxiliary = {pair_key(r): float(r["raw_logit"]) for r in _csv(ce_path)}
    if set(auxiliary) != {pair_key(r) for r in source}:
        raise ValueError("DL2019 assessor/auxiliary pairs differ")
    dl = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": r["prob_relevant"],
           "auxiliary_logit": auxiliary[pair_key(r)], "true_binary": r["true_binary"]} for r in source]
    cache = root/"cache/research/diras_795ef8d"
    needed = ["data/chatreport_test.csv", "data/chatreport_test_results.csv", "small_llm_results/gc_phi3_2e.csv"]
    downloads = json.loads((cache/"downloads.json").read_text(encoding="utf-8"))
    records = {r["path"]: r for r in downloads}
    for name in needed:
        if file_hash(cache/name) != records[name]["sha256"]:
            raise ValueError("DIRAS download hash mismatch")
    gold, retrieval, model = [_csv(cache/name) for name in needed]
    if not len(gold) == len(retrieval) == len(model) == 660:
        raise ValueError("Unexpected DIRAS test size")
    manifest_path = root/"data/research/chatreportretrieve.jsonl"
    manifest = read_manifest(manifest_path)
    ce_chat = root/"results/experiments/research_chatreport_minilm_v1/scores.csv"
    auxiliary_chat = {pair_key(r): float(r["raw_logit"]) for r in _csv(ce_chat)}
    if set(auxiliary_chat) != {pair_key(r) for r in manifest}:
        raise ValueError("Climate auxiliary predictions do not match manifest")
    climate = []
    ranking_fields = ("small_embed_score", "large_embed_score", "base_scores", "large_scores", "rerank_scores")
    for h, r, m, canonical in zip(gold, retrieval, model, manifest):
        if (h["Questions"] != r["Questions"] or h["Paragraphs"] != r["Paragraphs"]
                or h["Questions"] != canonical["query_text"] or h["Paragraphs"] != canonical["doc_text"]
                or h["Questions"] not in m["output"] or h["Paragraphs"] not in m["output"]):
            raise ValueError("DIRAS row/text alignment failed")
        matches = re.findall(r"\[Guess\]:\s*(Yes|No).*?\[Confidence\]:\s*([0-9.]+)", m["output"], re.I|re.S)
        if not matches:
            raise ValueError("Unparseable compact-model answer")
        guess, confidence = matches[-1]
        confidence = float(confidence.rstrip("."))
        if not .5 <= confidence <= 1:
            raise ValueError("Answer confidence below 0.5 requires explicit label handling")
        p = confidence if guess.lower() == "yes" else 1-confidence
        y = int(h["gold"].lower() == "yes" or (partial_positive and h["gold"].lower() == "partially"))
        climate.append({"query_id": canonical["query_id"], "doc_id": canonical["doc_id"], "prob_relevant": p,
                        "auxiliary_logit": auxiliary_chat[pair_key(canonical)], "true_binary": y,
                        "human_partial": h["gold"].lower() == "partially", "human_uncertain": int(h["hard"]),
                        "ranking_scores": {name: float(r[name]) for name in ranking_fields}})
    metadata = {"dl2019": {"pairs": len(dl), "queries": len({r['query_id'] for r in dl}),
                            "assessor": "Frozen Qwen3.5-4B canonical token probabilities",
                            "predictions_sha256": file_hash(pred_path), "auxiliary_sha256": file_hash(ce_path)},
                "chatreport": {"pairs": len(climate), "queries": len({r['query_id'] for r in climate}),
                               "assessor": "DIRAS gc_phi3_2e cached verbalized confidence; no new fine-tuning",
                               "partial_relevance_positive": partial_positive, "partial_pairs": 65,
                               "source_commit": "795ef8d20fe2729d313ca2d1f7cc6c096ea9c714",
                               "files": [records[name] for name in needed],
                               "manifest_sha256": file_hash(manifest_path), "auxiliary_sha256": file_hash(ce_chat),
                               "ranking_scope": "Five published scorer variants over the fixed 60-pair/query test candidate pools"},
                "comparison_scope": "Two-domain replication with DIFFERENT assessors; not an isolated same-model domain-shift experiment"}
    return {"dl2019": dl, "chatreport": climate}, metadata
