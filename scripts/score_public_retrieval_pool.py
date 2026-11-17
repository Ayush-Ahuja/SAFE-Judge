"""Create public DL2019 retrieval runs and rescore their candidates without gold."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cross_encoder import CrossEncoderConfig, MiniLMCrossEncoder
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


def execute(output):
    cache = ROOT/"cache/research/dl19_public_runs"
    sources = json.loads((cache/"downloads.json").read_text(encoding="utf-8"))
    pools, text = {}, {}
    for source in sources["files"]:
        path = cache/source["filename"]
        if file_hash(path) != source["sha256"]:
            raise ValueError("Public retrieval cache hash mismatch")
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        pools[source["method"]] = entries
        for entry in entries:
            qid, query = str(entry["query"]["qid"]), entry["query"]["text"]
            for candidate in entry["candidates"]:
                row = {"query_id": qid, "doc_id": str(candidate["docid"]), "query_text": query,
                       "doc_text": candidate["doc"]["contents"]}
                key = pair_key(row)
                if key in text and text[key] != row:
                    raise ValueError("Inconsistent public candidate text")
                text[key] = row
    output.mkdir(parents=True, exist_ok=False)
    config = CrossEncoderConfig()
    metadata = {"status": "running", "public_release": sources, "candidate_pairs": len(text),
                "scope": "Two public top-100 retrieval outputs and four rerankings; NOT all official TREC runs",
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__), ROOT/"src/baselines/cross_encoder.py")}}
    write_json(output/"experiment.json", metadata)
    backend = MiniLMCrossEncoder(config, "cpu", ROOT/"cache/huggingface", True)
    scored = []
    rows = sorted(text.values(), key=pair_key)
    start = time.perf_counter()
    for offset in range(0, len(rows), 32):
        scored.extend(backend.score_batch(rows[offset:offset+32]))
        if offset % 640 == 0:
            print(f"Scored {min(offset+32, len(rows))}/{len(rows)} public candidates", flush=True)
    write_csv(output/"scores.csv", scored)
    neural = {pair_key(r): r["raw_logit"] for r in scored}
    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, max_features=50000)
    matrix = vectorizer.fit_transform([r["doc_text"] for r in rows])
    query_matrix = vectorizer.transform([r["query_text"] for r in rows])
    cosine = np.asarray(matrix.multiply(query_matrix).sum(axis=1)).ravel()
    lexical = {pair_key(r): float(score) for r, score in zip(rows, cosine)}
    run_dir = output/"runs"
    run_dir.mkdir()
    for method, entries in pools.items():
        for scorer in ("original", "minilm", "tfidf"):
            lines = []
            tag = method.lower()+"_"+scorer
            for entry in entries:
                qid = str(entry["query"]["qid"])
                candidates = entry["candidates"]
                if scorer == "original":
                    ordered = sorted(candidates, key=lambda c: (float(c["score"]), str(c["docid"])), reverse=True)
                else:
                    lookup = neural if scorer == "minilm" else lexical
                    ordered = sorted(candidates, key=lambda c: (lookup[(qid, str(c["docid"]))], str(c["docid"])), reverse=True)
                for rank, candidate in enumerate(ordered, 1):
                    lines.append(f"{qid} Q0 {candidate['docid']} {rank} {101-rank} {tag}\n")
            atomic_text(run_dir/(tag+".trec"), "".join(lines))
    metadata.update(status="complete", elapsed_seconds=time.perf_counter()-start,
                    model=backend.provenance, truncated_pairs=sum(r["truncated"] for r in scored),
                    run_files=[p.name for p in sorted(run_dir.iterdir())])
    write_json(output/"experiment.json", metadata)
    outputs = sorted(p for p in output.rglob("*") if p.is_file())
    atomic_text(output/"checksums.sha256", "".join(f"{file_hash(p)}  {p.relative_to(output)}\n" for p in outputs))
    print(f"Created {len(metadata['run_files'])} retrieval-system runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
