"""Freeze judged pairs/text before GPU inference; missing text is a fatal error."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.data_loader import DATASETS, default_fields, select_text
from src.utils.io import atomic_text, file_hash, write_json


def reserve_audit_topics(topics, path, count=10, seed=42):
    topics = sorted(set(topics))
    if not 0 < count < len(topics):
        raise ValueError("Audit topic count must be positive and smaller than the collection")
    expected = {"dataset": "trec_covid", "seed": seed, "all_topic_ids": topics,
                "blind_audit_topic_ids": sorted(random.Random(seed).sample(topics, count))}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing blind-audit split differs. Preserve it; do not redraw audit topics.")
    else:
        write_json(path, expected)
    return set(expected["blind_audit_topic_ids"])


def select_qrels(qrels, seed=42, n_queries=0, max_pairs_per_query=0, excluded_topics=()):
    if n_queries < 0 or max_pairs_per_query < 0:
        raise ValueError("Sample sizes must be nonnegative")
    groups, seen = {}, set()
    for qrel in qrels:
        qid, did, grade = str(qrel.query_id), str(qrel.doc_id), int(qrel.relevance)
        if (qid, did) in seen:
            raise ValueError(f"Duplicate qrel {(qid, did)}")
        seen.add((qid, did))
        if grade >= 0 and qid not in excluded_topics:
            groups.setdefault(qid, []).append((qid, did, grade))
    rng = random.Random(seed)
    topics = sorted(groups)
    if n_queries > len(topics):
        raise ValueError(f"Requested {n_queries} queries, only {len(topics)} eligible")
    if n_queries:
        topics = sorted(rng.sample(topics, n_queries))
    selected = []
    for qid in topics:
        pairs = sorted(groups[qid])
        if max_pairs_per_query and len(pairs) > max_pairs_per_query:
            pairs = rng.sample(pairs, max_pairs_per_query)
        selected.extend(pairs)
    rng.shuffle(selected)
    if not selected:
        raise ValueError("No eligible judged pairs")
    return selected


def collect_document_text(dataset, document_ids, fields, access="store"):
    needed = set(document_ids)
    if access == "store":
        docs = dataset.docs_store()
        return {did: select_text(docs.get(did), fields) for did in sorted(needed)}
    if access != "stream":
        raise ValueError("Document access must be store or stream")
    # A one-off pilot need not build an index over all 8.8 million passages.
    # This still uses the canonical dataset text and may download the corpus.
    selected = {}
    for doc in dataset.docs_iter():
        did = str(doc.doc_id)
        if did in needed:
            selected[did] = select_text(doc, fields)
            if len(selected) == len(needed):
                break
    missing = needed - selected.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} selected documents from corpus")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[name for name in DATASETS if name != "msmarco_train"], default="trec_dl_2019")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-queries", type=int, default=0, help="0 selects every eligible query")
    parser.add_argument("--max-pairs-per-query", type=int, default=0, help="0 selects every judged pair")
    parser.add_argument("--doc-access", choices=("store", "stream"), default="store",
                        help="stream avoids building a full random-access document index; corpus download may still be needed")
    parser.add_argument("--query-fields", nargs="+")
    parser.add_argument("--doc-fields", nargs="+")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        parser.error("Output already exists. Reuse the frozen manifest or choose a new filename.")
    import ir_datasets
    dataset = ir_datasets.load(DATASETS[args.dataset])
    qrels = list(dataset.qrels_iter())
    excluded = set()
    audit_path = ROOT / "data/splits/trec_covid_blind_audit.json"
    if args.dataset == "trec_covid":
        excluded = reserve_audit_topics([str(r.query_id) for r in qrels if r.relevance >= 0], audit_path)
    selected = select_qrels(qrels, args.seed, args.n_queries, args.max_pairs_per_query, excluded)
    default_query, default_doc = default_fields(args.dataset)
    query_fields, doc_fields = args.query_fields or default_query, args.doc_fields or default_doc
    topic_ids = {qid for qid, _, _ in selected}
    queries = {str(q.query_id): select_text(q, query_fields) for q in dataset.queries_iter()
               if str(q.query_id) in topic_ids}
    if set(queries) != topic_ids:
        raise ValueError("Selected qrels contain query IDs with no query text")
    document_text = collect_document_text(dataset, {did for _, did, _ in selected}, doc_fields, args.doc_access)
    threshold = 1 if args.dataset == "trec_covid" else 2
    rows = [{"dataset": args.dataset, "query_id": qid, "doc_id": did,
             "query_text": queries[qid], "doc_text": document_text[did],
             "true_relevance": grade, "relevance_threshold": threshold,
             "true_binary": int(grade >= threshold)} for qid, did, grade in selected]
    atomic_text(args.output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(args.output.with_suffix(".metadata.json"), {
        "dataset": args.dataset, "ir_datasets_id": DATASETS[args.dataset],
        "seed": args.seed, "query_ids": sorted(topic_ids), "pairs": len(rows),
        "max_pairs_per_query": args.max_pairs_per_query, "query_fields": query_fields,
        "document_access": args.doc_access,
        "doc_fields": doc_fields, "relevance_threshold": threshold,
        "excluded_audit_topic_ids": sorted(excluded),
        "audit_split_sha256": file_hash(audit_path) if excluded else None,
        "unjudged_qrels_excluded": sum(r.relevance < 0 for r in qrels),
        "manifest_sha256": file_hash(args.output),
    })
    print(f"Frozen {len(rows)} pairs from {len(topic_ids)} queries -> {args.output}")


if __name__ == "__main__":
    main()
