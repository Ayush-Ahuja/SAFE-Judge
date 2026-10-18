"""Export queries/qrels: python scripts/download_data.py --dataset eval_only.

Documents are fetched by prepare_pilot.py. Download/filesystem failures stay visible.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.data_loader import DATASETS, default_fields, select_text
from src.utils.io import atomic_text, write_json


def export_dataset(dataset, name, output_dir):
    query_fields, _ = default_fields(name)
    queries, records = {}, {}
    for query in dataset.queries_iter():
        qid = str(query.query_id)
        if qid in queries:
            raise ValueError(f"Duplicate query ID {qid}")
        queries[qid] = select_text(query, query_fields)
        records[qid] = {field: getattr(query, field) for field in query._fields}
    if not queries:
        raise ValueError(f"No queries exported for {name}; existing artifacts were not replaced")
    qrels, seen = [], set()
    for qrel in dataset.qrels_iter():
        qid, did = str(qrel.query_id), str(qrel.doc_id)
        if qid not in queries:
            raise ValueError(f"Qrel query {qid} has no query text")
        if (qid, did) in seen:
            raise ValueError(f"Duplicate qrel {(qid, did)}")
        seen.add((qid, did))
        qrels.append({"query_id": qid, "doc_id": did, "relevance": int(qrel.relevance)})
    if not qrels:
        raise ValueError("No qrels exported")
    directory = Path(output_dir) / name
    write_json(directory / "queries.json", queries)
    write_json(directory / "query_records.json", records)
    write_json(directory / "qrels.json", qrels)
    atomic_text(directory / "qrels.trec", "".join(
        f"{r['query_id']} 0 {r['doc_id']} {r['relevance']}\n" for r in qrels))
    write_json(directory / "export_metadata.json", {
        "dataset": name, "ir_datasets_id": DATASETS[name], "query_fields": query_fields,
        "queries": len(queries), "qrels": len(qrels),
        "unjudged_qrels": sum(r["relevance"] < 0 for r in qrels),
    })
    print(f"{name}: {len(queries)} queries, {len(qrels)} qrels -> {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all", "eval_only"], default="eval_only")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    import ir_datasets
    names = list(DATASETS) if args.dataset == "all" else (
        [name for name in DATASETS if name != "msmarco_train"] if args.dataset == "eval_only" else [args.dataset])
    for name in names:
        export_dataset(ir_datasets.load(DATASETS[name]), name, args.output_dir)
    print("Official runs are separate downloads; place under data/runs/<collection>/.")


if __name__ == "__main__":
    main()
