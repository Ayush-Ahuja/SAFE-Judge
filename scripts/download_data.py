"""
Download datasets for SAFE-Judge using ir_datasets.

Datasets:
  - MS MARCO Passage (training set) — for fine-tuning the assessor
  - TREC DL 2019/2020 (judged)     — source domain (web search)
  - TREC-COVID                      — shifted domain (biomedical)

Usage:
  python data/download_data.py --dataset all
  python data/download_data.py --dataset trec_dl_2019
"""

import argparse
import json
import os
from pathlib import Path

import ir_datasets
import ir_datasets.util
import contextlib
import os
import shutil

# --- MONKEY PATCH TO FIX WINDOWS [WinError 5] & [WinError 32] ---
# ir_datasets uses os.replace on .tmp files which violently crashes on Windows 
# if antivirus or indexing holds a lock. We bypass the .tmp file renaming entirely.
@contextlib.contextmanager
def patched_finialized_file(path, mode):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, mode) as f:
        yield f
    try:
        if os.path.exists(path):
            os.remove(path)
        shutil.move(tmp_path, path)
    except PermissionError:
        # If still locked, fallback to reading and rewriting or ignore if it's just temp cleanup
        try:
            os.system(f'move /Y "{tmp_path}" "{path}" >nul 2>&1')
        except Exception:
            pass

ir_datasets.util.finialized_file = patched_finialized_file
# -----------------------------------------------------------------


DATASETS = {
    "trec_dl_2019": "msmarco-passage/trec-dl-2019/judged",
    "trec_dl_2020": "msmarco-passage/trec-dl-2020/judged",
    "trec_covid": "cord19/trec-covid",
    "msmarco_train": "msmarco-passage/train",
}


def download_and_export(name: str, ir_id: str, output_dir: Path) -> None:
    """Download a dataset via ir_datasets and export qrels + queries to JSON."""
    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({ir_id})")
    print(f"{'='*60}")

    dataset = ir_datasets.load(ir_id)
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Export queries
    if dataset.has_queries():
        queries = {}
        for q in dataset.queries_iter():
            queries[q.query_id] = q.text if hasattr(q, "text") else str(q)
        queries_path = ds_dir / "queries.json"
        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        print(f"  Exported {len(queries)} queries -> {queries_path}")

    # Export qrels
    if dataset.has_qrels():
        qrels = []
        for qrel in dataset.qrels_iter():
            qrels.append({
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": qrel.relevance,
            })
        qrels_path = ds_dir / "qrels.json"
        with open(qrels_path, "w", encoding="utf-8") as f:
            json.dump(qrels, f, indent=2)
        print(f"  Exported {len(qrels)} qrels -> {qrels_path}")

        # Also export in standard TREC qrel format for trec_eval
        qrels_trec_path = ds_dir / "qrels.trec"
        with open(qrels_trec_path, "w") as f:
            for qrel in qrels:
                f.write(f"{qrel['query_id']} 0 {qrel['doc_id']} {qrel['relevance']}\n")
        print(f"  Exported TREC-format qrels -> {qrels_trec_path}")

    # Report doc count (don't download full corpus yet, just count)
    if dataset.has_docs():
        doc_count = dataset.docs_count()
        print(f"  Document corpus available: {doc_count:,} documents")
        print(f"  (Documents are cached by ir_datasets; access via dataset.docs_iter())")

    print(f"  Done: {name}")


def main():
    parser = argparse.ArgumentParser(description="Download SAFE-Judge datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASETS.keys()) + ["all", "eval_only"],
        help="Which dataset to download. 'eval_only' skips MS MARCO train.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for exported files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        targets = DATASETS
    elif args.dataset == "eval_only":
        targets = {k: v for k, v in DATASETS.items() if k != "msmarco_train"}
    else:
        targets = {args.dataset: DATASETS[args.dataset]}

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Datasets to download: {list(targets.keys())}")

    for name, ir_id in targets.items():
        download_and_export(name, ir_id, output_dir)

    print(f"\n{'='*60}")
    print("All downloads complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Download official TREC runs separately (see data/download_runs.sh)")
    print("  2. Place run files in data/runs/<collection_name>/")


if __name__ == "__main__":
    main()
