"""Freeze Colab inputs from local public-source caches; never sample by gold label.

DL2020 archive: https://www.cs.unh.edu/~dietz/autograder/data-dl20.tar.xz
Official qrels: https://trec.nist.gov/data/deep/2020qrels-pass.txt?download=1
Download those two files into cache/research/transfer_inputs before running.
COVID and ChatReport reuse the project's already frozen research manifests.
"""

from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import atomic_text, file_hash, read_manifest, write_json


def main():
    cache = ROOT / "cache/research/transfer_inputs"
    target = ROOT / "data/transfer"
    target.mkdir(parents=True, exist_ok=True)
    archive = cache / "dl20_texts.tar.xz"
    qrels_file = cache / "2020qrels-pass.txt"
    official = {}
    for line in qrels_file.read_text(encoding="utf-8").splitlines():
        qid, _, did, grade = line.split()
        key = (qid, did)
        if key in official:
            raise ValueError(f"Duplicate official qrel {key}")
        official[key] = int(grade)
    assert len(official) == 11386 and len({q for q, _ in official}) == 54
    rows = []
    with tarfile.open(archive) as tar:
        queries = json.load(tar.extractfile("data/dl20/dl20-queries.json"))
        with gzip.GzipFile(fileobj=tar.extractfile("data/dl20/trecDL2020-qrels-runs-with-text.jsonl.gz")) as stream:
            for line in stream:
                qid, passages = json.loads(line)
                for passage in passages:
                    did = str(passage["paragraph_id"])
                    if (qid, did) not in official:
                        continue
                    grade = official[qid, did]
                    embedded = [int(j["relevance"]) for j in passage["paragraph_data"]["judgments"]
                                if str(j["query"]) == qid]
                    assert embedded == [grade], (qid, did, embedded, grade)
                    rows.append({"dataset": "trec_dl_2020", "rubric_name": "dl2020", "query_id": qid,
                                 "doc_id": did, "query_text": queries[qid].strip(), "doc_text": passage["text"].strip(),
                                 "true_relevance": grade, "true_binary": int(grade >= 2), "relevance_threshold": 2})
    assert len(rows) == len(official) and {(r["query_id"], r["doc_id"]) for r in rows} == set(official)
    rows.sort(key=lambda r: (int(r["query_id"]), int(r["doc_id"])))
    provenance = {
        "dl2020": {"text_source_url": "https://www.cs.unh.edu/~dietz/autograder/data-dl20.tar.xz",
                   "text_archive_sha256": file_hash(archive),
                   "qrels_source_url": "https://trec.nist.gov/data/deep/2020qrels-pass.txt?download=1",
                   "qrels_sha256": file_hash(qrels_file),
                   "validation": "Every pair and grade matched independently downloaded official NIST qrels. No autograder labels used.",
                   "text_preprocessing": "Strip surrounding query/document whitespace, matching the existing DL2019 exporter.",
                   "scope": "All 11,386 judged passage pairs across 54 queries; not full-corpus inference."},
        "trec_covid": {"original_manifest_sha256": file_hash(ROOT / "data/research/trec_covid_5000.jsonl"),
                       "sampling": json.loads((ROOT / "data/research/trec_covid_5000.metadata.json").read_text()),
                       "scope": "Frozen 100-pairs-per-query judged-pool sample across all 50 queries; title plus abstract, not full text.",
                       "label_mapping": "0 and 1 -> No; 2 -> Yes (strict fully-relevant target, not BEIR's >=1 convention)."},
        "chatreport": {"original_manifest_sha256": file_hash(ROOT / "data/research/chatreportretrieve.jsonl"),
                       "source_url": "https://raw.githubusercontent.com/EdisonNi-hku/DIRAS/795ef8d20fe2729d313ca2d1f7cc6c096ea9c714/data/chatreport_test.csv",
                       "source_sha256": file_hash(ROOT / "cache/research/diras_795ef8d/data/chatreport_test.csv"),
                       "scope": "All 660 expert-labeled pairs, 11 questions, 60 passages per question. Question-only input; expert background omitted, as in the existing manifest.",
                       "label_mapping": "no=0 -> No; partial=1 and yes=2 -> Yes. Human uncertainty retained, not used by the scorer."},
    }
    all_rows = {"dl2020": rows}
    for name, source in (("trec_covid", "trec_covid_5000"), ("chatreport", "chatreportretrieve")):
        all_rows[name] = [{**r, "rubric_name": name} for r in read_manifest(ROOT / f"data/research/{source}.jsonl")]
    suite = {"schema_version": 1, "datasets": {}}
    for name, data in all_rows.items():
        path = target / f"{name}.jsonl"
        atomic_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
        read_manifest(path)
        metadata = {"pairs": len(data), "queries": len({r["query_id"] for r in data}),
                    "grade_counts": dict(sorted(Counter(r["true_relevance"] for r in data).items())),
                    "relevance_threshold": data[0]["relevance_threshold"], "rubric_name": name,
                    "manifest_sha256": file_hash(path), **provenance[name]}
        write_json(target / f"{name}.metadata.json", metadata)
        suite["datasets"][name] = {"manifest": f"data/transfer/{name}.jsonl", **metadata}
        print(name, metadata["queries"], metadata["pairs"], metadata["manifest_sha256"])
    write_json(target / "suite.json", suite)


if __name__ == "__main__":
    main()
