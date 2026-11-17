"""Load checksum-pinned public Oosterhuis code and numeric benchmark data."""
import importlib
import json
from pathlib import Path
import sys
import types
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import file_hash


def load_reference(download=False):
    config_path = ROOT/"configs/oosterhuis_reference.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    cache = ROOT/settings["cache_directory"]
    for source in settings["files"]:
        target = cache/source["path"]
        if not target.exists() and download:
            payload = urllib.request.urlopen(source["url"], timeout=45).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if not target.exists() or file_hash(target) != source["sha256"]:
            raise ValueError(f"Missing/mismatched reference input: {source['path']}")
    # Preserve upstream source and its absolute package imports without patching.
    package_name = "high_confidence_ir_eval_using_genai"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(cache)]
        sys.modules[package_name] = package
    modules = [importlib.import_module(package_name+"."+name) for name in ("datasets", "methods", "utils")]
    return settings, cache, modules


def binary_top10(reference_dataset, threshold=2):
    selected = np.flatnonzero(reference_dataset.ranks <= 10)
    rows = []
    probabilities = reference_dataset.pred_label_dist[:, threshold:].sum(axis=1)
    gold_grades = reference_dataset.true_label_dist.argmax(axis=1)
    for index in selected:
        rows.append({"query_id": str(reference_dataset.query_segments[index]), "doc_id": str(index),
                     "prob_relevant": float(probabilities[index]), "true_binary": int(gold_grades[index] >= threshold),
                     "rank": int(reference_dataset.ranks[index]), "retrieval_score": float(reference_dataset.scores[index])})
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    settings, _, _ = load_reference(args.download)
    print(f"Validated {len(settings['files'])} pinned Oosterhuis reference files")
