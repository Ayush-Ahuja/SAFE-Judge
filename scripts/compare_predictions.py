"""Compare two complete reruns without requiring bit-for-bit GPU determinism."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import pair_key, read_predictions, write_json


def compare(reference, repeat):
    reference = {pair_key(r): r for r in reference}
    repeat = {pair_key(r): r for r in repeat}
    if not reference or reference.keys() != repeat.keys():
        raise ValueError("Reruns must contain exactly the same nonempty pair set")
    if any(reference[k]["true_relevance"] != repeat[k]["true_relevance"] for k in reference):
        raise ValueError("Gold labels changed between reruns")
    differences = [abs(reference[k]["prob_relevant"] - repeat[k]["prob_relevant"]) for k in reference]
    changed = [k for k in reference if reference[k]["predicted_label"] != repeat[k]["predicted_label"]]
    return {"pairs": len(reference), "changed_labels": len(changed),
            "label_agreement": 1 - len(changed) / len(reference),
            "mean_absolute_probability_difference": sum(differences) / len(differences),
            "max_absolute_probability_difference": max(differences),
            "changed_pair_ids": [{"query_id": q, "doc_id": d} for q, d in changed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiments = [json.loads((p.parent / "experiment.json").read_text(encoding="utf-8"))
                   for p in (args.reference, args.repeat)]
    if experiments[0] != experiments[1]:
        raise ValueError("Experiment identities differ; use the same manifest, code and frozen config")
    experiment = experiments[0]
    rows = [read_predictions(p, experiment["relevance_threshold"])[0] for p in (args.reference, args.repeat)]
    if any(len(r) != experiment["expected_pairs"] for r in rows):
        raise ValueError("Both runs must be complete before comparison")
    dtypes = [{r.get("compute_dtype") for r in group} for group in rows]
    if dtypes[0] != dtypes[1] or len(dtypes[0]) != 1 or None in dtypes[0]:
        raise ValueError("Runs must have one matching recorded compute dtype")
    result = compare(*rows)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
