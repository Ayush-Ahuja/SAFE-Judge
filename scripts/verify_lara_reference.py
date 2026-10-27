"""Compare binary LARA against a checksum-pinned upstream source snapshot.

Reference source is cached, not vendored. Only named class/function definitions
are loaded; upstream experiment dispatch and imports are not executed.
"""

import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import sys
import urllib.request

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_lara_baseline import load_inputs
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.utils.io import file_hash, pair_key, read_manifest, write_json


class QuietProgress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def update(self, *args):
        pass


def synthetic_case():
    probabilities = [.02, .05, .1, .15, .22, .3, .35, .4, .45, .49,
                     .51, .56, .62, .68, .75, .83, .9, .95, .98, .99]
    labels = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    pool = [{"query_id": f"q{i//5}", "doc_id": f"d{i:02d}", "prob_relevant": p}
            for i, p in enumerate(probabilities)]
    return pool, labels


def load_reference(cache, reference, download=False):
    # pandas is an optional verification dependency, absent from the CI baseline.
    import pandas as pd
    namespace = {"np": np, "pd": pd, "beta": beta, "random": random, "tqdm": QuietProgress,
                 "count_in_ranges": lambda x: np.histogram(x, bins=np.linspace(0, 1, 11))}
    for source, digest in reference["files"].items():
        path = cache / Path(source).name
        if download:
            cache.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/RikiyaT/LARA/{reference['commit']}/{source}"
            request = urllib.request.Request(url, headers={"User-Agent": "SAFE-Judge-reference-verification"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        if file_hash(path) != digest:
            raise ValueError(f"Pinned upstream checksum mismatch: {path}")
        names = reference["binary_symbols"] if source.endswith("algo.py") else [reference["driver_symbol"]]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if len(definitions) != len(names):
            raise ValueError("Pinned upstream symbols are missing")
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def compare_case(namespace, pool, labels, preprocessing, grouping, budget, seed=42):
    import pandas as pd
    data = pd.DataFrame({"topic_id": [r["query_id"] for r in pool], "doc_id": [r["doc_id"] for r in pool],
                         "pi": [r["prob_relevant"] for r in pool], "annotation": labels})
    if preprocessing == "upstream_smooth":
        data = namespace["smooth_normalize"](data, smoothness=5)
    original_class = namespace["GradualLogisticRegression"]
    instances, selections = [], []

    class TrackedCalibrator(original_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)
        def sample_next(self, available):
            selected = super().sample_next(available)
            selections.append(int(selected.name))
            return selected

    namespace["GradualLogisticRegression"] = TrackedCalibrator
    try:
        with redirect_stdout(io.StringIO()):
            if grouping == 1:
                _, final = namespace["ours"](data, annotation_budget=budget, learning_rate=.01,
                                               batch_size=10, lambda_=200, random_state=seed)
            else:
                count = data.topic_id.nunique() if grouping == "topics" else grouping
                _, final = namespace["ours_groups"](data, annotation_budget=budget, learning_rate=.01,
                                                       batch_size=10, lambda_=200, random_state=seed, n_groups=count)
    finally:
        namespace["GradualLogisticRegression"] = original_class
    oracle = LabelOracle({pair_key(r): int(y) for r, y in zip(pool, labels)}, budget)
    ours, acquisitions, state = run_lara(pool, oracle, budget, replace(LaraConfig(), preprocessing=preprocessing), seed, grouping)
    indices = {pair_key(r): i for i, r in enumerate(pool)}
    actual_selection = [indices[pair_key(r)] for r in acquisitions]
    expected_probability = np.asarray(instances[0].predict_proba(data.pi), dtype=float)
    actual_probability = np.asarray([r["prob_relevant"] for r in ours])
    if selections != actual_selection:
        raise AssertionError(f"Acquisition trace differs: {preprocessing}, groups={grouping}, B={budget}")
    np.testing.assert_allclose(actual_probability, expected_probability, rtol=0, atol=1e-12)
    if [r["final_label"] for r in ours] != final.pi.astype(int).tolist():
        raise AssertionError("Final mixed labels differ from upstream")
    return {"preprocessing": preprocessing, "grouping": grouping, "budget": budget, "seed": seed,
            "selected_indices": selections, "probabilities": expected_probability.tolist(),
            "final_labels": final.pi.astype(int).tolist(),
            "maximum_absolute_probability_difference": float(np.max(np.abs(actual_probability-expected_probability)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "cache/lara-reference")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-fixture", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    reference = json.loads((ROOT / "configs/lara_reference.json").read_text(encoding="utf-8"))
    namespace = load_reference(args.source_cache, reference, args.download)
    pool, labels = synthetic_case()
    cases = [compare_case(namespace, pool, labels, mode, groups, budget)
             for mode in ("raw", "upstream_smooth") for groups in (1, 3, "topics") for budget in (0, 1, 7, 12)]
    if args.write_fixture:
        write_json(args.write_fixture, {"upstream": reference, "pool": pool, "labels": labels, "cases": cases})
    real_cases = []
    if (args.manifest is None) != (args.predictions is None):
        parser.error("Manifest and prediction arguments must be supplied together")
    if args.manifest:
        manifest = read_manifest(args.manifest)
        probabilities, _ = load_inputs(args.predictions, manifest, "project")
        counts, selected = {}, []
        for row in sorted(manifest, key=pair_key):
            qid = row["query_id"]
            if counts.get(qid, 0) < 3:
                selected.append(row)
                counts[qid] = counts.get(qid, 0)+1
        real_pool = [{"query_id": r["query_id"], "doc_id": r["doc_id"], "prob_relevant": probabilities[pair_key(r)]}
                     for r in selected]
        real_labels = [int(r["true_binary"]) for r in selected]
        real_cases = [compare_case(namespace, real_pool, real_labels, mode, groups, budget)
                      for mode in ("raw", "upstream_smooth") for groups in (1, 3, "topics") for budget in (0, 1, 7, 23)]
    report = {"status": "passed", "upstream": reference, "synthetic_cases": len(cases),
              "real_cases": len(real_cases), "exact_acquisition_and_final_label_agreement": True,
              "maximum_absolute_probability_difference": max(c["maximum_absolute_probability_difference"] for c in cases+real_cases),
              "exclusions": "N-1 budget shortcut and constant normalization have explicit correctness fixes; not claimed identical.",
              "cases": cases+real_cases}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "upstream")}))


if __name__ == "__main__":
    main()
