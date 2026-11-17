"""Reproduce selected Figure 2 points with the unmodified released methods."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.oosterhuis_inputs import load_reference
from src.utils.io import atomic_text, file_hash, write_csv, write_json


def execute(output):
    settings, cache, (datasets, methods, utils) = load_reference()
    design = settings["reference_experiment"]
    output.mkdir(parents=True, exist_ok=False)
    data = datasets.read_datasets(cache/"data")
    metadata = {"status": "running", "settings": design, "reference_revision": settings["revision"],
                "reference_file_sha256": {r["path"]: r["sha256"] for r in settings["files"]},
                "config_sha256": file_hash(ROOT/"configs/oosterhuis_reference.json"),
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
                    (Path(__file__), ROOT/"scripts/oosterhuis_inputs.py")},
                "evaluation": "Author-provided validation/test splits, graded Flan-UL2 probabilities, BM25 DCG@10, unmodified bootstrap/PPI/CRC; interval coverage differs from automatic-label coverage",
                "datasets": {name: {"validation_queries": int(ds["vali"].num_queries), "test_queries": int(ds["test"].num_queries),
                    "test_gold_dcg": ds["test"].true_dcg(), "test_predicted_dcg": ds["test"].pred_dcg()} for name, ds in data.items()}}
    write_json(output/"experiment.json", metadata)
    records = []
    start = time.perf_counter()
    algorithms = {"bootstrap": methods.bootstrap, "ppi": methods.ppi, "crc": methods.crc_bootstrap}
    for name, ds in data.items():
        for size in design["calibration_queries"]:
            for seed in range(design["first_seed"], design["first_seed"]+design["repetitions"]):
                for method, function in algorithms.items():
                    result = utils.run_method(ds["vali"], ds["test"], function, seed=seed, vali_size=size)
                    lower, upper = float(result["lower"]), float(result["upper"])
                    assert np.isfinite([lower, upper]).all() and lower <= upper
                    records.append({"dataset": name, "calibration_queries": size, "seed": seed, "method": method,
                                    "lower": lower, "upper": upper, "width": upper-lower,
                                    "covered": bool(result["coverage"]), "true_dcg": float(result["true_dcg"]),
                                    "raw_predicted_dcg": float(result["pred_dcg"]),
                                    "midpoint_absolute_error": abs((lower+upper)/2-float(result["true_dcg"]))})
            print(f"Reference {name}: {size} calibration queries complete", flush=True)
    summary = []
    for name in data:
        for size in design["calibration_queries"]:
            for method in algorithms:
                rows = [r for r in records if r["dataset"] == name and r["calibration_queries"] == size and r["method"] == method]
                count = sum(r["covered"] for r in rows)
                interval = binomtest(count, len(rows)).proportion_ci()
                summary.append({"dataset": name, "calibration_queries": size, "method": method, "repetitions": len(rows),
                                "interval_coverage": count/len(rows), "coverage_binomial_lower": interval.low,
                                "coverage_binomial_upper": interval.high, "mean_width": np.mean([r["width"] for r in rows]),
                                "mean_midpoint_absolute_error": np.mean([r["midpoint_absolute_error"] for r in rows])})
    write_csv(output/"runs.csv", records)
    write_csv(output/"summary.csv", summary)
    metadata.update(status="complete", evaluations=len(records), elapsed_seconds=time.perf_counter()-start)
    write_json(output/"experiment.json", metadata)
    atomic_text(output/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file()))
    print(f"Completed {len(records)} reference interval evaluations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    execute(parser.parse_args().output_dir)
