"""Compare verbalized and token confidence from the SAME DIRAS generations."""

import argparse
import csv
import io
import json
from pathlib import Path
import pickle
import re
import sys
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_inputs import load_datasets
from scripts.run_diagnostic_confirmation import probability_diagnostics
from scripts.run_error_predictor_ablation import aggregate, complete_budget
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.diagnostic_review import ReviewConfig, run_diagnostic_review
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import atomic_text, file_hash, pair_key, write_csv, write_json


class NumericOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "scalar":
            return np._core.multiarray.scalar
        if module == "numpy" and name == "dtype":
            return np.dtype
        raise ValueError(f"Rejected pickle global: {module}.{name}")


def read_token_verdicts(path):
    outputs = NumericOnlyUnpickler(io.BytesIO(path.read_bytes())).load()
    result = []
    for output in outputs:
        cumulative, found = "", None
        for token, confidence in output:
            if not isinstance(token, str) or not isinstance(confidence, (float, int, np.floating)):
                raise ValueError("Unexpected token-cache schema")
            cumulative += token
            if "Guess" in cumulative and ("yes" in token.lower() or "no" in token.lower()):
                found = int("yes" in token.lower()), float(confidence)
                break
        if found is None or not 0 <= found[1] <= 1:
            raise ValueError("Missing or invalid verdict token")
        result.append(found)
    return result


def execute(args):
    root = ROOT/"cache/research/diras_795ef8d"
    sources = json.loads((ROOT/"configs/diras_token_sources.json").read_text())
    for source in sources["files"]:
        path = root/source["path"]
        if not path.exists() and args.download:
            payload = urllib.request.urlopen(source["url"], timeout=45).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if not path.exists() or file_hash(path) != source["sha256"]:
            raise ValueError("Missing/mismatched confidence source; use --download")
    base = load_datasets(ROOT)[0]["chatreport"]
    config = ReviewConfig(**json.loads((ROOT/"configs/diagnostic_review_selected.json").read_text())["review_config"])
    views, clipped = {}, {}
    for model in ("phi3", "llama"):
        with (root/f"small_llm_results/gc_{model}_2e.csv").open(encoding="utf-8-sig", newline="") as stream:
            generations = list(csv.DictReader(stream))
        token = read_token_verdicts(root/f"small_llm_results/gc_{model}_2e.pkl")
        if not len(token) == len(generations) == len(base) == 660:
            raise ValueError("Confidence-view length mismatch")
        verbal = []
        for row in generations:
            matches = re.findall(r"\[Guess\]:\s*(Yes|No).*?\[Confidence\]:\s*([0-9.]+)", row["output"], re.I|re.S)
            if not matches:
                raise ValueError("Unparseable verbal confidence")
            label, score = matches[-1]
            verbal.append((int(label.lower() == "yes"), float(score.rstrip("."))))
        if any(a[0] != b[0] for a, b in zip(verbal, token)):
            raise ValueError("Token and verbal readers disagree about model verdicts")
        for name, scores in (("verbal", verbal), ("token", token)):
            rows = []
            clipped[f"{model}_{name}"] = sum(c <= .5 for _, c in scores)
            for original, (label, confidence) in zip(base, scores):
                # Preserve the generated verdict even when its full-vocabulary
                # token probability is below 0.5. These are confidence proxies.
                c = max(.500001, confidence)
                rows.append({**original, "prob_relevant": c if label else 1-c})
            views[f"{model}_{name}"] = sorted(rows, key=pair_key)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "sources": sources, "clipped_confidences": clipped,
                "settings": {"first_seed": 8000, "repetitions": 10, "budget_fractions": [.1, .25, .5]},
                "probability_semantics": "Verdict-conditioned confidence proxies. Token scores are chosen-token probabilities, not renormalized Yes/No probabilities. Four Phi-3 and two Llama scores below 0.5 are placed at the bottom without flipping verdicts.",
                "scope": "Same-generation confidence ablation and second cached assessor, not stochastic self-consistency or independent domain replication",
                "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (Path(__file__),
                    ROOT/"src/calibration/diagnostic_review.py", ROOT/"src/evaluation/review_metrics.py",
                    ROOT/"scripts/experiment_inputs.py", ROOT/"src/baselines/lara.py")}}
    write_json(args.output_dir/"experiment.json", metadata)
    records, queries, calibration = [], [], []
    for name, rows in views.items():
        evaluator = PolicyEvaluator(rows)
        pool = [{k: r[k] for k in ("query_id", "doc_id", "prob_relevant")} for r in rows]
        labels = {pair_key(r): int(r["true_binary"]) for r in rows}
        index = {pair_key(r): i for i, r in enumerate(rows)}
        p = np.asarray([r["prob_relevant"] for r in rows])
        calibration.append({"view": name, "semantics": "confidence proxy", **probability_diagnostics(rows, p)})
        for seed in range(8000, 8010):
            for fraction in (.1, .25, .5):
                budget = int(len(rows)*fraction)
                def record(method, human, predicted):
                    metric, qrows = evaluator.evaluate(human, predicted)
                    ids = {"dataset": name, "seed": seed, "budget": budget, "method": method, "pilot_labels": 0}
                    records.append({**ids, **metric})
                    queries.extend({**ids, **q} for q in qrows)
                record("raw_confidence", complete_budget(pool, [], 1-np.maximum(p, 1-p), budget, seed), (p >= .5).astype(int))
                human, predicted, _, _, _ = run_diagnostic_review(pool, LabelOracle(labels, budget), budget, config, seed)
                record("selected_local", human, predicted)
                for grouping, preprocessing in ((1, "raw"), ("topics", "upstream_smooth")):
                    predictions, _, _ = run_lara(pool, LabelOracle(labels, budget), budget, LaraConfig(preprocessing=preprocessing), seed, grouping)
                    human, predicted = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=int)
                    for row in predictions:
                        i = index[pair_key(row)]
                        human[i], predicted[i] = row["label_source"] == "human", row["predicted_label"]
                    record(f"lara_{grouping}_{preprocessing}", human, predicted)
        print(f"Confidence view {name} complete", flush=True)
    write_csv(args.output_dir/"runs.csv", records)
    write_csv(args.output_dir/"per_query.csv", queries)
    write_csv(args.output_dir/"summary.csv", aggregate(records, ("dataset", "budget", "method")))
    write_csv(args.output_dir/"confidence_proxy_diagnostics.csv", calibration)
    metadata.update(status="complete", policies=len(records))
    write_json(args.output_dir/"experiment.json", metadata)
    outputs = sorted(p for p in args.output_dir.iterdir() if p.is_file())
    atomic_text(args.output_dir/"checksums.sha256", "".join(f"{file_hash(p)}  {p.name}\n" for p in outputs))
    print(f"Completed {len(records)} confidence-view allocations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    execute(parser.parse_args())
