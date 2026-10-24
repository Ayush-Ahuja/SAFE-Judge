# Calibrated neural cross-encoder baseline

The neural baseline uses a frozen MS MARCO MiniLM cross-encoder, positive
temperature scaling on query-disjoint calibration data, and confidence-based
human escalation. It corresponds to baseline 2 in the [proposal](../reports/proposal.tex).
Scoring supports CPU or CUDA; subsequent calibration and evaluation run on CPU.

## Model and calibration references

The checkpoint is [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
pinned to revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The model card
provides a Transformers inference example returning a scalar score. The baseline
follows that implementation using `AutoTokenizer` and
`AutoModelForSequenceClassification`, with evaluation mode, float32 weights and
inference without gradients. Model weights are loaded from safetensors.

[Guo et al. (ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html) define
temperature scaling as division of logits by a positive scalar fitted to
calibration negative log-likelihood. Their [reference implementation](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py)
also operates on logits before the probability activation. For the scalar model
score `z`, the binary probability is `sigmoid(z/T)`. This implementation minimizes
binary NLL using SciPy root finding on its monotone inverse-temperature derivative.
The positive bounds in the configuration preserve the decision boundary.

The scorer was checked against the official
[Sentence Transformers CrossEncoder implementation](https://github.com/huggingface/sentence-transformers/blob/v3.4.1/sentence_transformers/cross_encoder/CrossEncoder.py)
with an explicit identity activation. Raw scores are saved before any sigmoid;
neither default wrapper activation nor double sigmoid is part of the pipeline.
The implementation adds no intercept or relevance-threshold fitting. Consequently,
temperature scaling cannot correct a misplaced zero-score decision boundary.

## Environment and scoring

Inference dependencies are separate from the lightweight cached-score evaluator.
The following PowerShell example installs the pinned inference stack and scores
the frozen DL2019 manifest from the repository root:

```powershell
python -m venv .venv-cross-encoder
.\.venv-cross-encoder\Scripts\python.exe -m pip install -r requirements-cross-encoder.txt
.\.venv-cross-encoder\Scripts\python.exe scripts/score_cross_encoder.py --manifest data/pilot/dl2019.jsonl --output-dir results/baselines/cross_encoder/dl2019_minilm_scores
```

The corresponding Linux interpreter path is `.venv-cross-encoder/bin/python`.
The first scoring invocation downloads approximately 91 MB of model weights and
the tokenizer into `cache/huggingface/`. `--local-files-only` supports execution
after the model has been cached. `--device cuda` selects an available CUDA GPU;
the default is CPU. `configs/cross_encoder.json` specifies batch size 32, a
512-token cap and four CPU threads. A GPU environment requires a compatible
PyTorch installation.

Each pair is encoded jointly with the tokenizer's `longest_first` truncation,
matching Sentence Transformers. Original and retained token counts are saved,
including whether truncation occurred. Text is stripped at the model boundary.
Official labels are removed before inference. The raw score cache contains IDs,
logits, token counts and text hashes; it contains no gold labels.

`--split-json` optionally restricts scoring to the specified calibration and test
topics, excluding audit topics. The SQLite checkpoint records each completed
batch. Repeating an identical invocation resumes missing pairs. Changes to the
manifest, configuration, source code, dependencies or scoring device invalidate
checkpoint reuse. Existing completed caches can be exported without reloading
the model. Two simultaneous writers to one directory are rejected.

## Calibration and evaluation

The evaluator requires an explicit query split or a separate source-domain
calibration collection. The saved Stage 2 split contains ten calibration queries
(1,932 pairs) and 33 test queries (7,328 pairs):

```powershell
python scripts/run_cross_encoder_baseline.py --manifest data/pilot/dl2019.jsonl --scores results/baselines/cross_encoder/dl2019_minilm_scores --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --output-dir results/baselines/cross_encoder/dl2019_shared_test
```

Cached-score evaluation requires only `requirements-lexical.txt`, without Torch,
Transformers or a GPU. The cache identity, manifest hash, score checksum, input
texts and requested pair coverage are checked before evaluation. Existing output
directories are refused.

Temperature is fitted exclusively on calibration labels. The default objective
weights pairs equally; `calibration_weighting: query` gives each calibration
query equal total weight. The configured temperature interval is `[0.01, 1000]`.
An optimum on either boundary is recorded explicitly. All-zero calibration scores
select `T=1`. Calibration NLL is checked against the unscaled value.

`calibration_metrics.csv` compares raw and scaled probabilities on identical test
pairs using accuracy, macro-F1, binary NLL, Brier score and ten-bin probability ECE.
Positive scaling preserves labels and absolute-score ordering. It can change
probability quality and acceptance at a fixed numerical confidence threshold;
it does not improve the ordering of pairs for an exact escalation budget.

The allocation policy escalates the least confident test pairs. Ranking uses
absolute raw logits to avoid artificial confidence ties caused by floating-point
sigmoid saturation. Equal absolute logits are resolved with the saved seed.
The seeds are tie-break repetitions of one query split, not independent datasets
or alternative calibration splits.

## Human-budget accounting

For `C` calibration pairs, `N` test pairs and total human budget `B`:

- Calibration costs `C` human judgments.
- Test escalations cost `B-C` judgments.
- Automatic coverage is `(N-(B-C))/N` for feasible budgets.
- Budgets below `C` are recorded as `infeasible_calibration_budget`; no performance
  value is assigned to them.

Fractional budgets use `floor(fraction * N)` as the **total** human-label cap,
matching the lexical baseline's test-pool denominator. With the saved DL2019
split, calibration alone costs 26.36% of the test-pool size. A 50% total budget
is 3,664 judgments: 1,932 for calibration and 1,732 for escalation, leaving 5,596
automatic labels (76.36% coverage). It is distinct from 50% automatic coverage.

The default sweep also includes `B=C` (no test escalation) and `B=C+N` (all test
pairs labeled by humans). The latter costs 126.36% of test-pool size for this
split. Exact counts are supported by `--budgets`:

```powershell
python scripts/run_cross_encoder_baseline.py --manifest data/pilot/dl2019.jsonl --scores results/baselines/cross_encoder/dl2019_minilm_scores --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --budgets 1932 3664 5596 7328 9260 --output-dir results/baselines/cross_encoder/dl2019_exact_budgets
```

Cross-method comparisons match `budget` or `total_human_labels` against the
lexical baseline's human-label count. In per-test-pair metrics, `human_labels`
and `test_human_labels` count test escalations only; `total_human_labels` includes
calibration. Per-query tables contain test costs, while calibration costs and
their query IDs are recorded separately.

## Domain transfer and audit topics

The same evaluation interface supports source/target experiments with separate
manifests and score caches:

```powershell
python scripts/run_cross_encoder_baseline.py --calibration-manifest data/processed/dl2019.jsonl --calibration-scores results/baselines/cross_encoder/dl2019_scores --manifest data/processed/dl2020.jsonl --scores results/baselines/cross_encoder/dl2020_scores --output-dir results/baselines/cross_encoder/dl2019_to_dl2020
```

Without a split file, all source queries are used for calibration and all target
queries for testing. With a split file, `calibration_query_ids` select source
topics and `test_query_ids` select target topics. Calibration, test and optional
audit query IDs are disjoint; absent topics and duplicates are rejected. Dataset
binary cutoffs come from each manifest, including DL grades >=2 and TREC-COVID
grades >=1. All source calibration labels remain part of the total budget.
Source calibration provides no guarantee of validity under domain shift.

## Output files

The raw-score directory contains `scores.csv`, `experiment.json` and
`checkpoint.sqlite3`. Its experiment record includes the pinned model revision,
source hashes, dependencies, device, scoring settings, manifest identity and score
checksum. Checkpoint files remain local.

The evaluation root contains:

| File | Contents |
|---|---|
| `experiment.json` | Settings, query IDs, source/input hashes, budgets and completion status |
| `temperature.json` | Fitted temperature, bounds, objective, NLL and boundary diagnostics |
| `calibration_labels.csv` | Every purchased calibration label |
| `calibration_metrics.csv` | Raw/scaled test-set probability and classification metrics |
| `raw_predictions.csv`, `temperature_scaled_predictions.csv` | Held-out model scores and probabilities |
| `*_reliability.csv` | Ten-bin probability reliability data |
| `*_risk_coverage.csv` | Descriptive numerical-threshold sweeps; no risk guarantee |
| `budget_curve.csv` | Total-budget results, including explicitly infeasible points |
| `checksums.sha256` | Hashes of completed evaluation artifacts |

Each feasible `seed_*/budget_*` directory contains model predictions, human/automatic
allocation, final mixed labels, test acquisition logs, per-query metrics, a summary
and binary gold/mixed TREC qrels. `predicted_label` always refers to the model;
`final_label` contains the purchased human label on escalated pairs. Primary
selective risk and auto-error count errors on automatic test labels only.
Undefined risks are `null`; absent automatic classes remain explicit.

Optional `--run-files` arguments enable the shared binary nDCG@10 and Kendall's
tau-b evaluator. It uses the same topics, zero gain for unjudged documents and
zero scores for missing topics as the lexical baseline. Original graded nDCG
belongs to a separate evaluation scale.

## Plots

The plotter reads a completed evaluation and generates probability reliability,
risk/coverage, auto-error/cost and predicted-class risk figures:

```powershell
python -m pip install -r requirements-lexical-plots.txt
python scripts/plot_cross_encoder_baseline.py --run-dir results/baselines/cross_encoder/dl2019_shared_test
```

PNG and editable SVG figures are saved under `reports/figures/cross_encoder/`.
Their metadata records input and output hashes. Seed ranges describe tie-breaking
variability within one fixed query split and are not confidence intervals.

## Verification

The offline suite exercises calibration against an analytical optimum, objective
weighting, extreme logits, positive boundaries, invariance of labels and budget
ordering, query/audit isolation, source/target input, test-label mutation, cache
checksums, interrupted scoring, exact calibration-inclusive costs and output
compatibility. Scoring tests use an explicit synthetic backend; CI loads no model
weights. Real-model and real-data validation are recorded separately in
[cross_encoder_validation.md](cross_encoder_validation.md).
