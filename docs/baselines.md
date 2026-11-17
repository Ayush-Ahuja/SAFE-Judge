# Baseline setup

The final comparison uses the same 33 DL 2019 test topics (7,328 pairs), three
seeds and 3,664 total human labels. Saved results are in `artifacts/final/`;
the following runners need the original manifests and score exports.

| Baseline | Implementation | Configuration |
|---|---|---|
| Lexical active learning | `src/baselines/active_learning.py` | `configs/lexical_active_learning.json` |
| MiniLM + temperature scaling | `src/baselines/cross_encoder.py` | `configs/cross_encoder.json` |
| LARA | `src/baselines/lara.py` | `configs/lara.json` |

## Lexical

Logistic regression uses BM25, query-term overlap and document length. Budgeted
uncertainty sampling buys labels used for both fitting and final human judgments.
There are no extra training labels. Install `requirements-lexical.txt`, then use:

```powershell
python scripts/run_lexical_baseline.py --manifest data/pilot/dl2019.jsonl --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --output-dir results/reproduction/lexical
```

## MiniLM

The frozen checkpoint is `cross-encoder/ms-marco-MiniLM-L6-v2`; its revision and
512-token limit are pinned in the configuration. Install
`requirements-cross-encoder.txt`, then score and evaluate:

```powershell
python scripts/score_cross_encoder.py --manifest data/pilot/dl2019.jsonl --output-dir results/reproduction/minilm_scores --device cpu
python scripts/run_cross_encoder_baseline.py --manifest data/pilot/dl2019.jsonl --scores results/reproduction/minilm_scores --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --output-dir results/reproduction/minilm
```

Temperature is fitted on disjoint calibration queries. All 1,932 calibration
labels count toward the budget, leaving 1,732 test-review labels at the reported
half budget. Scoring supports CPU or CUDA; calibration runs on CPU. Saved numerical
validation compared 65 pairs with Sentence Transformers, with full label agreement
and maximum raw-logit difference below 0.000001.

See [LARA](lara.md) for its grouping and reference checks. All runners record
settings, splits, seeds, outputs and checksums. Use a new output directory for
a fresh run; the [reproduction guide](reproduce_final.md) rebuilds the published
tables without running these assessors again.
