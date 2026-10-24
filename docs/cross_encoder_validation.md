# Neural cross-encoder validation

Validation was completed on 23 October 2026 using Python 3.12.3, PyTorch
2.6.0+cpu, Transformers 4.49.0, NumPy 2.2.6 and SciPy 1.15.3. The checkpoint was
`cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Settings were fixed in
`configs/cross_encoder.json`; no model fine-tuning or hyperparameter search was
performed on the test queries.

## Software and reference checks

All 49 repository tests passed, including 14 cross-encoder tests. The complete
offline suite also passed in the lightweight CI environment without Torch or
Transformers. Dependency consistency checks passed in both environments.

The real scorer was compared with Sentence Transformers 3.4.1
`CrossEncoder.predict` using identity activation, the same pinned model and the
same 512-token limit. The comparison covered 64 DL2019 pairs and one synthetic
long-document pair, using batch sizes 16 and 8 respectively:

| Measurement | Result |
|---|---:|
| Compared pairs | 65 |
| Maximum absolute raw-logit difference | 0.000000954 |
| Mean absolute raw-logit difference | 0.0000000667 |
| Binary-label agreement | 100% |
| Synthetic long pair, original / retained tokens | 4,009 / 512 |

The score difference reflects a numerical batching check rather than a
statistical estimate. The record is stored in
`results/baselines/cross_encoder/validation/inference_reference.json`.

Tests verified the fitted temperature against an analytical binary optimum,
pair/query weighting, boundary optima, extreme logits, stable probabilities,
query/audit isolation, source/target inputs, cache integrity, resumable scoring,
calibration-inclusive costs, binary qrels and synthetic system-ranking fixtures.
Changing only test gold labels left the temperature, model probabilities and
allocation unchanged. Calibration is applied to raw logits exactly once.

## Completed DL2019 experiment

The frozen manifest supplied all 9,260 judged pairs across 43 queries. Scoring
used CPU float32, four threads and batches of 32. The scoring invocation took
210.38 seconds, including model loading, tokenization and checkpoint writes but
excluding dependency installation and model download. No actual DL2019 pair was
truncated. The raw cache is:

`results/baselines/cross_encoder/dl2019_minilm_scores_v1/`

The evaluation reused the Stage 2 split: ten calibration queries containing
1,932 pairs and 33 test queries containing 7,328 pairs. Calibration labels were
charged to every feasible total budget. The fitted temperature was **6.291521**,
an interior optimum of the prespecified `[0.01, 1000]` interval. Calibration NLL
decreased from 1.360493 to 0.585432.

### Held-out probability calibration

| Test metric | Raw scores, T=1 | Temperature scaled |
|---|---:|---:|
| Accuracy | 72.98% | 72.98% |
| Macro-F1 | 0.6957 | 0.6957 |
| Binary NLL | 1.105061 | 0.531877 |
| Brier score | 0.226353 | 0.177234 |
| ECE, ten relevance-probability bins | 0.215550 | 0.152806 |

Probability metrics improved on this split, while class decisions remained
unchanged. ECE remains substantial; the result does not establish a formal
calibration or selective-error guarantee. Positive temperature scaling preserves
confidence ordering, so exact-budget allocation is unchanged relative to ranking
the same raw scores.

### Total-budget sweep

The primary execution is
`results/baselines/cross_encoder/dl2019_shared_test_v1/`.
It contains 33 budget/seed records: 18 feasible evaluations and 15 explicitly
infeasible points. Seeds 42, 43 and 44 resolve score ties; results were identical
across these seeds. They are repetitions within one query split rather than
independent statistical samples.

| Total human labels | Calibration | Test escalations | Automatic coverage | Selective risk | Auto-error |
|---:|---:|---:|---:|---:|---:|
| 1,932 | 1,932 | 0 | 100.00% | 27.02% | 27.02% |
| 2,198 | 1,932 | 266 | 96.37% | 26.17% | 25.22% |
| 3,664 | 1,932 | 1,732 | 76.36% | 20.94% | 15.99% |
| 5,496 | 1,932 | 3,564 | 51.36% | 14.43% | 7.41% |
| 7,328 | 1,932 | 5,396 | 26.36% | 8.02% | 2.12% |
| 9,260 | 1,932 | 7,328 | 0.00% | Undefined | 0.00% |

At the 3,664-label total budget, automatic predictions include both classes:
3,603 predicted non-relevant pairs with 9.74% selective risk, and 1,993 predicted
relevant pairs with 41.19% selective risk. This class disparity is retained in
the summaries and plots.

The additional coverage-matched execution is
`results/baselines/cross_encoder/dl2019_coverage_matched_v1/`.
It contains 21 feasible budget/seed evaluations. At exactly 50% automatic coverage,
selective risk is **14.11%**, but total human cost is **5,596 judgments**: 1,932
calibration labels plus 3,664 test escalations. The lexical baseline's 50%-coverage
point purchases 3,664 labels; the two points therefore have different total costs.
The primary matched-cost comparison uses total human labels, as defined in the
[baseline protocol](cross_encoder.md#human-budget-accounting).

## Artifact verification and plots

For the primary execution, all 120 recorded artifact checksums and current source
hashes were verified. Every feasible allocation was reopened to recompute coverage,
selective risk and auto-error, verify the calibration-inclusive cost and confirm
that escalated pairs had the smallest absolute logits. Raw cache keys matched all
9,260 manifest pairs. An independent bounded minimization over log-temperature
returned 6.291521024, agreeing with the derivative-root solution.

All 21 allocations in the coverage-matched execution were also reopened to verify
input/artifact hashes, gold labels, total costs, coverage and error metrics. In
total, 39 feasible allocations passed these checks across the two executions.

The plotter generated ten PNG/SVG artifacts under `reports/figures/cross_encoder/`,
including a four-panel overview. All figure hashes were verified. The plots show
held-out probability reliability, risk versus coverage, auto-error versus total
cost and risk by predicted class. Infeasible budgets have no performance points;
seed ranges are not confidence intervals.

The validation artifacts are retained under
`results/baselines/cross_encoder/validation/`. Generated experiment results follow
the repository's existing exclusion from version control by default.

## Scope and remaining evaluation

The measured benchmark is DL2019 with one fixed query split. No official
retrieval-system runs were evaluated, so no empirical system-ranking tau is
reported. DL2020, TREC-COVID, blind-audit evaluation, alternative query splits,
query-bootstrap confidence analysis and cross-method significance tests remain
separate experiments. The implementation supports separate source/target
calibration, shared splits and official TREC run evaluation without rescoring
previously cached pairs.
