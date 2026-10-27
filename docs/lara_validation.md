# Binary LARA validation

Validation was completed on 27 October 2026. The measured baseline uses cached
Qwen3.5-4B Yes/No probabilities from the frozen DL2019 pilot, with the binary
relevance cutoff at grade >=2. The calibration and allocation code runs on CPU.
No additional LLM inference or model fine-tuning was performed.

## Upstream verification

The checked author snapshot is commit
`482cb834c103a043b932a6e22d988945be5832bd` of
[RikiyaT/LARA](https://github.com/RikiyaT/LARA). Exact source hashes are recorded
in `configs/lara_reference.json`. Verification used Python 3.12.3, NumPy 2.2.6,
SciPy 1.15.3 and pandas 2.2.3.

The optional verifier executed the pinned binary calibrator/acquisition functions
on the same inputs as the independent implementation:

| Check | Cases | Outcome |
|---|---:|---|
| Synthetic pool, raw/normalized input, 1/3/per-topic groups, four budgets | 24 | Exact agreement |
| Cached real Qwen scores, raw/normalized input, 1/3/per-topic groups, four budgets | 24 | Exact agreement |

Acquisition order, final mixed labels and all calibrated probabilities matched.
The maximum observed absolute probability difference was **0.0**. The real-score
comparison selected three pairs per DL2019 query. The comparison report is
`results/baselines/lara/validation/upstream_comparison.json`; versioned synthetic
outputs are stored in `tests/fixtures/lara_reference.json`.

The strict `N-1` budget endpoint and constant-normalization fallback are explicit
correctness changes described in the [method guide](lara.md#reference-semantics).
They are tested separately and excluded from exact upstream-equivalence claims.
The reference's fixed initial strength, unused driver arguments and round-robin
group schedule are retained rather than silently replaced with another estimator.

## Software checks

All **62 repository tests passed**, including 13 LARA tests. These also passed
in the lightweight CI environment without pandas, Torch, model weights or source
downloads. The LARA tests include the 24 pinned reference cases, label isolation,
zero-label access, exact budgets including `N-1`, one-pair pools, constant inputs,
group-capacity redistribution, reproducibility, unchanged global RNG state,
official binary CSV validation, shared query/audit isolation and synthetic
retrieval-system evaluation. Dependency consistency checks passed.

## Completed DL2019 sweeps

Three executions used seven budget fractions (0, 0.01, 0.05, 0.1, 0.2, 0.5, 1),
groupings 1/3/per-topic and seeds 42/43/44. Each execution contains 63 allocations,
for **189 completed allocations** overall.

| Directory under `results/baselines/lara/` | Queries | Pairs | Preprocessing |
|---|---:|---:|---|
| `dl2019_full_v1` | 43 | 9,260 | Upstream normalization |
| `dl2019_shared_test_v1` | 33 | 7,328 | Upstream normalization |
| `dl2019_shared_raw_v1` | 33 | 7,328 | Original assessor probabilities |

The shared test pool is the same 33-topic split used by the other baselines.
The ten reserved calibration queries are excluded. LARA purchases labels within
the selected annotation pool, and each purchased label both trains the calibrator
and supplies its final human judgment. Every acquisition is charged once.

Seeds describe query grouping variation within a fixed evaluation pool. Global
selection is deterministic for canonical input order, so repeating LARA(1) with
different seeds gives identical results. These repetitions are not independent
query splits, confidence intervals or significance tests.

### Selective risk at 50% automatic coverage

Values are means across the three grouping seeds. At this coverage the human
budget is 4,630 labels for the full pool or 3,664 labels for either shared pool.

| Variant | Full pool, normalized | Shared test, normalized | Shared test, raw |
|---|---:|---:|---:|
| LARA(1) | 10.19% | **8.65%** | 7.78% |
| LARA(3) | 10.28% | 8.78% | 7.74% |
| LARA(topics) | 12.38% | 11.90% | 11.79% |

For the shared normalized experiment, LARA(3) ranged from 8.38% to 9.20% and
LARA(topics) from 11.87% to 11.93%. These are seed ranges rather than confidence
intervals. Both automatic predicted classes were represented at 50% coverage in
all variants and executions.

### Primary shared-pool LARA(1) results

The primary setting retains the author's driver normalization. The results below
are identical across its three global-selection seeds.

| Human labels | Automatic coverage | Selective risk | Auto-error |
|---:|---:|---:|---:|
| 0 | 100.00% | 21.74% | 21.74% |
| 73 | 99.00% | 21.53% | 21.32% |
| 366 | 95.01% | 20.14% | 19.13% |
| 732 | 90.01% | 18.68% | 16.81% |
| 1,465 | 80.01% | 15.59% | 12.47% |
| 3,664 | 50.00% | **8.65%** | **4.33%** |
| 7,328 | 0.00% | Undefined | 0.00% |

At the 3,664-label point, 317 automatic judgments are wrong. The accepted subset
contains 2,950 predicted non-relevant pairs with 6.85% selective risk and 714
predicted relevant pairs with 16.11% selective risk. Mixed human/automatic label
accuracy is 95.67%. These are observed results without a formal error guarantee.

The zero-budget normalized result differs from the raw assessor because the
driver changes the probability scale before calibration. The raw-probability
control has 20.24% zero-budget risk and 7.78% risk at 50% coverage. The two
preprocessing settings remain separate sensitivity results.

## Artifact verification and plots

All 189 allocations were reopened to verify source/input hashes, artifact
checksums, prediction keys, source probabilities, official gold labels, exact
human costs, group quotas, selective risk and auto-error. Final probabilities
were independently reconstructed from the saved coefficients and Beta counts.
The artifact-verification report is
`results/baselines/lara/validation/artifact_validation.json`.

The plotter generates a four-panel overview and four individual figures in PNG
and SVG under `reports/figures/lara/`. Its metadata records source and output
hashes. The figures show the primary shared normalized experiment. The model-map
diagnostic uses the first seed and includes acquired training examples; it is
not a held-out reliability curve.

## Reproduction scope

This is a verified binary author-code implementation evaluated with the project's
compact assessor. The original paper's Llama-3.1-8B/70B TREC-COVID system-ranking
numbers have **not** been reproduced. The implementation accepts externally
generated binary probabilities, but the measured results do not include new
Llama inference, graded LARA, TREC-COVID, DL2020 or official retrieval-system runs.
No empirical ranking tau, domain-shift result or query-bootstrap significance
claim follows from these label-quality sweeps.

The original paper reports different benchmark/metric settings, including binary
MAP and graded nDCG. The project's optional shared scorer reports binary nDCG@10;
its output is a project comparison rather than an automatic replication of those
published tables. Original-setting reproduction and matched-budget comparisons
across all methods remain distinct evaluation tasks.
