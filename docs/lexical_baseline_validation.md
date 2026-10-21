# Lexical baseline validation

Validated on 20 October 2026 with Python 3.12.3, NumPy 2.2.6, SciPy 1.15.3 and
scikit-learn 1.6.1 on CPU. Settings were fixed to
`configs/lexical_active_learning.json`: seed size 50, acquisition batch 50,
logistic C=1, BM25 k1=1.2 and b=0.75. No hyperparameter search was performed.

## Code checks

- All **35 repository tests passed**, including 17 lexical baseline tests.
- An independent scikit-learn fit verified uncertainty acquisition.
- A hand calculation verified BM25; model reconstruction matched saved probabilities.
- Changing only unqueried gold labels left acquisitions and model output unchanged.
- Zero budget performed zero oracle accesses; one-class seeds did not purchase free
  labels. Budgets 0, 1, seed-sized, non-batch-sized, N-1 and N were checked.
- Analytical TREC fixtures checked binary nDCG and opposite/identical system rankings;
  the CLI path also produced system-score files using synthetic run fixtures.
- `pip check` and `git diff --check` passed.

## Completed real-data runs

Both executions used the frozen, official judged DL2019 manifest and seven budget
fractions (0, 0.01, 0.05, 0.1, 0.2, 0.5, 1) across seeds 42, 43 and 44.

| Output directory under `results/baselines/lexical/` | Queries | Pairs | Budget/seed runs | Mean selective risk at 50% coverage |
|---|---:|---:|---:|---:|
| `dl2019_full_v2` | 43 | 9,260 | 21 | 18.22% |
| `dl2019_shared_test_v1` | 33 | 7,328 | 21 | 19.10% |

At 50% coverage, risk ranges across the three random seeds were 18.14-18.27%
for the full pool and 19.02-19.13% for the saved test split. These are seed ranges,
not confidence intervals. Measured experiment-loop times were approximately 28.6
and 20.3 seconds, respectively, excluding common feature preparation and imports.
These are local CPU measurements, not HPC runtime estimates.

**At this coverage, all automatically labeled pairs are predicted non-relevant**
in all six results. The summaries correctly report
`both_predicted_classes_accepted: false`; the relevant-prediction class risk is
undefined. This does not demonstrate low risk in both classes or establish a
formal bound. It is a limitation of this baseline on the observed pool.

The full-pool run is descriptive. The shared-test run uses exactly the 33 test
query IDs from the existing Stage 2 temperature-scaling split and excludes its
ten calibration topics. Within that test pool, the baseline still purchases
labels for active learning; those labels are all charged to its annotation budget.
This is a transductive annotation experiment. Cross-method comparisons account
for all purchased labels, including any calibration labels acquired outside the
test pool, as defined in the [baseline protocol](lexical_active_learning.md#protocol-and-budget-accounting).

After completion, all **42 run outputs** were independently reopened and checked:

- Status was `complete`; all recorded source hashes matched current source files.
- All 151 hashed artifacts in each execution matched `checksums.sha256`.
- Prediction keys, query IDs and gold labels matched the frozen manifest/split.
- Every acquisition was unique, its label matched official qrels, and its count
  equaled the requested budget.
- Coverage, selective risk and auto-error recomputed from predictions matched
  every saved summary.

The completed full-pool execution is `dl2019_full_v2`. The incomplete
`dl2019_full_v1` directory has no completion or checksum record and is excluded
from the reported results.

These executions evaluate label quality without official retrieval-system runs;
system-ranking tau was not measured. DL2020, TREC-COVID, blind-audit testing,
query-bootstrap inference and cross-method significance tests remain final-experiment
work. The runner accepts manifests, shared splits and official TREC runs for those
comparisons; per-query outputs support paired query-level analysis.
