# Binary LARA baseline

LARA combines cached LLM relevance probabilities with budgeted human judgments.
After each purchased label, an online calibrator changes both the uncertainty
ranking used for subsequent acquisitions and the automatic labels assigned to
the remaining pool. This implementation provides the binary comparison required
by the [project proposal](../reports/proposal.tex), including global, three-group
and per-topic allocation variants. It runs on CPU without loading an LLM.

## Sources and implementation scope

The method is based on Takehi et al.,
[LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?](https://doi.org/10.1145/3726302.3729916),
SIGIR 2025, and the [authors' repository](https://github.com/RikiyaT/LARA).
Algorithm 1 describes repeated uncertainty selection and calibration; Section 3.4
introduces grouping for multiple assessors.

The source snapshot is pinned to commit
`482cb834c103a043b932a6e22d988945be5832bd`. The implementation follows the binary
`GradualLogisticRegression`, `ours` and `ours_groups` functions in
[`src/utils/algo.py`](https://github.com/RikiyaT/LARA/blob/482cb834c103a043b932a6e22d988945be5832bd/src/utils/algo.py)
and the normalization in
[`src/test-al.py`](https://github.com/RikiyaT/LARA/blob/482cb834c103a043b932a6e22d988945be5832bd/src/test-al.py).
`configs/lara_reference.json` records their exact SHA256 hashes.

The algorithm code is an independent implementation. The original source is
downloaded only for optional verification and remains in the ignored cache.
Direct comparisons passed on 24 synthetic and 24 real-score cases with identical
acquisition sequences, calibrated probabilities and final mixed labels. Versioned
synthetic reference outputs support offline CI without upstream downloads.

The measured experiments adapt LARA to the project's frozen Qwen3.5-4B binary
probabilities. They do not reproduce the paper's published Llama/TREC-COVID tables.
The paper's graded-label estimator is outside this binary implementation.

## Reference semantics

Several details of the released binary code differ from a generic logistic
regression fit or a literal reading of the paper's initialization:

- The calibrator starts with zero intercept/slope and Beta(1,1) counts. Each new
  label triggers one logistic-gradient update; it is not a batch refit.
- Its output blends the incoming probability with the logistic mapping. The
  reference's `initial_strength=200` remains fixed; it does not increase with
  the acquisition count. The transition scale is 1,000.
- The driver passes batch size 10 and lambda 200, but its binary functions do
  not use those arguments. The effective defaults above are represented directly
  in `configs/lara.json`.
- Grouped allocation uses a shared calibrator and round-robin acquisitions with
  fixed group quotas, matching the released function.

Two explicit correctness changes protect experimental validity. The upstream
`N-1` shortcut returns all labels as human; this implementation buys exactly
`N-1` labels and retains one automatic prediction. Only budget `N` uses the
full-human shortcut. Constant input to the driver's min-max normalization has
zero spread; an identity fallback retains finite probabilities instead of NaNs.
Reference-equivalence claims exclude these two intentionally corrected cases.

## Environment and execution

The baseline uses the existing lightweight CPU dependencies:

```powershell
python -m pip install -r requirements-lara.txt
python scripts/run_lara_baseline.py --manifest data/pilot/dl2019.jsonl --predictions results/stage2/dl2019_qwen35_full_v1/predictions.csv --output-dir results/baselines/lara/dl2019_full
```

The shared-split execution selects exactly the 33 test queries used by the
Stage 2 calibration comparison:

```powershell
python scripts/run_lara_baseline.py --manifest data/pilot/dl2019.jsonl --predictions results/stage2/dl2019_qwen35_full_v1/predictions.csv --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --output-dir results/baselines/lara/dl2019_shared_test
```

The input manifest and prediction CSV are checked for matching pairs and labels.
The model receives only IDs and probabilities. Gold labels are available through
a budget-limited oracle that rejects duplicates and over-budget acquisitions.
All splits are defined by query. With a split file, calibration and audit topics
are excluded from the annotation pool, and their labels are not used for fitting.

The default configuration evaluates seven human-budget fractions, three seeds
and three grouping settings. Each run starts independently with a fresh oracle.
Fractions are floored in decimal arithmetic. Existing output directories are
refused; a successful run records `status: complete` and artifact checksums.
There is no resume mode for the inexpensive cached-probability allocation stage.

## Input probabilities and preprocessing

The default `upstream_smooth` option reproduces the driver's preprocessing:

1. Center the selected pool's probabilities by their mean.
2. Apply `sigmoid(5 * centered_probability)`.
3. Rescale the resulting range to [0,1].

These are label-free pool statistics. The normalized values become the inputs
to the calibrator; original LLM probabilities remain in the output. This
transformation can change labels even at zero human budget. Consequently, the
zero-budget driver-compatible result is distinct from the raw LLM reference.

The `raw` sensitivity setting retains the assessor's original probabilities:

```powershell
python scripts/run_lara_baseline.py --manifest data/pilot/dl2019.jsonl --predictions results/stage2/dl2019_qwen35_full_v1/predictions.csv --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --preprocessing raw --output-dir results/baselines/lara/dl2019_shared_raw
```

The raw and normalized settings remain separately labeled. Their results are
not pooled or silently substituted for one another.

An external binary prediction CSV is supported through
`--predictions-format official_binary`, with the authors' documented columns:

```text
topic_id,doc_id,annotation,prob_yes,prob_no
```

Annotations are binary integers matching the manifest's relevance definition.
Yes/No masses are normalized to `prob_yes/(prob_yes+prob_no)`; nonfinite values,
zero total mass, duplicate pairs and incomplete coverage are rejected. This
adapter constructs the `pi` input needed by the upstream method. Graded
probability vectors are not treated as binary inputs implicitly.

## Acquisition, grouping and costs

For an incoming probability `p`, the calibrator's logistic mapping is
`sigmoid(intercept + slope*p)`. One acquired label updates the intercept, slope
and Beta counts. The resulting probability is a mixture of `p` and that mapping,
with the mixture weight computed from the reference defaults and Beta variance.
The configuration rejects settings that would produce an invalid probability
mixture. Original probability, transformed input and calibrated probability are
recorded separately.

The next acquisition minimizes `abs(calibrated_probability - 0.5)`. Equal
uncertainties select the first pair in canonical `(query_id, doc_id)` order,
matching the reference dataframe's first-minimum rule.

- `--groups 1`: unrestricted global selection, LARA(1).
- `--groups 3`: three query groups with approximately balanced pool sizes.
- `--groups topics`: one group per query, LARA(topics).

Grouping uses an isolated seeded shuffle. Group quotas sum to the exact budget
without exceeding group capacities. The shared calibrator is updated after every
purchase. Grouping seeds change assignment, order and budget remainders; they
are repetitions of the same evaluation pool rather than independent query splits.
Global selection is deterministic for a fixed canonical pool.

Every purchased label serves two purposes: calibrator supervision and the final
human label for its pair. Its cost is counted once. There is no separate fixed
calibration set or free random seed. Thus a total budget of 3,664 on the 7,328-pair
shared test pool yields 50% automatic coverage. This can be compared with the
lexical baseline at the same total budget; the cross-encoder's separate calibration
cost remains part of its own total budget.

Exact budgets and selected variants are supported explicitly:

```powershell
python scripts/run_lara_baseline.py --manifest data/pilot/dl2019.jsonl --predictions results/stage2/dl2019_qwen35_full_v1/predictions.csv --groups 1 topics --budgets 0 100 1000 4630 9259 9260 --output-dir results/baselines/lara/dl2019_exact_budgets
```

## Outputs and comparison metrics

The experiment root contains `experiment.json`, `raw_llm_reference.json`,
`budget_curve.csv` and `checksums.sha256`. The experiment record captures assessor
identity when supplied by the prediction sidecar, input/source hashes, the pinned
upstream reference, preprocessing, budgets, groups and seeds.

Each `groups_*/seed_*/budget_*` directory contains:

| File | Contents |
|---|---|
| `predictions.csv` | Original/transformed probabilities, calibrated predictions, human/automatic source, acquisition rank and final mixed label |
| `acquisitions.csv` | Every purchased label, group, probability before acquisition and acquisition order |
| `model.json` | Calibrator coefficients/counts, normalization state, grouping and quotas |
| `summary.json` | Coverage, selective risk, auto-error, class risks, human costs and probability diagnostics |
| `per_query.csv` | Query-level costs and errors for paired analysis |
| `mixed.qrels`, `gold.qrels` | Binary TREC qrels for identical evaluation topics |

Primary error metrics use only automatic labels. Undefined selective risk is
`null`; absent automatic predicted classes are represented explicitly. All-pool
model diagnostics include acquired training pairs and are not held-out estimates.
At the full-human endpoint, calibrator fitting is skipped because no model label
is retained. The saved state marks this shortcut.

Optional `--run-files` arguments enable the shared binary nDCG@10/Kendall tau-b
evaluation, including system-level and per-query score CSVs. The labels, topics,
unjudged-document treatment and total human budget are shared with other baseline
comparisons. The paper's graded nDCG or binary MAP results require their respective
benchmark protocol and are not interchangeable with these binary nDCG@10 outputs.

## Reference verification and CI

The optional verifier installs its separate pandas dependency and retrieves only
the pinned source files:

```powershell
python -m pip install -r requirements-lara-reference.txt
python scripts/verify_lara_reference.py --download --output results/baselines/lara/validation/upstream_comparison.json --manifest data/pilot/dl2019.jsonl --predictions results/stage2/dl2019_qwen35_full_v1/predictions.csv
```

Source hashes are checked before the named definitions are loaded. The verifier
does not execute upstream dataset loading or experiment dispatch. CI uses saved
synthetic reference fixtures and performs no source downloads, LLM inference or
pandas import. The software tests also cover exact endpoint budgets, one-pair
pools, label isolation, constant normalization, group quotas and input adapters.

## Plots and measured scope

The plotter requires the existing plotting dependencies:

```powershell
python -m pip install -r requirements-lexical-plots.txt
python scripts/plot_lara_baseline.py --run-dir results/baselines/lara/dl2019_shared_test_v1
```

PNG and editable SVG outputs under `reports/figures/lara/` show risk/coverage,
auto-error/budget, class risk at 50% coverage and a learned mapping diagnostic.
Seed bands describe grouping variation, not confidence intervals. The mapping
panel is a model diagnostic for the first seed rather than a reliability estimate.

The [validation record](lara_validation.md) distinguishes code/reference checks
from the measured Qwen/DL2019 adaptation. Original Llama benchmark reproduction,
graded LARA, domain-shift evaluation, official-run rankings and query-level
significance analysis remain separate experiments. LARA supplies no formal
selective-error guarantee.
