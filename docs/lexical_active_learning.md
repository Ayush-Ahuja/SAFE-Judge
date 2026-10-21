# Classical lexical active-learning baseline

The classical baseline combines logistic regression on BM25, query-term overlap
and document length with uncertainty sampling and mixed human/automatic labels.
The implementation runs entirely on CPU and corresponds to baseline 1 in the
[project proposal](../reports/proposal.tex).

## Method and references

Section 4.2 of LARA describes SAL(hybrid): uncertain examples are selected for
human annotation, and the learned model labels the remaining pool. SAL differs
from CAL, which selects the most likely relevant examples. The methodological
references and implementation sources are:

- [LARA paper](https://doi.org/10.1145/3726302.3729916).
- [Official `sal` and `OnlineLogisticRegression` implementation](https://github.com/RikiyaT/LARA/blob/main/src/utils/algo.py).
- [Official experiment dispatcher](https://github.com/RikiyaT/LARA/blob/main/src/test-al.py).

This baseline is a **lexical SAL(hybrid) variant**. The LARA implementation uses
preprocessed document feature indices and online weight updates; this variant
uses three standardized query-document features and regularized batch logistic
regression. All final labels and ranking metrics follow the binary relevance
definition used in the Stage 2 pilot. These feature, optimization and evaluation
choices distinguish the experiment from an exact reproduction of LARA's SAL
results. The classifier does not use LLM probabilities.

## Environment and execution

The pinned dependencies support Python 3.10-3.13; validation used Python 3.12.
The following PowerShell example creates a CPU environment and runs the baseline
from the repository root:

```powershell
python -m venv .venv-lexical
.\.venv-lexical\Scripts\python.exe -m pip install -r requirements-lexical.txt
.\.venv-lexical\Scripts\python.exe scripts/run_lexical_baseline.py --manifest data/pilot/dl2019.jsonl --output-dir results/baselines/lexical/dl2019_run
```

The corresponding Linux interpreter path is `.venv-lexical/bin/python`.
`requirements-lexical.txt` also supports installation in an existing CPU environment.
The default JSON configuration runs seven human-budget fractions for seeds 42, 43,
and 44. Fractional budgets become `floor(fraction * selected_pairs)` using decimal
arithmetic. Each budget/seed run is an independent experiment; a budget sweep is
not assumed to be a single nested annotation trajectory.

The shared-split execution evaluates the same test topics as the Stage 2
calibration experiment:

```powershell
.\.venv-lexical\Scripts\python.exe scripts/run_lexical_baseline.py --manifest data/pilot/dl2019.jsonl --split-json results/stage2/dl2019_qwen35_full_v1/evaluation/calibration/split.json --output-dir results/baselines/lexical/dl2019_shared_test
```

This selects the 33 test queries (7,328 pairs), excludes the ten calibration
queries, and does not read calibration labels. The split schema requires
`test_query_ids`; optional `calibration_query_ids` and `audit_query_ids` are
disjoint. Unknown test topics, duplicates and overlapping groups are rejected.
The same interface supports prespecified DL2020 and TREC-COVID splits. Blind audit
topics are represented separately from calibration and test topics and excluded
from these runs.

Exact matched human-label counts can replace fractions:

```powershell
.\.venv-lexical\Scripts\python.exe scripts/run_lexical_baseline.py --manifest data/pilot/dl2019.jsonl --budgets 0 100 500 1000 --seeds 42 --output-dir results/baselines/lexical/dl2019_exact_budgets
```

Each invocation requires a new output directory. Existing directories are refused,
including incomplete runs. An interrupted or failed execution retains
`status: running`; successful completion publishes the final budget curve,
checksums and `status: complete`.
There is no resume mode. New results follow the repository's existing `.gitignore`
policy and are excluded from version control by default.

## Protocol and budget accounting

This is **transductive pool annotation**. At deployment, all candidate query and
document texts are available. A budgeted subset is manually annotated, a model
is trained on those revealed labels, and the remaining labels are predicted.
Acquired labels can therefore come from test topics: they are the simulated human
judgments included in the annotation budget. The resulting evaluation measures
budgeted annotation of an available pool rather than zero-label generalization
to unseen topics.

1. Pairs are sorted by string `(query_id, doc_id)`, and label-free features are computed.
2. `StandardScaler` is fitted on available pool features without labels.
3. A random seed of `min(50, budget)` labels is revealed.
4. Logistic regression is fitted on revealed labels only. Up to 50 remaining pairs
   closest to probability 0.5 are selected for annotation before each refit. This
   cycle continues until the budget is exhausted.
5. Acquired pairs retain official binary labels; remaining pairs receive model labels.

Seed size, batch size, regularization and BM25 parameters are in
`configs/lexical_active_learning.json`. These settings define the configuration
fixed for an experiment. The learner receives text/IDs only and rejects gold
fields. The label oracle rejects
duplicate or over-budget queries. Every seed and subsequent acquisition counts as
one human judgment. No class-balanced seed is selected by inspecting gold labels.
There are no additional training or calibration labels.

Before both classes have been observed, the model uses the Laplace-smoothed constant
`(observed_positives + 1)/(observed_labels + 2)`. Ties use a seeded fixed random
priority, so one-class seeds continue to explore without free labels. At budget
zero the prior is 0.5 and prediction ties map to relevant. That endpoint is an
**untrained reference**. At budget `N-1`, exactly one label remains automatic;
budget `N` gives full human labels.

Comparisons with SAFE-Judge use the same total human-label cap, including
calibration labels and escalations. Calibration costs remain part of that cap
when the labels originate outside the test pool. Comparable results share the
same test pairs, relevance scale and treatment of unjudged documents.

## Feature definitions

Tokenization is Unicode alphanumeric matching after casefolding, without stemming
or stopword removal. Repeated query terms are counted once. The features are:

- **BM25:** sum over unique query terms of
  `log(1 + (N - df + 0.5)/(df + 0.5)) * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))`,
  with defaults `k1=1.2`, `b=0.75`.
- **Query-term overlap:** number of unique query terms appearing in the document
  divided by the number of unique query terms; zero for an empty tokenized query.
- **Document length:** `log1p(number_of_document_tokens)`.

By default, BM25 document frequencies and mean length use **unique documents in
the selected judged pool**, not the full 8.8-million-passage corpus. This avoids
building a new index and requires no labels. Documents appearing under multiple
queries count once. Conflicting text for an identical document/query ID is rejected.
The scope and full statistics are saved with each execution. The default BM25
feature therefore represents pool-level rather than full-corpus statistics.
The optional `--bm25-stats stats.json` argument accepts external corpus statistics
computed with the same tokenizer and the following schema:

```json
{
  "tokenizer": "unicode_alphanumeric_casefold_v1",
  "document_count": 100,
  "average_document_length": 80.0,
  "document_frequencies": {"example": 12},
  "scope": "description of the actual reference corpus"
}
```

External document frequencies cover every query term matching a scored document.
The runner rejects missing matching terms, nonfinite parameters and invalid counts.
The saved BM25 statistics can also be reused across prespecified splits.

## Output files and metrics

The experiment root contains `experiment.json` (configuration, query IDs, input
and source hashes, library versions), `features.csv`, `bm25_stats.json`,
`budget_curve.csv` and `checksums.sha256`. Each `seed_*/budget_*` directory contains:

| File | Meaning |
|---|---|
| `predictions.csv` | All pairs, model probabilities/labels, official gold labels for evaluation, human/automatic source, acquisition rank and `final_label` |
| `acquisitions.csv` | Ordered audit log of exactly the purchased labels, rounds and probabilities before each query |
| `model.json` | Final coefficients/intercept or prior, feature names, scaler and settings; sufficient to reconstruct model probabilities |
| `summary.json` | Coverage, total human cost, selective risk, auto-error, per-class risks, automatic-only classification metrics and mixed-label accuracy |
| `per_query.csv` | Query-level costs and errors for paired comparisons and query bootstrap analysis |
| `mixed.qrels` | Binary TREC qrels combining purchased human labels and automatic labels |
| `gold.qrels` | Full binary human qrels for exactly the same evaluation pool |

`predicted_label` and `prob_relevant` always refer to the model, even on human
pairs. `final_label` is the mixed label. All-pool model accuracy includes acquired
training pairs and is recorded as a diagnostic rather than a held-out estimate.
The primary error statistics use automatic pairs only:

- Coverage = automatic pairs / all selected pairs.
- Selective risk = wrong automatic labels / automatic pairs.
- Auto-error = wrong automatic labels / all selected pairs.
- Mixed-label accuracy = 1 - auto-error (simulated humans supply official labels).

Undefined selective risk is JSON `null`, never zero. Class risks are grouped by
the automatic predicted label. `worst_observed_class_selective_risk` is accompanied
by `both_predicted_classes_accepted`. A class with no automatic predictions has
undefined selective risk, so the observed-class maximum alone does not establish
low risk in both classes. `query_mean_auto_error_rate` and pooled
`auto_error_rate` have different weighting. This baseline makes no formal guarantee.

The optional `--run-files` argument enables evaluation of official TREC runs:

```powershell
.\.venv-lexical\Scripts\python.exe scripts/run_lexical_baseline.py --manifest data/pilot/dl2019.jsonl --seeds 42 --budget-fractions 0.1 0.5 1 --run-files data/runs/system_a.run data/runs/system_b.run --output-dir results/baselines/lexical/dl2019_rankings
```

The runner then writes `system_scores.csv` and `system_query_scores.csv`, and adds
Kendall's tau-b to summaries and the budget curve. Scores are binary nDCG@10 on
the same selected topics, with unjudged gain zero and missing-query score zero.
Runs sort by descending score, then descending document ID for ties. Undefined
tau (fewer than two systems or constant score vectors) is `null`. Binary IDCG uses
all judged relevant documents for each query. Original graded nDCG measures a
different relevance scale and is not directly comparable with this binary output.
No official runs are bundled, so a label-only execution does not establish
system-ranking fidelity.

## Plots

The completed DL2019 results can be plotted without retraining:

```powershell
.\.venv-lexical\Scripts\python.exe -m pip install -r requirements-lexical-plots.txt
.\.venv-lexical\Scripts\python.exe scripts/plot_lexical_baseline.py
```

Figures are saved under `reports/figures/lexical/` as PNG and editable SVG:
`overview`, `risk_coverage`, `auto_error_budget`, `mixed_accuracy_budget`, and
`class_composition`. `plot_data.csv` contains the aggregated means and seed
ranges; `plot_metadata.json` records input and output hashes. Bands show ranges
across seeds, not confidence intervals. The untrained zero-budget reference and
undefined selective risk at full human coverage are excluded where appropriate.
Lines join evaluated budgets; they do not establish results at intermediate budgets.

## Verification

```powershell
.\.venv-lexical\Scripts\python.exe -m unittest discover -s tests -v
```

The baseline tests cover a hand-computed BM25 value, unique-document frequencies,
independent uncertainty-selection verification, unseen-gold mutation, zero-label
access, one-class seeds, exact budgets including `N-1`, deterministic model
reconstruction, split/audit isolation, analytical ranking fixtures, CSV/qrels
compatibility, checksum validation and refusal to overwrite results. Existing Stage 2
regression tests are included. The [validation record](lexical_baseline_validation.md)
documents the completed real-data runs.
