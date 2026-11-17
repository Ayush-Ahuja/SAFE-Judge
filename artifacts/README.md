# Experimental artifacts

`final/` contains the paper's saved scores, metrics and statistical results for
DL 2019, DL 2020 and TREC-COVID, plus the qualified Oosterhuis reference study.

| Files | Contents |
|---|---|
| `*_predictions.csv`, `raw_assessor_metrics.csv` | Pair IDs, grades, probabilities and raw model metrics |
| `policy_runs.csv`, `policy_summary.csv`, `policy_per_query.csv` | Budgets, seeds, label/class/query metrics and policy times |
| `paired_query_intervals.csv`, `ranking_query_intervals.csv` | Conditional query-bootstrap comparisons |
| `official_dl2020_*` | Official-run scores and rank changes |
| `supplement_summary.csv`, `target_probability_calibration.csv` | COVID thresholds and paid target calibration |
| `dl2019_*`, `dl2020_*` | Discovery studies, ablations and alternative methods |
| `manual_failure_review.csv` | Fifty completed manual case reviews |
| `evidence_sources.json`, `allocation_checksums.json`, `SHA256SUMS.txt` | Source provenance and checksums |

The separate `dist/safe_judge_final_evidence.zip` also contains inference receipts,
logits and purchase ledgers. It is excluded from Git and intended as a release
attachment. Its size and hash are in `reports/final/submission_manifest.json`.
Historical source paths in receipts identify the original experiments; archived
paths are not required for the default replay or figure build.

See the [experiment index](../docs/experiment_index.md),
[reproduction guide](../docs/reproduce_final.md) and [data notice](../DATA_NOTICE.md).
