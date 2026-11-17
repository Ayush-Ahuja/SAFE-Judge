# Experiment index

Results below are in `artifacts/final/`. The paper gives the main comparison;
the supplement summarizes additional experiments and limitations.

| Experiment | Evidence files |
|---|---|
| Raw assessor and query-error concentration | `raw_assessor_metrics.csv`, `query_error_diagnosis.csv`, `query_error_concentration.csv` |
| SAFE-Judge, LARA and budget curves | `policy_runs.csv`, `policy_summary.csv`, `policy_per_query.csv` |
| Shared-only and shuffled-query controls | `dl2019_mechanism_*`, `dl2020_component_summary.csv` |
| Separate class thresholds | `dl2020_threshold_*`, `supplement_summary.csv`, `covid_threshold_*` |
| Paid target calibration | `target_probability_calibration.csv`, `supplement_summary.csv` |
| Class-coverage constraints | `dl2020_class_aware_summary.csv` |
| Graded models and ranking diagnostics | `dl2020_density_*`, `dl2020_grade_distribution.csv`, `dl2020_graded_ranking_scores.csv` |
| Official DL 2020 run fidelity | `official_dl2020_*`, `ranking_query_intervals.csv`, `dl2020_maximum_rank_drop.csv` |
| Lexical/MiniLM baselines | `dl2019_*_baseline.csv`, `common_baseline_metrics.csv` |
| Oosterhuis/PPI/CRC | `oosterhuis_trecdl_*` |
| Early error predictors, repair and numerical sensitivity | `dl2019_early_error_models.csv`, `dl2019_discovery_summary.csv`, `dl2019_precision.csv` |
| Sparse calibration and release audits | `dl2019_sparse_crc.csv`, `dl2019_target_pool_crc.csv`, `dl2019_*audit*.csv`, `dl2020_progressive_audit.csv` |
| Manual failure analysis | `manual_failure_review.csv` and metadata |
| Inference efficiency | `inference_efficiency.csv` |
| Relevance-boundary study | `dl2020_boundary_*.csv` |

The implementation is `src/calibration/safe_judge.py`, with frozen settings in
`configs/safe_judge.json`. Run it with `scripts/run_safe_judge.py --method safe_judge`.
The runner maps this name to `selected_local` in saved result tables. Discovery
tables use the original identifier `local_unguarded`.

DL 2019 informed discovery. DL 2020 and reserved COVID use frozen policy settings,
but all are previously inspected public benchmarks. Full-50-topic COVID overlaps
the reserved forty topics. The Oosterhuis study uses an auxiliary released pool.
The boundary study lacks its original fitting source and is descriptive.
No fresh blind test, teacher/QLoRA training or economical release guarantee is claimed.
