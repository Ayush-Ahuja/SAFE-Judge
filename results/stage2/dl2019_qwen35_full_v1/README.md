# Completed Stage 2 experiment

This is the full DL2019 run reported in `reports/stage2_report.pdf`: 9,260 pairs,
43 queries, Qwen3.5-4B, NF4 4-bit weights, FP16, 512-token inputs and seed 42.
Main scoring used batch 8 on a Tesla T4. No teacher generation or adapter training
was executed in this run.

## Recorded artifacts

- `predictions.csv`: raw probabilities, logits, labels and input diagnostics.
- `experiment.json`: frozen manifest/model/scoring identity.
- `environment.json` / `environment.txt`: actual Colab hardware and package versions.
- `sessions.json`, `progress.json`, `batch_benchmark.json`: completion, runtime,
  throughput, memory and batch measurements.
- `consistency.csv` / `consistency.json`: the 100-pair batch-size-1 repeat.
- `preparation.json` / `storage.json`: measured preparation and cache/storage use.
- `evaluation/`: saved metrics, bootstrap intervals and query-disjoint calibration.
- `inference.log` / `workflow.log`: original execution logs.
- `failure_review.csv` / `failure_review.metadata.json`: original unfilled 50-error
  export. Completed personal annotations are in `reports/failure_review.csv`.
- `provenance/colab.yaml`: original session configuration, including then-unused
  future-method settings. `experiment.json` identifies the scoring settings actually used.
- `provenance/executed_notebook.ipynb`: notebook source exported with the experiment.

The active notebook has progress-display and documentation updates. The four scoring
files named in `scripts/run_pilot.py::code_hash` remain byte-identical to the exported
run and reproduce its recorded pipeline hash. The original source-ZIP fingerprint
in `environment.json` is historical; regenerated upload ZIPs have different hashes.

The frozen manifest is in `data/pilot/dl2019.jsonl`, with preparation metadata beside
it. Predictions and manifest retain their original bytes. Check from the repository root:

```bash
sha256sum --check results/stage2/dl2019_qwen35_full_v1/checksums.sha256
```

The SQLite checkpoint, when present locally, is ignored by Git. It supports resuming
inference but is unnecessary for reproducing the reported analyses. Temporary report
scaffolds, duplicate diagnostic images and incomplete earlier runs are not part of
this published result. The final report uses the separate figures in `reports/figures/stage2/`.
