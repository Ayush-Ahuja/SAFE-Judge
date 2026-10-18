# Stage 2 Colab workflow

`notebooks/stage2_colab.ipynb` runs the full DL2019 judged pool: 9,260 pairs across
43 queries. The completed experiment is retained in
`results/stage2/dl2019_qwen35_full_v1/`; use a new run name when reproducing it.

## Start a run

1. From the repository root, run `python scripts/package_pilot.py`.
2. Open `notebooks/stage2_colab.ipynb` in Colab and select a GPU runtime.
3. Set a new `RUN` in the first code cell for a fresh experiment. Keep the name
   unchanged only when resuming that same experiment.
4. Run cells in order, authorize Drive and upload `dist/pilot_code.zip` when prompted.
5. Inspect pair counts, throughput, probability mass, truncation and completion status.
   Completion requires 9,260 unique predictions.
6. Review the failure CSV using [the protocol](failure_review.md). Replace the saved
   Drive copy with your completed CSV; rerun the final export cell.
7. Download `stage2_artifacts.zip`; a copy is also saved in Drive.

A new run's automatic report is a scaffold. The reviewed submission report, discussion
and proposed future budget are maintained separately in `reports/stage2_report.tex`.
The original run's frozen inputs are preserved under its `provenance/` directory.

## Storage and resume

Outputs persist under `MyDrive/SAFE-Judge/<RUN>/`. SQLite is written on the local VM;
backup snapshots are copied to Drive about every 120 seconds after completed batches.
Model and corpus caches remain on the VM. A restarted VM restores the manifest,
configuration and checkpoint. Model revision, scoring code, package versions and dtype
must match to resume. Use only one active runtime per run.

Drive is authoritative for manual annotations. Existing review CSVs are not overwritten
by an inference restart. Check the final artifact download before discarding a runtime.

## Model and progress

The model uses its native chat template with thinking disabled. The upload package
includes the published frozen manifest, so reproducing that pool needs no corpus
download. If the manifest is absent, preparation scans the canonical passage collection
without building a retrieval index; this fallback may download the corpus. Model
weights still need downloading unless already cached.

The progress display reads committed SQLite rows every two seconds. Zero completed
pairs during model loading or batch benchmarking is expected. The model loads once,
benchmarks batches 1/2/4/8/16 and chooses the fastest successful batch. CUDA OOM retries
halve batch size without dropping pairs or changing input length.

The first cell sets a five-hour session budget, including setup, with 20 minutes
reserved for evaluation/export. An interrupted run keeps committed pairs and resumes
the same manifest. The 100-pair batch-size-1 repeat measures numerical consistency;
it is not an independent full inference run.

## Analyses

The notebook saves accuracy, macro-F1, confusion counts, Brier score, ECE, per-query
metrics, overall/class-specific confidence sweeps, constant-label references and
10,000 whole-query bootstrap intervals. Temperature scaling is fitted on 10 queries
and evaluated on 33 disjoint queries. Threshold sweeps are descriptive and provide
no conformal guarantee.

Teacher generation, adapter training, CRC and domain-shift/system-ranking experiments
are planned Stage 3 work. A GPU run cannot automatically complete personal reviews
or validate the report's written interpretation.

## Local analysis and final report

The published run is already in the repository. Follow the CPU commands in the
[root README](../README.md#check-the-saved-experiment-without-a-gpu). No GPU or
checkpoint database is needed to regenerate evaluation from its predictions.

For another downloaded archive, extract outside the source tree or under ignored
`dist/`, then use:

```bash
python scripts/build_stage2_report.py --artifacts /path/to/extracted/artifacts --review reports/failure_review.csv
```

The submitted narrative is tied to the published prediction hash. Another experiment
requires a separately reviewed report, rather than reusing the existing interpretation.
The builder defaults to the repository's completed manual review and validates it
against the original sample. It leaves predictions and annotation text unchanged.

The LaTeX source compiles from `reports/` and loads the generated tables plus separate
vector PDFs in `reports/figures/stage2/`. Keep both the report source and these assets
when sharing a source bundle. The PNG versions support separate viewing and reuse.
