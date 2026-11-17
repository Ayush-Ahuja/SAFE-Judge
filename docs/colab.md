# GPU workflows

Use [CPU reproduction](reproduce_final.md) to rebuild results without inference.
For fresh inference, choose a GPU runtime and run notebook cells in order.

## DL 2019 pilot

1. Run `python scripts/package_pilot.py` from the repository root.
2. Open `notebooks/stage2_colab.ipynb` in Colab.
3. Choose a new `RUN` name, authorize Drive and upload `dist/pilot_code.zip`.
4. Run all cells. Completion requires 9,260 unique predictions over 43 queries.
5. Complete the [manual review](failure_review.md), rerun the export cell and
   download `stage2_artifacts.zip`.

Outputs persist in `MyDrive/SAFE-Judge/<RUN>/`. Local SQLite snapshots are copied
to Drive during inference. Resume with the same run name only when the manifest,
configuration, model and scoring code are unchanged; use one runtime per run.
Existing manual-review CSVs are preserved.

## TREC-COVID

After restoring the canonical inference export from the evidence package, run
`python scripts/build_proposal_covid_colab.py`. Upload the generated ZIP and use
`notebooks/qwen_covid_proposal_colab.ipynb`. Completion requires 5,000 unique
predictions over fifty topics. The rubric counts grades 1 and 2 as relevant.

Both workflows use frozen four-bit Qwen3.5-4B inference with a 512-token total
input limit. Exported receipts record source hashes, prompts, package versions,
truncation and measured GPU time. Download and inspect the completed artifacts
before ending the runtime; model weights and resume databases stay outside Git.
