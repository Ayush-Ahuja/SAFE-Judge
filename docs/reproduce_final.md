# Reproduction

Run commands from the repository root with Python 3.12. The default workflow uses
released probabilities and tables, without GPU inference or a research cache.

## Tests, figures and replay

```powershell
python -m pip install -r requirements-final.txt
python -m unittest discover -s tests -v
python scripts/build_final_report.py
python scripts/verify_final_artifacts.py
python scripts/run_safe_judge.py --dataset dl2020 --method safe_judge --seed 6000
python scripts/run_safe_judge.py --dataset dl2020 --method lara_1_raw --seed 6000
python scripts/run_safe_judge.py --dataset covid_reserved --method safe_judge --seed 6000
```

Datasets: `dl2019`, `dl2020`, `covid_reserved`, `covid_all`.
Methods: `safe_judge` (SAFE-Judge), `lara_1_raw`, `lara_1_upstream_smooth`,
`lara_topics_raw`, `lara_topics_upstream_smooth`, `raw_confidence`, `random_review`.
Use `--human-fraction` for the budget (default `0.5`). Published DL 2020/COVID
allocations use seeds 6000-6009 and fractions 0.1, 0.25, 0.5, 0.75 and 0.9.
DL 2019 discovery uses its earlier seed block.

Replays write to ignored `results/reproduction/` and check exact budgets, unique
purchases and published metrics when a matching seed/budget row exists. Figure
generation reads `artifacts/final/`; PNG previews are ignored, vector PDFs retained.

## PDFs and evidence

From `reports/final/`, using Tectonic (recorded build: 0.17.0):

```powershell
tectonic paper.tex
tectonic supplement.tex
```

Tectonic downloads standard TeX packages on first use. The paper uses ACM
`acmart`; the supplement uses the proposal/Stage 2 article style. Then, from
the repository root:

```powershell
python scripts/package_final_submission.py --skip-evidence-zip
```

This merges the PDFs and refreshes checksums. Building the full evidence ZIP
without that flag also requires the original inference exports and historical
ledgers. Verify a supplied ZIP with:

```powershell
python scripts/verify_final_artifacts.py --zip dist/safe_judge_final_evidence.zip
```

## Additional experiments

Extract the evidence ZIP's `replay_inputs/` contents into the repository root,
preserving paths, then run:

```powershell
python scripts/run_final_supplement.py
python scripts/analyze_final_supplement.py
```

This resumes the COVID threshold/calibration studies and aligned baseline check.
Input/source guards reject changed designs. Settings and seeds are recorded in
the experiment receipts. Other drivers may require their original input exports;
see the [experiment index](experiment_index.md) and [baselines](baselines.md).

GPU inference is described in [colab.md](colab.md). No inference is needed for
the default replay or paper build. The supplement explains the conditional
confidence intervals, overlapping COVID cohorts and exploratory evaluation scope.
