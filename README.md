# SAFE-Judge

**Calibrated and Selective LLM Relevance Assessment under Domain Shift**

This repository contains the completed Stage 2 pilot:
frozen Qwen3.5-4B relevance assessment on **all 9,260 judged TREC DL 2019 pairs
across 43 queries**. The model uses NF4 4-bit weights, FP16 computation, a 512-token
input limit and the canonical Yes/No answer probabilities. Grades 2 and 3 count
as relevant. No inputs were truncated in the completed run.

## Results

| Measurement | Completed pilot |
| --- | ---: |
| Accuracy | 79.40% |
| Macro-F1 | 0.7368 |
| Brier score | 0.1485 |
| ECE, 10 equal-width probability bins | 0.0688 |
| Main scoring time | 33.12 minutes |
| GPU | Tesla T4 |
| Manually reviewed failures | 50 |

The [report](reports/stage2_report.pdf) presents confidence sweeps, query-bootstrap
intervals, query-disjoint temperature scaling, numerical consistency, efficiency
and the manual error analysis. Raw predictions, logits, splits and execution logs
are retained in the [completed run](results/stage2/dl2019_qwen35_full_v1/).

These are frozen-model pilot results. Teacher generation, adapter training, conformal
risk control, domain-shift validation and retrieval-system ranking experiments are
planned work, described in the report. The pilot does not establish a conformal guarantee.

## Classical lexical baseline

The classical lexical baseline uses CPU logistic regression on BM25,
query-term overlap and document length, with budgeted uncertainty sampling and
mixed human/automatic labels. It includes
shared query splits, exact seed-inclusive human budgets, acquisition logs,
binary qrels and optional system-ranking evaluation. The
[baseline guide](docs/lexical_active_learning.md) documents installation, provenance,
protocol and experiment commands. The
[validation record](docs/lexical_baseline_validation.md) describes completed checks.

## Repository layout

| Path | Contents |
| --- | --- |
| `notebooks/stage2_colab.ipynb` | Full Colab workflow, progress display and Drive snapshots |
| `configs/default.yaml` | Stage 2 settings pinned to the recorded model revision |
| `src/` | Prompting, model scoring, evaluation metrics and checkpoint/data utilities |
| `scripts/` | Data preparation, inference, evaluation, review export and report building |
| `data/pilot/` | Frozen DL2019 manifest and sampling metadata |
| `results/stage2/dl2019_qwen35_full_v1/` | Completed run, analyses and execution provenance |
| `reports/` | Stage 1 proposal, Stage 2 report/source, completed reviews and separate figures |
| `docs/` | Colab reproduction instructions and manual-review protocol |
| `tests/` | Offline correctness, interruption/resume and notebook persistence tests |
| `dist/` | Generated ZIPs and LaTeX builds; excluded from Git |

## Continuous integration

The [CI workflow](.github/workflows/ci.yml) runs on pushes, pull requests and manual
dispatches in GitHub Actions. It checks Python 3.12 on GitHub-hosted Ubuntu and
Windows machines, installs the CPU test/plot dependencies, validates the frozen
DL2019 manifest, runs all offline tests and checks plotting CLI imports. Dependency
downloads are cached. The checks require no GPU, model weights, API keys or new
datasets. Full inference and experiment sweeps are independent of CI.

## CPU validation

Manifest validation needs only Python. The full test suite also covers the lexical
baseline and requires its CPU dependencies. The following commands perform local
validation from the repository root:

```bash
python scripts/run_pilot.py --manifest data/pilot/dl2019.jsonl --validate-only
python -m pip install -r requirements-lexical.txt
python -m unittest discover -s tests -v
```

Frozen experiment files and scoring code have recorded SHA256 checksums.
The WSL/Linux verification command is:

```bash
sha256sum --check results/stage2/dl2019_qwen35_full_v1/checksums.sha256
```

Evaluation can be regenerated in a separate directory while preserving the
published run:

```bash
python scripts/evaluate_pilot.py \
  --manifest data/pilot/dl2019.jsonl \
  --predictions results/stage2/dl2019_qwen35_full_v1/predictions.csv \
  --output-dir dist/evaluation \
  --bootstrap-samples 10000 --calibration-queries 10 --seed 42
```

Calibration uses 10 queries and tests on the other 33. This is a temperature-scaling
experiment, not query-level conformal calibration.

## Report generation

Only the plotting dependency is needed for this CPU step; PyTorch and model downloads
are unnecessary:

```bash
python -m pip install matplotlib==3.10.1
python scripts/build_stage2_report.py
mkdir -p dist/report-build
cd reports
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../dist/report-build stage2_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../dist/report-build stage2_report.tex
cp ../dist/report-build/stage2_report.pdf stage2_report.pdf
```

The builder validates the manifest, prediction hash, metrics and completed reviews
before generating `reports/stage2_results.tex` and nine separate PDF/PNG figures in
`reports/figures/stage2/`. The narrative remains in `reports/stage2_report.tex`.
The `--run-dir results/stage2/dl2019_qwen35_full_v1` option explicitly selects the
published run when other run directories are present.

## Inference reproduction

The [Colab guide](docs/colab.md) describes the inference environment and execution
protocol. The upload package is generated by:

```bash
python scripts/package_pilot.py
```

The upload package includes the frozen manifest, avoiding a new corpus download.
The notebook installs the inference packages, runs the offline tests, validates the
full judged pool, scores it, evaluates saved predictions and exports the run.
Each experiment uses a distinct `RUN` name. The notebook provides resumable
checkpoints and periodic Drive backups. Failure review is performed manually.

Other GPU environments require a compatible CUDA PyTorch build and the dependencies
in `requirements-pilot.txt`; the command-line scripts are shared across environments.
The actual Colab versions and configuration are recorded with the completed run; HPC execution is
not claimed as part of this pilot.

## Data and manual review

The frozen manifest contains benchmark queries, MS MARCO passage texts and official
TREC DL 2019 grades, obtained through
[`ir_datasets`](https://ir-datasets.com/msmarco-passage.html#msmarco-passage/trec-dl-2019/judged).
It is a judged pool, not the complete MS MARCO corpus or a retrieval index.

[Completed reviews](reports/failure_review.csv) contain the 50 manually reviewed
errors; [sampling metadata](reports/failure_review.metadata.json) ties them to the
manifest and predictions. The original unfilled export is preserved with the run.
Official evaluation labels were retained, including cases marked as judgment disagreement.
The [review protocol](docs/failure_review.md) and [AI-use statement](reports/ai_usage_statement.md)
describe the annotation procedure and use of AI assistance.

Tests use synthetic predictions and simulated failures. They validate software behavior;
they are separate from the recorded GPU experiment and manual annotations.
