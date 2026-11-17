# SAFE-Judge

**Shared Adaptive Feedback for Escalation in Relevance Assessment**

Ayush Ahuja | COL764, IIT Delhi | November 2026

SAFE-Judge allocates human review around a frozen Qwen3.5-4B relevance assessor.
It combines shared and query-specific error estimates to choose the next pair
for review. Purchased judgments replace model labels; other labels stay unchanged.

## Paper and reproduction

- [Paper](reports/final/paper.pdf) and [LaTeX](reports/final/paper.tex)
- [Experiment summary](reports/final/supplement.pdf) and [combined report](reports/stage3_report.pdf)
- [Reproduction commands](docs/reproduce_final.md)
- [Experiment index](docs/experiment_index.md) and [baseline setup](docs/baselines.md)
- [Results and data files](artifacts/README.md)
- [AI usage](reports/ai_usage_statement.md), [data notice](DATA_NOTICE.md) and [citation](CITATION.cff)

## Results

At 50% human review, including calibration and sentinel purchases:

| Pool | SAFE-Judge query-average error | LARA global/raw | SAFE-Judge selective error |
|---|---:|---:|---:|
| DL 2019, discovery | 2.128% | 4.452% | See supplement |
| DL 2020 | 0.539% | 1.086% | 1.333% |
| COVID, 40 reserved topics | 4.458% | 5.250% | 8.915% |

Query-average error divides each query's automatic mistakes by all its pairs;
selective error divides mistakes by automatic pairs. The conditional comparison
with LARA favors SAFE-Judge on DL 2020; its interval includes zero on reserved
COVID topics. DL 2020 retains only 0.30% of original positives automatically.
LARA has lower binary retrieval-score MAE despite similar system ordering.

## Quick start

Python 3.12, from the repository root:

```powershell
python -m pip install -r requirements-final.txt
python -m unittest discover -s tests -v
python scripts/build_final_report.py
python scripts/run_safe_judge.py --dataset covid_reserved --method safe_judge
```

These commands use saved probabilities and need no GPU or model download.

## Code

| Path | Purpose |
|---|---|
| `src/calibration/safe_judge.py` | SAFE-Judge |
| `configs/safe_judge.json` | Frozen method settings |
| `src/baselines/` | Lexical, MiniLM and LARA baselines |
| `src/evaluation/` | Label, query and retrieval metrics |
| `scripts/`, `tests/` | Experiment runners, report generation and correctness checks |
| `artifacts/final/`, `reports/final/` | Numerical evidence and paper sources |
| `notebooks/` | GPU workflows |

DL relevance is grade >=2; COVID relevance is grade >=1. COVID uses 100 judged
pairs per topic and a fixed 10-development/40-evaluation split. Human review is
simulated with existing judgments.

Weights, corpus caches, environments and scratch files are ignored. Full ledgers
and inference receipts are in the separate evidence ZIP. Reported ablations and
negative results retain their code; obsolete utilities are archived outside Git.
