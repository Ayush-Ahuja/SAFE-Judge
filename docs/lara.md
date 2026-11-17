# LARA baseline

`src/baselines/lara.py` implements the released binary LARA calibrator and
uncertainty acquisition. Reference revision and hashes are in
`configs/lara_reference.json`; settings are in `configs/lara.json`.

Global acquisition selects across the pool. Topic grouping assigns approximately
equal query quotas and visits groups round-robin. Both update one shared online
calibrator after each purchased label. Raw versus normalized probabilities is a
separate preprocessing choice. Every purchase counts toward the human budget.

Two deliberate endpoint fixes differ from upstream: budget `N-1` buys exactly
`N-1` labels, and constant min-max input falls back to unchanged probabilities.
Upstream parity claims exclude those cases. Saved checks covered 48 synthetic/
real-score settings with identical acquisition order, labels and probabilities;
offline fixtures are in `tests/fixtures/lara_reference.json`.

The final replay runs directly on released scores:

```powershell
python scripts/reproduce_final.py --dataset dl2020 --method lara_1_raw
python scripts/reproduce_final.py --dataset dl2020 --method lara_topics_upstream_smooth
```

For raw manifests, use `scripts/run_lara_baseline.py --help`. Optional upstream
verification uses `requirements-lara-reference.txt` and
`scripts/verify_lara_reference.py --download`. This is a same-score adaptation;
it does not reproduce the original paper's assessor or published table values.
