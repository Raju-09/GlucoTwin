# GlucoTwin — Scripts

All scripts accept `--help` for usage information.

| Script | Purpose |
|---|---|
| `audit_data.py` | Inspect raw dataset; produce audit report |
| `build_dataset.py` | Build supervised windows and train/val/test splits |
| `evaluate.py` | Evaluate a model on a split; compare against persistence |

## Common options

```bash
# Audit
python scripts/audit_data.py --data-dir data/raw --out reports/audit.json

# Build dataset
python scripts/build_dataset.py --data-dir data/raw --out data/processed/

# Evaluate persistence baseline
python scripts/evaluate.py --model persistence --split test

# Evaluate learned model
python scripts/evaluate.py --model rf --split test
```

## Notes

- Scripts never modify raw data.
- Scripts write outputs to `reports/` or `artifacts/`.
- All scripts accept `--seed` to override the global random seed.
- Scripts log to stdout using `rich` for readability.
