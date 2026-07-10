# P0 Migration Manifest

Date: 2026-06-16

No datasets, checkpoints, or outputs were deleted. Former paths were initially
preserved with compatibility symlinks where they were likely to be referenced by
older commands or other projects. Root data/output symlinks were retired on
2026-06-29 to keep the repository root clean; use the canonical paths below.

## Main moves

| Category | Old location pattern | New location |
| --- | --- | --- |
| Training scripts | `train*.py` | `scripts/training/` |
| Benchmark scripts | `benchmark_*.py`, `run_all_benchmarks.sh`, benchmark helpers | `scripts/benchmarks/` |
| Dataset generation scripts | `generate*.py`, `build_union_dataset.py` | `scripts/datasets/` |
| Dataset audit scripts | `dataset_leaks.py`, `fix_leaks.py`, `run_dataset_audit.py` | `scripts/audit/` |
| Analysis scripts | `analyze_noise.py`, `infer_doublets.py`, `pub_utils.py`, SNR helpers | `scripts/analysis/` |
| Plotting scripts | `scripts/plot_*`, plotting helpers | `scripts/plotting/` |
| Legacy documentation files | historical root docs | `docs/` |
| Root raw C1 folders | `C1_HF_5_10_*` | `data/raw/` |
| Existing datasets under `data/` | `data/<dataset>` | `data/processed/<dataset>` |
| Training outputs | `output` | `outputs/training/output` |
| Benchmark outputs | `results` | `outputs/benchmarks/results` |
| Dataset audits | `audit_results` | `outputs/audits/audit_results` |
| Logs | `logs` | `outputs/logs/logs` |
| Artifacts | `artifacts` | `outputs/artifacts/artifacts` |
| W&B runs | `wandb` | `outputs/wandb/wandb` |

## Retired root compatibility aliases

These root aliases were removed on 2026-06-29 without deleting their canonical
targets:

- `C1_HF_5_10_*` -> `data/raw/`
- `output` -> `outputs/training/output`
- `results` -> `outputs/benchmarks/results`
- `audit_results` -> `outputs/audits/audit_results`
- `logs` -> `outputs/logs/logs`
- `artifacts` -> `outputs/artifacts/artifacts`
- `wandb` -> `outputs/wandb/wandb`
