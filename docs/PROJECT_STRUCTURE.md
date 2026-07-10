# P0 Project Structure

Date: 2026-06-16

P0 is the supervised classification and dataset-generation baseline project.
It now separates code, data, generated outputs, and historical material.

## Layout

| Folder | Purpose |
| --- | --- |
| `p0/` | Importable package for shared data transforms, training loops, profiling utilities, plotting aliases, and model-zoo aliases. |
| `models/` | 1D classification model definitions. |
| `scripts/training/` | Training entry points. |
| `scripts/benchmarks/` | Benchmark launchers and benchmark measurement scripts. |
| `scripts/datasets/` | Dataset generation and dataset UI scripts. |
| `scripts/audit/` | Dataset leak and audit utilities. |
| `scripts/analysis/` | Noise, SNR, inference, and publication utility scripts. |
| `scripts/plotting/` | Figure and report plotting scripts. |
| `configs/` | Dataset/model configuration files. |
| `docs/` | Documentation and reports. |
| `data/raw/` | Raw C1 OFI folders. |
| `data/processed/` | Processed classification, noise, YOLO staging, and derived datasets. |
| `outputs/training/` | Historical `output/` training artifacts. |
| `outputs/benchmarks/` | Historical `results/` benchmark artifacts. |
| `outputs/audits/` | Dataset audit outputs. |
| `outputs/figures/` | Figures moved from the project root. |
| `outputs/logs/` | Logs. |
| `outputs/artifacts/` | Local W&B/artifact exports. |
| `outputs/wandb/` | W&B run folders. |
| `archive/` | Historical code and reports kept for traceability. |

## Compatibility

Former high-use Python commands are kept as root wrapper files, so commands such
as `python train4classes.py` work without setting `PYTHONPATH`. Data and output
compatibility symlinks were retired on 2026-06-29 to keep the repository root
clean; use canonical `data/...` and `outputs/...` paths instead. Reusable Python
code should prefer `p0.*` imports; legacy imports such as
`from training_utils import measure_cpu_latency` remain supported for P1
compatibility.

## Root command wrappers

The root stays intentionally small. Keep wrappers only for high-use commands:
`train.py`, `train4classes.py`, `benchmark_zoo.py`, `benchmark_base.py`,
`generate_dataset.py`, `generate_ui.py`, `analyze_noise.py`,
`dataset_leaks.py`, `fix_leaks.py`, and `run_dataset_audit.py`.

Less common tools should be run from their canonical `scripts/` paths, for
example `python scripts/training/train3classes_proof.py` or
`python scripts/analysis/infer_doublets.py`. Shared code should be imported
from `p0.*`; root modules such as `train.py`, `training_utils.py`, and
`pub_utils.py` exist for compatibility only.
