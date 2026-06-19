# Canonical P0 Artifacts

Date: 2026-06-16

P0 is mostly a source of classification baselines and reusable datasets for
P1/P2/P3. This registry records the paths that should be preferred in new work.

## Data

| Role | Canonical path | Compatibility path |
| --- | --- | --- |
| Raw 2um C1 folder | `data/raw/C1_HF_5_10_2um_doublet2` | `C1_HF_5_10_2um_doublet2` |
| Raw 4um C1 folder | `data/raw/C1_HF_5_10_4um_doublet` | `C1_HF_5_10_4um_doublet` |
| Raw 10um C1 folder | `data/raw/C1_HF_5_10_10um_doublet` | `C1_HF_5_10_10um_doublet` |
| Noise background dataset | `data/processed/Noise` | `data/Noise` |
| Processed classification datasets | `data/processed/` | `data/<dataset_name>` symlinks |

## Outputs

| Role | Canonical path | Compatibility path |
| --- | --- | --- |
| Historical training outputs | `outputs/training/output` | `output` |
| Benchmark results | `outputs/benchmarks/results` | `results` |
| Dataset audits | `outputs/audits/audit_results` | `audit_results` |
| Logs | `outputs/logs/logs` | `logs` |

## Rule

For current paper-facing detection/segmentation claims, prefer P1 registries.
Use P0 primarily to cite classification baselines, raw/source data provenance,
or reusable noise/background data.
