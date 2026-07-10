# Canonical classification inputs and artifacts

Dataset versions are registered in the workspace `datasets/registry/index.yaml`.
Use IDs in run metadata rather than mutable aliases.

| Role | Dataset ID |
| --- | --- |
| Baseline three-class classification | `p0-baseline-3class@v1` |
| Baseline four-class classification | `p0-baseline-4class@v1` |
| Active dual-clean class folders | `particles2snr-f-dual-clean-c1-class-folders@v1` |
| Event classification | `particles2snr-f-dual-clean-c1-events@v1` |
| Noise/background control | `noise@v1` |

All run material belongs under `artifacts/SMI_CNN_limitations/<kind>/<run-id>`
in the parent workspace. A paper-facing run must include `run.json`, the dataset
ID/version, repository revisions, command, timestamps, and status.
