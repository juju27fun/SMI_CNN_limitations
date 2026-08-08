---
name: p0-wandb
description: Use for P0 Weights & Biases run setup, canonical logging, summaries, artifacts, sweeps, reports, and offline or disabled behavior.
---

# P0 W&B

Use this skill when modifying W&B instrumentation or benchmark reporting.

## Project

- W&B project: `particle-benchmark`.
- Group runs by `model_name`.
- Tags should include dataset/version/experiment identifiers.
- `job_type` should be `training` or `evaluation`.

## Run Config

Every run config should include:

- `model_name`
- `model_size_params`
- `dataset`
- `dataset_size`
- `epochs`
- `batch_size`
- `learning_rate`
- `optimizer`
- `seed`
- `patience`

Also log architecture-specific parameters such as depth, width, dropout, kernel size, or family.

## Metric Definitions

Define epoch as the x-axis:

```python
run.define_metric("epoch")
run.define_metric("train/*", step_metric="epoch")
run.define_metric("val/*", step_metric="epoch")
run.define_metric("val/accuracy", summary="max", goal="maximize")
run.define_metric("val/loss", summary="min", goal="minimize")
```

## Logging

- Per-epoch metrics must use canonical keys from the `metrics` skill.
- Summary metrics belong in `run.summary`, not only in logged history.
- Do not call `wandb.log()` in tight inner loops unless rate-limited.
- For `--no-wandb` or offline paths, provide a no-op run object or conditional logging so training still works.

## Artifacts And Reports

- Dataset artifacts should preserve dataset identity and version.
- Best checkpoints may be logged as model artifacts when W&B is enabled.
- Reports should include leaderboard tables, loss/accuracy curves, confusion matrices, scatter plots, and short conclusions.

## Validation

- Validate both online/offline code paths when behavior differs.
- For dry validation, prefer offline mode or a `_NullRun` style stub.
- Check that required config and summary fields are populated before `run.finish()`.
