---
name: benchmarking
description: Use for P0 multi-model benchmark orchestration, model-vs-dataset matrices, leaderboards, W&B comparisons, sweeps, and benchmark exports.
---

# Benchmarking

Use this skill when coordinating full benchmark runs or editing scripts that aggregate model results.

## Workflow

1. Define the model x dataset matrix.
2. Build a config for each run with canonical metric fields.
3. Train each model/dataset pair with W&B logging or offline result capture.
4. Evaluate each checkpoint on the relevant test sets.
5. Export summaries to CSV/markdown and optionally W&B tables.
6. Generate comparison plots from raw exported data.

## Leaderboards

Typical columns:

- rank
- model
- dataset
- best validation accuracy
- final validation accuracy
- macro F1
- total training time
- convergence time
- model size
- best epoch

Rank by the project-specific primary metric unless the task specifies another criterion.

## Sweeps

- Use W&B sweeps for hyperparameter search when online tracking is wanted.
- Supported methods are grid, random, and bayesian search.
- Use early termination such as Hyperband for expensive spaces.
- Keep sweep configs reproducible and commit-worthy when they define scientific results.

## Validation

- Prefer `--dry-run`, `--skip-existing`, or a tiny run matrix before launching long benchmarks.
- Check that aggregation scripts ignore incomplete runs or report them explicitly.
- For published results, regenerate leaderboards from raw run outputs rather than manual edits.
