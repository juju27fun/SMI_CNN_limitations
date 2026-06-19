---
name: training
description: Use for P0 PyTorch training-loop work, validation loops, checkpoints, convergence metrics, parameter counts, profiling utilities, and compatibility with P1's use of P0/training_utils.py.
---

# Training

Use this skill when changing model training, evaluation passes, checkpointing, profiling helpers, or training utility code.

## Core Rules

- Keep training loops deterministic when a seed is provided.
- Use `model.train()` for training and `model.train(False)` or `model.eval()` under `torch.no_grad()` for validation/inference.
- Zero gradients before backpropagation, then call `loss.backward()` and `optimizer.step()`.
- Track epoch duration and current learning rate when a scheduler is active.
- Count trainable parameters with `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- Preserve `training_utils.py` as a shared utility surface because P1 imports its profiling helpers.

## Standard Outputs

- Per-epoch values: train loss, train accuracy, val loss, val accuracy, epoch duration, learning rate.
- Summary values: best validation accuracy, best epoch, final validation accuracy/loss, total training time, convergence time, parameter count, dataset size.
- Checkpoints should encode model, dataset, and run identity clearly enough to be reused by evaluation scripts.

## Convergence

- Convergence time is the cumulative epoch duration until validation accuracy reaches a fixed fraction of the best validation accuracy.
- Use `NaN` or omit the value only when the metric is impossible to define for the run.

## Profiling

- Profiling helpers should remain generic across P0 and P1.
- Preserve helpers for MACs, inference latency, peak RAM, and state-dict size.
- CPU latency should be measured with warmup and repeated runs; report median and p95 when possible.

## Validation

- Run focused tests for the touched utility or training path.
- For expensive training paths, run a one-epoch or smoke variant when available.
- If a full training run is skipped, state the command that would validate it and why it was skipped.
