---
name: metrics
description: Use for P0 metric naming, required W&B keys, run summary conventions, OOD metrics, and benchmark config schema.
---

# Metrics

Use this skill when adding, renaming, consuming, or validating metrics.

## Naming

- Use `/` separators for namespaces: `train/loss`, `val/accuracy`.
- Use snake_case for metric components.
- Avoid hyphens, spaces, and commas.
- Keep metric names stable because W&B dashboards and exports depend on them.

## Per-Epoch Metrics

Required keys:

- `epoch`
- `train/loss`
- `train/accuracy`
- `val/loss`
- `val/accuracy`
- `epoch_time_sec`
- `learning_rate` when a scheduler exists

## Summary Metrics

Required or expected keys:

- `best_val_accuracy`
- `best_epoch`
- `total_training_time_sec`
- `convergence_time_sec`
- `final_val_accuracy`
- `final_val_loss`
- `model_size_params`
- `dataset_size`
- `early_stopped_epoch` when early stopping triggers
- `generalization_gap` when a separate test set makes it meaningful

## Evaluation Metrics

Use namespaced prefixes for test sets, for example `test_synthetic/accuracy` or `test_real/loss`.

Expected evaluation artifacts:

- confusion matrix
- per-class F1 table
- F1 bar chart
- PR curve
- ROC curve

## OOD Metrics

Use the `noise_ood/` namespace for noise/OOD evaluation:

- `noise_ood/auroc_msp`
- `noise_ood/auroc_energy`
- `noise_ood/fpr95_msp`
- `noise_ood/fpr95_energy`
- `noise_ood/avg_max_softmax_id`
- `noise_ood/avg_max_softmax_noise`
- `noise_ood/avg_entropy_noise`
- `noise_ood/num_noise_samples`

Put scalar OOD metrics in `run.summary` when they represent final run outcomes.

## Validation

- Confirm all required per-epoch metrics are emitted once per epoch.
- Confirm final summaries are set after training and evaluation.
- Update consumers, exports, and plotting scripts when renaming any metric.
