# Particle Benchmark — Implementation & W&B Reference

> The benchmark pipeline is **implemented** in `benchmark.py`. It wraps the training loop from `train.py` with structured W&B metric logging, convergence tracking, early stopping, and full post-training evaluation on both synthetic and (optionally) real test sets.

## Implemented Pipeline

The benchmark runs in **5 phases**:

| Phase | Function | What it does |
|-------|----------|--------------|
| Phase 1 | `log_pre_training()` | Logs config, model size, dataset size to W&B and stdout |
| Phase 2 | `run_training_loop()` | Full training with per-epoch metrics, best-model tracking, convergence timing, early stopping |
| Phase 3 | `run_post_evaluation()` | Evaluates best model on synthetic test set (and optionally real test set), logs confusion matrix, F1 table, bar charts, PR/ROC curves |
| Phase 4 | `extract_features()` + `plot_dimensionality_reduction()` | PCA and t-SNE visualizations of the latent space (fc1 features) for test sets and noise separation |
| Phase 5 | `run_ood_evaluation()` | OOD noise evaluation with 5 methods (MSP, Energy, ODIN, Mahalanobis, Energy_tuned). Logs AUROC, FPR@95, AUPR, score histograms, ROC comparison, temperature sweep, per-class analysis, and silhouette score |

### Quick start

```bash
# Synthetic test set only
python benchmark.py --data-dir dataset

# With real test set (measures generalization gap)
python benchmark.py --data-dir dataset --real-test-dir dataset_real/test

# With noise samples for OOD evaluation (Phase 5)
python benchmark.py --data-dir dataset --noise-dir Noise

# Full benchmark: synthetic + real test + OOD noise
python benchmark.py --data-dir dataset --real-test-dir dataset_real/test --noise-dir Noise

# Offline W&B mode
python benchmark.py --data-dir dataset --wandb-offline

# Custom hyperparameters
python benchmark.py --data-dir dataset --epochs 200 --lr 1e-3 --batch-size 64 --patience 20

# Disable LR scheduler (constant LR)
python benchmark.py --data-dir dataset --scheduler none
```

### Logged metrics summary

| Category | Metrics |
|----------|---------|
| Per-epoch | `epoch`, `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `epoch_time_sec`, `learning_rate` |
| Summary | `best_val_accuracy`, `best_epoch`, `total_training_time_sec`, `convergence_time_sec`, `final_val_accuracy`, `final_val_loss`, `model_size_params`, `dataset_size` |
| Evaluation | `{prefix}/accuracy`, `{prefix}/loss`, `{prefix}/confusion_matrix`, `{prefix}/f1_per_class`, `{prefix}/f1_bar_chart`, `{prefix}/pr_curve`, `{prefix}/roc_curve` |
| Dim. reduction | `test_synthetic/pca`, `test_synthetic/tsne`, `test_real/pca`, `test_real/tsne`, `noise_separation/pca`, `noise_separation/tsne` |
| OOD (summary) | `noise_ood/auroc_{method}`, `noise_ood/fpr95_{method}`, `noise_ood/aupr_{method}` (for msp, energy, odin, mahalanobis, energy_tuned), `noise_ood/avg_max_softmax_id`, `noise_ood/avg_max_softmax_noise`, `noise_ood/avg_entropy_id`, `noise_ood/avg_entropy_noise`, `noise_ood/silhouette_score`, `noise_ood/num_noise_samples` |
| OOD (plots) | `noise_ood/msp_histogram`, `noise_ood/energy_histogram`, `noise_ood/odin_histogram`, `noise_ood/mahalanobis_histogram`, `noise_ood/prediction_distribution`, `noise_ood/roc_comparison`, `noise_ood/temperature_sweep`, `noise_ood/threshold_analysis` |
| OOD (tables) | `noise_ood/summary_table`, `noise_ood/operating_points`, `noise_ood/per_class_analysis` |
| Conditional | `early_stopped_epoch` (if triggered), `generalization_gap` (if real test set provided) |

All runs are logged to the W&B project **`particle-benchmark`** (see `.claude/rules/metrics-conventions.md` for the full specification).

---

# Weights & Biases — Reference Guide

> This section is a **reference guide** covering every W&B feature relevant to the particle-classification benchmark: how each feature works, the Python API to use, and concrete code patterns.

---

## 1. Foundational Concepts

### 1.1 Run (`wandb.Run`)

A **Run** is the atomic unit of computation in W&B. Each training or evaluation script creates one Run.

**Key attributes:**

| Attribute | Description |
| --- | --- |
| `run.config` | Dictionary-like object storing **hyperparameters & input settings** (learning rate, batch size, model architecture…). Logged once at init. |
| `run.summary` | Dictionary storing **single aggregate values** per metric (by default the *last* logged value). Can be overridden to store *best* value. |
| `run.name` | Human-readable display name (can be customized). |
| `run.group` | Groups related runs together (e.g. same model, different seeds). |
| `run.tags` | List of string tags for filtering and organizing. |
| `run.id` | Unique identifier. |

### 1.2 Initializing a Run

```python
import wandb

config = {
    "model_name": "Conv1DClassifier",
    "model_size_params": num_params,
    "dataset": "synthetic_v1",
    "dataset_size": train_size,
    "epochs": 150,
    "batch_size": 32,
    "learning_rate": 6e-4,
    "optimizer": "Adam",
    "weight_decay": 0.0001,
    "decimate": 4,
    "input_length": 625,
    "num_classes": 3,
    "dropout_conv": 0.2,
    "dropout_fc": 0.5,
    "val_split": 0.2,
    "convergence_threshold": 0.95,
    "has_real_test": False,
    "patience": 0,
    "seed": 42,
    "scheduler": "cosine",
}

run = wandb.init(
    project="particle-benchmark",
    config=config,
    group="Conv1DClassifier",                     # group runs by model
    tags=["synthetic_v1", "benchmark"],            # filterable tags
    name="Conv1D-synthetic_v1-run1",               # {model}-{dataset}-{run_id}
    job_type="training",
)
```

### 1.3 `wandb.Run.config` — Experiment Configuration

Stores all **independent variables** of the experiment. This is what appears in the config columns of the Runs Table and can be used to **group, filter and compare** runs.

```python
# Set at init (benchmark.py passes the full config dict)
wandb.init(config=config)

# Or update later
run.config["optimizer"] = "AdamW"
run.config.update({"scheduler": "cosine", "weight_decay": 1e-4})
```

**Current benchmark config includes:** `model_name`, `model_size_params`, `dataset`, `dataset_size`, `epochs`, `batch_size`, `learning_rate`, `optimizer`, `seed`, `patience`, `scheduler`, plus architecture-specific params (`decimate`, `input_length`, `num_classes`, `dropout_conv`, `dropout_fc`, `val_split`, `convergence_threshold`, `has_real_test`, `weight_decay`).

---

## 2. Logging Metrics — `wandb.Run.log()`

This is the **core API** for sending time-series data to W&B.

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch_time_sec": epoch_duration,
    })
```

**Important rules:**

- Each call to `wandb.log()` creates a new **step** (auto-incrementing integer).
- Use the `/` separator to organize metrics into **sections** in the UI (e.g. `train/loss` vs `val/loss`).
- Do not call `wandb.log()` more than a few times per second.
- Metric names must match `/^[_a-zA-Z][_a-zA-Z0-9]*$/` — avoid hyphens, spaces, commas.

### 2.1 Custom X-Axis with `define_metric()`

By default, charts use the internal `_step` as x-axis. To use **epoch** instead:

```python
run.define_metric("epoch")
run.define_metric("train/*", step_metric="epoch")
run.define_metric("val/*", step_metric="epoch")
```

Now all `train/` and `val/` metrics will plot against `epoch` automatically.

### 2.2 Summary Metrics — Best Values

By default, `run.summary` holds the **last** value logged for each key. Override it to store the **best** value:

```python
# Track best accuracy manually
if val_acc > best_val_acc:
    best_val_acc = val_acc
    run.summary["best_val_accuracy"] = best_val_acc
    run.summary["best_epoch"] = epoch

# Or use define_metric for automatic tracking
run.define_metric("val/accuracy", summary="max", goal="maximize")
run.define_metric("val/loss", summary="min", goal="minimize")
```

Summary values appear in the **Runs Table** and can be used for sorting, filtering and bar charts.

### 2.3 All Metrics Logged by `benchmark.py`

```python
# ---- Per-epoch metrics (in run_training_loop) ----
run.log({
    "epoch": epoch,
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "epoch_time_sec": epoch_time,
    "learning_rate": args.lr,
})

# ---- Pre-training summary (in log_pre_training) ----
run.summary["model_size_params"] = num_params
run.summary["dataset_size"] = train_size

# ---- End-of-training summary (in run_training_loop) ----
run.summary["best_val_accuracy"] = best_val_acc
run.summary["best_epoch"] = best_epoch
run.summary["total_training_time_sec"] = total_time
run.summary["final_val_accuracy"] = val_acc
run.summary["final_val_loss"] = val_loss
run.summary["convergence_time_sec"] = convergence_time  # NaN if never reached

# ---- Conditional summary ----
run.summary["early_stopped_epoch"] = epoch              # only if early stopping triggered
run.summary["generalization_gap"] = synth_acc - real_acc # only if real test set provided
```

---

## 3. `wandb.Table` — Structured Tabular Data

Tables let you log **any structured data** — per-class metrics, confusion matrices, predictions, etc. — and visualize/query it interactively in the W&B UI.

### 3.1 Creating & Logging a Table

```python
# From lists
columns = ["class_name", "precision", "recall", "f1_score", "support"]
data = []
for cls_name, metrics in per_class_metrics.items():
    data.append([cls_name, metrics["precision"], metrics["recall"], metrics["f1"], metrics["support"]])

table = wandb.Table(columns=columns, data=data)
run.log({"per_class_metrics": table})

# From a Pandas DataFrame
import pandas as pd
df = pd.DataFrame(per_class_metrics)
table = wandb.Table(dataframe=df)
run.log({"per_class_metrics_df": table})
```

### 3.2 Table Features

- **Max rows**: 200,000 (can be overridden with `wandb.Table.MAX_ARTIFACT_ROWS`).
- **Supported types**: scalars, strings, booleans, numpy arrays, `wandb.Image`, `wandb.Audio`, `wandb.Video`, `wandb.Html`, `wandb.Plotly`.
- **Operations in UI**: sort, filter, group columns, compute aggregates.
- **Comparison**: compare Tables from different runs side-by-side or merged.

### 3.3 Table for F1 Scores per Class (Benchmark Use Case)

```python
# From run_post_evaluation() in benchmark.py
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

rows = []
for cls in class_names:
    rows.append([
        cls,
        round(report[cls]["precision"], 4),
        round(report[cls]["recall"], 4),
        round(report[cls]["f1-score"], 4),
        int(report[cls]["support"]),
    ])

run.log({
    f"{prefix}/f1_per_class": wandb.Table(
        columns=["Class", "Precision", "Recall", "F1", "Support"],
        data=rows,
    )
})
```

### 3.4 Table Comparison Across Runs

When you log a Table to the **same key** across multiple runs, you can:

- **Merge view**: see all runs' tables merged (with run index), diff values as histograms.
- **Side-by-side view**: compare two tables independently.
- **Join key**: set a column (e.g. `class_name`) as join key to align rows.

### 3.5 Joined Tables (`wandb.JoinedTable`)

Join two artifact tables on a common key:

```python
join_table = wandb.JoinedTable(table_1, table_2, join_key="class_name")
artifact = wandb.Artifact("combined_metrics", type="analysis")
artifact.add(join_table, "combined")
run.log_artifact(artifact)
```

---

## 4. Built-in Plots — `wandb.plot`

W&B provides **one-line** chart creation methods. These are logged like any other metric.

### 4.1 Confusion Matrix

```python
# From run_post_evaluation() in benchmark.py
cm = wandb.plot.confusion_matrix(
    y_true=y_true.tolist(),
    preds=y_pred.tolist(),
    class_names=class_names,
)
run.log({f"{prefix}/confusion_matrix": cm})
```

**Inputs:**

- `y_true`: list of true labels (converted from numpy with `.tolist()`)
- `preds`: list of predicted labels (OR `probs`: array of shape `(n_samples, n_classes)`)
- `class_names`: list of string names (e.g. `["2um", "4um", "10um"]`)

### 4.2 Precision-Recall Curve

```python
# From run_post_evaluation() in benchmark.py
run.log({
    f"{prefix}/pr_curve": wandb.plot.pr_curve(
        y_true.tolist(), y_proba.tolist(), labels=class_names
    )
})
```

### 4.3 ROC Curve

```python
# From run_post_evaluation() in benchmark.py
run.log({
    f"{prefix}/roc_curve": wandb.plot.roc_curve(
        y_true.tolist(), y_proba.tolist(), labels=class_names
    )
})
```

### 4.4 Custom Line Plot

```python
data = [[x, y] for x, y in zip(x_values, y_values)]
table = wandb.Table(data=data, columns=["epoch", "metric"])
run.log({"custom_line": wandb.plot.line(table, "epoch", "metric", title="My Metric")})
```

### 4.5 Multi-Line Plot

```python
run.log({"multi_line": wandb.plot.line_series(
    xs=[0, 1, 2, 3, 4],
    ys=[[0.9, 0.8, 0.7, 0.6, 0.5], [0.3, 0.4, 0.5, 0.6, 0.7]],
    keys=["train_loss", "val_loss"],
    title="Loss Curves",
    xname="epoch",
)})
```

### 4.6 Bar Chart

```python
# From run_post_evaluation() in benchmark.py
f1_data = [[cls, report[cls]["f1-score"]] for cls in class_names]
f1_table = wandb.Table(data=f1_data, columns=["class", "f1"])
run.log({
    f"{prefix}/f1_bar_chart": wandb.plot.bar(
        f1_table, "class", "f1", title=f"F1 per Class ({prefix})"
    )
})
```

### 4.7 Histogram

```python
data = [[s] for s in confidence_scores]
table = wandb.Table(data=data, columns=["confidence"])
run.log({"confidence_hist": wandb.plot.histogram(table, "confidence", title="Prediction Confidence")})
```

### 4.8 Scatter Plot

```python
data = [[x, y] for x, y in zip(metric_a, metric_b)]
table = wandb.Table(data=data, columns=["accuracy", "training_time"])
run.log({"scatter": wandb.plot.scatter(table, "accuracy", "training_time")})
```

### 4.9 Logging Matplotlib / Plotly Directly

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(epochs, train_losses, label="train")
ax.plot(epochs, val_losses, label="val")
ax.legend()
run.log({"loss_curves_matplotlib": wandb.Image(fig)})  # as image
run.log({"loss_curves_plotly": fig})  # auto-converted to Plotly (interactive)
plt.close(fig)
```

You can also log Plotly figures as interactive HTML inside Tables:

```python
import plotly.express as px
fig = px.bar(df, x="class", y="f1_score", title="F1 per class")
fig.write_html("plot.html", auto_play=False)
table = wandb.Table(columns=["plotly_chart"])
table.add_data(wandb.Html("plot.html"))
run.log({"interactive_chart": table})
```

---

## 5. PyTorch Integration — `run.watch()`

Monitor **gradients and parameters** of your PyTorch model:

```python
run.watch(
    models=model,               # nn.Module or list of modules
    criterion=loss_fn,          # optional: loss function
    log="all",                  # "gradients" | "parameters" | "all"
    log_freq=100,               # log every N batches
    log_graph=True,             # log computational graph
)
```

This creates histograms of gradient and parameter distributions in the UI. Useful for debugging vanishing/exploding gradients.

Call `run.unwatch(model)` when done.

---

## 6. Grouping & Organizing Runs

### 6.1 Groups

Use `group` in `wandb.init()` to group runs by model or experiment:

```python
wandb.init(group="Conv1DClassifier")  # all Conv1D runs grouped together
```

In the UI, grouped runs can be **collapsed** and their metrics **averaged** automatically.

### 6.2 Tags

```python
wandb.init(tags=["synthetic_v1", "benchmark"])
# Or add later
run.tags = run.tags + ("best_model",)
```

Tags allow filtering in the Runs Table.

### 6.3 Job Type

```python
wandb.init(job_type="training")    # or "evaluation", "preprocessing"
```

### 6.4 Runs Table Operations

In the W&B UI Runs Table you can:

- **Sort** by any config or summary metric
- **Filter** using expressions (e.g. `config.model_name == "ResNet18"`)
- **Group** by any column to see aggregated statistics
- **Show/Hide** columns for clarity

---

## 7. Run Comparer

The **Run Comparer** panel lets you view config and metric differences across runs side-by-side.

**How to use:**

1. Add panel → Evaluation → **Run comparer**
2. Shows up to 10 visible runs in columns
3. Toggle **"Diff only"** to hide identical values and focus on differences
4. Search for specific config keys or metadata
5. Filter/sort the runs table to select which runs to compare

**Use case for benchmark:** compare hyperparameters and final metrics between your top-N models to understand what drives performance.

---

## 8. Line Plots in the UI

Line plots are auto-created for every metric logged with `wandb.log()`.

**Key features:**

- **Multi-metric plots**: use regex to combine e.g. `train/.*` into one chart
- **Grouping**: group runs by a config key → shows **mean ± std** as shaded area
- **Smoothing**: exponential smoothing slider to see trends in noisy data
- **Custom x-axis**: switch between `_step`, `epoch`, relative time, or wall time
- **Zoom**: click-drag to zoom into a region
- **Log scale**: toggle for loss curves
- **Outlier handling**: toggle to clip or remove outliers

**For the benchmark (`benchmark.py` auto-logs these):**

- Plot `train/loss` and `val/loss` over `epoch` for each run → compare convergence speed
- Group by `Conv1DClassifier` → see averaged curves across runs
- Compare `val/accuracy` across dataset variants (filter by `dataset` config key)

---

## 9. Scatter Plots in the UI

Add via: **Add panel → Scatter plot**

Useful for:

- Plotting `best_val_accuracy` vs `total_training_time` across runs
- Plotting `model_size` vs `accuracy` to see efficiency frontiers
- Color-coding points by `dataset_name` or `model_name`
- Adding min/max/average trend lines
- Using log scale on axes

---

## 10. Bar Plots in the UI

Bar plots appear when all logged values are **single values** (length 1). Also:

- Can be grouped by runs table grouping → shows **box plots** or **violin plots**
- Useful for comparing `best_val_accuracy` or `f1_score` across models

---

## 11. Parallel Coordinates

Automatically created during Sweeps, but can be added manually.

- Each vertical axis = one hyperparameter or metric
- Each line = one run
- Color = target metric value
- Instantly see which hyperparameter combos lead to best results

---

## 12. Parameter Importance

Auto-generated in Sweep dashboards:

- Shows which hyperparameters are **most correlated** with your target metric
- Uses feature importance (correlation + importance) analysis
- Helps prioritize what to tune

---

## 13. W&B Reports

Reports are **collaborative documents** that combine:

- Run visualizations (line plots, tables, scatter plots…)
- Markdown text, LaTeX equations
- Embedded panels from your workspace
- Cross-project comparisons

### 13.1 Creating a Report

**UI method:**

1. Navigate to project workspace
2. Click **"Create report"**
3. Select panels to include
4. Add text, analysis, conclusions
5. **Publish** and **Share**

**Programmatic method:**

```python
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="particle-benchmark")
report.save()
```

### 13.2 Report Use Cases for Benchmark

- **Model leaderboard**: Table of all models ranked by best accuracy
- **Per-dataset analysis**: Sections for each dataset showing loss curves, confusion matrices
- **Cross-model comparison**: Side-by-side line plots grouped by model
- **Summary findings**: Text + visualizations of best performing configurations
- **Export**: PDF or LaTeX for academic publication

### 13.3 Embedding Reports

Reports can be embedded in Notion or any HTML page via IFrame.

---

## 14. W&B Sweeps — Hyperparameter Optimization

Automate hyperparameter search across your models.

### 14.1 Sweep Configuration

```yaml
program: benchmark.py
method: bayes  # grid | random | bayes
metric:
  name: val/accuracy
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-2
  batch_size:
    values: [16, 32, 64, 128]
  epochs:
    value: 150  # fixed
  decimate:
    values: [2, 4, 8]
  patience:
    value: 20
early_terminate:
  type: hyperband
  min_iter: 5
  eta: 3
run_cap: 50
```

**Search methods:**

- **Grid**: exhaustive, all combinations (expensive)
- **Random**: random sampling from distributions
- **Bayesian**: informed search using probabilistic model (best for continuous params)

**Early termination** (Hyperband): stops poorly performing runs before they finish → saves compute.

### 14.2 Running a Sweep

```bash
wandb sweep sweep_config.yaml  # returns SWEEP_ID
wandb agent SWEEP_ID           # start agent
```

Or from Python:

```python
sweep_id = wandb.sweep(sweep_config, project="particle-benchmark")
wandb.agent(sweep_id, function=train_function, count=50)
```

### 14.3 Sweep Visualizations (Auto-Generated)

- **Parallel coordinates plot**: hyperparams vs metric, color-coded
- **Parameter importance plot**: which params matter most
- **Scatter plot**: runs plotted by metric value

---

## 15. W&B Artifacts — Dataset & Model Versioning

Artifacts let you **version and track** datasets and models as inputs/outputs of runs.

### 15.1 Logging a Dataset Artifact

```python
with wandb.init(project="particle-benchmark", job_type="upload-data") as run:
    artifact = wandb.Artifact(name="particle_dataset_A", type="dataset")
    artifact.add_dir("./data/set_A/")
    run.log_artifact(artifact)
```

### 15.2 Using a Dataset in Training

```python
with wandb.init(project="particle-benchmark", job_type="training") as run:
    artifact = run.use_artifact("particle_dataset_A:latest")
    data_dir = artifact.download()
    # ... load data from data_dir
```

### 15.3 Logging a Model Checkpoint

```python
# From benchmark.py main()
run.log_model(
    path=str(output_dir / "best_model.pth"),
    name=f"Conv1D-{args.dataset_name}",
)
```

### 15.4 Lineage Graphs

W&B auto-generates **DAG lineage graphs** showing: Dataset → Training Run → Model → Evaluation Run. Full traceability.

---

## 16. Custom Charts (Vega)

For fully custom visualizations beyond the built-in presets:

```python
table = wandb.Table(data=data, columns=["step", "value"])
fields = {"x": "step", "value": "value"}
my_chart = wandb.plot_table(
    vega_spec_name="wandb/line/v0",  # or your own saved preset
    data_table=table,
    fields=fields,
    string_fields={"title": "Custom Chart"},
)
run.log({"custom_chart": my_chart})
```

Uses **Vega** grammar for full control over fonts, colors, tooltips, axes, transforms.

---

## 17. System Metrics (Auto-Logged)

W&B automatically captures:

- **CPU utilization** (per core)
- **GPU utilization** (via `nvidia-smi`): usage %, memory, temperature
- **Memory usage** (RAM)
- **Network I/O**
- **Disk I/O**

Useful for profiling training efficiency and identifying bottlenecks.

---

## 18. Implementation Status

> Status of the benchmark pipeline as implemented in `benchmark.py`.

### Phase 1 — Setup & Config

- [x]  Install `wandb` via pip (in `requirements.txt`)
- [x]  Create project `particle-benchmark` on W&B
- [x]  Define a standard config dict for all runs (22 keys including model, dataset, hyperparams, architecture)
- [ ]  Log datasets as Artifacts for traceability

### Phase 2 — Training Loop Integration

- [x]  Call `wandb.init()` with config, group, tags, name, job_type
- [x]  Use `run.define_metric()` to set `epoch` as x-axis + summary aggregation goals
- [x]  Log per-epoch: `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `epoch_time_sec`, `learning_rate`
- [x]  LR scheduler support (`--scheduler cosine|plateau|none`, default: cosine). `learning_rate` logs the actual LR from the optimizer each epoch
- [ ]  Use `run.watch(model)` for gradient monitoring (optional, not yet added)
- [x]  At end: set `run.summary` for `best_val_accuracy`, `best_epoch`, `total_training_time_sec`, `convergence_time_sec`, `final_val_accuracy`, `final_val_loss`
- [x]  Early stopping with configurable patience
- [x]  Best model checkpoint saved and reloaded for evaluation

### Phase 3 — Evaluation & Per-Class Metrics

- [x]  Log `wandb.plot.confusion_matrix()` for each test set (synthetic + real)
- [x]  Log `wandb.Table` with per-class Precision, Recall, F1, Support
- [x]  Log `wandb.plot.pr_curve()` and `wandb.plot.roc_curve()`
- [x]  Log bar charts for F1 per class
- [x]  Generalization gap computed when real test set is provided
- [x]  Best model saved as W&B artifact via `run.log_model()`

### Phase 4 — Dimensionality Reduction (PCA / t-SNE)

- [x]  Extract fc1 features (256-dim) via forward hook for all test sets
- [x]  PCA and t-SNE scatter plots for synthetic test set, logged to W&B
- [x]  PCA and t-SNE scatter plots for real test set (if provided)
- [x]  Noise separation visualization: combined ID + noise latent space (PCA + t-SNE)

### Phase 5 — OOD Noise Evaluation

- [x]  Load noise samples from `--noise-dir` with truncation + bandpass + decimation
- [x]  MSP (Max Softmax Probability) scoring
- [x]  Energy score (`-logsumexp(logits)`)
- [x]  ODIN (temperature scaling + input perturbation, T=1000, ε=0.0012)
- [x]  Mahalanobis distance (multi-layer: pool1, pool2, pool3, fc1 with GAP, class-conditional Gaussian, tied covariance)
- [x]  Temperature scaling sweep (T ∈ [1, 2, 5, 10, 50, 100, 500, 1000]) for MSP and Energy
- [x]  Energy_tuned: re-evaluation with optimal temperature from sweep
- [x]  AUROC, FPR@95, AUPR for all methods
- [x]  Score distribution histograms (ID vs noise) for all methods
- [x]  Overlaid ROC curves comparison across methods
- [x]  Threshold analysis with operating points table (TPR=90%, 95%, 99%)
- [x]  Per-class OOD analysis (AUROC per class vs noise using MSP)
- [x]  Silhouette score for latent space separability (ID vs noise)
- [x]  Prediction distribution on noise (class assignment bar chart)
- [x]  Summary table logged to W&B (all methods × all metrics)

### Future — Comparison & Reporting

- [ ]  Use **Run Comparer** to diff top models
- [ ]  Use **Grouping** (by model name) to see averaged curves
- [ ]  Create scatter plots: accuracy vs training time, accuracy vs model size
- [ ]  Build a summary `wandb.Table` as leaderboard (all models × all datasets × key metrics)
- [ ]  Create a **W&B Report** per dataset and one global summary
- [ ]  Export as PDF if needed

---

## 19. Quick Reference — Code Cheat Sheet

```python
# This mirrors the actual benchmark.py implementation.
# See benchmark.py for the full runnable code.

import time
import wandb
import torch
import numpy as np
from sklearn.metrics import classification_report
from train import (RAW_SIGNAL_LENGTH, ParticleDataset, Conv1DClassifier,
                   BandpassFilter, Decimate, train_one_epoch, evaluate)

# ============ INIT ============
config = {
    "model_name": "Conv1DClassifier",
    "model_size_params": num_params,
    "dataset": args.dataset_name,
    "dataset_size": train_size,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "optimizer": "Adam",
    "seed": args.seed,
    "patience": args.patience,
    "scheduler": args.scheduler,
    # ... plus architecture-specific params
}
run = wandb.init(
    project="particle-benchmark",
    config=config,
    group="Conv1DClassifier",
    tags=[args.dataset_name, "benchmark"],
    name=f"Conv1D-{args.dataset_name}-{args.run_id}",
    job_type="training",
)
run.define_metric("epoch")
run.define_metric("train/*", step_metric="epoch")
run.define_metric("val/*", step_metric="epoch")
run.define_metric("val/accuracy", summary="max", goal="maximize")
run.define_metric("val/loss", summary="min", goal="minimize")

# ============ PRE-TRAINING ============
run.summary["model_size_params"] = num_params
run.summary["dataset_size"] = train_size

# ============ TRAINING LOOP ============
best_val_acc = 0.0
convergence_time = None
total_start = time.time()

for epoch in range(args.epochs):
    epoch_start = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
    epoch_time = time.time() - epoch_start

    run.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch_time_sec": epoch_time,
        "learning_rate": current_lr,  # actual LR from optimizer (tracks scheduler)
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        run.summary["best_val_accuracy"] = best_val_acc
        run.summary["best_epoch"] = best_epoch
        torch.save(model.state_dict(), output_dir / "best_model.pth")

    if convergence_time is None and val_acc >= args.convergence_threshold:
        convergence_time = time.time() - total_start
        run.summary["convergence_time_sec"] = convergence_time

    # Early stopping
    if args.patience > 0 and epochs_without_improvement >= args.patience:
        run.summary["early_stopped_epoch"] = epoch
        break

# ============ SUMMARY ============
run.summary["total_training_time_sec"] = time.time() - total_start
run.summary["final_val_accuracy"] = val_acc
run.summary["final_val_loss"] = val_loss
if convergence_time is None:
    run.summary["convergence_time_sec"] = float("nan")

# ============ EVALUATION (per test set) ============
model.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))
loss, acc, y_pred, y_true, y_proba = evaluate(model, test_loader, criterion, device)

prefix = "test_synthetic"  # or "test_real"
run.summary[f"{prefix}/accuracy"] = acc
run.summary[f"{prefix}/loss"] = loss

# Confusion matrix
run.log({f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
    y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=class_names
)})

# PR & ROC curves
run.log({f"{prefix}/pr_curve": wandb.plot.pr_curve(
    y_true.tolist(), y_proba.tolist(), labels=class_names)})
run.log({f"{prefix}/roc_curve": wandb.plot.roc_curve(
    y_true.tolist(), y_proba.tolist(), labels=class_names)})

# Per-class F1 table
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
rows = [[c, round(report[c]["precision"], 4), round(report[c]["recall"], 4),
         round(report[c]["f1-score"], 4), int(report[c]["support"])] for c in class_names]
run.log({f"{prefix}/f1_per_class": wandb.Table(
    columns=["Class", "Precision", "Recall", "F1", "Support"], data=rows)})

# F1 bar chart
f1_data = [[c, report[c]["f1-score"]] for c in class_names]
f1_table = wandb.Table(data=f1_data, columns=["class", "f1"])
run.log({f"{prefix}/f1_bar_chart": wandb.plot.bar(
    f1_table, "class", "f1", title=f"F1 per Class ({prefix})")})

# Model artifact
run.log_model(path=str(output_dir / "best_model.pth"), name=f"Conv1D-{args.dataset_name}")

run.finish()
```
