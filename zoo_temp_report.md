# Model Zoo Benchmark — Temporary Report

> Companion to `benchmark_zoo.py` (Benchmark 2). Covers the 9 architectures
> evaluated, their justification, and a full **Materials & Methods** section
> describing every figure emitted under `results/benchmark2/figures/`.

> **Recent updates (2026-04-08):**
>
> 1. **New Conv1DGAP family.** Same three-block Conv1D backbone as the
>    original `Conv1D` control, but the flatten → dense head is replaced
>    by a Global Average Pooling over the temporal axis. This shrinks
>    the latent vector from `c3 × (L/8)` to `c3` and, at the base (M)
>    tier, takes the parameter count from 5.3 M → 273 K (≈5 %) while
>    keeping the same Tier-1 accuracy (0.971). At Tier 1 the new family
>    immediately claimed leaderboard ranks **#2** (`Conv1DGAP-L`) and
>    **#3** (`Conv1DGAP-S`), and two Pareto-front slots on each of the
>    latency and on-disk-size axes. It is registered as the 9th family in
>    `models/conv1d_gap.py` / `models/__init__.py` with the same 7-tier
>    `width_mult` schedule as Conv1D, and its visual channels (color
>    `#E6AB02`, marker `h`, linestyle `(0, (5,1,1,1,1,1,1,1))`) are in
>    `benchmark_zoo.py`. See §1 and §1.1 for sizing tables and §3.6
>    for its impact on the scaling/Pareto figures.
> 2. **Dynamic small-multiples grid.** `generate_scaling_grid` and
>    `generate_tier_grid` are no longer hard-coded to 2×4 = 8 panels.
>    They now compute `(n_rows, n_cols) = _grid_layout(len(families))`
>    with a near-square layout (`n_cols = ceil(sqrt(n_panels))`), so
>    the 9-family zoo renders as a clean **3×3** grid (instead of
>    leaving a half-empty third row on a 3×4 layout), and any future
>    family count automatically gets the tightest square layout
>    without silently clipping a panel. See §3.6 / §3.7 / §3.4.
> 3. **Dedicated latency Pareto renderer.** `pareto_latency.pdf` uses
>    `generate_pareto_latency_focus` — log error rate on the y-axis,
>    clipped y-window, kernel-launch floor shading, numbered Pareto
>    badges with a lower-right side-key table, and iso-accuracy guides
>    on the right edge. See §3.9 for the full rationale and
>    `doc/variant_plotting.md` §3.4 for the design notes.

---

## 1 — Models in the Zoo

All architectures are 1-D CNNs operating on `(batch, 1, 625)` particle traces
(2500-sample raw signal, decimated ×4 after a 5–100 Hz Butterworth bandpass).
Every family **except `Conv1DGAP`** was budgeted to a **~5.3 M-parameter "M"
variant** so that Tier-1 comparisons between families isolate *architectural
priors* from raw capacity; additional S/M/L/XS/XXS/Nano/Pico variants are
trained separately for the scaling curves. The 9 families were selected to
span the design space that matters for FPGA deployment (depth vs. width,
dense vs. sparse connectivity, depthwise vs. standard convolutions, fixed
vs. multi-scale receptive fields, dense head vs. global-average-pooling
head). `Conv1DGAP` is deliberately *not* renormalised to the 5.3 M budget:
its whole point is to show what the Conv1D backbone can deliver once the
dense head is removed, so it is registered with the **same `width_mult`
schedule as Conv1D** and its base M variant comes out at ≈273 K parameters
(≈5 % of the equivalent Conv1D head) by construction.

| # | Family | Why it is in the zoo | Paper |
|---|--------|----------------------|-------|
| 1 | **Conv1D** | Plain 3-layer conv + dense head. The "null-architecture" control — every claim about a more sophisticated design must beat it on the same parameter budget. | No canonical paper (generic CNN baseline; closest ancestor: LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998 — [IEEE 726791](https://ieeexplore.ieee.org/document/726791)). |
| 2 | **LeNet1D** | Canonical two-conv + three-FC topology — tests whether the oldest CNN prior still competes once widened to 5 M params on a modern signal task. | LeCun, Bottou, Bengio, Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proc. IEEE 1998 — [IEEE 726791](https://ieeexplore.ieee.org/document/726791). |
| 3 | **VGG1D** | Deep stack of 3×3 convs with doubling channels — probes whether *uniform, deep, small-kernel* feature hierarchies help on short 1-D traces. | Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, ICLR 2015 — [arXiv:1409.1556](https://arxiv.org/abs/1409.1556). |
| 4 | **ResNet1D** | BasicBlock residual connections — the single most impactful modern CNN prior. Included as the de-facto strong baseline for 1-D time-series classification. | He, Zhang, Ren, Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). |
| 5 | **InceptionTime1D** | Multi-scale parallel kernels (11/21/41) with 1×1 bottlenecks — designed *specifically* for time-series on the UCR archive and currently the published SOTA for UCR classification. | Fawaz, Lucas, Forestier, Pelletier, Schmidt, Weber, Webb, Idoumghar, Muller, Petitjean, *InceptionTime: Finding AlexNet for Time Series Classification*, DMKD 2020 — [arXiv:1909.04939](https://arxiv.org/abs/1909.04939). |
| 6 | **MobileNet1D** | Inverted-residual blocks with depthwise-separable convs — the standard "edge" topology, optimal MAC/accuracy trade-off for fixed-point FPGA inference. | Sandler, Howard, Zhu, Zhmoginov, Chen, *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018 — [arXiv:1801.04381](https://arxiv.org/abs/1801.04381). |
| 7 | **EfficientNet1D** | MBConv with Squeeze-and-Excitation, SiLU, and compound width/depth scaling — tests whether SE channel attention recovers accuracy that depthwise-separable convs lose. | Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019 — [arXiv:1905.11946](https://arxiv.org/abs/1905.11946). |
| 8 | **DenseNet1D** | Dense connectivity with growth-rate control — probes whether feature reuse compensates for shallower gradient paths at equal parameter cost. | Huang, Liu, van der Maaten, Weinberger, *Densely Connected Convolutional Networks*, CVPR 2017 — [arXiv:1608.06993](https://arxiv.org/abs/1608.06993). |
| 9 | **Conv1DGAP** | Same three-block Conv1D backbone as family #1, but the flatten → dense head is replaced with Global Average Pooling over the temporal axis. Tests the head-side ablation in isolation: *"how much of Conv1D's capacity is actually spent in the dense head, not in the convolutional feature extractor?"*. On this benchmark the answer is **~95 % of the parameters** — the GAP head preserves Tier-1 accuracy while shrinking the model 20× on disk. | Lin, Chen, Yan, *Network In Network*, ICLR 2014 — [arXiv:1312.4400](https://arxiv.org/abs/1312.4400) (the paper that introduced GAP classifier heads). |

Measured "M" (base) variant sizes under the current pipeline (3 classes,
`input_length = 625`):

| Family            | Params      | MACs (forward) |
|-------------------|------------:|---------------:|
| Conv1DGAP         |    273,027  |     39,104,576 |
| LeNet1D           | 5,255,107   |      8,698,400 |
| VGG1D             | 5,270,739   |    158,941,632 |
| InceptionTime1D   | 5,289,575   |  3,217,065,412 |
| ResNet1D          | 5,289,733   |    146,317,342 |
| DenseNet1D        | 5,298,107   |    281,430,724 |
| Conv1D            | 5,319,299   |     44,130,624 |
| MobileNet1D       | 5,364,035   |    134,932,976 |
| EfficientNet1D    | 5,393,205   |     66,243,952 |

The parameter counts of the first eight families are deliberately matched
to within <3 % (the "iso-capacity" comparison). **Conv1DGAP is the one
deliberate outlier**: it shares the Conv1D backbone but its GAP head
collapses the latent from `c3 × (L/8) = 256 × 78 = 19 968` features to
just `c3 = 256`, which brings the total down to 273 K at the same
`width_mult = 1.0`. It is placed in the zoo specifically to expose how
much of the "M"-tier parameter budget the Conv1D baseline is burning in
the dense head alone. MACs diverge across families by two orders of
magnitude because they reflect the *arithmetic structure* of each prior
— exactly the quantity that drives FPGA footprint — which is what the
scaling and Pareto figures below are meant to expose.

### 1.1 — Full variant reference table

All 58 models actually trained in the sweep, with their measured parameter
count and MACs (single forward pass at `input_length = 625`, taken from
`results/benchmark2/summary.csv`). Suffixes are ordered smallest → largest:
**Pico < Nano < XXS < XS < S < M (no suffix) < L**. Conv1D / Conv1DGAP /
LeNet1D / VGG1D expose a Pico tier; the other five families bottom out at
Nano because their architectural minimums (a single residual block, a
single dense block, etc.) already exceed the Pico budget.

| Family | Variant | Size | Params | MACs |
|---|---|---|---:|---:|
| Conv1D | Conv1D-Pico | Pico | 757 | 13,356 |
| Conv1D | Conv1D-Nano | Nano | 2,975 | 34,659 |
| Conv1D | Conv1D-XXS | XXS | 50,872 | 459,471 |
| Conv1D | Conv1D-XS | XS | 332,963 | 2,885,712 |
| Conv1D | Conv1D-S | S | 1,330,499 | 11,202,720 |
| Conv1D | Conv1D | M | 5,319,299 | 44,130,624 |
| Conv1D | Conv1D-L | L | 21,271,811 | 175,161,984 |
| Conv1DGAP | Conv1DGAP-Pico | Pico | 64 | 12,900 |
| Conv1DGAP | Conv1DGAP-Nano | Nano | 203 | 32,361 |
| Conv1DGAP | Conv1DGAP-XXS | XXS | 2,747 | 413,321 |
| Conv1DGAP | Conv1DGAP-XS | XS | 17,571 | 2,575,376 |
| Conv1DGAP | Conv1DGAP-S | S | 68,931 | 9,951,264 |
| Conv1DGAP | Conv1DGAP | M | 273,027 | 39,104,576 |
| Conv1DGAP | Conv1DGAP-L | L | 1,086,723 | 155,017,344 |
| DenseNet1D | DenseNet1D-Nano | Nano | 33,179 | 1,103,092 |
| DenseNet1D | DenseNet1D-XXS | XXS | 90,131 | 3,599,880 |
| DenseNet1D | DenseNet1D-XS | XS | 405,827 | 19,076,341 |
| DenseNet1D | DenseNet1D-S | S | 1,423,607 | 72,339,858 |
| DenseNet1D | DenseNet1D | M | 5,298,107 | 281,430,724 |
| DenseNet1D | DenseNet1D-L | L | 20,402,867 | 1,109,867,784 |
| EfficientNet1D | EfficientNet1D-Nano | Nano | 45,585 | 1,381,200 |
| EfficientNet1D | EfficientNet1D-XXS | XXS | 90,305 | 1,718,032 |
| EfficientNet1D | EfficientNet1D-XS | XS | 388,637 | 4,984,496 |
| EfficientNet1D | EfficientNet1D-S | S | 1,672,787 | 18,783,504 |
| EfficientNet1D | EfficientNet1D | M | 5,393,205 | 66,243,952 |
| EfficientNet1D | EfficientNet1D-L | L | 20,885,409 | 252,650,480 |
| InceptionTime1D | InceptionTime1D-Nano | Nano | 9,641 | 2,998,630 |
| InceptionTime1D | InceptionTime1D-XXS | XXS | 41,315 | 18,023,104 |
| InceptionTime1D | InceptionTime1D-XS | XS | 190,123 | 99,098,664 |
| InceptionTime1D | InceptionTime1D-S | S | 1,363,621 | 807,483,090 |
| InceptionTime1D | InceptionTime1D | M | 5,289,575 | 3,217,065,412 |
| InceptionTime1D | InceptionTime1D-L | L | 20,831,563 | 12,842,530,056 |
| LeNet1D | LeNet1D-Pico | Pico | 822 | 9,229 |
| LeNet1D | LeNet1D-Nano | Nano | 1,999 | 10,395 |
| LeNet1D | LeNet1D-XXS | XXS | 49,297 | 101,529 |
| LeNet1D | LeNet1D-XS | XS | 328,819 | 592,520 |
| LeNet1D | LeNet1D-S | S | 1,314,275 | 2,239,760 |
| LeNet1D | LeNet1D | M | 5,255,107 | 8,698,400 |
| LeNet1D | LeNet1D-L | L | 21,016,451 | 34,272,320 |
| MobileNet1D | MobileNet1D-Nano | Nano | 28,275 | 1,474,432 |
| MobileNet1D | MobileNet1D-XXS | XXS | 81,683 | 2,423,680 |
| MobileNet1D | MobileNet1D-XS | XS | 388,339 | 10,059,328 |
| MobileNet1D | MobileNet1D-S | S | 1,642,483 | 36,658,800 |
| MobileNet1D | MobileNet1D | M | 5,364,035 | 134,932,976 |
| MobileNet1D | MobileNet1D-L | L | 20,272,963 | 523,536,864 |
| ResNet1D | ResNet1D-Nano | Nano | 9,181 | 142,150 |
| ResNet1D | ResNet1D-XXS | XXS | 62,326 | 1,422,405 |
| ResNet1D | ResNet1D-XS | XS | 343,981 | 8,899,446 |
| ResNet1D | ResNet1D-S | S | 1,363,996 | 36,907,935 |
| ResNet1D | ResNet1D | M | 5,289,733 | 146,317,342 |
| ResNet1D | ResNet1D-L | L | 20,829,511 | 582,642,876 |
| VGG1D | VGG1D-Pico | Pico | 842 | 38,297 |
| VGG1D | VGG1D-Nano | Nano | 3,205 | 114,128 |
| VGG1D | VGG1D-XXS | XXS | 51,750 | 1,615,303 |
| VGG1D | VGG1D-XS | XS | 330,807 | 10,211,184 |
| VGG1D | VGG1D-S | S | 1,319,531 | 40,105,184 |
| VGG1D | VGG1D | M | 5,270,739 | 158,941,632 |
| VGG1D | VGG1D-L | L | 21,068,195 | 632,808,320 |

Totals: **58 variants across 9 families**, spanning **64 → 21.3 M** parameters
(≈5.5 orders of magnitude, with `Conv1DGAP-Pico` setting the new floor) and
9.2 k → 12.8 G MACs (≈6 orders of magnitude). Conv1DGAP's seven variants
share the same `width_mult ∈ {0.0125, 0.025, 0.1, 0.25, 0.5, 1.0, 2.0}`
schedule as Conv1D, so their parameter ratios (Conv1DGAP ≈ 5 % of Conv1D at
the same tier) are directly readable from the two blocks above.

---

## 2 — Materials and Methods

### 2.1 — Dataset and preprocessing

Primary dataset: `data/dataset` (3 classes: `2um`, `4um`, `10um`), split
on disk into `train/` and `test/`. Each signal is a 2500-sample raw trace
from the self-mixing optical sensor.

For every run:

1. **Bandpass filter** — 2-nd-order Butterworth, 5–100 Hz (`BandpassFilter`
   in `train.py`), removing DC drift and high-frequency noise that lies
   outside the particle-transit spectrum.
2. **Decimation** — factor 4 (`Decimate`), yielding a 625-sample input.
3. **Stratified train/val split** — `val_split = 0.2` via
   `sklearn.model_selection.StratifiedShuffleSplit`, re-seeded per run to
   keep class proportions identical between the two halves.

Noise and time-mask transforms are applied **only on validation and test
sets**, never on training data. Early stopping uses the transformed
validation loader so the selected checkpoint is the one that performs best
under the tier's deployment conditions.

Tier definitions (`create_tier_loaders`, `benchmark_zoo.py:255`):

| Tier | Train size | Val/Test transform            | Purpose                               |
|------|-----------:|-------------------------------|---------------------------------------|
| 1    | full       | none                          | clean baseline                        |
| 2    | 50/class   | none                          | small-data regime                     |
| 3    | full       | Gaussian noise, SNR = 10 dB   | noise robustness                      |
| 4    | 50/class   | Gaussian noise, SNR = 10 dB   | small-data × noise                    |
| 5    | 20/class   | Real noise ∈ [−3, 3] dB + 15 % time mask | worst-case field deployment |
| 6    | full synth | base only                     | sim-to-real domain shift (test on `data/S7_pure_real`) |

### 2.2 — Training protocol

Identical for every (model, tier, seed) triple:

- Optimizer: **Adam**, `lr = 6e-4`, `weight_decay = 1e-4`.
- Scheduler: **cosine annealing** over the full horizon.
- Loss: cross-entropy on raw logits.
- Batch size: 32.
- Epochs: **150** with early stopping `patience = 20` on `val/accuracy`.
- Seeds: **42, 123, 7** (three independent runs per cell).
- Determinism: `torch.manual_seed`, `np.random.seed`, `random.seed`,
  `cudnn.deterministic = True`, `cudnn.benchmark = False`.

Per-epoch W&B logs (`training_utils.py:169`): `epoch`, `train/loss`,
`train/accuracy`, `val/loss`, `val/accuracy`, `epoch_time_sec`,
`learning_rate`. Summary scalars: `best_val_accuracy`, `best_epoch`,
`total_training_time_sec`, `convergence_time_sec`, `model_size_params`,
`dataset_size`, `inference_latency_median_ms`, `peak_ram_mb`, `model_size_mb`.

### 2.3 — Efficiency measurements

All three measured on the same machine (RTX A1000 GPU + CPU host) right
after loading the best checkpoint, so they are free of training-time
allocator state:

- **MACs** — `thop.profile` on a synthetic `(1, 1, 625)` input
  (`compute_model_macs`, `training_utils.py:35`). `thop`'s `total_ops`
  and `total_params` side-effects are scrubbed before the model is used
  for anything else.
- **Latency** — 50 warmup passes, 1000 timed passes at batch size 1,
  `cuda.synchronize` around each (`measure_inference_latency`,
  `training_utils.py:49`). Report median and p95.
- **Peak RAM** — GPU path: `cuda.reset_peak_memory_stats` + single forward
  + `cuda.max_memory_allocated`; CPU fallback uses forward-hook tracing
  (`measure_peak_ram`, `training_utils.py:81`).
- **Model size** — parameter count + on-disk size of a `torch.save`-d
  `state_dict` to a throwaway `tempfile` (`measure_model_size`).

### 2.4 — Accuracy statistics

For every test loader, the pipeline computes a **1000-draw bootstrap 95 %
confidence interval** on accuracy (`bootstrap_accuracy_ci`,
`benchmark_zoo.py:369`). Aggregation across seeds uses the per-cell mean
(`Acc_Mean`) and sample standard deviation (`Acc_Std`), both written to
`summary.csv` and rendered as error bands on every publication figure.

---

## 3 — Figure catalogue

Each subsection answers three questions:

- **Why** — what claim the figure is meant to support.
- **How to interpret** — what a reader should look for.
- **Compute & data** — exactly how the figure is assembled from the
  per-run JSONs under `results/benchmark2/runs/`.

All PDFs use the publication `rcParams` declared at `benchmark_zoo.py:151`
(7 pt labels, 1.2 pt lines, TrueType fonts, fixed canvas size so no
`bbox="tight"` rescaling happens). They are emitted to
`results/benchmark2/figures/`, and simultaneously logged as `wandb.Image`
previews to the consolidated `benchmark2-report-<timestamp>` W&B run plus
bundled as a `benchmark2-report` artifact.

### 3.1 — `tier_heatmap.pdf`
*Single-column heatmap of base-model accuracy across all 6 tiers.*

- **Why.** One-glance answer to "which architecture breaks first as the
  task gets harder?". A heatmap lets the reader compare 9 families × 6
  tiers = 54 cells without ever reading an axis twice; gradients of colour
  along rows expose per-family degradation, gradients along columns
  expose per-tier difficulty.
- **Interpret.** Darker = higher accuracy (YlGnBu colormap, `vmin = 0.5`).
  Rows are alphabetically sorted families; columns are tiers T1…T6.
  A row that stays dark everywhere is "robust"; a row that lightens
  sharply in T5 or T6 has a specific robustness failure.
- **Compute.** Loads every `runs/*.json`, filters to `size_tag == "M"`,
  pivots to `(model_name, tier) → mean accuracy` over seeds, and
  `sns.heatmap`-s the result. See `generate_tier_heatmap`
  (`benchmark_zoo.py:1337`).

### 3.2 — `tier_robustness.pdf`
*Single-column line plot — accuracy vs. difficulty tier, one line per family.*

- **Why.** The heatmap answers "who" and "where"; the line plot answers
  "by how much". Slopes between tiers are directly readable and the ±σ
  band makes seed stability visible.
- **Interpret.** x-axis is discrete tiers T1…T6; y-axis is mean accuracy
  with a shaded band of width ±σ across seeds. A line that stays flat is
  tier-robust; a line that collapses at T5/T6 has brittle data-or-domain
  priors. Each family has a distinct (colour, marker, linestyle) triple
  so the figure is colour-blind-safe and B&W-printable.
- **Compute.** `_load_all_tiers_aggregated` pools per-seed JSONs into
  `(model_name, tier) → (acc_mean, acc_std)`, filtered to the base "M"
  variants only — scaling variants are excluded here so the figure
  stays on the *architecture* axis. Plotting: `generate_tier_robustness`
  (`benchmark_zoo.py:1147`).

### 3.3 — `tier_grid.pdf`
*Small-multiples version of the tier-robustness curve (near-square grid, `ceil(sqrt(n))` columns).*

- **Why.** When every family is drawn on the same axes, lines cross
  and shaded bands overlap. The small-multiples grid sacrifices
  direct comparability for per-family readability — each panel shows
  a single family's tier trajectory on its own axes.
- **Interpret.** Each panel shares the same axis limits so panel shapes
  are directly comparable; panels are titled with the family name in
  its canonical colour. Read the panels row-major to compare "shapes of
  collapse" (steep at T5 vs. gentle linear decay vs. flat). With the
  9th family (Conv1DGAP) the layout is a **3 × 3 grid** (9 cells
  exactly filled); a 10th family would trigger a 4 × 3 layout
  automatically via `ceil(sqrt(n))`.
- **Compute.** Same aggregation as §3.2; layout via
  `_grid_layout(len(families))` — which picks
  `n_cols = ceil(sqrt(n_families))` and
  `n_rows = ceil(n_families / n_cols)` — fed to
  `plt.subplots(..., sharex=True, sharey=True, squeeze=False)` in
  `generate_tier_grid` (`benchmark_zoo.py:1211`). The near-square
  layout keeps the canvas compact and never silently clips a family.

### 3.4 — `tier6_domain_gap.pdf`
*Slope chart of synthetic-vs-real accuracy on Tier 6.*

- **Why.** Tier 6 is the only tier that produces two test accuracies
  per run (synthetic test split + real `S7_pure_real` test split),
  because the training distribution is synthetic but deployment is
  real. A slope chart maps that pair of points to a single segment per
  family — the slope is literally the domain gap.
- **Interpret.** Left anchor = synthetic-test accuracy, right anchor =
  real-test accuracy. A near-horizontal line = high transferability;
  a steep downward line = big sim-to-real gap. Families are drawn in
  order of increasing domain gap (smallest gap drawn last, i.e. on top)
  so the most transferable family is visually salient.
- **Compute.** Pulls rows where `tier == 6` from the aggregated dataframe
  and uses the per-seed means of `accuracy_synthetic`, `accuracy_real`
  and `domain_gap` that `run_single` writes into each JSON. See
  `generate_tier6_domain_gap` (`benchmark_zoo.py:1274`).

### 3.5 — `scaling_macs.pdf`
*Single-column scaling curve — Tier-1 accuracy vs. MACs, per-family upper envelope.*

- **Why.** Answers the FPGA-deployment question directly: "how much
  accuracy does this family buy per order of magnitude of compute?".
  MACs are used instead of parameter count because MACs drive FPGA
  area and power; accuracy is the only quantity the end user cares
  about. Latency is handled separately in `pareto_latency.pdf`
  because kernel-launch overhead creates latency clumps that break
  envelope curves (see docstring at `benchmark_zoo.py:807`).
- **Interpret.** Each family line is its *own* monotone upper envelope
  (cumulative max over the size variants sorted by MACs), shown as a
  solid line + ±σ band. Variants that sit *below* the envelope of their
  own family are plotted as hollow markers of the same colour and shape
  — the reader can immediately see which size variants are dominated
  by larger siblings. A family whose envelope plateaus quickly has
  hit its intrinsic ceiling for the task.
- **Compute.** `_load_tier1_aggregated` pools per-seed results per
  `(model_name)` into `(acc_mean, acc_std, macs)`; then for each family
  the envelope is extracted with a left-to-right `cummax` on `acc_mean`.
  Code: `generate_scaling_curves` (`benchmark_zoo.py:807`).

### 3.6 — `scaling_grid.pdf`
*Small-multiples scaling curve (near-square grid, `ceil(sqrt(n))` columns), **x = MACs** (compute view).*

- **Why.** Same trade-off as §3.3: individual family readability at the
  cost of cross-family comparison. Useful because envelope lines in
  `scaling_macs.pdf` cluster tightly in the "good" region and become
  hard to distinguish.
- **Interpret.** Shared log-x and linear-y axes, so *shapes* of scaling
  curves are directly comparable across panels: a family whose panel
  slopes upward steeply gains accuracy cheaply per added MAC; a
  family whose panel is already flat at its smallest variant has
  little to gain from scaling. ±σ bands are drawn per panel.
- **Compute.** Same per-`(model_name)` aggregation as §3.5; each panel
  is sorted by `macs` (not by size tag, which can disagree with MAC
  order) so the connecting line is monotone in x. Grid layout is
  computed dynamically by `_grid_layout(len(families))` — near-square
  with `n_cols = ceil(sqrt(n_families))` — so the figure always fits
  the current number of families (9 → **3 × 3** at the time of
  writing). See `generate_scaling_grid` (`benchmark_zoo.py:914`).
  Conv1DGAP's panel is the visually most distinctive: its envelope
  sits to the *left* of every other family's at comparable accuracy
  because its MACs are close to Conv1D's but its parameter count is
  ≈20× smaller.

### 3.7 — `scaling_grid_size.pdf`
*Same small-multiples layout as §3.6 but **x = on-disk model size (MB)** (storage view).*

- **Why.** The three resource axes — MACs (compute), latency
  (real-time), on-disk size (storage) — do **not** rank variants
  identically. A wider-shallow variant can have *fewer* MACs than a
  narrower-deeper sibling but a much *larger* on-disk footprint
  (the parameter tensors dominate the file). For an FPGA deployment
  whose binding constraint is BRAM/flash rather than DSP cycles,
  the storage axis is the relevant one — and §3.6 cannot answer it.
- **Interpret.** Read exactly like §3.6: shared log-x / linear-y, ±σ
  bands per panel, family colour conventions identical. The interesting
  comparison is *cross-figure*: a family whose §3.6 curve climbs
  sharply but whose §3.7 curve is shallow gets a lot of accuracy per
  MAC but pays for it in storage; the converse points to a memory-
  efficient family worth shortlisting for storage-constrained targets.
- **Compute.** `_load_tier1_aggregated` already aggregates `size_mb`
  (median across seeds is unnecessary because the on-disk size is
  deterministic per-architecture); `generate_scaling_grid` is then
  called with `x_col="size_mb", x_label="Model size (MB)",
  fname="scaling_grid_size.pdf"` — exactly the same code path as
  §3.6, just a different x-axis. See `benchmark_zoo.py:914` and the
  call site at `benchmark_zoo.py:1520`. On the storage axis Conv1DGAP
  is the clearest demonstration of the head-side ablation thesis: at
  every width tier its panel is shifted ~1.3 decades left of Conv1D
  without losing height, which is the direct visual translation of
  "the dense head owned 95 % of the parameters".

### 3.8 — `pareto.pdf`
*Single-column scatter of every Tier-1 variant with Pareto front highlighted (x = MACs, **compute view**).*

- **Why.** The scaling curves show what each family can *individually*
  achieve; the global Pareto front shows what *any* architecture can
  achieve at a given compute budget. This is the actual "which model
  should I put on the FPGA?" figure: each point on the dashed line is
  undominated in (accuracy, MACs) space.
- **Interpret.** Every colour+marker is a family; filled markers are
  all variants; the dashed black line connects Pareto-optimal points;
  these are circled and tagged with a numeric badge; a boxed legend in
  the lower-right lists the model name corresponding to each badge.
  Reading top-down along the front recovers the recommended
  architecture for every compute budget.
- **Compute.** `_pareto_front` implements the standard "maximize y,
  minimize x" sweep: sort by ascending x, keep points whose y strictly
  exceeds the running max. Same aggregation function as the scaling
  curves. See `generate_pareto_publication` (`benchmark_zoo.py:1007`).
  With the Conv1DGAP family added, Conv1DGAP variants now sit between
  the Conv1D line and the MobileNet1D line on the MACs axis — MACs
  are similar to Conv1D's but accuracy is preserved, so Conv1DGAP does
  *not* change the MACs front materially (the front is still dominated
  by MobileNet1D / EfficientNet1D at the low end and
  InceptionTime1D at the high end), which is expected because MACs
  measure compute and Conv1DGAP changes the *parameter budget*, not
  the arithmetic work. The storage and latency views below show where
  it actually moves the front.

### 3.9 — `pareto_latency.pdf`
*Specialised log-error / log-latency view — **real-time deployment figure**.*

- **Why a dedicated helper.** This is the figure the deployment
  decision actually rests on, so it gets its own renderer
  (`generate_pareto_latency_focus`, `benchmark_zoo.py:1131`) instead of
  reusing the generic `generate_pareto_publication`. Five concrete
  problems made the generic layout unreadable:
  1. **Linear accuracy wastes the canvas.** Every competitive variant
     sits in [0.94, 0.99] accuracy, so a linear y-axis squashes the
     interesting differences (1.7 % vs 2.7 % vs 4.0 % error) into
     ~5 % of the plot height. We plot **error rate (1 − acc) on a log
     y-axis** instead — those three numbers now span half the canvas.
  2. **Pico outliers blow up the y-range.** A handful of models land
     at 40–50 % error and would crush the rest of the data into a
     thin band. The y-window is **clipped at err = 0.15** (acc =
     0.85); anything outside is drawn as an upward arrow + italic
     name at the top edge so the clipping is fully transparent.
  3. **Kernel-launch floor.** On the RTX A1000 the per-call CUDA
     overhead is ~0.13 ms, creating a vertical clump of small
     variants at the leftmost x-edge that share latency but not
     accuracy. The shaded band + "kernel-launch floor" italic label
     makes that hardware artefact explicit.
  4. **Pareto label collisions.** Three of the six Pareto points sit
     inside the kernel-launch clump and overlap visually. We use
     **numbered gold-star badges (1–6)** + a fixed-width **side-key
     table** anchored to the lower-right corner, listing model name,
     latency, and accuracy. The table doubles as a deployment
     cheat-sheet.
  5. **Iso-accuracy translation.** Horizontal dotted lines at acc =
     {0.90, 0.95, 0.97, 0.98, 0.99} with right-edge labels let the
     reader translate any error reading on the left axis into the
     familiar accuracy reading they expect, without a second twin
     axis.
- **Interpret.** Read each point as `(latency, error)`. Lower-left is
  better. Family colour + marker matches every other figure in the
  report. The dashed line through the gold stars is the global Pareto
  front computed on the full dataset (truthful even when some points
  lie outside the displayed y-window). Bold black numbers `1`–`6`
  next to each star map onto the side-key, which spells out the
  model name and exact `(lat, acc)` numbers — so the figure is fully
  self-contained without going back to the leaderboard.
- **Compute.** `_load_tier1_aggregated` (Tier 1, mean over seeds) →
  `_pareto_front` (maximise acc, minimise latency) → custom
  matplotlib layout described above (log-log axes, custom y-tick
  locator at `{1, 2, 3, 5} × 10^k` so the left axis labels 0.02 /
  0.03 / 0.05 / 0.1, kernel-floor `axvspan`, side-key text box, iso-
  acc `axhline`s). Implemented in `generate_pareto_latency_focus`
  (`benchmark_zoo.py:1132`) and wired in
  `regenerate_and_publish_figures` (`benchmark_zoo.py:1726`).
  Conv1DGAP variants appear on the latency front: because their
  convolutional backbone is the same as Conv1D, their latency is
  very close to Conv1D's, but they carry a flat FC head with `c3 →
  fc_hidden` instead of `c3 × 78 → fc_hidden`, so the second Linear
  call is dramatically cheaper and the Pareto-optimal slot at low
  error rate goes to a Conv1DGAP variant rather than a Conv1D one.

### 3.10 — `pareto_size.pdf`
*Same layout as `pareto.pdf` but x = on-disk model size (MB) — **storage view**.*

- **Why.** Closes the third resource axis: storage / on-disk footprint.
  This is the binding constraint when the deployment target is an
  embedded board whose flash, BRAM, or weight-tile cache cannot hold
  the entire model. A variant that is MAC-optimal but stores 84 MB of
  weights is useless on a 32 MB device, no matter how fast it runs.
- **Interpret.** Same conventions as §3.8 — colour, marker, badge,
  boxed legend listing the model name behind each Pareto badge. The
  diagnostic move is to compare the **three** Pareto fronts side-by-
  side (`pareto.pdf` ↔ `pareto_latency.pdf` ↔ `pareto_size.pdf`):
  models that survive on all three are universally efficient; models
  that drop off only the storage front are compute-cheap but
  parameter-heavy and should be shortlisted only when memory is
  abundant. Note that `pareto_latency.pdf` (§3.9) uses a different
  custom layout (log-error y-axis, kernel-floor shading, side-key
  table) because the latency view is the deployment-decision figure
  and warranted its own renderer.
- **Compute.** Identical to §3.8 except `x_col="size_mb"`,
  `x_label="Model size (MB)"`, `fname="pareto_size.pdf"`. The
  `_load_tier1_aggregated` helper exposes `size_mb` directly from each
  per-run JSON's `size_mb` field (set by `measure_model_size`,
  `training_utils.py`). Wired at `benchmark_zoo.py:1727`. On the
  storage axis, Conv1DGAP is the clearest winner: at base width its
  on-disk size is ≈1.1 MB (vs ≈20 MB for Conv1D at iso-accuracy),
  and `Conv1DGAP-S` / `Conv1DGAP` claim two slots on the storage
  Pareto front outright — exactly the slots that were previously
  owned by MobileNet1D and EfficientNet1D.

### 3.11 — Diagnostic PNGs (not publication, emitted by `generate_plots`)
*`confusion_matrix_best_<model>.png`, `confusion_matrix_worst_<model>.png`,
`f1_heatmap.png`, `seed_stability_boxplot.png`.*

- **Why.** Sanity checks to make sure the aggregated metrics in the
  publication figures actually reflect the per-class behaviour. These
  are intentionally raster PNGs to make it obvious they are *not* part
  of the paper's figure set (see `doc/variant_plotting.md` §8.5).
- **Interpret.**
  - Confusion matrices — pick out class confusions the scalar accuracy
    hides (typically 2 µm ↔ 4 µm on this dataset).
  - `f1_heatmap.png` — per-class F1 for every Tier-1 model; looks for
    models that are high-accuracy overall but collapse on a single
    rare class.
  - `seed_stability_boxplot.png` — per-model distribution of Tier-1
    accuracy across seeds. Tall boxes = unstable training and should
    trigger more seeds or a higher epoch budget.
- **Compute.** `generate_plots` (`benchmark_zoo.py:687`) loads
  Tier-1-only results, picks best/worst by mean accuracy, renders the
  confusion matrices with `sns.heatmap`, and boxplots seed-level
  accuracy with `sns.boxplot`.

---

## 4 — W&B layout

All per-run trainings go to project **`particle-benchmark`** under
`group = <family_name>`, `tags ⊇ {benchmark2, tier<N>, seed<N>}`,
`name = <model>-<dataset>-tier<N>-seed<N>`.

Publication figures are pushed to a separate
`benchmark2-report-<timestamp>` run with
`group = benchmark2-report`, `job_type = benchmark-report`. Each figure
is logged under a stable media key:

- `figures/tier/heatmap` — §3.1
- `figures/tier/robustness` — §3.2
- `figures/tier/robustness_grid` — §3.3
- `figures/tier/domain_gap_t6` — §3.4
- `figures/variant/scaling_macs` — §3.5 (compute view, envelope)
- `figures/variant/scaling_grid` — §3.6 (compute view, small-multiples)
- `figures/variant/scaling_grid_size` — §3.7 (storage view, small-multiples)
- `figures/variant/pareto_macs` — §3.8 (compute Pareto)
- `figures/variant/pareto_latency` — §3.9 (real-time Pareto)
- `figures/variant/pareto_size` — §3.10 (storage Pareto)

The original PDFs plus `summary.csv` + `leaderboard.md` are bundled
into a `benchmark2-report` artifact attached to the same run. This
keeps the per-model training runs free of figure clutter while still
making the publication assets discoverable from one place.

The three Pareto figures (§3.8 / §3.9 / §3.10) are intentionally a
**triptych over the three resource axes**: MACs (computing budget),
latency (real-time throughput), on-disk size (storage / BRAM
footprint). Reading them together is the canonical "which model
should I put on this device?" workflow.
