# Variant Plotting — Design and Justification

> Documentation for the publication-quality plotting system used by the
> *Model Zoo Scaling* benchmark (`benchmark_zoo.py --scaling`).
> Covers the figures produced under `results/<run>/figures/`, the design
> constraints they satisfy, and how they are meant to be read.

---

## 1 — Motivation

Benchmark 2 evaluates 8 architecture families across up to 7 size variants
each (51 trained models in total). The first iteration of the plotting
code reused the generic matplotlib defaults from Benchmark 1 (PNG, tight
bbox, default Tab10 palette, inline labels). It produced figures that:

1. Did not fit cleanly into a LaTeX two-column layout — `bbox_inches="tight"`
   makes every figure a different size, so `\includegraphics{...}` ends up
   with inconsistent on-page font sizes between figures.
2. Were unreadable to colorblind reviewers (color was the only visual
   channel separating 8 families).
3. Suffered from visual "asymptotes" on the latency scaling curve because
   small variants are kernel-launch-bound (~0.13 ms on the test hardware)
   and several variants land on top of each other with very different
   accuracies, producing zig-zag artefacts.
4. Cluttered the Pareto front with overlapping inline text labels.

This document explains the design choices that fixed each of those
problems and the conventions every new variant figure must follow.

---

## 2 — Hard constraints

Every figure produced by `benchmark_zoo.py` for the variant analysis must
satisfy the following:

| Constraint              | Why                                                          |
|-------------------------|--------------------------------------------------------------|
| **Fixed canvas size**   | Side-by-side LaTeX placement requires identical `\includegraphics` widths so font sizes match across the paper. |
| **PDF output**          | Vector graphics, infinitely zoomable, and editable in Illustrator (`pdf.fonttype=42`). |
| **Colorblind-safe**     | Reviewers may have CVD; B&W printers exist. |
| **Three visual channels** | Color + marker shape + linestyle, all redundant. Any single channel is enough to identify a family. |
| **No overlapping labels** | Annotations must not collide. Use boxed keys + numeric badges instead. |
| **Single-source style** | All figures share `apply_publication_style()` so fonts, tick widths, and grid alpha never drift. |

These constraints are encoded directly in `benchmark_zoo.py` as module-level
constants:

- `COL_W = 3.39` — IEEE/Elsevier single-column width (inches).
- `DCOL_W = 7.00` — full text width.
- `FIG_SINGLE_TALL = (3.39, 3.22)` — single-column figure with bottom legend.
- `FIG_GRID = (7.00, 3.50)` — double-column 2×4 small-multiples.
- `PUB_RC` — matplotlib rcParams (font size 8, serif, `savefig.bbox="standard"`).
- `FAMILY_COLORS`, `FAMILY_MARKERS`, `FAMILY_LINESTYLES` — three fully
  independent per-family encoding dicts (Okabe–Ito palette, 8 unique
  marker shapes, 8 unique linestyles).

`savefig.bbox="standard"` is critical: it preserves the canvas, in
contrast to `"tight"` which crops it to whatever the current artist
bounds happen to be. Without this, every figure would have a slightly
different on-page size in LaTeX.

---

## 3 — Figure inventory

All figures are emitted to `results/<run>/figures/` by
`generate_scaling_curves`, `generate_scaling_grid` (called twice —
compute + storage view), `generate_pareto_publication` (called twice —
MACs + size view), and `generate_pareto_latency_focus` (the dedicated
log-error latency renderer).

| File                     | Function                           | Format           | Purpose |
|--------------------------|------------------------------------|------------------|---------|
| `scaling_macs.pdf`       | `generate_scaling_curves`          | single column    | Per-family upper envelope of accuracy vs MACs (log x). The "scaling law" view. |
| `scaling_grid.pdf`       | `generate_scaling_grid`            | double column    | 2×4 small-multiples — one panel per family, shared axes, for quick visual comparison of scaling shapes (compute view). |
| `scaling_grid_size.pdf`  | `generate_scaling_grid`            | double column    | Same 2×4 small-multiples layout but with on-disk model size (MB) on the x-axis — storage view, complementary to the MACs view. |
| `pareto.pdf`             | `generate_pareto_publication`      | single column    | Accuracy vs MACs scatter + global Pareto front. Numbered badges + boxed key list every front member. |
| `pareto_size.pdf`        | `generate_pareto_publication`      | single column    | Same layout as `pareto.pdf` but with on-disk size (MB, log x) — storage / BRAM-footprint view. |
| `pareto_latency.pdf`     | `generate_pareto_latency_focus`    | single column    | **Specialised** log-error / log-latency renderer — clipped y-window, kernel-launch floor shading, numbered Pareto badges + side-key table, iso-accuracy guides. The deployment-decision figure. |

Note that `scaling_latency.pdf` was **deliberately removed** — see §6.

### 3.1 — `scaling_macs.pdf` — Per-family upper envelope

For each family, sort variants by MACs ascending, then keep only those
points whose accuracy ties or beats the cumulative max along the x-axis.
This is the family's own Pareto front in *(MACs, accuracy)* space, and
is what "scaling curve" actually means: the best accuracy achievable at
this compute budget.

Variants that fall *below* their family's envelope (i.e., a smaller
sibling already matched or beat them) are still drawn as faint hollow
markers so the reader can see they exist without polluting the line.

Why an envelope rather than a line through every variant?
A naive `plot(sort_by_x(variants))` produces a zig-zag any time two
variants share roughly the same x-value but differ in accuracy — and
that happens routinely with small variants that are kernel-launch-bound.
Cumulative max is monotone non-decreasing by construction, so the line
is always interpretable as "best you can do at ≤ this budget".

### 3.2 — `scaling_grid.pdf` — Small multiples

Tufte-style 2×4 grid of single-family scaling curves with shared x and
y axes. Each panel shows one architecture family across all its
variants. This is the "is the family well-behaved?" view — a curve that
plateaus, dips, or spikes is immediately visible because every panel
has the same scale.

Points inside each panel are sorted by **MACs** (the x-axis), not by
size-tag ordinal. Sorting by size tag would be wrong because size order
and MAC order can diverge — a wider shallow variant may have fewer MACs
than a narrower deep one — which would produce the same zig-zag
pathology on a log-MACs axis that the envelope filter fixes in the
main plot. Unlike `scaling_macs.pdf`, the grid does not apply the upper
envelope: every variant is drawn so that per-family non-monotonicities
remain visible, which is the entire point of the small-multiples view.

Used as a complement to `scaling_macs.pdf`, not a replacement.
The combined plot answers "which family scales best?", the grid
answers "does each family scale at all?".

### 3.3 — `pareto.pdf` and `pareto_size.pdf`

Single-column scatter of every variant in *(x, accuracy)* space, with
the *global* Pareto front computed across all families and overlaid as
a dashed black line. Pareto-optimal points are circled, annotated with
a tiny numeric badge (1, 2, 3, …), and listed in a small boxed key in
the lower-right corner of the axes that maps each badge to its full
model name.

The boxed key is the key innovation that lets us name front members
without cluttering the data area. Earlier versions placed the model
name next to each circled point, which produced overlapping text on
crowded fronts.

`pareto.pdf` uses MACs (log x) and `pareto_size.pdf` uses on-disk
size in MB (log x). They share the function
`generate_pareto_publication(output_dir, x_col, x_label, x_log, fname)`,
which is called twice from the wiring in
`regenerate_and_publish_figures` (`benchmark_zoo.py`).

### 3.4 — `pareto_latency.pdf` — specialised log-error view

The latency view uses its own renderer (`generate_pareto_latency_focus`)
rather than the generic `generate_pareto_publication`. A boss-facing
design iteration revealed that the generic layout wasted the canvas:
every competitive variant sits in [0.94, 0.99] accuracy, and a linear
y-axis compresses the most interesting differences (1.7 % vs 2.7 %
vs 4.0 % error) into ~5 % of the plot height. Five changes together
make the latency figure actually readable:

1. **Log error rate, not linear accuracy.** Plotting `1 − acc` on a
   log y-axis spreads the 94 %–99 % band across half the canvas.
2. **Clipped y-window at err = 0.15.** Pico outliers at 40–50 % error
   would otherwise crush the interesting band into a thin strip. Any
   point outside the window gets an explicit upward arrow + italic
   name label at the top edge so the clipping is transparent.
3. **Kernel-launch floor shading.** The RTX A1000's per-call CUDA
   overhead (~0.13 ms) creates a vertical clump of small variants at
   the leftmost x-edge that share latency but not accuracy. A shaded
   band + "kernel-launch floor" italic label makes the hardware
   artefact explicit.
4. **Numbered Pareto badges + lower-right side-key table.** Three of
   the six Pareto points sit inside the kernel-launch clump and would
   overlap with inline labels. Gold-star markers carry bold numeric
   badges (1–6) that map onto a monospace table in the lower-right
   listing `(model, latency, accuracy)` — a self-contained
   deployment cheat-sheet.
5. **Iso-accuracy guides.** Horizontal dotted lines at acc =
   {0.90, 0.95, 0.97, 0.98, 0.99} with right-edge labels let the
   reader translate any error reading back into a familiar accuracy
   reading without a twin axis.

The custom y-tick locator (subs = `(1, 2, 3, 5) × 10^k`) labels the
left axis at 0.02 / 0.03 / 0.05 / 0.1 rather than only at the
decade, which matches the density of the data. The Pareto front is
computed on the **full** dataset (not just the clipped window) so
the dashed line stays truthful when some variants lie outside the
displayed y-range.

---

## 4 — Colorblind-safe encoding

Every figure uses **three fully independent visual channels** so any
single channel can identify a family on its own:

1. **Color** — Okabe–Ito 8-color palette (CVD-safe under deuteranopia,
   protanopia, and tritanopia).
2. **Marker shape** — circle, square, triangle, diamond, etc. (8 distinct
   shapes for 8 families).
3. **Linestyle** — 8 distinct patterns: the four standard matplotlib
   strings (`-`, `--`, `-.`, `:`) for Conv1D/DenseNet1D/EfficientNet1D/
   InceptionTime1D, plus four custom `(offset, on_off_seq)` tuples for
   LeNet1D (dash-dot-dot), MobileNet1D (dense long-dash), ResNet1D
   (dot-dot-dash), and VGG1D (dash-dot-dot-dot). Each pattern was chosen
   to stay visually distinct at the 1.2 pt publication linewidth defined
   in `PUB_RC`.

This redundancy means the figures stay readable when:

- Printed in black and white.
- Viewed by a colorblind reader.
- Photocopied or projected at low contrast.

The encoding tables live in `benchmark_zoo.py` lines 97–136 and are
imported by every plotting function that draws a family-grouped series.

---

## 5 — Plot interpretation guide

Each figure answers a specific question. Read them in this order.

### 5.1 — `scaling_macs.pdf` — *"How does each family scale?"*
- **X-axis**: MACs (log) — the compute budget.
- **Y-axis**: Accuracy on the held-out real test set.
- **Solid line + filled markers**: family upper envelope.
- **Hollow markers**: dominated variants in the same family.
- **Shaded band**: ±1 std across seeds on envelope points.
- **What to look for**: a steeper slope means the family converts compute
  into accuracy more efficiently. A flat line means more compute does
  not buy more accuracy — the family has plateaued.

### 5.2 — `scaling_grid.pdf` — *"Is the scaling well-behaved per family?"*
- One panel per family, shared axes.
- **What to look for**: monotone curves (good), saturating curves
  (the family has hit its limit), or non-monotone curves (a variant is
  poorly tuned and should be retrained).

### 5.3 — `pareto.pdf` — *"Which model should I pick at a given MAC budget?"*
- **X-axis**: MACs (log).
- **Y-axis**: Accuracy.
- **Dashed black line**: global Pareto front across *all* families.
- **Circled points + numbered badges**: front members.
- **Boxed key**: badge → model name mapping.
- **What to look for**: the front member directly above your MAC budget
  is the best model you can deploy at that budget. Models far above the
  front are dominated and should not be considered.

### 5.4 — `pareto_latency.pdf` — *"Which model should I pick at a given latency budget?"*
- **X-axis**: Latency in milliseconds (log).
- **Y-axis**: Error rate `1 − acc` on a log scale, clipped at
  err = 0.15 (acc = 0.85). The right edge carries iso-accuracy
  reference labels {0.90, 0.95, 0.97, 0.98} so the reader can read
  either error or accuracy from the same dotted lines.
- **Grey band on the left**: the kernel-launch floor (~0.13 ms on
  RTX A1000). Any variant inside the band is bottlenecked by CUDA
  launch overhead, not by its own arithmetic — the vertical clump
  there is an expected hardware artefact, not a modelling result.
- **Upward arrows at the top edge**: points that fell outside the
  clipped y-window (typically `LeNet1D-Pico`). The italic name next
  to the arrow identifies the variant.
- **Gold stars + bold numeric badges**: global Pareto-optimal points
  across all families, computed on the full dataset. The dashed
  black line connects them in latency order.
- **Lower-right side-key table**: maps each badge to its model name,
  exact median latency, and exact mean accuracy — the figure is
  fully self-contained, no cross-reference to the leaderboard
  required.
- **What to look for**: scan upward from your latency budget along a
  vertical line, find the first Pareto star above it, and read its
  number off the side-key. That is the lowest-error model that
  meets your latency constraint on the target hardware. This is the
  figure that drives the FPGA deployment decision.

### 5.5 — `pareto_size.pdf` — *"Which model should I pick at a given storage budget?"*
- **X-axis**: on-disk model size in MB (log).
- Otherwise identical to `pareto.pdf`.
- **What to look for**: the binding constraint when the deployment
  target is an embedded board whose flash / BRAM / weight-tile
  cache cannot hold the full model. Comparing the MACs / latency /
  size fronts side-by-side is the canonical "universally efficient
  vs axis-specific" diagnostic.

---

## 6 — Why `scaling_latency.pdf` was removed

Three iterations of the latency scaling curve all failed to produce a
readable figure:

1. **v1 — Sort by latency, draw line.** Zig-zag artefacts everywhere
   because several small variants share latencies near the kernel-launch
   floor (~0.13 ms) but have very different accuracies. The line ran
   straight up and down at those latencies — visual "vertical
   asymptotes".

2. **v2 — Switch to log-x.** Spread the cluster horizontally but the
   line still zig-zagged inside the cluster.

3. **v3 — Per-family upper envelope (the same trick as MACs).** Removed
   the zig-zags by construction, but produced a different pathology:
   most families collapse to 1–2 envelope points because their variants
   all live near the launch floor, and MobileNet1D collapses to a single
   point. The figure became a thicket of lone markers with no curves.

The fundamental problem is that **MACs and latency are not interchangeable**
on the small end of the scale. Below ~10⁵ MACs, every variant runs in
roughly the same wall-clock time because the GPU/CPU kernel-launch
overhead dominates the actual compute. A scaling-curve metaphor (line
through ordered points) cannot represent that, but a scatter + global
Pareto front handles it gracefully: vertical clumps simply appear as
stacked dots, with only the topmost landing on the front.

The Pareto plot subsumes the scaling-curve information for latency
without the readability problem, so `scaling_latency.pdf` was dropped
in favour of `pareto_latency.pdf`. The latency Pareto later grew its
own specialised renderer (§3.4) because the generic layout still
wasted the canvas on the linear-accuracy view. The current wiring is:

```python
# benchmark_zoo.py — regenerate_and_publish_figures
generate_scaling_curves(args.output_dir)            # only emits scaling_macs.pdf now
generate_scaling_grid(args.output_dir)              # compute view (MACs)
generate_scaling_grid(                              # storage view (MB)
    args.output_dir,
    x_col="size_mb", x_label="Model size (MB)",
    fname="scaling_grid_size.pdf",
)
generate_pareto_publication(args.output_dir)        # MACs (default)
generate_pareto_latency_focus(args.output_dir)      # specialised log-error
generate_pareto_publication(
    args.output_dir,
    x_col="size_mb", x_label="Model size (MB)",
    x_log=True, fname="pareto_size.pdf",
)
```

---

## 7 — Relationship between params, MACs, and latency

The three axes are correlated but **not redundant** — each captures a
different cost dimension and they can disagree:

| Axis    | Measures                     | Bottleneck            |
|---------|------------------------------|-----------------------|
| Params  | Memory footprint (weights)   | RAM / FPGA BRAM       |
| MACs    | Theoretical work per sample  | Compute / energy      |
| Latency | Actual wall-clock per sample | The deployment target |

Examples of where they disagree:

- **MobileNet1D vs ResNet1D at base size**: similar params (~5 M),
  similar MACs (~135 M vs ~146 M), but MobileNet1D is *slower* at
  latency because depthwise separable convolutions are kernel-launch
  heavy on CPU.
- **Small variants of any family**: 10–100× spread in MACs maps to a
  ~2× spread in latency because the kernel-launch overhead floor
  dominates.
- **Params vs MACs**: a deep narrow network can have many params but
  few MACs per sample. A wide shallow network is the opposite.

That is why the benchmark plots both `pareto.pdf` (MACs) and
`pareto_latency.pdf` (latency), and why `scaling_macs.pdf` exists as a
separate figure from the Pareto plots: MACs is the architectural
property to optimise during design, latency is the deployment-time
property to optimise during model selection.

---

## 8 — Adding a new variant figure

If you need to add another figure to the benchmark, follow this
checklist so it slots cleanly into the existing system.

1. **Call `apply_publication_style()`** at the top of the function.
2. **Use one of the `FIG_*` size constants** — never `figsize=(...)`
   inline. If none of the existing sizes fits, add a new constant
   alongside `FIG_SINGLE_TALL`.
3. **Color, marker, linestyle by family** must come from the three
   `FAMILY_*` dicts. Never hardcode an `ax.scatter(color="red")`.
4. **Set `fig.subplots_adjust(left=…, right=…, top=…, bottom=…)`
   manually** so the canvas matches sibling figures. Do not rely on
   `tight_layout` — it breaks `savefig.bbox="standard"`.
5. **Save via `_emit_pdf(fig, figures_dir, "name.pdf", wandb_run=wandb_run,
   wandb_key="figures/variant/<slug>", caption=...)`** — the helper
   writes the PDF, optionally logs a `wandb.Image` preview under the
   supplied media key, and closes the Figure. No PNG sibling file is
   written. The 4 stale PNGs (`scaling_curves.png`,
   `scaling_curves_latency.png`, `pareto_accuracy_vs_macs.png`,
   `pareto_accuracy_vs_latency.png`) were removed in the
   publication-quality refactor and should not come back.
6. **Accept `wandb_run=None` as a keyword-only parameter** on the
   `generate_*` function signature and forward it to `_emit_pdf`. This
   lets the benchmark-report orchestrator (§10) group every figure
   under one W&B run, while still keeping the function callable
   standalone with no W&B publishing.
7. **Add the new function to `regenerate_and_publish_figures`** in
   `benchmark_zoo.py` (around line 1433). The two wiring blocks in
   `main()` now dispatch through that single orchestrator, so a new
   figure only needs to be wired once.
8. **Document the new figure here** by adding a row to §3, an
   interpretation paragraph to §5, and — if it needs its own W&B
   panel — a row to the §10 media-key table.

---

## 9 — File and code references

- `benchmark_zoo.py:96-135` — `FAMILY_COLORS`, `FAMILY_MARKERS`,
  `FAMILY_LINESTYLES` (all three dicts have 8 unique entries).
- `benchmark_zoo.py:142-171` — figure size constants and `PUB_RC`.
- `benchmark_zoo.py:174-179` — `apply_publication_style()`.
- `benchmark_zoo.py:182-216` — `_emit_pdf` helper (saves PDF, closes
  `Figure`, optional `wandb.Image` logging).
- `benchmark_zoo.py:807-911` — `generate_scaling_curves` (only emits
  `scaling_macs.pdf`).
- `benchmark_zoo.py:913-980` — `generate_scaling_grid` (sorted by
  `macs`, not by size-tag ordinal).
- `benchmark_zoo.py:982-1091` — `generate_pareto_publication`
  (parameterised on `x_col`, `x_label`, `x_log`, `fname`; handles
  the MACs and on-disk-size fronts, dispatches the W&B key on
  `fname`).
- `benchmark_zoo.py:1124-1313` — `KERNEL_LAUNCH_FLOOR_MS` module
  constant (0.14 ms on RTX A1000) plus `generate_pareto_latency_focus`,
  the dedicated log-error renderer for `pareto_latency.pdf` (see
  §3.4 for the rationale).
- `benchmark_zoo.py:1395-1434` — `_attach_report_artifacts` (uploads
  PDFs + `summary.csv` + `leaderboard.md` as a single `wandb.Artifact`
  and writes headline metrics to `run.summary`).
- `benchmark_zoo.py:1433-1497` — `regenerate_and_publish_figures`
  (single orchestrator for every publication figure; opens the
  report run when `wandb_publish=True`).
- `benchmark_zoo.py:1691-1695` and `:1727-1731` — wiring in
  `main()`; both `--aggregate-only` and post-training paths dispatch
  through `regenerate_and_publish_figures`.

Regenerate every figure from cached JSON results without retraining:

```bash
source venv/bin/activate
python benchmark_zoo.py --aggregate-only --scaling --output-dir results/benchmark2
```

---

## 10 — Publishing to W&B

Every publication figure is also pushed to a dedicated **benchmark
report** Weights & Biases run whenever `regenerate_and_publish_figures`
runs without `--no-wandb-publish`. This happens automatically in both
the `--aggregate-only` path and the post-training path; no extra CLI
flag is required to *enable* publishing — it's the default.

### 10.1 — The report run

| W&B field  | Value                                  |
|------------|----------------------------------------|
| `project`  | `particle-benchmark` (same as training runs) |
| `name`     | `benchmark2-report-YYYYMMDDTHHMMSS` (ISO timestamp keeps it sortable and collision-free across aggregations) |
| `group`    | `benchmark2-report` (distinct from per-model training groups) |
| `job_type` | `benchmark-report` |
| `tags`     | `["benchmark2", "report", "publication", <dataset_name>]` where `dataset_name = Path(args.data_dir).name` |
| `mode`     | `"offline"` when `--wandb-offline` is set, otherwise `"online"` |

The report run is intentionally *not* a per-model training run: the
training runs fan out one per `(model, tier, seed)` and report
training-loop metrics, while the report run reports cross-model
aggregates. Keeping them in the same project but separate groups lets
you filter the W&B UI to either view.

### 10.2 — Variant-axis media keys

The variant figures share a single `figures/variant/*` namespace, so
they land together in the W&B media panel:

| Function                         | Local file                | W&B key                              |
|----------------------------------|---------------------------|--------------------------------------|
| `generate_scaling_curves`        | `scaling_macs.pdf`        | `figures/variant/scaling_macs`       |
| `generate_scaling_grid`          | `scaling_grid.pdf`        | `figures/variant/scaling_grid`       |
| `generate_scaling_grid`          | `scaling_grid_size.pdf`   | `figures/variant/scaling_grid_size`  |
| `generate_pareto_publication`    | `pareto.pdf`              | `figures/variant/pareto_macs`        |
| `generate_pareto_publication`    | `pareto_size.pdf`         | `figures/variant/pareto_size`        |
| `generate_pareto_latency_focus`  | `pareto_latency.pdf`      | `figures/variant/pareto_latency`     |

`generate_pareto_publication` dispatches its W&B key on `fname`
(`"size"` → `pareto_size`, else `pareto_macs`) so the two invocations
land on two distinct panels rather than overwriting each other.
`generate_scaling_grid` uses the same `fname`-based dispatch pattern
(`"size"` → `scaling_grid_size`, else `scaling_grid`). The latency
front has its own dedicated function and media key.

### 10.3 — The `benchmark2-report` artifact

Every report run also uploads a single
`wandb.Artifact(name="benchmark2-report", type="benchmark-report")`
that contains the *original vector PDFs* under `figures/*.pdf` plus
`summary.csv` and `leaderboard.md` at the artifact root. The PDFs are
what the paper actually `\includegraphics{}`s — the `wandb.Image`
previews logged under the `figures/` media keys are PNG rasterisations
that exist for quick browsing in the web UI.

A few headline metrics are also written to `run.summary`:
`num_models`, `num_families`, `num_tiers`, `num_results`,
`best_tier1_model`, `best_tier1_accuracy`.

### 10.4 — Opting out

Use `--no-wandb-publish` to skip the report run entirely. The local
PDFs are still emitted to `results/<run>/figures/` (that's the point
of the existing layout discipline), only the W&B side is disabled.
Pair with `--wandb-offline` if you want to publish but don't have
online credentials — both paths use the same offline cache under
`wandb/offline-run-*`.

```bash
# Publish locally only, no W&B side effect
python benchmark_zoo.py --aggregate-only --scaling \
    --output-dir results/benchmark2 --no-wandb-publish

# Publish to offline W&B cache (sync later with `wandb sync`)
python benchmark_zoo.py --aggregate-only --scaling \
    --output-dir results/benchmark2 --wandb-offline
```
