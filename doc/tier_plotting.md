# Tier Plotting — Design and Justification

> Documentation for the publication-quality plotting system used by the
> *Difficulty Tier* benchmark (`benchmark_zoo.py --tier all`). Covers the
> figures produced under `results/<run>/figures/` that live on the *tier
> axis* (robustness / degradation curves) rather than the size axis.
> Sibling document: [`variant_plotting.md`](variant_plotting.md) —
> same infrastructure, different question.

---

## 1 — Motivation

Benchmark 2 grades every architecture across 6 difficulty tiers:

| Tier | Name              | What it stresses                                     |
|------|-------------------|------------------------------------------------------|
| 1    | standard          | Baseline (full synthetic training set)               |
| 2    | data_starved      | 20 samples/class — small-data regime                 |
| 3    | noisy             | GaussianNoise SNR=10 dB at test time                 |
| 4    | combined          | Data-starved **and** noisy                           |
| 5    | noise_extreme     | Real captured noise (SNR ∈ [−3, 3] dB) + time mask   |
| 6    | domain_shift_real | Train on synthetic, test on real measurements        |

The variant plots (`scaling_macs.pdf`, `scaling_grid.pdf`, `pareto*.pdf`)
answer *"how does each family convert compute into accuracy?"* — they
live on the size axis and treat tier 1 as ground truth. They cannot
answer the tier-axis questions that drive model selection for a noisy
FPGA deployment:

1. **Which family degrades most gracefully as the operating conditions
   get harder?** (tier robustness curve)
2. **Does each family degrade smoothly or does it have a pathological
   tier where it collapses?** (small-multiples tier grid)
3. **How large is the sim-to-real gap on a per-family basis?**
   (tier 6 domain-gap slope chart)

This document describes the three tier-axis figures that complement the
pre-existing `tier_heatmap.pdf`, the design choices they share with the
variant figures, and the conventions every new tier figure must follow.

---

## 2 — Hard constraints

Every tier figure reuses the *same* publication infrastructure as the
variant figures, so the constraints are identical to
[`variant_plotting.md` §2](variant_plotting.md):

| Constraint              | Why                                                          |
|-------------------------|--------------------------------------------------------------|
| **Fixed canvas size**   | Side-by-side LaTeX placement requires identical `\includegraphics` widths so font sizes match across the paper. |
| **PDF output**          | Vector graphics, infinitely zoomable, and editable in Illustrator (`pdf.fonttype=42`). |
| **Colorblind-safe**     | Reviewers may have CVD; B&W printers exist. |
| **Three visual channels** | Color + marker shape + linestyle, all redundant. Any single channel is enough to identify a family. |
| **Single-source style** | All tier figures call `apply_publication_style()` — same `PUB_RC`, same fonts, same tick widths as their variant siblings. |

Concretely, the tier figures import the same module-level constants
from `benchmark_zoo.py`:

- `FIG_SINGLE_TALL = (3.39, 3.22)` — `tier_robustness.pdf` and `tier6_domain_gap.pdf`.
- `FIG_GRID = (7.00, 3.50)` — `tier_grid.pdf`.
- `FIG_SINGLE` — `tier_heatmap.pdf`.
- `FAMILY_COLORS`, `FAMILY_MARKERS`, `FAMILY_LINESTYLES` — identical family
  encoding to the variant figures, with 8 unique colors, 8 unique marker
  shapes and 8 unique linestyles, so the same line style means the same
  family across the whole paper.

Canvas verification: `tier_robustness.pdf` and `tier6_domain_gap.pdf`
both render at `244.08 × 231.876 pts` — bit-for-bit identical to
`scaling_macs.pdf`. `tier_grid.pdf` is `504 × 252 pts` — bit-for-bit
identical to `scaling_grid.pdf`. This is the key property that lets a
LaTeX paper place a variant figure and a tier figure side by side
without any font-size drift.

---

## 3 — Figure inventory

All four figures are emitted to `results/<run>/figures/` by
`generate_tier_heatmap`, `generate_tier_robustness`, `generate_tier_grid`,
and `generate_tier6_domain_gap`. They are wired unconditionally in
`main()` — unlike the variant figures, they do **not** require
`--scaling`, because tier-axis data is produced by every benchmark run.

| File                    | Function                         | Format          | Purpose |
|-------------------------|----------------------------------|-----------------|---------|
| `tier_heatmap.pdf`      | `generate_tier_heatmap`          | single column   | Compact overview: base models × tiers, colored by accuracy. The "at-a-glance" leaderboard. |
| `tier_robustness.pdf`   | `generate_tier_robustness`       | single column   | Per-family degradation curve — one line per family, tier on x, accuracy on y, ±σ band across seeds. The "which family ages best" view. |
| `tier_grid.pdf`         | `generate_tier_grid`             | double column   | 2×4 small-multiples — one panel per family, shared axes. The "does each family degrade smoothly" view. |
| `tier6_domain_gap.pdf`  | `generate_tier6_domain_gap`      | single column   | Slope chart: `synthetic → real` accuracy drop per family on tier 6. The "how painful is sim-to-real" view. |

### 3.1 — `tier_heatmap.pdf` — Base models × tiers

Pre-existing figure, kept as-is. Pivot table of accuracy with rows =
model names (base size only) and columns = tiers, colored with
`YlGnBu`. The annotation font is small (`annot_kws={"size": 6.5}`) so
the whole table fits in one column.

Shows what's easy to see in a table but pops visually: rows that go
dark on the right (large drop from T1 to T5/T6) are fragile families;
rows that stay light across all columns are robust.

### 3.2 — `tier_robustness.pdf` — Per-family robustness curve

Tier-axis analogue of `scaling_macs.pdf`. For each architecture family
we draw a line across the available tiers using the family's canonical
*base* variant (`size_tag == "M"`), with a ±σ band across seeds drawn
via `ax.fill_between`.

Why base variants only? If we plotted every S/M/L variant we would
produce the exact "vertical clutter" pathology that `scaling_macs.pdf`
solves with an upper envelope. On the tier axis there is no analogue of
"MACs" to sort by, so the upper envelope trick does not apply. Limiting
the plot to the base variant (one curve per family) keeps it readable.
This is the same convention used by the pre-existing
`tier_heatmap.pdf`, enforced at a single place:
`_load_all_tiers_aggregated` filters with `base = df[df["model_size_tag"] == "M"]`
(`benchmark_zoo.py:1083`).

The x-axis uses integer tier indices `[1..6]`, not the tier name, so
the spacing is regular and the visual slope corresponds to the actual
numeric drop in accuracy. Tier names are only used as labels:
`ax.set_xticklabels([f"T{t}" for t in tiers])`.

### 3.3 — `tier_grid.pdf` — Small multiples

Tufte-style 2×4 grid of single-family tier-robustness curves with
shared x and y axes. Same idea as `scaling_grid.pdf`: the combined
plot (`tier_robustness.pdf`) answers *"which family wins"*, the grid
answers *"does each family behave sensibly"*. A non-monotone curve in
a panel immediately flags a training pathology (e.g., the tier-5 run
crashed for this family and we're seeing a random-init score).

### 3.4 — `tier6_domain_gap.pdf` — Sim-to-real slope chart

Tier 6 is special: it trains on synthetic and tests on *both*
synthetic and real measurements. The dual-test layout produces two
accuracy numbers per model, `accuracy_synthetic` and `accuracy_real`,
and their difference `domain_gap = accuracy_synthetic − accuracy_real`
is what the paper cares about.

A slope chart makes the pairing visually obvious: for each family, we
draw a line from `(x=synthetic, y=acc_synth)` to
`(x=real, y=acc_real)`. Steeper = worse domain gap. Readers can rank
the families by gap size in a single glance because slope steepness is
preattentive in a way that two separate bar charts are not.

Sort order: `t6.sort_values("domain_gap_mean", ascending=False)` — the
family with the *smallest* gap is drawn last and therefore ends up on
top of the stack. This is deliberate: the most robust family is what
the reader wants to identify first.

---

## 4 — Colorblind-safe encoding

Identical to the variant figures. See
[`variant_plotting.md` §4](variant_plotting.md) for the full
justification. Every tier figure uses all three channels
simultaneously, and each channel is fully independent:

1. **Color** — Okabe–Ito 8-color palette (CVD-safe).
2. **Marker shape** — 8 distinct shapes, one per family.
3. **Linestyle** — 8 distinct patterns (the four standard matplotlib
   strings plus four custom `(offset, on_off_seq)` tuples). Linestyle
   alone is enough to identify a family, which was not the case in the
   initial implementation.

The encoding tables live in `benchmark_zoo.py:97–136` and are imported
by both the variant and tier plotting functions. Do not define a new
`FAMILY_*` dict locally — the whole point of a single source of truth
is that reviewers see the same family the same way whether they're
reading a scaling figure or a robustness figure.

---

## 5 — Plot interpretation guide

Each figure answers a specific question. Read them in this order.

### 5.1 — `tier_heatmap.pdf` — *"Give me the leaderboard."*
- **Rows**: base models, one per family.
- **Columns**: tiers T1..T6.
- **Cells**: mean accuracy across seeds, colored on `YlGnBu`.
- **What to look for**: dark columns (broadly hard tiers) vs dark rows
  (fragile families). A cell that is noticeably darker than its
  neighbours flags either a pathological (model, tier) interaction or
  a crashed run.

### 5.2 — `tier_robustness.pdf` — *"Which family ages best?"*
- **X-axis**: tier index (T1..T6).
- **Y-axis**: Accuracy.
- **Lines + filled markers**: base variant per family.
- **Shaded band**: ±1 std across seeds at each tier.
- **What to look for**: a line that stays flat as the tier index
  increases has *graceful degradation*. A line that cliffs between T4
  and T5 is fragile to real noise; a line that cliffs between T5 and
  T6 is fragile to domain shift. The flattest curve wins.

### 5.3 — `tier_grid.pdf` — *"Does each family degrade sensibly?"*
- One panel per family, shared axes.
- **What to look for**: monotone-decreasing curves (well-behaved
  family), curves with a sudden drop at a single tier (fragility
  concentrated there), or non-monotone curves (likely a training
  pathology on a specific seed/tier and should be investigated).

### 5.4 — `tier6_domain_gap.pdf` — *"How big is the sim-to-real gap?"*
- **X-axis**: two ticks, `synthetic` → `real`.
- **Y-axis**: Accuracy.
- **Each line**: one family, connecting its synthetic T6 accuracy to
  its real T6 accuracy.
- **What to look for**: the *slope* of each line. A near-horizontal
  line means the family transfers cleanly from synthetic to real; a
  steep downward slope means the family has memorised synthetic-only
  artefacts. Pick the family with the shallowest slope *at an
  acceptable real-accuracy level* — not just the one with the highest
  synthetic accuracy.

---

## 6 — Why a slope chart for tier 6 (and not scatter + diagonal)

Three alternatives were tried before landing on the slope chart:

1. **Scatter with `acc_synth` on x and `acc_real` on y, plus the
   `y = x` diagonal.** Points below the diagonal are sim-to-real
   losers. Problem: the *size* of the gap is encoded as perpendicular
   distance to the diagonal, which is hard to eyeball and gets harder
   for points near the corners. Family identity also competes with the
   diagonal for attention.

2. **Grouped bar chart with two bars per family (`synth`, `real`).**
   Encodes the two accuracies separately but not their *pairing* — the
   reader has to mentally subtract. Adds visual weight without
   information.

3. **Slope chart (the chosen design).** Each family becomes a single
   line whose slope *is* the domain gap. Slope steepness is
   preattentive: readers rank families by fragility in one glance,
   with no arithmetic. The shared y-axis also makes it easy to compare
   absolute accuracies at either end. The three-channel encoding
   (color + marker + linestyle) disambiguates the families where lines
   cross.

The slope chart requires only two x-ticks and no legend column for
"which point belongs to which family" because the line explicitly
draws the connection, which is why it fits in a single column.

---

## 7 — The ±σ band collapse on single-seed runs

`generate_tier_robustness` and `generate_tier_grid` draw ±σ bands
across seeds using `ax.fill_between(x, y - yerr, y + yerr, ...)` where
`yerr = acc_std`. The `acc_std` column is computed in
`_load_all_tiers_aggregated` via
`acc_std=("accuracy", "std")` followed by `.fillna(0.0)`
(`benchmark_zoo.py:1102`).

Consequences:

- Runs trained with **≥ 2 seeds** show genuine confidence bands. The
  width tells you how reproducible the family's tier behaviour is
  (wide band = unstable family).
- Runs trained with **a single seed** have `acc_std = NaN → 0`, which
  means `fill_between` produces a zero-width band that collapses onto
  the line. **This is a feature**: a flat line is visually honest
  about the lack of spread information, and the curve itself is still
  readable.

If you see the tier-robustness lines drawn without any shading at all
on what is supposed to be a multi-seed run, check that the expected
seeds are present in `results/<run>/runs/*.json` — a crashed seed is
the most common cause.

The canonical multi-seed invocation is:

```bash
python benchmark_zoo.py --all --tier all --seeds 42,123,7 \
    --epochs 150 --output-dir results/benchmark2
```

---

## 8 — Adding a new tier figure

If you need to add another tier-axis figure to the benchmark, follow
this checklist so it slots cleanly into the existing system.

1. **Load data through `_load_all_tiers_aggregated`**, not with a
   fresh pass over `runs/*.json`. The helper already enforces the
   base-models-only filter and handles the tier-6-only columns
   (`acc_synth_mean`, `acc_real_mean`, `domain_gap_mean`) as NaNs for
   other tiers. Re-rolling that logic is how inconsistencies creep in.
2. **Call `apply_publication_style()`** at the top of the function.
3. **Use one of the `FIG_*` size constants** — never `figsize=(...)`
   inline. If none of the existing sizes fits, add a new constant
   alongside `FIG_SINGLE_TALL`.
4. **Color, marker, linestyle by family** must come from the three
   `FAMILY_*` dicts. Never hardcode an `ax.plot(color="red")`. This is
   what guarantees the same family looks the same across variant *and*
   tier figures.
5. **Label tiers `T1..T6`**, not by tier name. The integer index
   controls line slope; the `T*` label is only the visual decoration.
6. **Handle `len(tiers) < 2`** gracefully — return early with a
   message. A single-tier run (typical during smoke tests) must not
   crash the whole plot phase.
7. **Set `fig.subplots_adjust(left=…, right=…, top=…, bottom=…)`
   manually** — never `tight_layout`, which defeats
   `savefig.bbox="standard"` and drifts the canvas size.
8. **Save with `fig.savefig(figures_dir / "name.pdf")`** — no PNG.
9. **Wire the new function into both call sites** in `main()` —
   the `--aggregate-only` path near line 1521 and the post-training
   path near line 1567. The tier figures are *not* gated on
   `--scaling` because tier-axis data is collected by every run.
10. **Document the new figure here** by adding a row to §3 and an
    interpretation paragraph to §5.

---

## 9 — File and code references

- `benchmark_zoo.py:97-136` — `FAMILY_COLORS`, `FAMILY_MARKERS`,
  `FAMILY_LINESTYLES` (shared with variant figures; all three dicts
  have 8 unique entries).
- `benchmark_zoo.py:146-175` — figure size constants and `PUB_RC`.
- `benchmark_zoo.py:178-183` — `apply_publication_style()`.
- `benchmark_zoo.py:1054-1107` — `_load_all_tiers_aggregated`
  (base-models-only, tier-6 column handling).
- `benchmark_zoo.py:1108-1165` — `generate_tier_robustness`.
- `benchmark_zoo.py:1166-1222` — `generate_tier_grid`.
- `benchmark_zoo.py:1223-1279` — `generate_tier6_domain_gap`.
- `benchmark_zoo.py:1280-1330` — `generate_tier_heatmap`.
- `benchmark_zoo.py:1521-1524` — wiring in the `--aggregate-only` path.
- `benchmark_zoo.py:1567-1570` — wiring in the post-training path.

Regenerate every tier figure from cached JSON results without
retraining:

```bash
source venv/bin/activate
python benchmark_zoo.py --aggregate-only --output-dir results/benchmark2
```

Note: `--scaling` is **not** required. The four tier figures are
always regenerated during aggregation; `--scaling` only controls the
additional variant figures documented in
[`variant_plotting.md`](variant_plotting.md).
