---
name: publication-figures
description: Use for P0 paper-quality plotting, matplotlib figure standards, W&B-to-PDF retracing, confusion matrices, heatmaps, Pareto plots, ROC/PR curves, and final figure review.
---

# Publication Figures

Use this skill when creating or reviewing figures intended for papers, reports, or thesis-quality exports.

## Output

- Vector PDF only for paper figures.
- Set `pdf.fonttype = 42` and `ps.fonttype = 42`.
- Use `savefig.dpi = 300` for internal rasterization.
- Use fixed canvas sizes and explicit `fig.subplots_adjust(...)`.
- Do not use `tight_layout()` or `constrained_layout()`.
- Do not use `ax.set_title()`; titles belong in captions.

## Typography

- Use serif publication style.
- Keep text at least 7 pt at final print size.
- Axis labels must include units where applicable.
- Prefer fewer ticks or rotated labels before shrinking fonts.

## Encodings

- Use project constants for family colors, markers, and linestyles.
- Categorical line/scatter plots should use color, marker, and linestyle together.
- Do not use rainbow colormaps such as `jet` or `hsv`.
- Use `Blues` or another perceptually reasonable sequential map for matrices.

## Matrices

- Confusion matrices should be square when practical.
- Annotate every cell.
- Use luminance-aware text color for readability.
- Zero cells should remain visible but visually de-emphasized.
- Include a horizontal colorbar with a label.

## Curves And Scatter

- Training curves should show smoothed data plus a faint raw overlay for publication.
- Multi-seed curves should show mean with a confidence band and state std/SE in the legend.
- Pareto plots should mark Pareto-front members clearly and keep legends out of the data area.
- ROC/PR curves should show AUC in the legend and include appropriate baselines.

## Checklist

- PDF vector output only.
- Fixed canvas size.
- No `tight_layout`, `constrained_layout`, or plot titles.
- Palette and marker conventions respected.
- Text legible at print size.
- Units present on axes.
- Legend outside data or semi-transparent if inside axes.
- Generated from raw data exports, not W&B raster exports.
