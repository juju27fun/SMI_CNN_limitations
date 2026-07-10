"""Two-panel horizontal bar chart: before/after doublet accuracy + latency.

Regenerates
``artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained/doublet_comparison_acc_latency.pdf``
with the original three CNN families plus the retrained transformer experiment.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from p0.benchmark_style import FAMILY_COLORS  # noqa: E402
from p0.plotting import apply_publication_style, DCOL_W  # noqa: E402


FAMILIES = ["Swin1D", "PatchTST", "EfficientNet1D", "ResNet1D", "Conv1DGAP"]
DISPLAY = {"PatchTST": "PatchTST", "Swin1D": "Swin-1D"}
ACC_BEFORE = {"Conv1DGAP": 45.1, "ResNet1D": 48.9, "EfficientNet1D": 49.2}
ACC_AFTER  = {"Conv1DGAP": 92.8, "ResNet1D": 91.0, "EfficientNet1D": 90.7}

OUT_DIR = _PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained"
_LAT_CMP_JSON = OUT_DIR / "_lat_comparison.json"
_TRANSFORMER_DIR = (
    _PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/doublet_transformers_retrained_lr1e4"
)
_TRANSFORMER_LAT_CMP_JSON = _TRANSFORMER_DIR / "_lat_comparison.json"
_TRANSFORMER_RT_JSON = _TRANSFORMER_DIR / "_rt_data.json"

BEFORE_COLOR = "0.75"


def _display_family(family):
    return DISPLAY.get(family, family)


def _load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path) as f:
        return json.load(f)


def _load_comparison_data():
    cnn_lat = _load_json(_LAT_CMP_JSON)
    transformer_lat = _load_json(_TRANSFORMER_LAT_CMP_JSON)
    transformer_rt = _load_json(_TRANSFORMER_RT_JSON)

    acc_before = dict(ACC_BEFORE)
    acc_after = dict(ACC_AFTER)
    for entry in transformer_rt:
        family = entry["family"]
        stage = entry.get("stage", "")
        acc_pct = entry["accuracy"] * 100.0
        if stage == "Short-trained":
            acc_before[family] = acc_pct
        elif stage == "Retrained" and entry.get("size_tag", "M") == "M":
            acc_after[family] = acc_pct

    lat_before = {**cnn_lat["LAT_BEFORE"], **transformer_lat["LAT_BEFORE"]}
    lat_after = {**cnn_lat["LAT_AFTER"], **transformer_lat["LAT_AFTER"]}

    missing = [
        family for family in FAMILIES
        if family not in acc_before
        or family not in acc_after
        or family not in lat_before
        or family not in lat_after
    ]
    if missing:
        raise KeyError(f"Missing comparison data for: {', '.join(missing)}")

    return acc_before, acc_after, lat_before, lat_after


def _save_pdf(fig, out_dir, out_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _make_figure(acc_before: dict, acc_after: dict,
                 lat_before: dict, lat_after: dict,
                 families: list[str] | None = None) -> plt.Figure:
    apply_publication_style()
    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(DCOL_W, 3.05))

    families = families or FAMILIES
    y = np.arange(len(families))
    bar_h = 0.38

    # ── Panel (a): accuracy ────────────────────────────────────────────────
    for i, fam in enumerate(families):
        col = FAMILY_COLORS.get(fam, "#333333")
        ax_acc.barh(y[i] + bar_h / 2, acc_after[fam],  height=bar_h,
                    color=col, edgecolor="none", zorder=3)
        ax_acc.barh(y[i] - bar_h / 2, acc_before[fam], height=bar_h,
                    color=BEFORE_COLOR, edgecolor="none", zorder=3)

        delta = acc_after[fam] - acc_before[fam]
        ax_acc.text(acc_after[fam] + 1.2, y[i] + bar_h / 2,
                    f"{acc_after[fam]:.1f}  (+{delta:.1f})",
                    va="center", ha="left", fontsize=7)
        ax_acc.text(acc_before[fam] - 1.2, y[i] - bar_h / 2,
                    f"{acc_before[fam]:.1f}",
                    va="center", ha="right", fontsize=7, color="0.35")

    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(
        [_display_family(fam) for fam in families],
        fontsize=8, fontweight="bold",
    )
    for tick_lbl, fam in zip(ax_acc.get_yticklabels(), families):
        tick_lbl.set_color(FAMILY_COLORS.get(fam, "black"))

    ax_acc.set_xlim(35, 108)
    ax_acc.set_xlabel("Accuracy (%)")
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    ax_acc.grid(True, axis="x", alpha=0.3, linewidth=0.4)
    ax_acc.set_axisbelow(True)

    # ── Panel (b): latency ─────────────────────────────────────────────────
    for i, fam in enumerate(families):
        col = FAMILY_COLORS.get(fam, "#333333")
        ax_lat.barh(y[i] + bar_h / 2, lat_after[fam],  height=bar_h,
                    color=col, edgecolor="none", zorder=3)
        ax_lat.barh(y[i] - bar_h / 2, lat_before[fam], height=bar_h,
                    color=BEFORE_COLOR, edgecolor="none", zorder=3)

        slowdown = lat_after[fam] / lat_before[fam] if lat_before[fam] else 1.0
        if lat_after[fam] == max(lat_after.values()):
            after_x = lat_after[fam] / 1.10
            after_ha = "right"
        else:
            after_x = lat_after[fam] * 1.08
            after_ha = "left"
        ax_lat.text(after_x, y[i] + bar_h / 2,
                    f"{lat_after[fam]:.2f}  ({slowdown:.1f}x)",
                    va="center", ha=after_ha, fontsize=7)
        ax_lat.text(lat_before[fam] / 1.08, y[i] - bar_h / 2,
                    f"{lat_before[fam]:.2f}",
                    va="center", ha="right", fontsize=7, color="0.35")

    ax_lat.set_yticks(y)
    ax_lat.set_yticklabels([])
    ax_lat.set_xscale("log")
    # Bounds derived from data: ~half a decade of headroom on each side so
    # the inline value labels (placed at lat * 1.08 / lat / 1.08) sit fully
    # inside the canvas.
    lat_lo = min(min(lat_before.values()), min(lat_after.values())) / 2.5
    lat_hi = max(max(lat_before.values()), max(lat_after.values())) * 2.5
    ax_lat.set_xlim(lat_lo, lat_hi)
    ax_lat.set_xlabel("Latency (ms) — CPU torch, batch = 1")
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)
    ax_lat.grid(True, axis="x", which="both", alpha=0.3, linewidth=0.4)
    ax_lat.set_axisbelow(True)

    ax_lat.text(
        0.99, 0.01, "CPU torch, batch = 1",
        transform=ax_lat.transAxes, ha="right", va="bottom",
        fontsize=6, color="0.50", style="italic",
    )

    # ── (a) / (b) panel labels OUTSIDE data area (above each axes) ─────────
    #   Placed in figure coordinates so they don't overlap the top bar.
    ax_acc.text(0.0, 1.04, "(a)", transform=ax_acc.transAxes,
                ha="left", va="bottom", fontsize=9, fontweight="bold")
    ax_lat.text(0.0, 1.04, "(b)", transform=ax_lat.transAxes,
                ha="left", va="bottom", fontsize=9, fontweight="bold")

    # ── Shared legend at the very bottom ───────────────────────────────────
    before_patch = mpatches.Patch(
        facecolor=BEFORE_COLOR, edgecolor="none",
        label="Before (short-trained / zoo, 625 samples)",
    )
    after_patch = mpatches.Patch(
        facecolor="0.35", edgecolor="none",
        label="After (retrained, 4096 samples)",
    )
    fig.legend(
        handles=[before_patch, after_patch],
        loc="lower center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, 0.0),
        fontsize=7, handlelength=1.4, columnspacing=2.0,
    )

    # Generous bottom margin so xlabel <-> legend don't touch; extra top
    # margin to host the (a)/(b) labels outside the axes.
    fig.subplots_adjust(
        left=0.17, right=0.97, top=0.90, bottom=0.24, wspace=0.08,
    )
    return fig


def main():
    acc_before, acc_after, lat_before, lat_after = _load_comparison_data()
    fig = _make_figure(acc_before, acc_after, lat_before, lat_after)
    out_path = _save_pdf(fig, OUT_DIR, "doublet_comparison_acc_latency.pdf")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
