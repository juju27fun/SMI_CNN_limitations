"""Doublet real-time factor (ρ) figure — accuracy vs ρ scatter.

Reproduces ``results/doublet_3fam_retrained/realtime_factor_doublet_3fam.pdf``
from ``_rt_data.json``, aligning the typography and the ρ = 1 demarcation
with P1's ``detseg/plotting.py:generate_real_time_factor`` so the two
figures sit side-by-side cleanly in a conjoined paper:

- Linear x-axis (was log), so the saturation threshold at ρ = 1 reads
  as a real visual cut rather than collapsing onto log(1) = 0.
- Saturated zone (ρ < 1) shaded in light grey + dashed threshold line
  + "ρ = 1" annotation in the top margin (matches P1).
- ρ = N_point / (τ · f_s) notation (was N_raw / (f_acq · t_median)) so
  the symbols are identical across both papers.
- pub_utils.PUB_RC font/size already match P1's PUB_RC; nothing more
  to do for typography beyond ``apply_publication_style()``.

The chart-type cannot match P1 exactly (P1's bar form would be
unreadable at 19 points) so this stays a scatter, but every other
visual element follows the P1 RTF convention.

Usage
-----
    python scripts/plot_realtime_factor_doublet.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark_zoo import FAMILY_COLORS, FAMILY_MARKERS, _emit_pdf  # noqa: E402
from p0.plotting import apply_publication_style, DCOL_W  # noqa: E402


FAMILIES = ["Conv1DGAP", "EfficientNet1D", "ResNet1D"]
DATA_JSON = Path("results/doublet_3fam_retrained/_rt_data.json")
OUT_DIR = Path("results/doublet_3fam_retrained")
OUT_NAME = "realtime_factor_doublet_3fam.pdf"


def _pareto_front(rho, acc):
    """Indices of points on the (max ρ, max accuracy) Pareto front."""
    order = sorted(range(len(rho)), key=lambda i: rho[i])
    front = []
    best_acc = -np.inf
    # Walk from low ρ to high ρ; keep points whose accuracy beats every
    # higher-ρ point seen later. So iterate from highest ρ downward and
    # track running max of accuracy.
    for i in reversed(order):
        if acc[i] > best_acc:
            front.append(i)
            best_acc = acc[i]
    return sorted(front, key=lambda i: rho[i])


def _make_figure(entries):
    apply_publication_style()
    rho = np.array([e["rt_factor"] for e in entries])
    acc = np.array([e["accuracy"] * 100 for e in entries])
    fams = [e["family"] for e in entries]
    names = [e["model_name"] for e in entries]

    fig, ax = plt.subplots(figsize=(DCOL_W, 3.2))

    # ── Saturated zone (ρ < 1) + ρ = 1 threshold. Log scale spreads the
    # 3-decade dynamic range (0.1 → 105) that the CPU latencies introduce;
    # without it 80 % of points collapse into the leftmost 5 % of the canvas.
    rho_lo = max(min(rho.min() * 0.7, 0.05), 1e-3)
    ax.axvspan(rho_lo, 1.0, alpha=0.10, color="0.45", zorder=1)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=4)
    ax.text(
        1.0, 1.02, r"$\rho = 1$",
        transform=ax.get_xaxis_transform(),
        fontsize=7, color="0.25", va="bottom", ha="center", style="italic",
    )

    # ── Per-family scatter
    for fam in FAMILIES:
        mask = [f == fam for f in fams]
        if not any(mask):
            continue
        ax.scatter(
            rho[mask], acc[mask],
            s=42, color=FAMILY_COLORS.get(fam, "#333"),
            marker=FAMILY_MARKERS.get(fam, "o"),
            edgecolor="white", linewidth=0.4, label=fam, zorder=3,
        )

    # ── Pareto front + badges + side-key
    front_idx = _pareto_front(rho, acc)
    if front_idx:
        ax.plot(rho[front_idx], acc[front_idx],
                color="black", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4)
        ax.scatter(rho[front_idx], acc[front_idx],
                   facecolor="none", edgecolor="black", linewidth=0.8,
                   s=46, zorder=5)

        # Inline labels on each Pareto point (model name + accuracy).
        # Log x-scale spreads the 6 Pareto points across ~2.5 decades
        # so direct inline labels stay readable; a side-key table would
        # collide with the high-ρ Conv1DGAP-{Nano,Pico} cluster on the
        # right or with the front itself if placed elsewhere.
        # Side selection: points in the right third of the canvas get
        # left-anchored labels so they don't run off the figure edge.
        label_bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.80)
        x_split = rho.max() / 5.0  # ρ above which labels go to the left of the point
        for k, i in enumerate(front_idx):
            on_right_side = rho[i] > x_split
            dx = -8 if on_right_side else 8
            # Alternate above/below to avoid label-on-line collisions
            # along the front.
            dy = 10 if (k % 2 == 0) else -12
            label = f"{names[i]}\n({acc[i]:.1f}%)"
            ax.annotate(
                label, xy=(rho[i], acc[i]),
                xytext=(dx, dy), textcoords="offset points",
                ha="right" if on_right_side else "left",
                va="bottom" if dy > 0 else "top",
                fontsize=6.5, color="0.15", linespacing=1.0,
                bbox=label_bbox, zorder=6,
            )

    # ── Axes formatting (log x, ρ spans ~3 orders of magnitude on CPU)
    ax.set_xscale("log")
    # Extra right padding to host the inline label of the highest-ρ point.
    ax.set_xlim(rho_lo, rho.max() * 1.8)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(
        r"Real-time factor  $\rho = N_\mathrm{point} / (\tau \cdot f_s)$  [log scale]"
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.30, linewidth=0.4)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)
    ax.set_axisbelow(True)

    # Family legend in lower-left, inside the saturated (ρ < 1) zone where
    # only Pareto-dominated points sit — keeps the high-acc band clear for
    # the front + side-key, and the lower-right empty for Conv1DGAP-Nano/Pico.
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.02, 0.03),
        frameon=True, framealpha=0.85, fontsize=7,
        handletextpad=0.4, borderaxespad=0.4,
    )

    ax.text(
        0.99, 0.01, "CPU torch, batch = 1",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6, color="0.50", style="italic",
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13)
    return fig


def main():
    if not DATA_JSON.exists():
        print(f"Missing {DATA_JSON}", file=sys.stderr)
        return 1
    entries = [e for e in json.load(open(DATA_JSON)) if e["family"] in FAMILIES]
    fig = _make_figure(entries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _emit_pdf(
        fig, OUT_DIR, OUT_NAME, wandb_run=None,
        caption=(
            "Doublet test accuracy vs real-time factor "
            r"$\rho = N_\mathrm{point} / (\tau \cdot f_s)$ for three "
            "model families. Dashed line at ρ = 1 marks the saturation "
            "threshold; shaded zone marks ρ < 1 (saturated)."
        ),
    )
    print(f"Wrote {OUT_DIR / OUT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
