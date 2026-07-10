"""Doublet real-time factor (ρ) figure — accuracy vs ρ scatter.

Regenerates
``artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained/realtime_factor_doublet_3fam.pdf``
from the retrained CNN-family data and the retrained transformer data. Family
symbols follow benchmark2 color/marker/linestyle conventions.

Usage
-----
    python scripts/plotting/plot_realtime_factor_doublet.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from p0.benchmark_style import FAMILY_COLORS, FAMILY_MARKERS, FAMILY_LINESTYLES  # noqa: E402
from p0.plotting import apply_publication_style, DCOL_W  # noqa: E402


FAMILIES = ["Conv1DGAP", "EfficientNet1D", "ResNet1D", "PatchTST", "Swin1D"]
DISPLAY = {"PatchTST": "PatchTST", "Swin1D": "Swin-1D"}
SIZE_ORDER = {"Pico": 0, "Nano": 1, "XXS": 2, "XS": 3, "S": 4, "M": 5, "L": 6}
CNN_DATA_JSON = (
    _PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained/_rt_data.json"
)
TRANSFORMER_DATA_JSON = (
    _PROJECT_ROOT.parent
    / "artifacts/SMI_CNN_limitations/benchmarks/doublet_transformers_retrained_lr1e4/_rt_data.json"
)
OUT_DIR = _PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained"
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


def _display_family(family):
    return DISPLAY.get(family, family)


def _stage(entry):
    return entry.get("stage", "Retrained")


def _size_tag(entry):
    if "size_tag" in entry:
        return entry["size_tag"]
    name = entry["model_name"]
    for tag in ("Pico", "Nano", "XXS", "XS", "S", "L"):
        if name.endswith(f"-{tag}"):
            return tag
    return "M"


def _save_pdf(fig, out_dir, out_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _load_entries():
    for path in (CNN_DATA_JSON, TRANSFORMER_DATA_JSON):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    with open(CNN_DATA_JSON) as f:
        cnn_entries = json.load(f)
    with open(TRANSFORMER_DATA_JSON) as f:
        transformer_entries = json.load(f)

    entries = []
    for entry in cnn_entries:
        if entry["family"] not in {"Conv1DGAP", "EfficientNet1D", "ResNet1D"}:
            continue
        merged = dict(entry)
        merged.setdefault("stage", "Retrained")
        entries.append(merged)

    for entry in transformer_entries:
        if entry["family"] in {"PatchTST", "Swin1D"} and _stage(entry) == "Retrained":
            entries.append(dict(entry))

    return entries


def _make_figure(entries):
    apply_publication_style()
    rho = np.array([e["rt_factor"] for e in entries])
    acc = np.array([e["accuracy"] * 100 for e in entries])
    fams = [e["family"] for e in entries]
    names = [e["model_name"] for e in entries]

    fig, ax = plt.subplots(figsize=(DCOL_W, 3.35))

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

    # ── Per-family scatter using benchmark2 family symbols.
    for fam in FAMILIES:
        idx = [i for i, f in enumerate(fams) if f == fam]
        if not idx:
            continue
        color = FAMILY_COLORS.get(fam, "#333")
        marker = FAMILY_MARKERS.get(fam, "o")

        for i in idx:
            ax.scatter(
                rho[i], acc[i],
                s=46,
                facecolor=color,
                edgecolor="white",
                marker=marker,
                linewidth=0.4,
                label=fam,
                zorder=3,
            )

        if fam in {"PatchTST", "Swin1D"}:
            ordered_idx = sorted(
                idx,
                key=lambda i: (
                    SIZE_ORDER.get(_size_tag(entries[i]), 99),
                    entries[i].get("params") or 0,
                ),
            )
            ax.plot(
                rho[ordered_idx], acc[ordered_idx],
                color=color,
                linestyle=FAMILY_LINESTYLES.get(fam, "-"),
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )
            for j, i in enumerate(ordered_idx):
                label = _size_tag(entries[i])
                dx = 6
                dy = 5 if j % 2 == 0 else -9
                ax.annotate(
                    label, xy=(rho[i], acc[i]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.1, color="0.18",
                    ha="left", va="bottom" if dy > 0 else "top",
                    linespacing=0.95, zorder=6,
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
    ax.set_ylim(40, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.30, linewidth=0.4)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)
    ax.set_axisbelow(True)

    # Family legend in lower-left, inside the saturated (ρ < 1) zone where
    # only Pareto-dominated points sit — keeps the high-acc band clear for
    # the front + side-key, and the lower-right empty for Conv1DGAP-Nano/Pico.
    plotted_families = [fam for fam in FAMILIES if fam in set(fams)]
    family_handles = [
        Line2D(
            [0], [0],
            color=FAMILY_COLORS.get(fam, "#333"),
            marker=FAMILY_MARKERS.get(fam, "o"),
            linestyle=FAMILY_LINESTYLES.get(fam, "-"),
            linewidth=1.0,
            markersize=4.5,
            markeredgecolor=FAMILY_COLORS.get(fam, "#333"),
            markerfacecolor=FAMILY_COLORS.get(fam, "#333"),
            label=_display_family(fam),
        )
        for fam in plotted_families
    ]
    family_legend = ax.legend(
        handles=family_handles,
        loc="lower right", bbox_to_anchor=(0.98, 0.03),
        frameon=True, framealpha=0.85, fontsize=7,
        handletextpad=0.4, borderaxespad=0.4, ncol=1,
    )

    ax.text(
        0.99, 0.01, "CPU torch, batch = 1",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6, color="0.50", style="italic",
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13)
    return fig


def main():
    try:
        entries = _load_entries()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    fig = _make_figure(entries)
    out_path = _save_pdf(fig, OUT_DIR, OUT_NAME)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
