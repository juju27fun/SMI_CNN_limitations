"""Publication plot: doublet accuracy vs input length, per (family, kernel).

Reads ``results/kernel_length_sweep_3fam/doublet_accuracy.json`` produced by
``infer_doublets.py --sweep`` and emits 1x3 PDF figures — one panel per
family, three curves per panel — using full B5 three-channel redundancy
(Blues sequential color + distinct markers + distinct linestyles), so each
curve remains identifiable in B&W or for colorblind readers.

Usage
-----
    python scripts/plot_acc_length.py
    python scripts/plot_acc_length.py --json custom.json --out my.pdf
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Allow running from either the project root or the scripts/ subdirectory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark_zoo import FAMILY_COLORS, FAMILY_MARKERS, _emit_pdf  # noqa: E402
from p0.plotting import apply_publication_style, COL_W, FIG_DOUBLE  # noqa: E402


# Same three families across all sweep variants.
FAMILIES = ["Conv1DGAP", "EfficientNet1D", "ResNet1D"]


def _auto_ylim(values, *, hard_floor=0.0, hard_ceil=1.0):
    """Tight, paper-friendly y-range around the data — rounded to 0.05.

    Avoids wasting whitespace when accuracies cluster in a narrow band
    (e.g. decimation sweep where all curves sit between 0.77 and 0.93).
    """
    vmin, vmax = min(values), max(values)
    span = max(vmax - vmin, 1e-6)
    pad = max(0.015, 0.03 * span)
    lo = max(hard_floor, np.floor((vmin - pad) * 20) / 20)
    hi = min(hard_ceil, np.ceil((vmax + pad) * 20) / 20)
    tick_span = hi - lo
    step = 0.05 if tick_span <= 0.30 else 0.10
    yticks = [round(lo + i * step, 2)
              for i in range(int(round(tick_span / step)) + 1)
              if lo + i * step <= hi + 1e-9]
    return (lo, hi), yticks


def _load_entries(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def _ordinal_style(values):
    """Map sorted ordinal values -> (color, marker, linestyle) for B5 redundancy.

    Three channels grow together (lightest+dotted+circle for smallest value,
    darkest+solid+diamond for largest), so each curve remains identifiable
    when printed in B&W or seen with red-green colorblindness.
    """
    sorted_vals = sorted(set(values))
    n = len(sorted_vals)
    cmap = plt.get_cmap("Blues")
    if n == 3:
        positions = [0.45, 0.70, 0.92]
        markers = ["o", "s", "D"]
        linestyles = [":", "--", "-"]
    elif n == 2:
        positions = [0.55, 0.92]
        markers = ["o", "D"]
        linestyles = ["--", "-"]
    elif n == 1:
        positions = [0.75]
        markers = ["o"]
        linestyles = ["-"]
    else:
        positions = list(np.linspace(0.42, 0.95, n))
        markers = (["o", "s", "D", "^", "v", "P", "X"] * 2)[:n]
        linestyles = ([":", "--", "-", "-.",
                       (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))] * 2)[:n]
    return {v: (cmap(p), m, ls)
            for v, p, m, ls in zip(sorted_vals, positions, markers, linestyles)}


def _add_family_header(ax, family, color):
    """Family identifier above the panel: bold colored text, white-stroked."""
    ax.text(
        0.5, 1.03, family, transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9, fontweight="bold", color=color,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )


def _style_acc_panel(ax, x_ticks, x_label, *, ylim, yticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_xlabel(x_label)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)


def _make_figure(entries: list[dict]) -> plt.Figure:
    """acc(L) per family — one curve per kernel (Blues + markers + linestyles)."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)

    grouped = defaultdict(list)
    for e in entries:
        grouped[(e["family"], e["kernel_size"])].append(
            (e["input_length"], e["accuracy"])
        )

    kernels = sorted({e["kernel_size"] for e in entries})
    lengths = sorted({e["input_length"] for e in entries})
    style = _ordinal_style(kernels)
    ylim, yticks = _auto_ylim([e["accuracy"] for e in entries])

    for ax, family in zip(axes, FAMILIES):
        family_color = FAMILY_COLORS.get(family, "black")
        for k in kernels:
            pts = sorted(grouped.get((family, k), []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color, marker, ls = style[k]
            ax.plot(
                xs, ys,
                linestyle=ls, linewidth=1.4,
                marker=marker, markersize=5.5,
                color=color,
                markeredgecolor="white", markeredgewidth=0.4,
                label=f"k = {k}" if ax is axes[0] else None,
            )
        _style_acc_panel(ax, lengths, "Input length (samples)",
                         ylim=ylim, yticks=yticks)
        _add_family_header(ax, family, family_color)

    axes[0].set_ylabel("Doublet test accuracy")
    fig.legend(
        loc="lower center", ncol=len(kernels),
        title="Kernel size", title_fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=2.6, columnspacing=2.0,
    )
    fig.subplots_adjust(
        bottom=0.30, top=0.92, left=0.08, right=0.985, wspace=0.14,
    )
    return fig


def _make_figure_acc_vs_kernel(entries: list[dict]) -> plt.Figure:
    """acc(k) per family — one curve per input length (Blues + markers + linestyles).

    Isolates kernel impact by plotting it on the X-axis; lengths become the
    controlled variable, encoded with three redundant channels (B5).
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)

    grouped = defaultdict(list)
    for e in entries:
        grouped[(e["family"], e["input_length"])].append(
            (e["kernel_size"], e["accuracy"])
        )

    lengths = sorted({e["input_length"] for e in entries})
    kernels = sorted({e["kernel_size"] for e in entries})
    style = _ordinal_style(lengths)
    ylim, yticks = _auto_ylim([e["accuracy"] for e in entries])

    for ax, family in zip(axes, FAMILIES):
        family_color = FAMILY_COLORS.get(family, "black")
        for L in lengths:
            pts = sorted(grouped.get((family, L), []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color, marker, ls = style[L]
            ax.plot(
                xs, ys,
                linestyle=ls, linewidth=1.4,
                marker=marker, markersize=5.5,
                color=color,
                markeredgecolor="white", markeredgewidth=0.4,
                label=f"L = {L}" if ax is axes[0] else None,
            )
        _style_acc_panel(ax, kernels, "Kernel size",
                         ylim=ylim, yticks=yticks)
        _add_family_header(ax, family, family_color)

    axes[0].set_ylabel("Doublet test accuracy")
    fig.legend(
        loc="lower center", ncol=len(lengths),
        title="Input length (samples)", title_fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=2.6, columnspacing=2.0,
    )
    fig.subplots_adjust(
        bottom=0.30, top=0.92, left=0.08, right=0.985, wspace=0.14,
    )
    return fig


def _load_latency_rows(summary_csv: Path) -> list[dict]:
    """Parse summary.csv and return one dict per (Model, Kernel_Size, Input_Length)."""
    rows = []
    with open(summary_csv) as f:
        for r in csv.DictReader(f):
            if not r.get("Kernel_Size") or not r.get("Input_Length"):
                continue
            rows.append({
                "model_name": r["Model"],
                "family": r["Model_Family"],
                "kernel_size": int(r["Kernel_Size"]),
                "input_length": int(r["Input_Length"]),
                "latency_ms": float(r["Latency_Median_ms"]),
            })
    return rows


def _make_figure_latency_vs_kernel(latency_rows: list[dict]) -> plt.Figure:
    """Median inference latency vs kernel size, per (family, input length).

    Complements the accuracy plots: same layout, same Blues-on-length scheme
    plus markers/linestyles (B5), log-scale Y to cover the >1-OOM spread.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)

    grouped = defaultdict(list)
    for r in latency_rows:
        grouped[(r["family"], r["input_length"])].append(
            (r["kernel_size"], r["latency_ms"])
        )

    lengths = sorted({r["input_length"] for r in latency_rows})
    kernels = sorted({r["kernel_size"] for r in latency_rows})
    style = _ordinal_style(lengths)

    for ax, family in zip(axes, FAMILIES):
        family_color = FAMILY_COLORS.get(family, "black")
        for L in lengths:
            pts = sorted(grouped.get((family, L), []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color, marker, ls = style[L]
            ax.plot(
                xs, ys,
                linestyle=ls, linewidth=1.4,
                marker=marker, markersize=5.5,
                color=color,
                markeredgecolor="white", markeredgewidth=0.4,
                label=f"L = {L}" if ax is axes[0] else None,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(kernels)
        ax.set_xticklabels([str(k) for k in kernels])
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 3, 5)))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.set_xlabel("Kernel size")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.4, which="both")
        ax.set_axisbelow(True)
        _add_family_header(ax, family, family_color)

    axes[0].set_ylabel("Median latency (ms)")
    fig.legend(
        loc="lower center", ncol=len(lengths),
        title="Input length (samples)", title_fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
        handlelength=2.6, columnspacing=2.0,
    )
    fig.subplots_adjust(
        bottom=0.30, top=0.92, left=0.09, right=0.985, wspace=0.14,
    )
    return fig


def _make_figure_delta_vs_length(entries: list[dict]) -> plt.Figure:
    """Single-panel narrative plot: Δacc = acc(k=49) − acc(k=3) vs length.

    Shows how much a larger kernel compensates for a shorter window, isolated
    into a one-number-per-(family, length) summary.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))

    # Index acc by (family, kernel, length) for quick lookup
    acc = {(e["family"], e["kernel_size"], e["input_length"]): e["accuracy"]
           for e in entries}
    lengths = sorted({e["input_length"] for e in entries})
    k_lo, k_hi = 3, 49

    for family in FAMILIES:
        deltas = []
        for L in lengths:
            a_lo = acc.get((family, k_lo, L))
            a_hi = acc.get((family, k_hi, L))
            deltas.append((a_hi - a_lo) if (a_lo is not None and a_hi is not None) else None)
        xs = [L for L, d in zip(lengths, deltas) if d is not None]
        ys = [d for d in deltas if d is not None]
        ax.plot(
            xs, ys,
            linestyle="-", linewidth=1.2,
            marker=FAMILY_MARKERS.get(family, "o"),
            markersize=5,
            color=FAMILY_COLORS.get(family, "black"),
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=family,
        )

    ax.axhline(0, color="0.5", linewidth=0.6, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Input length (samples)")
    ax.set_ylabel(r"$\Delta$ accuracy (k=49 $-$ k=3)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    ax.legend(
        loc="best", frameon=True,
        facecolor="white", edgecolor="0.7", framealpha=0.85,
    )
    fig.subplots_adjust(bottom=0.18, top=0.96, left=0.20, right=0.97)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=str,
        default="results/kernel_length_sweep_3fam/doublet_accuracy.json",
        help="Sweep doublet-accuracy JSON (from infer_doublets.py --sweep)",
    )
    parser.add_argument(
        "--summary-csv", type=str, default=None,
        help="Sweep summary.csv for latency plot (default: sibling of --json)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="results/kernel_length_sweep_3fam",
        help="Destination directory for the PDFs",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise SystemExit(f"JSON not found: {json_path}. Run the sweep first.")

    entries = _load_entries(json_path)
    if not entries:
        raise SystemExit(f"No entries in {json_path}")
    print(f"Loaded {len(entries)} entries from {json_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_length = _make_figure(entries)
    _emit_pdf(
        fig_length, out_dir, "acc_length_kernel_3fam.pdf",
        wandb_run=None,
        caption="Doublet test accuracy vs input length across kernel sizes, per model family (seed 42)",
    )

    fig_kernel = _make_figure_acc_vs_kernel(entries)
    _emit_pdf(
        fig_kernel, out_dir, "acc_kernel_length_3fam.pdf",
        wandb_run=None,
        caption="Doublet test accuracy vs kernel size across input lengths, per model family (seed 42)",
    )

    fig_delta = _make_figure_delta_vs_length(entries)
    _emit_pdf(
        fig_delta, out_dir, "delta_acc_length_3fam.pdf",
        wandb_run=None,
        caption="Kernel-induced accuracy gain (k=49 minus k=3) vs input length, per model family (seed 42)",
    )

    # Latency plot — sourced from summary.csv (latency is not in the JSON)
    summary_csv = Path(args.summary_csv) if args.summary_csv else json_path.parent / "summary.csv"
    if summary_csv.exists():
        latency_rows = _load_latency_rows(summary_csv)
        print(f"Loaded {len(latency_rows)} latency rows from {summary_csv}")
        fig_lat = _make_figure_latency_vs_kernel(latency_rows)
        _emit_pdf(
            fig_lat, out_dir, "latency_kernel_3fam.pdf",
            wandb_run=None,
            caption="Median inference latency vs kernel size across input lengths, per model family (seed 42)",
        )
    else:
        print(f"summary.csv not found at {summary_csv} — skipping latency plot")


if __name__ == "__main__":
    main()
