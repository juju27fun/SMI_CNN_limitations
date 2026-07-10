"""Publication-style doublet plots for Swin1D and PatchTST.

Outputs in ``artifacts/SMI_CNN_limitations/doublet_transformers_retrained_lr1e4``:
  - ``doublet_comparison_acc_latency.pdf``
  - ``realtime_factor_doublet_3fam.pdf``
  - ``_lat_comparison.json``
  - ``_rt_data.json``

The filenames intentionally mirror the existing 3-family figures while the
directory keeps the transformer results separate because Swin1D CPU latency is
far outside the CNN-scale plot range.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from p0.plotting import DCOL_W, apply_publication_style  # noqa: E402


OUT_DIR = _PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/doublet_transformers_retrained_lr1e4"
LAT_REMEASURE = OUT_DIR / "_latency_remeasure.json"

MODELS = ["PatchTST", "Swin1D"]
DISPLAY = {"PatchTST": "PatchTST", "Swin1D": "Swin-1D"}
SIGNAL_DURATION_MS = 8.192
SIZE_ORDER = {"Nano": 0, "XXS": 1, "XS": 2, "S": 3, "M": 4, "L": 5}

FAMILY_COLORS = {
    "PatchTST": "#332288",
    "Swin1D": "#CC79A7",
}


def _emit_pdf(fig, figures_dir: Path, fname: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")

# Long-sequence accuracy for checkpoints trained on short sequences.
ACC_BEFORE = {
    "PatchTST": 0.8307,
    "Swin1D": 0.4836,
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _rt_entry_from_run(run: dict) -> dict:
    latency_ms = run["latency_median_ms"]
    return {
        "model_name": run["model_name"],
        "family": run["model_family"],
        "stage": "Retrained",
        "size_tag": run.get("model_size_tag", "M"),
        "params": run.get("params"),
        "macs": run.get("macs"),
        "latency_ms": latency_ms,
        "latency_p95_ms": run.get("latency_p95_ms"),
        "accuracy": run["accuracy"],
        "rt_factor": SIGNAL_DURATION_MS / latency_ms,
        "latency_device": run.get("latency_device", "cpu"),
    }


def _load_retrained_rt_entries() -> list[dict]:
    entries = []
    for path in sorted((OUT_DIR / "runs").glob("*.json")):
        run = _load_json(path)
        if run.get("model_family") not in set(MODELS):
            continue
        if run.get("tier") != 1 or run.get("seed") != 42:
            continue
        if run.get("input_length") != 4096:
            continue
        entries.append(_rt_entry_from_run(run))

    return sorted(
        entries,
        key=lambda e: (
            MODELS.index(e["family"]),
            SIZE_ORDER.get(e.get("size_tag", "M"), 99),
            e["model_name"],
        ),
    )


def _load_inputs() -> tuple[dict, dict, dict, list[dict]]:
    lat = _load_json(LAT_REMEASURE)
    runs = {
        model: _load_json(
            OUT_DIR / "runs" / f"{model}-L4096-dataset_doublet-tier1-seed42.json"
        )
        for model in MODELS
    }

    lat_before = {
        model: lat[f"{model}:short_625"]["median_ms"]
        for model in MODELS
    }
    lat_after = {
        model: lat[f"{model}:retrained_4096"]["median_ms"]
        for model in MODELS
    }
    acc_after = {
        model: runs[model]["accuracy"]
        for model in MODELS
    }

    rt_entries = []
    for model in MODELS:
        short_long_lat = lat[f"{model}:short_4096"]["median_ms"]
        rt_entries.extend([
            {
                "model_name": f"{model} short-trained",
                "family": model,
                "stage": "Short-trained",
                "latency_ms": short_long_lat,
                "accuracy": ACC_BEFORE[model],
                "rt_factor": SIGNAL_DURATION_MS / short_long_lat,
                "latency_device": "cpu",
            },
        ])
    rt_entries.extend(_load_retrained_rt_entries())
    return lat_before, lat_after, acc_after, rt_entries


def _write_data(lat_before: dict, lat_after: dict, rt_entries: list[dict]) -> None:
    comparison = {
        "LAT_BEFORE": lat_before,
        "LAT_AFTER": lat_after,
        "device": "cpu",
        "input_length_before": 625,
        "input_length_after": 4096,
        "note": (
            "Before accuracy is long-sequence eval of short-trained checkpoints; "
            "before latency is measured at the short 625-sample input for parity "
            "with the existing doublet comparison figure."
        ),
    }
    (OUT_DIR / "_lat_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (OUT_DIR / "_rt_data.json").write_text(json.dumps(rt_entries, indent=2) + "\n")


def _plot_comparison(lat_before: dict, lat_after: dict, acc_after: dict) -> None:
    apply_publication_style()
    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(DCOL_W, 2.55))
    y = np.arange(len(MODELS))
    bar_h = 0.34
    before_color = "0.75"

    for i, model in enumerate(MODELS):
        color = FAMILY_COLORS.get(model, "#333333")
        ax_acc.barh(y[i] + bar_h / 2, acc_after[model] * 100, height=bar_h,
                    color=color, edgecolor="none", zorder=3)
        ax_acc.barh(y[i] - bar_h / 2, ACC_BEFORE[model] * 100, height=bar_h,
                    color=before_color, edgecolor="none", zorder=3)
        delta = (acc_after[model] - ACC_BEFORE[model]) * 100
        ax_acc.text(acc_after[model] * 100 + 1.0, y[i] + bar_h / 2,
                    f"{acc_after[model] * 100:.1f} (+{delta:.1f})",
                    va="center", fontsize=7)
        ax_acc.text(ACC_BEFORE[model] * 100 - 1.0, y[i] - bar_h / 2,
                    f"{ACC_BEFORE[model] * 100:.1f}",
                    va="center", ha="right", fontsize=7, color="0.35")

        ax_lat.barh(y[i] + bar_h / 2, lat_after[model], height=bar_h,
                    color=color, edgecolor="none", zorder=3)
        ax_lat.barh(y[i] - bar_h / 2, lat_before[model], height=bar_h,
                    color=before_color, edgecolor="none", zorder=3)
        slowdown = lat_after[model] / lat_before[model]
        ax_lat.text(lat_after[model] * 1.08, y[i] + bar_h / 2,
                    f"{lat_after[model]:.1f} ({slowdown:.1f}x)",
                    va="center", fontsize=7)
        ax_lat.text(lat_before[model] / 1.08, y[i] - bar_h / 2,
                    f"{lat_before[model]:.1f}",
                    va="center", ha="right", fontsize=7, color="0.35")

    labels = [DISPLAY[m] for m in MODELS]
    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels(labels, fontsize=8, fontweight="bold")
    ax_acc.set_xlim(35, 100)
    ax_acc.set_xlabel("Long-sequence accuracy (%)")
    ax_acc.grid(True, axis="x", alpha=0.3, linewidth=0.4)

    ax_lat.set_yticks(y)
    ax_lat.set_yticklabels([])
    ax_lat.set_xscale("log")
    ax_lat.set_xlim(min(lat_before.values()) / 2.5, max(lat_after.values()) * 2.8)
    ax_lat.set_xlabel("Latency (ms) - CPU torch, batch = 1")
    ax_lat.grid(True, axis="x", which="both", alpha=0.3, linewidth=0.4)

    for ax, label in [(ax_acc, "(a)"), (ax_lat, "(b)")]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.04, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, fontweight="bold")

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=before_color, edgecolor="none",
                          label="Short-trained"),
            plt.Rectangle((0, 0), 1, 1, facecolor="0.35", edgecolor="none",
                          label="Retrained on long"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0),
        fontsize=7, handlelength=1.4, columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.24, wspace=0.08)
    _emit_pdf(fig, OUT_DIR, "doublet_comparison_acc_latency.pdf")


def _plot_realtime(rt_entries: list[dict]) -> None:
    apply_publication_style()
    entries = sorted(rt_entries, key=lambda e: e["rt_factor"])
    labels = [
        f"{DISPLAY.get(e['family'], e['family'])} {e['stage'].lower()}"
        for e in entries
    ]
    rho = np.array([e["rt_factor"] for e in entries])
    colors = [FAMILY_COLORS.get(e["family"], "#333333") for e in entries]
    hatches = ["" if e["stage"] == "Retrained" else "///" for e in entries]

    fig_h = max(2.4, 0.34 * len(entries) + 1.15)
    fig, ax = plt.subplots(figsize=(DCOL_W * 0.78, fig_h))
    y_pos = np.arange(len(entries))
    bars = ax.barh(
        y_pos, rho,
        color=colors, edgecolor="white", linewidth=0.6,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    ax.axvspan(0, 1.0, alpha=0.10, color="0.45", zorder=1)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, zorder=4)
    ax.text(1.0, 1.02, r"$\rho = 1$", transform=ax.get_xaxis_transform(),
            fontsize=7, color="0.25", va="bottom", ha="center", style="italic")

    x_pad = max(rho.max() * 0.012, 0.01)
    for yi, value in zip(y_pos, rho):
        ax.text(
            value + x_pad, yi, f"{value:.3f}",
            va="center", ha="left", fontsize=7, color="0.15",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(rho.max() * 1.18, 2.0))
    ax.set_xlabel(
        r"$\rho = N_\mathrm{point} / (\tau \cdot f_s)$"
        rf"   ($N_\mathrm{{point}}/f_s = {SIGNAL_DURATION_MS:.3f}$ ms)"
    )
    ax.grid(True, axis="x", which="both", alpha=0.30, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.99, 0.02, "CPU torch, batch=1",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="0.35", style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.7", alpha=0.85))

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="0.75", edgecolor="white",
                          hatch="///", label="Short-trained"),
            plt.Rectangle((0, 0), 1, 1, facecolor="0.35", edgecolor="white",
                          label="Retrained"),
        ],
        loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015),
        fontsize=7, handlelength=1.4, columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.31, right=0.97, top=0.90, bottom=0.27)
    _emit_pdf(fig, OUT_DIR, "realtime_factor_doublet_3fam.pdf")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lat_before, lat_after, acc_after, rt_entries = _load_inputs()
    _write_data(lat_before, lat_after, rt_entries)
    _plot_comparison(lat_before, lat_after, acc_after)
    _plot_realtime(rt_entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
