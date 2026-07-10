"""PatchTST scratch vs pretrained comparison figure.

Generates a presentation-friendly accuracy-vs-size plot from
``outputs/benchmarks/results/patchtst_pretrained_p0_direct_20260702/summary.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from p0.plotting import DCOL_W, apply_publication_style  # noqa: E402


OUT_DIR = (
    _PROJECT_ROOT
    / "outputs/benchmarks/results/patchtst_pretrained_p0_direct_20260702/figures"
)
SUMMARY_CSV = OUT_DIR.parent / "summary.csv"

DISPLAY = {
    "PatchTST": "Scratch",
    "PatchTST-Compact": "Scratch compact",
    "PatchTSTPretrained": "Pretrained full",
    "PatchTSTPretrained-Frozen": "Pretrained frozen",
}

COLORS = {
    "PatchTST": "#332288",
    "PatchTST-Compact": "#AA4499",
    "PatchTSTPretrained": "#117733",
    "PatchTSTPretrained-Frozen": "#88CCEE",
}

MARKERS = {
    "PatchTST": ">",
    "PatchTST-Compact": "o",
    "PatchTSTPretrained": "<",
    "PatchTSTPretrained-Frozen": "s",
}

ORDER = [
    "PatchTSTPretrained-Frozen",
    "PatchTSTPretrained",
    "PatchTST-Compact",
    "PatchTST",
]


def _load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    return df[df["Model"].isin(ORDER)].copy()


def _make_figure(df: pd.DataFrame) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(DCOL_W * 0.78, 3.05))

    for model in ORDER:
        row = df[df["Model"] == model]
        if row.empty:
            continue
        item = row.iloc[0]
        params = item["Params"]
        acc = item["Acc_Mean"] * 100.0
        err = item["Acc_Std"] * 100.0
        label = DISPLAY[model]

        ax.errorbar(
            params,
            acc,
            yerr=err,
            fmt=MARKERS[model],
            markersize=7.2,
            color=COLORS[model],
            markerfacecolor=COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=0.9,
            capsize=2.4,
            label=label,
            zorder=4,
        )

        dx, dy = {
            "PatchTSTPretrained-Frozen": (14, -14),
            "PatchTSTPretrained": (-18, -20),
            "PatchTST-Compact": (14, 18),
            "PatchTST": (16, -18),
        }[model]
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            f"{label}\n{acc:.1f}%",
            xy=(params, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7,
            color="0.15",
            arrowprops=dict(
                arrowstyle="-",
                color="0.45",
                linewidth=0.45,
                shrinkA=0,
                shrinkB=4,
            ),
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.90,
            ),
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters [log scale]")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(df["Params"].min() / 1.6, df["Params"].max() * 1.6)
    ax.set_ylim(88.4, 99.6)
    ax.grid(True, which="major", alpha=0.30, linewidth=0.4)
    ax.grid(True, which="minor", axis="x", alpha=0.12, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.99,
        0.02,
        "PatchTST tier 1, 3 seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="0.45",
        style="italic",
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.18)
    return fig


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_summary()
    fig = _make_figure(df)
    pdf = OUT_DIR / "patchtst_pretrained_accuracy_size.pdf"
    png = OUT_DIR / "patchtst_pretrained_accuracy_size.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
