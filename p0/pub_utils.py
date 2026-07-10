"""Centralised publication-quality plotting utilities.

**Single source of truth** for all recurring plot types (confusion matrices,
etc.) so that style changes happen in exactly one place.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Canvas constants ────────────────────────────────────────────────────────
COL_W = 3.39          # single-column width (inches)
DCOL_W = 7.00         # double-column width
FIG_SINGLE = (COL_W, 2.10)
FIG_SINGLE_TALL = (COL_W, 3.22)
FIG_DOUBLE = (DCOL_W, 2.70)

# ── RC params ───────────────────────────────────────────────────────────────
PUB_RC = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
}


def apply_publication_style():
    """Set matplotlib rcParams for publication-quality figures (idempotent)."""
    plt.rcParams.update(PUB_RC)


# ── Confusion matrix ───────────────────────────────────────────────────────

CM_SIZE = 5.0         # square figure side (inches)
CM_ANNOT_PT = 18      # cell annotation font size
CM_LABEL_PT = 14      # axis label + tick label font size

def plot_confusion_matrix(cm, class_names, *, ax=None):
    """Draw a publication-quality confusion matrix on *ax*.

    Parameters
    ----------
    cm : ndarray, shape (n, n)
        Confusion matrix (integer counts).
    class_names : list[str]
        Class labels for both axes.
    ax : matplotlib Axes, optional
        If *None* a new ``(CM_SIZE, CM_SIZE)`` figure is created.

    Returns
    -------
    fig : matplotlib Figure
        The parent figure (caller is responsible for saving / closing).
    ax : matplotlib Axes

    Style rules (all enforced here, never override in callers):
    - Blues + LogNorm(vmin=1, vmax=max(cm.max(), 2))
    - Zeros -> NaN -> white via cmap.set_bad("white")
    - No colorbar, no gridlines
    - Geometric luminance threshold for text color
    - Zeros displayed "0" in black
    - No bold on diagonal — uniform weight
    """
    apply_publication_style()
    n_classes = cm.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(CM_SIZE, CM_SIZE))
    else:
        fig = ax.figure

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    data_color = np.where(cm == 0, np.nan, cm).astype(float)
    vmax = max(cm.max(), 2)
    norm = LogNorm(vmin=1, vmax=vmax)
    ax.imshow(data_color, norm=norm, cmap=cmap, aspect="equal")

    # Geometric midpoint luminance threshold
    mid_val = np.sqrt(float(vmax))
    mid_lum = (0.2126 * cmap(norm(mid_val))[0]
               + 0.7152 * cmap(norm(mid_val))[1]
               + 0.0722 * cmap(norm(mid_val))[2])

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if val == 0:
                txt_color = "black"
            else:
                rgba = cmap(norm(val))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "white" if lum < mid_lum else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=CM_ANNOT_PT, fontweight="bold", color=txt_color)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=CM_LABEL_PT)
    ax.set_yticklabels(class_names, fontsize=CM_LABEL_PT)
    ax.set_xlabel("Predicted", fontsize=CM_LABEL_PT)
    ax.set_ylabel("True", fontsize=CM_LABEL_PT)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.96, bottom=0.10)

    return fig, ax
