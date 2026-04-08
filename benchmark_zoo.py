"""Benchmark 2 — Model Zoo (data/dataset, 3 classes).

Evaluates all 8 model zoo architectures across 6 difficulty tiers to build
an accuracy-vs-efficiency Pareto front for FPGA deployment decisions.
Tiers 1-5 use data/dataset; tier 6 trains on synthetic and tests on real.

Tiers:
    1 - Standard:          full train, clean test
    2 - Data-starved:      50 samples/class, clean test
    3 - Noisy:             full train, Gaussian noise on test (SNR=10dB)
    4 - Combined:          50 samples/class + noisy test
    5 - Noise extreme:     20 samples/class + real noise (SNR ~ U[-3,3] dB) + 15% time mask
    6 - Domain shift real: full synthetic train, dual eval (synthetic + real)

Usage:
    # Single model, single tier
    python benchmark_zoo.py --model Conv1D --tier 1 --seed 42 --epochs 5 --wandb-offline

    # All models, tier 1, 3 seeds
    python benchmark_zoo.py --all --tier 1 --wandb-offline

    # All models, all tiers
    python benchmark_zoo.py --all --tier all --wandb-offline

    # Custom seeds
    python benchmark_zoo.py --all --tier 1 --seeds 42,123,7 --wandb-offline

    # Sanity check
    python benchmark_zoo.py --sanity-check --wandb-offline
"""

import argparse
import json
import math
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from scipy.stats import kendalltau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from models import create_model, get_family, list_models
from train import (
    RAW_SIGNAL_LENGTH,
    BandpassFilter,
    Decimate,
    GaussianNoise,
    ParticleDataset,
    RealNoise,
    TimeMasking,
    evaluate,
)
from training_utils import (
    add_common_training_args,
    compute_model_macs,
    create_optimizer,
    create_scheduler,
    measure_inference_latency,
    measure_model_size,
    measure_peak_ram,
    run_post_testing,
    run_training_loop,
)

CLASS_NAMES = ["2um", "4um", "10um"]
TIER_NAMES = {
    1: "standard",
    2: "data_starved",
    3: "noisy",
    4: "combined",
    5: "noise_extreme",
    6: "domain_shift_real",
}

# Tier 5 hyperparameters (kept here as constants — edit to recalibrate)
TIER5_N_PER_CLASS = 20
TIER5_SNR_RANGE = (-3.0, 3.0)
TIER5_MASK_RATIO = 0.15

# Colorblind-safe palette derived from Okabe-Ito (avoids pale yellow on white).
# These 8 colors remain distinguishable under deuteranopia, protanopia, and
# tritanopia simulations.
FAMILY_COLORS = {
    "Conv1D":          "#0072B2",  # blue
    "Conv1DGAP":       "#E6AB02",  # Brewer Dark2 dark yellow (CVD-safe)
    "DenseNet1D":      "#999999",  # medium grey
    "EfficientNet1D":  "#009E73",  # bluish green
    "InceptionTime1D": "#CC79A7",  # reddish purple
    "LeNet1D":         "#E69F00",  # orange
    "MobileNet1D":     "#56B4E9",  # sky blue
    "ResNet1D":        "#D55E00",  # vermillion
    "VGG1D":           "#000000",  # black
}

# Distinct marker shape per family — provides a visual channel independent
# of color so the figures remain readable when printed in B&W or for
# color-vision-deficient readers.
FAMILY_MARKERS = {
    "Conv1D":          "o",  # circle
    "Conv1DGAP":       "h",  # hexagon
    "DenseNet1D":      "s",  # square
    "EfficientNet1D":  "^",  # triangle up
    "InceptionTime1D": "D",  # diamond
    "LeNet1D":         "v",  # triangle down
    "MobileNet1D":     "P",  # filled plus
    "ResNet1D":        "X",  # filled x
    "VGG1D":           "*",  # star
}

# Distinct line style per family — third visual channel for B&W readability.
# All 8 entries are unique so linestyle alone can identify a family (alongside
# color and marker). Custom tuples use the matplotlib ``(offset, on_off_seq)``
# form; each pattern was chosen to stay visually distinct at the 1.2 pt
# publication linewidth defined in ``PUB_RC``.
FAMILY_LINESTYLES = {
    "Conv1D":          "-",                              # solid
    "Conv1DGAP":       (0, (5, 1, 1, 1, 1, 1, 1, 1)),    # long-dash-triple-dot
    "DenseNet1D":      "--",                             # dashed
    "EfficientNet1D":  "-.",                             # dash-dot
    "InceptionTime1D": ":",                              # dotted
    "LeNet1D":         (0, (3, 1, 1, 1)),                # dash-dot-dot (tight)
    "MobileNet1D":     (0, (5, 2)),                      # dense long-dash
    "ResNet1D":        (0, (1, 1, 1, 1, 5, 1)),          # dot-dot-dash
    "VGG1D":           (0, (3, 1, 1, 1, 1, 1)),          # dash-dot-dot-dot (tight)
}

# ──────────────────────────────────────────────
# Publication style (double-column LaTeX figures)
# ──────────────────────────────────────────────
# Save figures at exact target dimensions and use \includegraphics{fig.pdf}
# WITHOUT a width= override so font sizes stay consistent across the paper.
COL_W = 3.39          # IEEE/Elsevier single-column width (inches)
DCOL_W = 7.00         # full text width (inches)
GOLD = 1.61803

FIG_SINGLE = (COL_W, COL_W / GOLD)            # 3.39 x 2.10  — compact single column
FIG_SINGLE_TALL = (COL_W, COL_W * 0.95)       # 3.39 x 3.22  — single column with bottom legend
FIG_DOUBLE = (DCOL_W, DCOL_W / (GOLD * 1.6))  # 7.00 x 2.70  — wide, short
FIG_GRID_ROW_H = (DCOL_W * 0.50) / 2          # per-row height of the small-multiples grid


def _family_legend_handles(families):
    """Return matplotlib ``Line2D`` handles (one per family) for a shared legend.

    Each handle carries the same (color, marker, linestyle) triple used by
    the grid panels so the legend is a faithful key for every curve drawn
    in the small-multiples figure.
    """
    from matplotlib.lines import Line2D
    handles = []
    for family in families:
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ls = FAMILY_LINESTYLES.get(family, "-")
        handles.append(Line2D(
            [0], [0], color=color, marker=marker, linestyle=ls,
            linewidth=1.0, markersize=4.0,
            markeredgecolor="white", markeredgewidth=0.3,
            label=family,
        ))
    return handles


def _grid_layout(n_panels: int, n_cols: int | None = None) -> tuple[int, int, tuple[float, float]]:
    """Return ``(n_rows, n_cols, figsize)`` for a small-multiples grid.

    The layout is a (near-)square: ``n_cols = ceil(sqrt(n_panels))`` and
    ``n_rows = ceil(n_panels / n_cols)``, so a 9-family run renders as
    a 3x3 grid, a 4-family run as 2x2, and a 10-family run as 4x3.
    ``n_cols`` can be overridden explicitly if a figure needs a specific
    aspect (e.g. to match a sibling figure). The figure width stays at
    ``DCOL_W`` and the row height scales linearly so each sub-axes keeps
    roughly the same on-page footprint regardless of the grid shape.
    """
    if n_cols is None:
        n_cols = max(1, math.ceil(math.sqrt(n_panels)))
    n_rows = max(1, (n_panels + n_cols - 1) // n_cols)
    return n_rows, n_cols, (DCOL_W, FIG_GRID_ROW_H * n_rows)

PUB_RC = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "standard",   # NOT "tight" — keeps canvas size constant
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType, editable in Illustrator
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
    """Set matplotlib rcParams for publication-quality figures.

    Idempotent — safe to call multiple times.
    """
    plt.rcParams.update(PUB_RC)


def _emit_pdf(fig, figures_dir, fname, *, wandb_run=None,
              wandb_key=None, caption=None):
    """Save a publication PDF and (optionally) log it to a W&B run.

    The matplotlib ``Figure`` is also closed here so callers do not have
    to. When ``wandb_run`` is provided the figure is logged via
    ``wandb.Image(fig, caption=...)`` which rasterises the Figure to PNG
    in-memory — **no PNG sibling file is written to disk**, preserving
    the existing "no PNG variants" convention from
    ``doc/variant_plotting.md`` §8.5.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    figures_dir : pathlib.Path
        Output directory for the PDF.
    fname : str
        Output filename (must end with ``.pdf``).
    wandb_run : wandb.sdk.wandb_run.Run, optional
        Active W&B run. If ``None`` no W&B logging happens (the default,
        backwards-compatible behaviour).
    wandb_key : str, optional
        Media key (e.g. ``figures/tier/heatmap``) under which the figure
        is logged. Required when ``wandb_run`` is provided.
    caption : str, optional
        Caption shown on hover in the W&B media panel. Defaults to
        ``wandb_key`` when omitted.
    """
    fig.savefig(figures_dir / fname)
    if wandb_run is not None and wandb_key is not None:
        wandb_run.log({wandb_key: wandb.Image(fig, caption=caption or wandb_key)})
    plt.close(fig)
    print(f"  Saved {figures_dir / fname}")


def infer_size_tag(model_name: str) -> str:
    """Return size tag (Pico/Nano/XXS/XS/S/M/L) inferred from a model variant name."""
    for suffix in ("-Pico", "-Nano", "-XXS", "-XS", "-S", "-L"):
        if model_name.endswith(suffix):
            return suffix[1:]
    return "M"


# ──────────────────────────────────────────────
# Stratified split helper
# ──────────────────────────────────────────────
def stratified_split(dataset, val_fraction, seed):
    """Return (train_indices, val_indices) using stratified shuffle split."""
    labels = np.array(dataset.labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    return train_idx.tolist(), val_idx.tolist()


def stratified_subsample(dataset, n_per_class, seed):
    """Return indices for a balanced subsample of n_per_class samples per class."""
    rng = np.random.RandomState(seed)
    labels = np.array(dataset.labels)
    selected = []
    for cls in range(len(CLASS_NAMES)):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < n_per_class:
            chosen = cls_indices
        else:
            chosen = rng.choice(cls_indices, size=n_per_class, replace=False)
        selected.extend(chosen.tolist())
    return selected


# ──────────────────────────────────────────────
# Tier data loading
# ──────────────────────────────────────────────
def create_tier_loaders(tier, seed, args):
    """Create (train_loader, val_loader, test_loaders, train_size) for a tier.

    Tiers:
        1 (standard):          full train, clean test
        2 (data_starved):      50 samples/class, clean test
        3 (noisy):             full train, Gaussian noise on val+test (SNR=10dB)
        4 (combined):          50 samples/class + noisy val+test
        5 (noise_extreme):     20 samples/class + real noise + time masking on val+test
        6 (domain_shift_real): full synthetic train, dual eval on synthetic + real

    The validation set always uses the same transforms as the test set so that
    early stopping selects checkpoints suited to the actual evaluation conditions.

    Returns
    -------
    (train_loader, val_loader, test_loaders, train_size)
        ``test_loaders`` is always a dict ``{suffix: DataLoader}``. For tiers
        1-5 the dict has a single entry keyed by the empty string. For tier 6
        it contains ``{"synthetic": ..., "real": ...}``.
    """
    data_dir = Path(args.data_dir)
    bandpass = BandpassFilter(5.0, 100.0, 2.0)
    decimate = Decimate(args.decimate)
    base_transforms = [bandpass, decimate]

    # Noisy test transforms (always-on Gaussian noise at SNR=10dB) — tiers 3/4
    noisy_test_transforms = base_transforms + [GaussianNoise(snr_db=10.0, p=1.0)]

    is_domain_shift = tier == 6

    # ── Tier 6: train+val on synthetic, dual test (synthetic + real) ──
    if is_domain_shift:
        train_dataset = ParticleDataset(data_dir / "train", CLASS_NAMES, transforms=base_transforms)
        val_dataset = ParticleDataset(data_dir / "train", CLASS_NAMES, transforms=base_transforms)
        train_idx, val_idx = stratified_split(train_dataset, args.val_split, seed)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        train_size = len(train_idx)

        test_dataset_synthetic = ParticleDataset(
            data_dir / "test", CLASS_NAMES, transforms=base_transforms
        )
        real_test_dir = Path(args.real_test_dir) / "test"
        test_dataset_real = ParticleDataset(
            real_test_dir, CLASS_NAMES, transforms=base_transforms
        )

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loaders = {
            "synthetic": DataLoader(test_dataset_synthetic, batch_size=args.batch_size, shuffle=False, num_workers=4),
            "real":      DataLoader(test_dataset_real,      batch_size=args.batch_size, shuffle=False, num_workers=4),
        }
        return train_loader, val_loader, test_loaders, train_size

    # ── Tiers 1-5: single test loader ──
    use_subsample = tier in (2, 4, 5)
    use_noisy_test = tier in (3, 4)
    use_extreme_test = tier == 5

    # Choose val/test transforms
    if use_extreme_test:
        extreme_test_transforms = base_transforms + [
            RealNoise(noise_dir=args.noise_dir,
                      snr_range=TIER5_SNR_RANGE,
                      p=1.0,
                      seed=seed),
            TimeMasking(mask_ratio=TIER5_MASK_RATIO, p=1.0),
        ]
        test_transforms = extreme_test_transforms
    elif use_noisy_test:
        test_transforms = noisy_test_transforms
    else:
        test_transforms = base_transforms

    # Train dataset (always base transforms — noise only on val/test)
    train_dataset = ParticleDataset(data_dir / "train", CLASS_NAMES, transforms=base_transforms)

    # Test dataset
    test_dataset = ParticleDataset(data_dir / "test", CLASS_NAMES, transforms=test_transforms)

    # Val dataset: must match test distribution so early stopping selects
    # models that perform well under the same conditions as the test set.
    val_dataset = ParticleDataset(data_dir / "train", CLASS_NAMES, transforms=test_transforms)

    # Train/val split (full or subsampled)
    if use_subsample:
        n_per_class = TIER5_N_PER_CLASS if tier == 5 else 50
        sub_indices = stratified_subsample(train_dataset, n_per_class=n_per_class, seed=seed)
        sub_labels = np.array(train_dataset.labels)[sub_indices]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=seed)
        rel_train, rel_val = next(sss.split(np.zeros(len(sub_indices)), sub_labels))
        train_idx = [sub_indices[i] for i in rel_train]
        val_idx = [sub_indices[i] for i in rel_val]
    else:
        train_idx, val_idx = stratified_split(train_dataset, args.val_split, seed)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    train_size = len(train_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loaders = {"": test_loader}

    return train_loader, val_loader, test_loaders, train_size


# ──────────────────────────────────────────────
# Bootstrap confidence interval
# ──────────────────────────────────────────────
def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for accuracy."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        acc = np.mean(y_true[idx] == y_pred[idx])
        accs.append(acc)
    accs = np.sort(accs)
    alpha = 1 - ci
    lower = float(accs[int(alpha / 2 * n_bootstrap)])
    upper = float(accs[int((1 - alpha / 2) * n_bootstrap)])
    return lower, upper


# ──────────────────────────────────────────────
# Hardware info
# ──────────────────────────────────────────────
def get_hardware_info():
    """Collect basic hardware info."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)),
        "os": f"{platform.system()} {platform.release()}",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# ──────────────────────────────────────────────
# Single run pipeline
# ──────────────────────────────────────────────
def run_single(model_name, tier, seed, args):
    """Run training + evaluation for a single (model, tier, seed) combo.

    Returns the result dict.
    """
    tier_name = TIER_NAMES[tier]
    dataset_name = Path(args.data_dir).name
    run_tag = f"{model_name}-{dataset_name}-tier{tier}-seed{seed}"
    print(f"\n{'=' * 70}")
    print(f"  {run_tag}")
    print(f"{'=' * 70}")

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = RAW_SIGNAL_LENGTH // args.decimate

    # Create model
    model = create_model(model_name, input_length=input_length, num_classes=len(CLASS_NAMES)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create loaders
    train_loader, val_loader, test_loaders, train_size = create_tier_loaders(tier, seed, args)
    print(f"  Model: {model_name} ({num_params:,} params)")
    test_sizes = ", ".join(
        f"test{('_' + suf) if suf else ''}={len(loader.dataset)}"
        for suf, loader in test_loaders.items()
    )
    print(f"  Tier {tier} ({tier_name}): train={train_size}, {test_sizes}")

    # Output directory for checkpoint
    output_dir = Path(args.output_dir) / "checkpoints" / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B init
    wandb_mode = "offline" if args.wandb_offline else "online"
    model_macs = compute_model_macs(model, (1, 1, input_length), device)

    config = {
        "model_name": model_name,
        "model_size_params": num_params,
        "dataset": dataset_name,
        "dataset_size": train_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "seed": seed,
        "patience": args.patience,
        "num_classes": len(CLASS_NAMES),
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "decimate": args.decimate,
        "input_length": input_length,
        "model_macs": model_macs,
        "tier": tier,
        "tier_name": tier_name,
    }

    run = wandb.init(
        project="particle-benchmark",
        config=config,
        group=model_name,
        tags=[tier_name, "benchmark2", f"tier{tier}", f"seed{seed}"],
        name=run_tag,
        job_type="training",
        mode=wandb_mode,
    )

    result = {}
    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("val/accuracy", summary="max", goal="maximize")
        run.define_metric("val/loss", summary="min", goal="minimize")

        run.summary["model_size_params"] = num_params
        run.summary["dataset_size"] = train_size

        # Train
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, args)
        scheduler = create_scheduler(optimizer, args)

        best_val_acc, best_epoch, total_time, convergence_time = run_training_loop(
            run, model, train_loader, val_loader, criterion, optimizer, device, args,
            output_dir=output_dir, scheduler=scheduler,
        )

        # Load best model
        model.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))

        # Evaluate on each test loader (single entry for tiers 1-5, dual for tier 6)
        test_results = {}
        for suffix, loader in test_loaders.items():
            prefix = f"test_tier{tier}_{suffix}" if suffix else f"test_tier{tier}"
            run_post_testing(
                run, model, loader, criterion, device, CLASS_NAMES, prefix=prefix,
            )
            _, acc_s, y_pred_s, y_true_s, _ = evaluate(model, loader, criterion, device)
            ci_lower_s, ci_upper_s = bootstrap_accuracy_ci(y_true_s, y_pred_s, seed=seed)
            report_s = classification_report(
                y_true_s, y_pred_s, target_names=CLASS_NAMES, output_dict=True
            )
            test_results[suffix] = {
                "accuracy": acc_s,
                "ci_lower": ci_lower_s,
                "ci_upper": ci_upper_s,
                "per_class_f1": {cls: round(report_s[cls]["f1-score"], 4) for cls in CLASS_NAMES},
                "confusion_matrix": confusion_matrix(y_true_s, y_pred_s).tolist(),
            }

        # Pick "primary" metrics depending on tier:
        #   - Tier 6: real-test accuracy is the headline (the whole point of the tier)
        #   - Other tiers: only one entry, use it.
        if tier == 6:
            primary = test_results["real"]
            synth = test_results["synthetic"]
            domain_gap = synth["accuracy"] - primary["accuracy"]
            run.summary["domain_gap"] = round(domain_gap, 4)
            run.summary["accuracy_synthetic"] = round(synth["accuracy"], 4)
            run.summary["accuracy_real"] = round(primary["accuracy"], 4)
        else:
            primary = next(iter(test_results.values()))
            domain_gap = None

        acc = primary["accuracy"]
        ci_lower = primary["ci_lower"]
        ci_upper = primary["ci_upper"]
        per_class_f1 = primary["per_class_f1"]
        cm = primary["confusion_matrix"]

        # Efficiency metrics
        latency = measure_inference_latency(model, (1, 1, input_length), device, warmup=50, n_runs=1000)
        peak_ram = measure_peak_ram(model, (1, 1, input_length), device)
        _, size_mb = measure_model_size(model)

        run.summary["inference_latency_median_ms"] = latency["median_ms"]
        run.summary["inference_latency_p95_ms"] = latency["p95_ms"]
        run.summary["peak_ram_mb"] = peak_ram
        run.summary["model_size_mb"] = size_mb

        # Build result dict
        model_family = get_family(model_name)
        model_size_tag = infer_size_tag(model_name)
        result = {
            "model_name": model_name,
            "model_family": model_family,
            "model_size_tag": model_size_tag,
            "seed": seed,
            "tier": tier,
            "tier_name": tier_name,
            "params": num_params,
            "macs": model_macs,
            "size_mb": round(size_mb, 2),
            "peak_ram_mb": round(peak_ram, 2),
            "latency_median_ms": round(latency["median_ms"], 4),
            "latency_p95_ms": round(latency["p95_ms"], 4),
            "accuracy": round(acc, 4),
            "accuracy_ci_lower": round(ci_lower, 4),
            "accuracy_ci_upper": round(ci_upper, 4),
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_acc, 4),
            "total_training_time_sec": round(total_time, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hardware_info": get_hardware_info(),
        }
        if tier == 6:
            result["accuracy_synthetic"] = round(test_results["synthetic"]["accuracy"], 4)
            result["accuracy_real"] = round(test_results["real"]["accuracy"], 4)
            result["domain_gap"] = round(domain_gap, 4)

        # Save JSON
        runs_dir = Path(args.output_dir) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        json_path = runs_dir / f"{run_tag}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Result saved to {json_path}")

    finally:
        run.finish()

    return result


# ──────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────
def aggregate_results(output_dir):
    """Load all JSON results, aggregate across seeds, write summary.csv."""
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        print("No runs directory found, skipping aggregation.")
        return None

    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))

    if not results:
        print("No JSON result files found.")
        return None

    df = pd.DataFrame(results)

    # Group by (model, tier) and compute stats across seeds
    rows = []
    for (model, tier), grp in df.groupby(["model_name", "tier"]):
        row = {
            "Model": model,
            "Model_Family": grp["model_family"].iloc[0] if "model_family" in grp.columns else get_family(model),
            "Model_Size_Tag": grp["model_size_tag"].iloc[0] if "model_size_tag" in grp.columns else "M",
            "Tier": int(tier),
            "Tier_Name": TIER_NAMES.get(int(tier), "unknown"),
            "Params": int(grp["params"].iloc[0]),
            "MACs": int(grp["macs"].iloc[0]) if grp["macs"].iloc[0] is not None else None,
            "Size_MB": round(grp["size_mb"].mean(), 2),
            "Peak_RAM_MB": round(grp["peak_ram_mb"].mean(), 2),
            "Latency_Median_ms": round(grp["latency_median_ms"].mean(), 4),
            "Acc_Mean": round(grp["accuracy"].mean(), 4),
            "Acc_Std": round(grp["accuracy"].std(), 4),
            "Acc_CI_Lower": round(grp["accuracy_ci_lower"].mean(), 4),
            "Acc_CI_Upper": round(grp["accuracy_ci_upper"].mean(), 4),
            "Num_Seeds": len(grp),
        }
        if row["MACs"] is not None and row["MACs"] > 0:
            row["Efficiency_Score"] = round(row["Acc_Mean"] / math.log10(row["MACs"]), 4)
        else:
            row["Efficiency_Score"] = None
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary written to {csv_path}")
    print(summary_df.to_string(index=False))

    # Generate leaderboard markdown
    leaderboard_path = Path(output_dir) / "leaderboard.md"
    with open(leaderboard_path, "w") as f:
        f.write("# Benchmark 2 — Model Zoo Leaderboard\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        for tier_num in sorted(summary_df["Tier"].unique()):
            tier_df = summary_df[summary_df["Tier"] == tier_num].sort_values("Acc_Mean", ascending=False)
            f.write(f"## Tier {tier_num}: {TIER_NAMES.get(tier_num, 'unknown')}\n\n")
            f.write("| Rank | Model | Params | MACs | Acc Mean | Acc Std | Latency (ms) | Efficiency |\n")
            f.write("|------|-------|--------|------|----------|---------|-------------|------------|\n")
            for rank, (_, r) in enumerate(tier_df.iterrows(), 1):
                macs_str = f"{r['MACs']:,}" if r["MACs"] else "N/A"
                eff_str = f"{r['Efficiency_Score']:.4f}" if r["Efficiency_Score"] else "N/A"
                f.write(f"| {rank} | {r['Model']} | {r['Params']:,} | {macs_str} | "
                        f"{r['Acc_Mean']:.4f} | {r['Acc_Std']:.4f} | "
                        f"{r['Latency_Median_ms']:.3f} | {eff_str} |\n")
            f.write("\n")
    print(f"Leaderboard written to {leaderboard_path}")

    return summary_df


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
def _pareto_front(xs, ys):
    """Return indices of points on the Pareto front (maximize y, minimize x)."""
    points = sorted(range(len(xs)), key=lambda i: xs[i])
    pareto = []
    best_y = -float("inf")
    for i in points:
        if ys[i] > best_y:
            pareto.append(i)
            best_y = ys[i]
    return pareto


def generate_plots(output_dir):
    """Generate diagnostic PNG plots (confusion matrices, F1 heatmap, seed
    boxplot).

    These are *diagnostic* artefacts, not publication figures. The
    publication PDFs (``scaling_*.pdf``, ``pareto*.pdf``, ``tier_*.pdf``) are
    produced by their dedicated functions and follow the layout discipline
    documented in ``doc/variant_plotting.md`` and ``doc/tier_plotting.md``.
    Anything that has a publication equivalent (accuracy-by-tier bar chart,
    raster scaling curves, raster Pareto fronts) has been removed from
    here — keeping it would create a bifurcated, inconsistent output set.
    """
    runs_dir = Path(output_dir) / "runs"
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))

    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)

    # All remaining diagnostic plots operate on Tier 1 only.
    t1 = df[df["tier"] == 1].copy()
    if t1.empty:
        return
    # Derive family/size columns (backward compat with old JSONs missing them)
    if "model_family" not in t1.columns:
        t1["model_family"] = t1["model_name"].apply(get_family)
    if "model_size_tag" not in t1.columns:
        t1["model_size_tag"] = t1["model_name"].apply(infer_size_tag)

    # ── Confusion matrices: best and worst Tier-1 model (diagnostic) ──
    t1_acc = t1.groupby("model_name")["accuracy"].mean()
    for label, mname in (("best", t1_acc.idxmax()), ("worst", t1_acc.idxmin())):
        row = t1[t1["model_name"] == mname].iloc[0]
        cm = np.array(row["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Tier 1 {label.capitalize()} Model: {mname} (acc={row['accuracy']:.4f})")
        plt.tight_layout()
        fig.savefig(figures_dir / f"confusion_matrix_{label}_{mname}.png", dpi=150)
        plt.close(fig)

    # ── Per-class F1 heatmap (diagnostic) ──
    f1_rows = []
    for _, row in t1.groupby("model_name").first().reset_index().iterrows():
        f1_rows.append({"Model": row["model_name"], **row["per_class_f1"]})
    f1_df = pd.DataFrame(f1_rows).set_index("Model")
    fig, ax = plt.subplots(figsize=(8, max(4, len(f1_df) * 0.6)))
    sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Tier 1: Per-Class F1 Score")
    plt.tight_layout()
    fig.savefig(figures_dir / "f1_heatmap.png", dpi=150)
    plt.close(fig)

    # ── Seed stability boxplot (diagnostic; needs ≥ 2 seeds per model) ──
    if t1.groupby("model_name").size().max() > 1:
        models_sorted = t1.groupby("model_name")["accuracy"].mean().sort_values(ascending=False).index
        t1_sorted = t1.set_index("model_name").loc[models_sorted].reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=t1_sorted, x="model_name", y="accuracy", ax=ax, order=models_sorted)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title("Tier 1: Seed Stability (Accuracy Spread)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(figures_dir / "seed_stability_boxplot.png", dpi=150)
        plt.close(fig)

    print(f"\nDiagnostic plots saved to {figures_dir}/")


# ──────────────────────────────────────────────
# Scaling curves (v1.1)
# ──────────────────────────────────────────────
def _load_tier1_aggregated(output_dir):
    """Helper: load tier 1 results and aggregate across seeds.

    Returns (agg_df, families) or (None, []) if no data.
    """
    runs_dir = Path(output_dir) / "runs"
    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))
    if not results:
        return None, []

    df = pd.DataFrame(results)
    t1 = df[df["tier"] == 1].copy()
    if t1.empty:
        return None, []

    if "model_family" not in t1.columns:
        t1["model_family"] = t1["model_name"].apply(get_family)
    if "model_size_tag" not in t1.columns:
        t1["model_size_tag"] = t1["model_name"].apply(infer_size_tag)

    agg = t1.groupby("model_name").agg(
        macs=("macs", "first"),
        latency=("latency_median_ms", "mean"),
        size_mb=("size_mb", "first"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        family=("model_family", "first"),
        size_tag=("model_size_tag", "first"),
    ).reset_index()
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    families = sorted(agg["family"].unique())
    return agg, families


def generate_scaling_curves(output_dir, *, wandb_run=None):
    """Publication-quality scaling curves (single-column figure).

    Produces:
        scaling_macs.pdf — Accuracy vs MACs (log x)

    NOTE: the latency variant of this curve was dropped because kernel-launch
    overhead (~0.13 ms on our hardware) creates clumps of small variants that
    share identical latency but different accuracies, which no per-family
    envelope can render cleanly. Accuracy-vs-latency is now shown as
    ``pareto_latency.pdf`` (scatter + global Pareto front), which handles
    such clumps gracefully.

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under the media key ``figures/variant/scaling_macs``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, families = _load_tier1_aggregated(output_dir)
    if agg is None:
        print("No Tier 1 results for scaling curves.")
        return

    # Y-axis range
    ymin = max(0.78, float(agg["acc_mean"].min()) - 0.02)
    ymax = min(1.005, float(agg["acc_mean"].max()) + 0.015)

    panels = [
        ("macs", "MACs", True, "scaling_macs.pdf"),
    ]

    for x_col, x_label, x_log, fname in panels:
        fig, ax = plt.subplots(figsize=FIG_SINGLE_TALL)
        for family in families:
            fam = (agg[agg["family"] == family]
                   .dropna(subset=[x_col])
                   .sort_values(x_col)
                   .reset_index(drop=True))
            if fam.empty:
                continue
            color = FAMILY_COLORS.get(family, "#333333")
            marker = FAMILY_MARKERS.get(family, "o")
            ls = FAMILY_LINESTYLES.get(family, "-")

            # Per-family upper envelope: keep only variants whose accuracy
            # ties or beats the cumulative max along increasing x. This is
            # the family's own Pareto front in (x, accuracy) space and is
            # what "scaling curve" actually means — the best accuracy
            # achievable at this compute budget. Sorting alone is not
            # enough because some small variants happen to have nearly
            # identical latencies but very different accuracies, producing
            # zig-zag artefacts (visual "asymptotes"). The envelope is
            # monotone non-decreasing by construction.
            cmax = fam["acc_mean"].cummax()
            fam_env = fam[fam["acc_mean"] == cmax]
            # Dominated variants: still plotted as faint hollow markers so
            # the reader sees that some variants exist below the frontier.
            fam_dom = fam[fam["acc_mean"] < cmax]

            x = fam_env[x_col].values
            y = fam_env["acc_mean"].values
            yerr = fam_env["acc_std"].values

            # Error band on the envelope
            ax.fill_between(x, y - yerr, y + yerr,
                            color=color, alpha=0.15, linewidth=0)
            # Envelope line + markers (distinct shape and linestyle per
            # family for colorblind-safe and B&W-printable plots).
            ax.plot(x, y,
                    color=color, marker=marker, markersize=4.0,
                    linewidth=1.2, linestyle=ls,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label=family)
            # Dominated variants — small open markers, no line.
            if not fam_dom.empty:
                ax.scatter(fam_dom[x_col].values, fam_dom["acc_mean"].values,
                           facecolor="none", edgecolor=color,
                           marker=marker, s=12, linewidth=0.6,
                           alpha=0.55, zorder=2)

        if x_log:
            ax.set_xscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3, linewidth=0.4)

        # Legend below the axes — keeps the data area clean.
        # 3 cols × 3 rows fits the 8 families without clipping at 3.39" width.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=3, columnspacing=0.8, handletextpad=0.3,
                  borderaxespad=0)

        # Manual margins so the canvas size matches pareto.pdf exactly,
        # enabling side-by-side placement in LaTeX subfigures.
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.32)
        _emit_pdf(
            fig, figures_dir, fname,
            wandb_run=wandb_run,
            wandb_key="figures/variant/scaling_macs",
            caption="Scaling curve: per-family upper envelope (Accuracy vs MACs)",
        )


def generate_scaling_grid(output_dir, x_col="macs", x_label="MACs",
                          fname="scaling_grid.pdf", *, wandb_run=None):
    """2x4 small-multiples: one scaling curve per family, shared axes.

    Designed as a double-column figure for an overview of all family
    scaling behaviours without the visual clutter of the combined plot.

    Parameterized on the x-axis so the same layout can be reused to
    probe each resource axis:

        x_col="macs"      -> compute view (``scaling_grid.pdf``)
        x_col="size_mb"   -> storage view (``scaling_grid_size.pdf``)

    The reason this is useful is that the three resource axes — MACs
    (computing), latency (real-time), and on-disk size (storage) — do
    **not** rank variants identically. An FPGA deployment decision that
    optimises for BRAM usage (storage) may land on a different winner
    than one optimising for DSP cycles (MACs).

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under a key dispatched on ``fname``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, families = _load_tier1_aggregated(output_dir)
    if agg is None:
        return

    agg = agg.dropna(subset=[x_col])
    if agg.empty:
        print(f"No data for scaling grid with x_col={x_col!r}.")
        return

    # Grid sized to the current family count — near-square layout via
    # `_grid_layout`, so the 9-family zoo renders as 3x3. Add an extra
    # slice of canvas height for the shared family legend drawn under
    # the grid.
    n_rows, n_cols, (fig_w, fig_h) = _grid_layout(len(families))
    legend_h = 0.55  # inches reserved for the shared bottom legend
    figsize = (fig_w, fig_h + legend_h)
    n_panels = n_rows * n_cols
    ymin = max(0.78, float(agg["acc_mean"].min()) - 0.02)
    ymax = min(1.005, float(agg["acc_mean"].max()) + 0.015)
    xmin = max(1e-3, float(agg[x_col].min()) * 0.5)
    xmax = float(agg[x_col].max()) * 2.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for ax, family in zip(axes.flat, families):
        # Sort by the x-axis quantity so the connecting line is monotone
        # in x. Size-tag ordinal sorting would be wrong here because size
        # order and resource order can diverge (a wider shallow variant
        # may have fewer MACs than a narrower deep one), which would
        # produce zig-zag inside a panel on the log-x axis.
        fam = (agg[agg["family"] == family]
               .dropna(subset=[x_col])
               .sort_values(x_col))
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ls = FAMILY_LINESTYLES.get(family, "-")
        x = fam[x_col].values
        y = fam["acc_mean"].values
        yerr = fam["acc_std"].values

        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.20, linewidth=0)
        ax.plot(x, y, color=color, marker=marker, linestyle=ls,
                markersize=3.4, linewidth=1.0,
                markeredgecolor="white", markeredgewidth=0.3)

        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(family, color=color, pad=2, fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=6)

    # Hide unused axes if the family count does not fill the grid.
    for ax in axes.flat[len(families):n_panels]:
        ax.set_visible(False)

    # Reserve the lower `legend_frac` of the figure for the shared legend,
    # then pack the subplot grid into the remaining upper band. This keeps
    # the subplot tile size independent of the added legend strip.
    legend_frac = legend_h / figsize[1]
    fig.supxlabel(x_label, fontsize=8, y=legend_frac + 0.02)
    fig.supylabel("Accuracy", fontsize=8, x=0.005)
    fig.subplots_adjust(left=0.06, right=0.99,
                        top=1.0 - (0.06 * fig_h / figsize[1]),
                        bottom=legend_frac + 0.08,
                        wspace=0.12, hspace=0.40)
    fig.legend(handles=_family_legend_handles(families),
               loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=min(len(families), 5), columnspacing=1.2,
               handletextpad=0.4, handlelength=2.2,
               frameon=False, fontsize=7)
    # Dispatch the W&B media key on fname so the MACs grid and the
    # size grid land under separate panels in the report run.
    if "size" in fname:
        wandb_key = "figures/variant/scaling_grid_size"
        caption = "Scaling grid: per-family small-multiples (Accuracy vs on-disk size)"
    else:
        wandb_key = "figures/variant/scaling_grid"
        caption = "Scaling grid: per-family small-multiples (Accuracy vs MACs)"
    _emit_pdf(
        fig, figures_dir, fname,
        wandb_run=wandb_run, wandb_key=wandb_key, caption=caption,
    )


def generate_pareto_publication(output_dir, x_col="macs", x_label="MACs",
                                x_log=True, fname="pareto.pdf", *,
                                wandb_run=None):
    """Publication-quality Pareto plot (single column).

    Highlights points that lie on the Pareto front. No per-point text labels.
    Same colors as scaling_curves so the figures can be cross-referenced.

    Parameterized on the x-axis so the same layout can be reused for
    accuracy-vs-MACs (``pareto.pdf``) and accuracy-vs-size
    (``pareto_size.pdf``). Accuracy-vs-latency is handled by the
    dedicated :func:`generate_pareto_latency_focus`, whose custom
    log-error layout copes with kernel-launch clumps that this generic
    helper cannot render cleanly.

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under either ``figures/variant/pareto_macs`` or
    ``figures/variant/pareto_size``, dispatched on ``fname``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, families = _load_tier1_aggregated(output_dir)
    if agg is None:
        return

    valid = agg.dropna(subset=[x_col]).copy()
    if valid.empty:
        return

    pareto_idx = _pareto_front(valid[x_col].values, valid["acc_mean"].values)

    ymin = max(0.78, float(valid["acc_mean"].min()) - 0.02)
    ymax = min(1.005, float(valid["acc_mean"].max()) + 0.015)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_TALL)
    # Scatter — distinct color AND marker shape per family for colorblind safety.
    for family in families:
        fam = valid[valid["family"] == family]
        if fam.empty:
            continue
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ax.scatter(fam[x_col], fam["acc_mean"],
                   color=color, marker=marker,
                   s=22, edgecolor="white", linewidth=0.4,
                   label=family, zorder=3)

    # Pareto front line
    pdf_pts = valid.iloc[pareto_idx].sort_values(x_col)
    if len(pdf_pts) > 1:
        ax.plot(pdf_pts[x_col], pdf_pts["acc_mean"],
                color="black", linestyle="--", linewidth=0.8,
                alpha=0.7, zorder=2, label="Pareto front")
    # Highlight Pareto points
    ax.scatter(pdf_pts[x_col], pdf_pts["acc_mean"],
               facecolor="none", edgecolor="black", linewidth=0.8,
               s=42, zorder=4)

    # Tag each Pareto-optimal point with a tiny numeric badge and list the
    # full model names in a boxed key anchored in the lower-right corner of
    # the axes. This keeps the front uncluttered while still identifying
    # every recommended architecture.
    pdf_pts_r = pdf_pts.reset_index(drop=True)
    for i, row in pdf_pts_r.iterrows():
        ax.annotate(str(i + 1),
                    xy=(row[x_col], row["acc_mean"]),
                    xytext=(5, 4), textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=6, color="black", fontweight="bold",
                    zorder=5)

    key_text = "\n".join(
        f"{i + 1}. {row['model_name']}"
        for i, row in pdf_pts_r.iterrows()
    )
    ax.text(0.975, 0.03, key_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=6, family="serif", linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="0.5",
                      linewidth=0.4),
            zorder=6)

    if x_log:
        ax.set_xscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, columnspacing=0.8, handletextpad=0.3,
              borderaxespad=0)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.32)
    # This function is called twice with different ``fname`` values —
    # ``pareto.pdf`` (compute / MACs) and ``pareto_size.pdf`` (storage /
    # on-disk MB). Each gets a distinct W&B media key so they show up
    # as separate panels in the report run. The real-time latency view
    # is generated separately by ``generate_pareto_latency_focus``.
    if "size" in fname:
        wandb_key = "figures/variant/pareto_size"
        caption = "Pareto front: Accuracy vs on-disk size (MB)"
    else:
        wandb_key = "figures/variant/pareto_macs"
        caption = "Pareto front: Accuracy vs MACs"
    _emit_pdf(
        fig, figures_dir, fname,
        wandb_run=wandb_run, wandb_key=wandb_key, caption=caption,
    )


# Hardware-specific kernel-launch floor on RTX A1000. Latency below
# this value is dominated by the per-call CUDA launch overhead, not by
# the model's arithmetic — variants in this region cluster regardless
# of compute. We shade the region in latency-focused figures so the
# clump is interpreted correctly.
KERNEL_LAUNCH_FLOOR_MS = 0.14


def generate_pareto_latency_focus(output_dir, *, wandb_run=None):
    """Latency-focused Pareto plot — accuracy vs latency in **error rate** space.

    A specialised replacement for the generic
    :func:`generate_pareto_publication` when ``x_col="latency"``. Built
    after extensive iteration on the Acc-vs-Latency view because that
    figure is the primary deployment-decision artefact for embedded
    inference.

    Why a dedicated helper rather than reusing
    :func:`generate_pareto_publication` ?

    1. **Log error rate, not linear accuracy.** With every competitive
       variant landing in [0.94, 0.99] accuracy, a linear y-axis
       compresses the most interesting differences into ~5 % of the
       canvas. Plotting :math:`1 - \\mathrm{acc}` on a log axis spreads
       1.7 % vs 2.7 % vs 4.0 % error into clearly distinct bands —
       which is the entire point of the figure.
    2. **Clipped y-window.** A handful of pico variants land at
       40–50 % error and would otherwise blow up the y-range. We clip
       the window at err = 0.15 (acc = 0.85) and annotate excluded
       outliers with explicit upward arrows + italic name labels at
       the top edge, so the clipping is transparent.
    3. **Kernel-launch floor.** On the RTX A1000 the per-call CUDA
       launch overhead is ~0.13 ms, creating a clump of small variants
       at the leftmost x-edge that share latency but not accuracy.
       The shaded floor band makes that hardware artefact explicit so
       the reader does not over-interpret the clump.
    4. **Numbered Pareto badges + side-key table.** With six Pareto
       points, three of which sit inside the kernel-launch clump,
       inline labels collide unreadably. Numeric badges next to the
       gold stars + a fixed-width table in the lower-right corner
       solve this and double as a deployment cheat-sheet (model,
       latency, accuracy).
    5. **Iso-accuracy guides.** Horizontal dotted lines at acc =
       {0.90, 0.95, 0.97, 0.98, 0.99} let the reader translate any
       error reading on the left axis into a familiar accuracy
       reading on the right.

    Output: ``pareto_latency.pdf`` under ``<output_dir>/figures``.
    W&B media key: ``figures/variant/pareto_latency``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, families = _load_tier1_aggregated(output_dir)
    if agg is None:
        return

    df = agg.dropna(subset=["latency"]).copy()
    if df.empty:
        return
    df["err"] = 1.0 - df["acc_mean"]

    # ── y-window: clip below 0.85 accuracy (err = 0.15) so the worst
    # pico outliers don't dominate the canvas. Anything excluded gets
    # an explicit "↑ name" annotation at the top edge.
    err_hi = 0.15
    err_lo = 0.013
    in_view = df[df["err"] <= err_hi]
    clipped = df[df["err"] > err_hi]

    # 7.0 x 4.2 → full text width, taller than FIG_DOUBLE so the
    # side-key + family legend + iso-acc rail all fit without cramping.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # ── kernel-launch floor band + label
    ax.axvspan(0, KERNEL_LAUNCH_FLOOR_MS, color="0.85", alpha=0.45, zorder=0)
    ax.text(KERNEL_LAUNCH_FLOOR_MS * 0.96, err_lo * 1.20,
            "kernel-launch\nfloor",
            ha="right", va="bottom", fontsize=5.5, color="0.30",
            style="italic")

    # ── clipped outliers as upward arrows at the top edge
    for _, row in clipped.iterrows():
        c = FAMILY_COLORS.get(row["family"], "#333")
        ax.annotate("",
                    xy=(row["latency"], err_hi * 0.985),
                    xytext=(row["latency"], err_hi * 0.78),
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
        ax.text(row["latency"] * 1.08, err_hi * 0.84, row["model_name"],
                ha="left", va="center", fontsize=5.5, color=c,
                style="italic")

    # ── per-family scatter (no zigzag connecting lines)
    for fam in families:
        sub = in_view[in_view["family"] == fam]
        if sub.empty:
            continue
        c = FAMILY_COLORS.get(fam, "#333")
        ax.scatter(sub["latency"], sub["err"],
                   s=38, color=c, marker=FAMILY_MARKERS.get(fam, "o"),
                   edgecolor="white", linewidth=0.4, label=fam, zorder=3)

    # ── Pareto front (computed on the FULL dataset so it's truthful
    # even when some points lie outside the displayed y-window)
    pareto_idx = _pareto_front(df["latency"].values, df["acc_mean"].values)
    pdf_pts = df.iloc[pareto_idx].sort_values("latency").reset_index(drop=True)
    ax.plot(pdf_pts["latency"], pdf_pts["err"],
            color="black", linestyle="--", linewidth=1.5, alpha=0.9, zorder=4)
    ax.scatter(pdf_pts["latency"], pdf_pts["err"],
               facecolor="gold", edgecolor="black", linewidth=1.0,
               s=140, zorder=5, marker="*", label="Pareto front")

    # numeric badges next to each star
    for i, row in pdf_pts.iterrows():
        ax.annotate(str(i + 1),
                    xy=(row["latency"], row["err"]),
                    xytext=(7, 6), textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=7.5, color="black", fontweight="bold",
                    zorder=6)

    # ── side-key box anchored to the empty lower-right quadrant
    key_lines = [
        f"{'#':<2s}{'Model':<22s}{'Lat (ms)':>10s}{'Acc':>9s}",
        "─" * 43,
    ]
    for i, row in pdf_pts.iterrows():
        key_lines.append(
            f"{i + 1:<2d}{row['model_name']:<22s}"
            f"{row['latency']:>10.2f}{row['acc_mean']:>9.3f}"
        )
    ax.text(0.985, 0.025, "\n".join(key_lines),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.7, family="monospace", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="0.4",
                      linewidth=0.5, alpha=0.95),
            zorder=7)

    # ── iso-accuracy guides — horizontal lines + right-edge labels
    for a in (0.90, 0.95, 0.97, 0.98, 0.99):
        e = 1 - a
        if err_lo <= e <= err_hi:
            ax.axhline(e, color="0.55", linestyle=":", linewidth=0.5, zorder=1)
            ax.text(1.005, e, f"  {a:.2f}",
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left",
                    fontsize=5.8, color="0.30")

    # ── axes formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(df["latency"].min() * 0.85, df["latency"].max() * 1.15)
    ax.set_ylim(err_lo, err_hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    def _fmt_err(v, _):
        if v >= 0.1:
            return f"{v:.1f}"
        if v >= 0.01:
            return f"{v:.2f}"
        return f"{v:.3f}"

    # Custom log locator on the y-axis: show 1×10^k AND 2,3,5×10^k so the
    # reader sees ticks at 0.02 / 0.03 / 0.05 / 0.1 instead of just 0.1.
    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=12)
    )
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_err))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.30, linewidth=0.4)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)
    ax.set_xlabel("Median inference latency (ms)  —  batch = 1, RTX A1000")
    ax.set_ylabel("Error rate  (1 − accuracy)   [log scale]")

    # ── family legend in the upper-right (sparser quadrant — only the
    # high-error pico outliers live up there and they're tagged inline)
    ax.legend(loc="upper right", ncol=3, columnspacing=0.7,
              handletextpad=0.35, borderaxespad=0.5, fontsize=6.3,
              frameon=True, framealpha=0.92)

    fig.subplots_adjust(left=0.085, right=0.93, top=0.97, bottom=0.13)
    _emit_pdf(
        fig, figures_dir, "pareto_latency.pdf",
        wandb_run=wandb_run,
        wandb_key="figures/variant/pareto_latency",
        caption=("Pareto front: Accuracy vs Latency "
                 "(log-error space, kernel-floor annotated, side-keyed)"),
    )


def _load_all_tiers_aggregated(output_dir):
    """Load all runs across all tiers, aggregate base models across seeds.

    Mirrors :func:`_load_tier1_aggregated` but keeps the tier axis. Restricted
    to base-size models (``size_tag == "M"``) so the tier curves stay legible
    — same convention as :func:`generate_tier_heatmap`.

    Returns
    -------
    (agg_df, families) or (None, [])
        ``agg_df`` has columns: ``model_name``, ``family``, ``size_tag``,
        ``tier``, ``acc_mean``, ``acc_std``, ``macs``, ``latency``. Tier-6
        rows additionally carry ``acc_synth_mean``, ``acc_real_mean`` and
        ``domain_gap_mean`` (NaN for other tiers).
    """
    runs_dir = Path(output_dir) / "runs"
    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))
    if not results:
        return None, []

    df = pd.DataFrame(results)
    if "model_family" not in df.columns:
        df["model_family"] = df["model_name"].apply(get_family)
    if "model_size_tag" not in df.columns:
        df["model_size_tag"] = df["model_name"].apply(infer_size_tag)

    base = df[df["model_size_tag"] == "M"].copy()
    if base.empty:
        return None, []

    for col in ("accuracy_synthetic", "accuracy_real", "domain_gap"):
        if col not in base.columns:
            base[col] = np.nan

    agg = base.groupby(["model_name", "tier"]).agg(
        family=("model_family", "first"),
        size_tag=("model_size_tag", "first"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        macs=("macs", "first"),
        latency=("latency_median_ms", "mean"),
        acc_synth_mean=("accuracy_synthetic", "mean"),
        acc_real_mean=("accuracy_real", "mean"),
        domain_gap_mean=("domain_gap", "mean"),
    ).reset_index()
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    agg["tier"] = agg["tier"].astype(int)
    families = sorted(agg["family"].unique())
    return agg, families


def generate_tier_robustness(output_dir, *, wandb_run=None):
    """Publication-quality tier-robustness curve (single column).

    Analogue of :func:`generate_scaling_curves` with the tier axis instead
    of the size axis: one line per family (base model), markers at each
    available tier, shaded ±σ band across seeds.

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under the media key ``figures/tier/robustness``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, families = _load_all_tiers_aggregated(output_dir)
    if agg is None:
        print("No base-model results for tier robustness curve.")
        return

    tiers = sorted(agg["tier"].unique())
    if len(tiers) < 2:
        print("Need at least 2 tiers for tier robustness curve.")
        return

    ymin = max(0.0, float(agg["acc_mean"].min()) - 0.05)
    ymax = min(1.005, float(agg["acc_mean"].max()) + 0.03)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_TALL)
    for family in families:
        fam = agg[agg["family"] == family].sort_values("tier")
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ls = FAMILY_LINESTYLES.get(family, "-")
        x = fam["tier"].to_numpy(dtype=float)
        y = fam["acc_mean"].to_numpy()
        yerr = fam["acc_std"].to_numpy()

        ax.fill_between(x, y - yerr, y + yerr,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y,
                color=color, marker=marker, markersize=4.0,
                linewidth=1.2, linestyle=ls,
                markeredgecolor="white", markeredgewidth=0.4,
                label=family)

    ax.set_xticks(tiers)
    ax.set_xticklabels([f"T{t}" for t in tiers])
    ax.set_xlim(min(tiers) - 0.25, max(tiers) + 0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Difficulty tier")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, columnspacing=0.8, handletextpad=0.3,
              borderaxespad=0)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.32)
    _emit_pdf(
        fig, figures_dir, "tier_robustness.pdf",
        wandb_run=wandb_run,
        wandb_key="figures/tier/robustness",
        caption="Tier robustness: per-family degradation curves",
    )


def generate_tier_grid(output_dir, *, wandb_run=None):
    """2x4 small-multiples tier-robustness curve (double column).

    One panel per family, shared axes. Analogue of
    :func:`generate_scaling_grid` but along the tier axis. Highlights
    per-family degradation patterns without the clutter of the combined
    plot in :func:`generate_tier_robustness`.

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under the media key ``figures/tier/robustness_grid``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, families = _load_all_tiers_aggregated(output_dir)
    if agg is None:
        return
    tiers = sorted(agg["tier"].unique())
    if len(tiers) < 2:
        return

    n_rows, n_cols, (fig_w, fig_h) = _grid_layout(len(families))
    legend_h = 0.55  # inches reserved for the shared bottom legend
    figsize = (fig_w, fig_h + legend_h)
    n_panels = n_rows * n_cols
    ymin = max(0.0, float(agg["acc_mean"].min()) - 0.05)
    ymax = min(1.005, float(agg["acc_mean"].max()) + 0.03)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for ax, family in zip(axes.flat, families):
        fam = agg[agg["family"] == family].sort_values("tier")
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ls = FAMILY_LINESTYLES.get(family, "-")
        x = fam["tier"].to_numpy(dtype=float)
        y = fam["acc_mean"].to_numpy()
        yerr = fam["acc_std"].to_numpy()

        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.20, linewidth=0)
        ax.plot(x, y, color=color, marker=marker, linestyle=ls,
                markersize=3.4, linewidth=1.0,
                markeredgecolor="white", markeredgewidth=0.3)

        ax.set_ylim(ymin, ymax)
        ax.set_xticks(tiers)
        ax.set_xticklabels([f"T{t}" for t in tiers])
        ax.set_title(family, color=color, pad=2, fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[len(families):n_panels]:
        ax.set_visible(False)

    legend_frac = legend_h / figsize[1]
    fig.supxlabel("Difficulty tier", fontsize=8, y=legend_frac + 0.02)
    fig.supylabel("Accuracy", fontsize=8, x=0.005)
    fig.subplots_adjust(left=0.06, right=0.99,
                        top=1.0 - (0.06 * fig_h / figsize[1]),
                        bottom=legend_frac + 0.08,
                        wspace=0.12, hspace=0.40)
    fig.legend(handles=_family_legend_handles(families),
               loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=min(len(families), 5), columnspacing=1.2,
               handletextpad=0.4, handlelength=2.2,
               frameon=False, fontsize=7)
    _emit_pdf(
        fig, figures_dir, "tier_grid.pdf",
        wandb_run=wandb_run,
        wandb_key="figures/tier/robustness_grid",
        caption="Tier robustness grid: per-family small-multiples",
    )


def generate_tier6_domain_gap(output_dir, *, wandb_run=None):
    """Slope chart showing sim-to-real accuracy drop per family (tier 6).

    For each family we draw a sloped line from ``(synthetic, acc_synth)``
    to ``(real, acc_real)``. Steeper = larger domain gap. Colors, markers
    and line styles match the rest of the publication figure set so the
    family identity is consistent across the paper.

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under the media key ``figures/tier/domain_gap_t6``.
    """
    apply_publication_style()
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg, _ = _load_all_tiers_aggregated(output_dir)
    if agg is None:
        return
    t6 = agg[agg["tier"] == 6].dropna(subset=["acc_synth_mean", "acc_real_mean"])
    if t6.empty:
        print("No tier-6 results for domain gap plot.")
        return

    x_synth, x_real = 0.0, 1.0
    ymin = max(0.0, float(min(t6["acc_synth_mean"].min(),
                               t6["acc_real_mean"].min())) - 0.03)
    ymax = min(1.005, float(max(t6["acc_synth_mean"].max(),
                                 t6["acc_real_mean"].max())) + 0.03)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_TALL)
    # Sort so that the smallest domain gap is drawn last (on top) — lets
    # readers locate the most robust family immediately.
    for _, row in t6.sort_values("domain_gap_mean", ascending=False).iterrows():
        family = row["family"]
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        ls = FAMILY_LINESTYLES.get(family, "-")
        ax.plot([x_synth, x_real],
                [row["acc_synth_mean"], row["acc_real_mean"]],
                color=color, marker=marker, markersize=4.5,
                linewidth=1.2, linestyle=ls,
                markeredgecolor="white", markeredgewidth=0.4,
                label=family)

    ax.set_xticks([x_synth, x_real])
    ax.set_xticklabels(["synthetic", "real"])
    ax.set_xlim(x_synth - 0.25, x_real + 0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Evaluation domain")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3, linewidth=0.4, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, columnspacing=0.8, handletextpad=0.3,
              borderaxespad=0)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.32)
    _emit_pdf(
        fig, figures_dir, "tier6_domain_gap.pdf",
        wandb_run=wandb_run,
        wandb_key="figures/tier/domain_gap_t6",
        caption="Tier 6 domain gap: synthetic to real slope chart",
    )


def generate_tier_heatmap(output_dir, *, wandb_run=None):
    """Single-column heatmap of accuracy across tiers (base models only).

    When ``wandb_run`` is provided, the figure is also logged to W&B
    under the media key ``figures/tier/heatmap``.
    """
    apply_publication_style()
    runs_dir = Path(output_dir) / "runs"
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))
    if not results:
        return

    df = pd.DataFrame(results)
    if "model_size_tag" not in df.columns:
        df["model_size_tag"] = df["model_name"].apply(infer_size_tag)

    # Restrict to base models (size tag = "M") to keep the heatmap legible
    base = df[df["model_size_tag"] == "M"]
    if base.empty:
        print("No base models for tier heatmap.")
        return

    pivot = base.groupby(["model_name", "tier"])["accuracy"].mean().unstack("tier")
    if pivot.empty or pivot.shape[1] < 2:
        return
    pivot = pivot.reindex(sorted(pivot.index))

    vmin = max(0.50, float(pivot.min().min()) - 0.05)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu",
                vmin=vmin, vmax=1.0,
                cbar_kws={"label": "Accuracy", "shrink": 0.85,
                          "pad": 0.02},
                annot_kws={"size": 6.5},
                linewidths=0.3, linecolor="white",
                ax=ax)
    ax.set_xlabel("Tier")
    ax.set_ylabel("")
    ax.set_xticklabels([f"T{int(t)}" for t in pivot.columns], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.subplots_adjust(left=0.32, right=0.94, top=0.96, bottom=0.18)
    _emit_pdf(
        fig, figures_dir, "tier_heatmap.pdf",
        wandb_run=wandb_run,
        wandb_key="figures/tier/heatmap",
        caption="Tier heatmap: base models x tiers",
    )


# ──────────────────────────────────────────────
# W&B benchmark-report run
# ──────────────────────────────────────────────
def _attach_report_artifacts(run, output_dir, summary_df):
    """Upload publication PDFs, summary.csv and leaderboard.md to the
    current W&B run as a single ``benchmark-report`` artifact, and write
    a few headline metrics to ``run.summary``.

    The artifact preserves the original PDFs for download; the W&B UI
    already shows the rasterised PNG previews logged via ``_emit_pdf``,
    but the vector PDFs are what the paper actually includes.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    artifact = wandb.Artifact(
        name="benchmark2-report",
        type="benchmark-report",
        description="Publication-quality figures (PDF), summary CSV, leaderboard",
    )
    if figures_dir.exists():
        for pdf_path in sorted(figures_dir.glob("*.pdf")):
            artifact.add_file(str(pdf_path), name=f"figures/{pdf_path.name}")
    for extra in ("summary.csv", "leaderboard.md"):
        p = output_dir / extra
        if p.exists():
            artifact.add_file(str(p), name=extra)
    run.log_artifact(artifact)

    if summary_df is not None and not summary_df.empty:
        run.summary["num_models"] = int(summary_df["Model"].nunique())
        run.summary["num_families"] = int(summary_df["Model_Family"].nunique())
        run.summary["num_tiers"] = int(summary_df["Tier"].nunique())
        run.summary["num_results"] = int(len(summary_df))
        t1 = summary_df[summary_df["Tier"] == 1]
        if not t1.empty:
            best = t1.sort_values("Acc_Mean", ascending=False).iloc[0]
            run.summary["best_tier1_model"] = str(best["Model"])
            run.summary["best_tier1_accuracy"] = float(best["Acc_Mean"])


def regenerate_and_publish_figures(args, summary_df=None, *, wandb_publish=True):
    """Regenerate every publication figure and (optionally) push to W&B.

    Always emits the local PDFs into ``<output_dir>/figures``. When
    ``wandb_publish`` is True, also opens a dedicated "benchmark report"
    W&B run, logs each figure as a ``wandb.Image`` under
    ``figures/<axis>/<name>``, and attaches the original PDFs +
    ``summary.csv`` + ``leaderboard.md`` as a single ``wandb.Artifact``.

    The report run is intentionally distinct from the per-model training
    runs (different ``group``, ``job_type``, and ``name`` template) so
    it's instantly filterable in the W&B UI.
    """
    run = None
    if wandb_publish:
        run_name = f"benchmark2-report-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        dataset_name = Path(args.data_dir).name
        # ``reinit="finish_previous"`` ensures any lingering per-model
        # wandb.Run from the post-training path is closed cleanly before
        # the report run starts. This is the modern (wandb >= 0.20)
        # spelling of the older ``reinit=True`` flag, which now emits a
        # deprecation warning but still works.
        run = wandb.init(
            project="particle-benchmark",
            name=run_name,
            group="benchmark2-report",
            job_type="benchmark-report",
            tags=["benchmark2", "report", "publication", dataset_name],
            mode="offline" if args.wandb_offline else "online",
            reinit="finish_previous",
            config={
                "output_dir": str(args.output_dir),
                "scaling": bool(args.scaling),
                "epochs": args.epochs,
                "seeds": args.seeds,
                "dataset": dataset_name,
            },
        )
    try:
        # Diagnostic PNGs — kept W&B-free (they're debug artefacts, not
        # publication figures) per doc/variant_plotting.md §8.5.
        generate_plots(args.output_dir)

        # Tier-axis figures (no --scaling dependency)
        generate_tier_heatmap(args.output_dir, wandb_run=run)
        generate_tier_robustness(args.output_dir, wandb_run=run)
        generate_tier_grid(args.output_dir, wandb_run=run)
        generate_tier6_domain_gap(args.output_dir, wandb_run=run)

        # Variant-axis figures (only meaningful with --scaling)
        if args.scaling:
            generate_scaling_curves(args.output_dir, wandb_run=run)
            # Small-multiples scaling grid — one curve per family, over
            # each resource axis. MACs = compute view; on-disk size =
            # storage view (complementary because a wider shallow variant
            # can have fewer MACs but a much larger on-disk footprint).
            generate_scaling_grid(args.output_dir, wandb_run=run)
            generate_scaling_grid(
                args.output_dir,
                x_col="size_mb", x_label="Model size (MB)",
                fname="scaling_grid_size.pdf",
                wandb_run=run,
            )
            # Pareto fronts over the three resource axes:
            #   MACs     -> computing (FLOPs)
            #   latency  -> real-time  (wall-clock inference)
            #   size_mb  -> storage    (on-disk / BRAM footprint)
            generate_pareto_publication(args.output_dir, wandb_run=run)
            # Latency uses a specialised log-error design (kernel-floor
            # shading, side-key, clipped-outlier arrows, iso-acc guides).
            generate_pareto_latency_focus(args.output_dir, wandb_run=run)
            generate_pareto_publication(
                args.output_dir,
                x_col="size_mb", x_label="Model size (MB)",
                x_log=True, fname="pareto_size.pdf",
                wandb_run=run,
            )

        if run is not None:
            _attach_report_artifacts(run, args.output_dir, summary_df)
    finally:
        if run is not None:
            run.finish()


# ──────────────────────────────────────────────
# Ranking stability
# ──────────────────────────────────────────────
def compute_ranking_stability(output_dir):
    """Check Kendall's tau between seed-based rankings on Tier 1."""
    runs_dir = Path(output_dir) / "runs"
    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))

    df = pd.DataFrame(results)
    t1 = df[df["tier"] == 1]
    if t1.empty:
        print("No Tier 1 results for ranking stability.")
        return

    seeds = sorted(t1["seed"].unique())
    if len(seeds) < 2:
        print("Need at least 2 seeds for ranking stability.")
        return

    # For each seed, rank models by accuracy
    rankings = {}
    for s in seeds:
        seed_df = t1[t1["seed"] == s].sort_values("accuracy", ascending=False)
        rankings[s] = list(seed_df["model_name"])

    # Compute Kendall's tau between all pairs
    taus = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            # Map model names to ranks
            rank1 = {m: r for r, m in enumerate(rankings[s1])}
            rank2 = {m: r for r, m in enumerate(rankings[s2])}
            common = sorted(set(rank1) & set(rank2))
            if len(common) < 2:
                continue
            r1 = [rank1[m] for m in common]
            r2 = [rank2[m] for m in common]
            tau, _ = kendalltau(r1, r2)
            taus.append(tau)
            print(f"  Kendall's tau (seed {s1} vs {s2}): {tau:.4f}")

    if taus:
        avg_tau = np.mean(taus)
        print(f"  Average Kendall's tau: {avg_tau:.4f}")
        if avg_tau < 0.8:
            print("  WARNING: Ranking stability is low (tau < 0.8). "
                  "Consider more epochs or seeds.")


# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────
def run_sanity_checks(args):
    """Run baseline sanity checks before main benchmark."""
    print("\n" + "=" * 70)
    print("  SANITY CHECKS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = RAW_SIGNAL_LENGTH // args.decimate

    # 1. Random baseline
    print(f"\n1. Random baseline: {1.0 / len(CLASS_NAMES):.4f} (floor)")

    # 2. Majority baseline
    bandpass = BandpassFilter(5.0, 100.0, 2.0)
    decimate = Decimate(args.decimate)
    data_dir = Path(args.data_dir)
    test_dataset = ParticleDataset(data_dir / "test", CLASS_NAMES, transforms=[bandpass, decimate])
    labels = np.array(test_dataset.labels)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    majority_acc = counts.max() / len(labels)
    print(f"2. Majority baseline: {majority_acc:.4f} "
          f"(class '{CLASS_NAMES[counts.argmax()]}' = {counts.max()}/{len(labels)})")

    # 3. Determinism check
    print("\n3. Determinism check: training Conv1D with seed=42 twice...")
    accs = []
    for trial in range(2):
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = create_model("Conv1D", input_length=input_length, num_classes=len(CLASS_NAMES)).to(device)

        train_loader, val_loader, test_loaders, _ = create_tier_loaders(1, 42, args)
        test_loader = next(iter(test_loaders.values()))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Train for a few epochs
        for epoch in range(args.epochs):
            model.train()
            for signals, labels_batch in train_loader:
                signals, labels_batch = signals.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                out = model(signals)
                loss = criterion(out, labels_batch)
                loss.backward()
                optimizer.step()

        _, acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        accs.append(acc)
        print(f"   Trial {trial + 1}: accuracy = {acc:.6f}")

    if abs(accs[0] - accs[1]) < 1e-6:
        print("   PASS: Results are deterministic.")
    else:
        print(f"   WARNING: Results differ by {abs(accs[0] - accs[1]):.6f}")

    print("\nSanity checks complete.\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 2: Model Zoo (data/dataset, 3 classes)"
    )
    add_common_training_args(parser, data_dir_default="data/dataset")

    # Override defaults for benchmark
    parser.set_defaults(patience=20, epochs=150, scheduler="cosine", output_dir="results/benchmark2")

    # Benchmark-specific args
    parser.add_argument("--all", action="store_true",
                        help="Run all models in the zoo")
    parser.add_argument("--tier", type=str, default="1",
                        help="Tier: 1, 2, 3, 4, 5, 6, or 'all' (default: 1)")
    parser.add_argument("--seeds", type=str, default="42,123,7",
                        help="Comma-separated seeds (default: 42,123,7)")
    parser.add_argument("--noise-dir", type=str, default="data/Noise",
                        help="Directory of real noise .npy files (tier 5)")
    parser.add_argument("--real-test-dir", type=str, default="data/S7_pure_real",
                        help="Directory of real measurements for tier 6 "
                             "(must contain test/<class>/*.npy)")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity checks before benchmark")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only run aggregation and plots (no training)")
    parser.add_argument("--scaling", action="store_true",
                        help="Run S/M/L variants for scaling curves (with --all: all 24 models; "
                             "with --model: S/M/L of that family)")
    parser.add_argument("--no-wandb-publish", action="store_true",
                        help="Skip pushing the publication figures to a W&B "
                             "'benchmark report' run (still emits PDFs locally).")

    args = parser.parse_args()

    # Parse tiers
    if args.tier.lower() == "all":
        tiers = [1, 2, 3, 4, 5, 6]
    else:
        tiers = [int(t.strip()) for t in args.tier.split(",")]

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Parse models — expand to S/M/L variants when --scaling is set
    if args.all:
        if args.scaling:
            models_to_run = list_models(include_variants=True)
        else:
            models_to_run = list_models()
    else:
        if args.scaling:
            family = get_family(args.model)
            # Expand to S/M/L for this family
            models_to_run = [
                v for v in list_models(include_variants=True)
                if get_family(v) == family
            ]
        else:
            models_to_run = [args.model]

    print(f"Benchmark 2 — Model Zoo")
    print(f"  Models: {models_to_run}")
    print(f"  Tiers: {tiers}")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Output: {args.output_dir}")
    if args.scaling:
        print(f"  Scaling: enabled ({len(models_to_run)} variants)")

    # Aggregate only mode
    if args.aggregate_only:
        summary_df = aggregate_results(args.output_dir)
        if summary_df is not None:
            regenerate_and_publish_figures(
                args, summary_df,
                wandb_publish=not args.no_wandb_publish,
            )
            compute_ranking_stability(args.output_dir)
        return

    # Sanity checks
    if args.sanity_check:
        run_sanity_checks(args)

    # Main benchmark loop
    all_results = []
    total_runs = len(models_to_run) * len(tiers) * len(seeds)
    run_idx = 0

    for model_name in models_to_run:
        for tier in tiers:
            for seed in seeds:
                run_idx += 1
                print(f"\n>>> Run {run_idx}/{total_runs}")
                try:
                    result = run_single(model_name, tier, seed, args)
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {model_name} tier={tier} seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()

    # Aggregation & plots
    if all_results:
        print("\n" + "=" * 70)
        print("  AGGREGATION & ANALYSIS")
        print("=" * 70)
        summary_df = aggregate_results(args.output_dir)
        if summary_df is not None:
            regenerate_and_publish_figures(
                args, summary_df,
                wandb_publish=not args.no_wandb_publish,
            )
            compute_ranking_stability(args.output_dir)

    print(f"\nBenchmark complete. {len(all_results)}/{total_runs} runs succeeded.")


if __name__ == "__main__":
    main()
