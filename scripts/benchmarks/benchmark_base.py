"""Benchmark — Base model comparison (9 families, standard tier, no W&B).

Trains the base (M) variant of all 9 model families on data/dataset
(3 classes: 2um, 4um, 10um) with 3 seeds, measures accuracy + latency,
and generates:
  - results/benchmark_base/runs/*.json       (per-run results)
  - results/benchmark_base/summary.csv       (aggregated table)
  - results/benchmark_base/leaderboard.md    (markdown leaderboard)
  - results/benchmark_base/figures/pareto_latency.pdf

Usage:
    # All 9 base models, 3 seeds
    python benchmark_base.py

    # Single model
    python benchmark_base.py --model Conv1D

    # Custom seeds / epochs
    python benchmark_base.py --seeds 42,123,7 --epochs 150

    # Only regenerate plots + leaderboard from existing JSONs
    python benchmark_base.py --aggregate-only
"""

import argparse
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from p0.models import create_model, get_family, list_models
from p0.data import RAW_SIGNAL_LENGTH, BandpassFilter, Decimate, ParticleDataset
from p0.training import evaluate, train_one_epoch
from p0.training_utils import (
    compute_model_macs,
    measure_cpu_latency,
    measure_model_size,
    measure_peak_ram,
)
from p0.plotting import apply_publication_style, COL_W, DCOL_W, FIG_DOUBLE

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["2um", "4um", "10um"]

BASE_MODELS = [
    "Conv1DGAP", "ResNet1D",
    "InceptionTime1D", "MobileNet1D", "EfficientNet1D", "DenseNet1D",
]

FAMILY_COLORS = {
    "Conv1DGAP":       "#E6AB02",
    "DenseNet1D":      "#999999",
    "EfficientNet1D":  "#009E73",
    "InceptionTime1D": "#CC79A7",
    "MobileNet1D":     "#56B4E9",
    "ResNet1D":        "#D55E00",
}
FAMILY_MARKERS = {
    "Conv1DGAP":       "h",
    "DenseNet1D":      "s",
    "EfficientNet1D":  "^",
    "InceptionTime1D": "D",
    "MobileNet1D":     "P",
    "ResNet1D":        "X",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_hardware_info():
    info = {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)),
        "os": f"{platform.system()} {platform.release()}",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def stratified_split(dataset, val_fraction, seed):
    labels = np.array(dataset.labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    return train_idx.tolist(), val_idx.tolist()


def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        accs.append(np.mean(y_true[idx] == y_pred[idx]))
    accs = np.sort(accs)
    alpha = 1 - ci
    return float(accs[int(alpha / 2 * n_bootstrap)]), float(accs[int((1 - alpha / 2) * n_bootstrap)])


def _pareto_front(xs, ys):
    """Indices of Pareto-optimal points (minimize x, maximize y)."""
    points = sorted(range(len(xs)), key=lambda i: xs[i])
    pareto = []
    best_y = -float("inf")
    for i in points:
        if ys[i] > best_y:
            pareto.append(i)
            best_y = ys[i]
    return pareto


def _remove_chartjunk(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _emit_pdf(fig, figures_dir, fname):
    path = figures_dir / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Data loading (tier 1: standard) ─────────────────────────────────────────

def create_loaders(data_dir, seed, batch_size=32, val_split=0.2, decimate=4):
    bandpass = BandpassFilter(5.0, 100.0, 2.0)
    dec = Decimate(decimate)
    transforms = [bandpass, dec]

    train_dataset = ParticleDataset(data_dir / "train", CLASS_NAMES, transforms=transforms)
    val_dataset = ParticleDataset(data_dir / "train", CLASS_NAMES, transforms=transforms)
    test_dataset = ParticleDataset(data_dir / "test", CLASS_NAMES, transforms=transforms)

    train_idx, val_idx = stratified_split(train_dataset, val_split, seed)
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, len(train_idx)


# ── Training loop (no W&B) ──────────────────────────────────────────────────

def train_loop(model, train_loader, val_loader, criterion, optimizer, device,
               epochs, patience, output_dir, scheduler=None,
               convergence_threshold=0.95):
    best_val_acc = 0.0
    best_epoch = 0
    convergence_time = None
    no_improve = 0
    total_start = time.time()

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            no_improve += 1

        if convergence_time is None and val_acc >= convergence_threshold:
            convergence_time = time.time() - total_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {dt:.1f}s")

        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs)")
            break

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
    print(f"  Total training time: {total_time:.1f}s")

    return best_val_acc, best_epoch, total_time, convergence_time


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(model_name, seed, args):
    run_tag = f"{model_name}-seed{seed}"
    print(f"\n{'=' * 70}")
    print(f"  {run_tag}")
    print(f"{'=' * 70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = RAW_SIGNAL_LENGTH // args.decimate

    model = create_model(model_name, input_length=input_length,
                         num_classes=len(CLASS_NAMES)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader, val_loader, test_loader, train_size = create_loaders(
        Path(args.data_dir), seed, args.batch_size, args.val_split, args.decimate,
    )
    print(f"  Model: {model_name} ({num_params:,} params)")
    print(f"  Data: train={train_size}, test={len(test_loader.dataset)}")

    ckpt_dir = Path(args.output_dir) / "checkpoints" / run_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_macs = compute_model_macs(model, (1, 1, input_length), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_epoch, total_time, convergence_time = train_loop(
        model, train_loader, val_loader, criterion, optimizer, device,
        args.epochs, args.patience, ckpt_dir, scheduler=scheduler,
    )

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", weights_only=True))
    _, acc, y_pred, y_true, _ = evaluate(model, test_loader, criterion, device)
    ci_lower, ci_upper = bootstrap_accuracy_ci(y_true, y_pred, seed=seed)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n  Test accuracy: {acc:.4f}  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Efficiency (CPU canonical, see doc/metrics_conventions.md)
    latency = measure_cpu_latency(model, (1, 1, input_length),
                                  warmup=20, n_runs=200)
    peak_ram = measure_peak_ram(model, (1, 1, input_length), device)
    _, size_mb = measure_model_size(model)

    result = {
        "model_name": model_name,
        "model_family": get_family(model_name),
        "seed": seed,
        "params": num_params,
        "macs": model_macs,
        "size_mb": round(size_mb, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "latency_median_ms": round(latency["median_ms"], 4),
        "latency_p95_ms": round(latency["p95_ms"], 4),
        "latency_device": latency["latency_device"],
        "accuracy": round(acc, 4),
        "accuracy_ci_lower": round(ci_lower, 4),
        "accuracy_ci_upper": round(ci_upper, 4),
        "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in CLASS_NAMES},
        "confusion_matrix": cm,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "total_training_time_sec": round(total_time, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hardware_info": get_hardware_info(),
    }

    runs_dir = Path(args.output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / f"{run_tag}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved to {json_path}")

    return result


# ── Aggregation + leaderboard ────────────────────────────────────────────────

def aggregate_results(output_dir):
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        print("No runs directory found.")
        return None

    results = []
    for jf in sorted(runs_dir.glob("*.json")):
        with open(jf) as f:
            results.append(json.load(f))
    if not results:
        print("No JSON result files found.")
        return None

    df = pd.DataFrame(results)

    rows = []
    for model_name, grp in df.groupby("model_name"):
        row = {
            "Model": model_name,
            "Family": grp["model_family"].iloc[0],
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

    # Leaderboard
    leaderboard_path = Path(output_dir) / "leaderboard.md"
    sorted_df = summary_df.sort_values("Acc_Mean", ascending=False)
    with open(leaderboard_path, "w") as f:
        f.write("# Benchmark Base — Leaderboard\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| Rank | Model | Params | MACs | Acc Mean | Acc Std "
                "| Latency (ms) | Efficiency |\n")
        f.write("|------|-------|--------|------|----------|---------|"
                "-------------|------------|\n")
        for rank, (_, r) in enumerate(sorted_df.iterrows(), 1):
            macs_str = f"{r['MACs']:,}" if r["MACs"] else "N/A"
            eff_str = f"{r['Efficiency_Score']:.4f}" if r["Efficiency_Score"] else "N/A"
            f.write(f"| {rank} | {r['Model']} | {r['Params']:,} | {macs_str} | "
                    f"{r['Acc_Mean']:.4f} | {r['Acc_Std']:.4f} | "
                    f"{r['Latency_Median_ms']:.3f} | {eff_str} |\n")
    print(f"Leaderboard written to {leaderboard_path}")

    return summary_df


# ── Pareto plot (accuracy vs latency) ────────────────────────────────────────

def generate_pareto_latency(output_dir):
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
    agg = df.groupby("model_name").agg(
        latency=("latency_median_ms", "mean"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        family=("model_family", "first"),
    ).reset_index()
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    agg["err"] = 1.0 - agg["acc_mean"]

    families = sorted(agg["family"].unique())

    # y-window
    err_hi = 0.15
    err_lo = max(0.005, agg["err"].min() * 0.7)
    in_view = agg[agg["err"] <= err_hi]
    clipped = agg[agg["err"] > err_hi]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Clipped outliers
    for _, row in clipped.iterrows():
        c = FAMILY_COLORS.get(row["family"], "#333")
        ax.annotate("", xy=(row["latency"], err_hi * 0.985),
                    xytext=(row["latency"], err_hi * 0.78),
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.9))

    # Per-family scatter
    for fam in families:
        sub = in_view[in_view["family"] == fam]
        if sub.empty:
            continue
        c = FAMILY_COLORS.get(fam, "#333")
        ax.scatter(sub["latency"], sub["err"],
                   s=38, color=c, marker=FAMILY_MARKERS.get(fam, "o"),
                   edgecolor="white", linewidth=0.4, label=fam, zorder=3)

    # Pareto front
    pareto_idx = _pareto_front(agg["latency"].values, agg["acc_mean"].values)
    pdf_pts = agg.iloc[pareto_idx].sort_values("latency").reset_index(drop=True)
    ax.plot(pdf_pts["latency"], pdf_pts["err"],
            color="black", linestyle="--", linewidth=1.5, alpha=0.9, zorder=4)
    ax.scatter(pdf_pts["latency"], pdf_pts["err"],
               facecolor="none", edgecolor="black", linewidth=0.8,
               s=42, zorder=5)

    # Numbered badges
    badge_offsets = [(7, 7), (7, -10), (-14, 7), (-14, -10)]
    badge_bbox = dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.85)
    for i, row in pdf_pts.iterrows():
        dx, dy = badge_offsets[i % len(badge_offsets)]
        va = "bottom" if dy > 0 else "top"
        ha = "left" if dx > 0 else "right"
        ax.annotate(str(i + 1), xy=(row["latency"], row["err"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=8, color="black", fontweight="bold",
                    bbox=badge_bbox, zorder=6)

    # Side-key table
    key_lines = [
        f"{'#':<2s}{'Model':<22s}{'Lat (ms)':>10s}{'Acc':>9s}",
        "\u2500" * 43,
    ]
    for i, row in pdf_pts.iterrows():
        key_lines.append(
            f"{i + 1:<2d}{row['model_name']:<22s}"
            f"{row['latency']:>10.2f}{row['acc_mean']:>9.3f}"
        )
    ax.text(0.985, 0.025, "\n".join(key_lines),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, family="monospace", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.4", linewidth=0.5, alpha=0.85),
            zorder=7)

    # Iso-accuracy guides
    for a in (0.90, 0.95, 0.97, 0.98, 0.99):
        e = 1 - a
        if err_lo <= e <= err_hi:
            ax.axhline(e, color="0.55", linestyle=":", linewidth=0.5, zorder=1)
            ax.text(1.005, e, f"  {a:.2f}",
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7, color="0.30")

    # Axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(agg["latency"].min() * 0.85, agg["latency"].max() * 1.15)
    ax.set_ylim(err_lo, err_hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    def _fmt_err(v, _):
        if v >= 0.1:
            return f"{v:.1f}"
        if v >= 0.01:
            return f"{v:.2f}"
        return f"{v:.3f}"

    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0), numticks=12))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_err))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.30, linewidth=0.4)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)
    ax.set_xlabel("Median inference latency (ms)  \u2014  CPU torch, batch = 1")
    ax.set_ylabel("Error rate  (1 \u2212 accuracy)   [log scale]")
    _remove_chartjunk(ax)

    ax.legend(loc="upper right", ncol=3, columnspacing=0.7,
              handletextpad=0.35, borderaxespad=0.5, fontsize=7,
              frameon=True, framealpha=0.85)

    ax.text(
        0.99, 0.01, "CPU torch, batch = 1",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6, color="0.50", style="italic",
    )

    fig.subplots_adjust(left=0.085, right=0.93, top=0.97, bottom=0.13)
    _emit_pdf(fig, figures_dir, "pareto_latency.pdf")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Base: 9 model families, standard tier, no W&B")
    parser.add_argument("--model", type=str, default=None,
                        choices=BASE_MODELS,
                        help="Single model to run (default: all 9)")
    parser.add_argument("--data-dir", type=str, default="data/dataset",
                        help="Dataset root (must contain train/ and test/)")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_base",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--decimate", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,123,7",
                        help="Comma-separated seeds")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip training, only aggregate + plot")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = [args.model] if args.model else BASE_MODELS

    print("=" * 70)
    print("  Benchmark Base — 9 Families")
    print(f"  Models: {', '.join(models)}")
    print(f"  Seeds:  {seeds}")
    print(f"  Epochs: {args.epochs}, Patience: {args.patience}")
    print(f"  Data:   {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    if not args.aggregate_only:
        for model_name in models:
            for seed in seeds:
                run_tag = f"{model_name}-seed{seed}"
                json_path = Path(args.output_dir) / "runs" / f"{run_tag}.json"
                if json_path.exists():
                    print(f"\n  SKIP {run_tag} (already exists)")
                    continue
                run_single(model_name, seed, args)

    # Aggregate + plots
    summary_df = aggregate_results(args.output_dir)
    if summary_df is not None:
        generate_pareto_latency(args.output_dir)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
