"""Re-generate all OOD plots with updated tab10 color palette and log to existing W&B runs.

For each benchmark run, this script:
  1. Loads the saved best_model.pth
  2. Re-computes all OOD scores (MSP, Energy, ODIN, Mahalanobis)
  3. Re-generates every plot that uses hardcoded colors
  4. Resumes the existing W&B run and overwrites the plots (commit=False)

Usage:
    python recolor_ood_plots.py                          # all 12 datasets
    python recolor_ood_plots.py --datasets S0_baseline   # single dataset
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from train import (
    RAW_SIGNAL_LENGTH,
    ParticleDataset,
    Conv1DClassifier,
    BandpassFilter,
    Decimate,
    Truncate,
)
from benchmark import (
    compute_ood_scores,
    compute_odin_scores,
    compute_mahalanobis_scores,
    _compute_fpr_at_tpr,
    extract_features,
)


# ── Constants ────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    "dataset",
    "S0_baseline",
    "S1_white",
    "S2_colored",
    "S3_realistic",
    "S4_real_noise",
    "S5_signal_realism",
    "S6_noise_realism",
    "S7_pure_real",
    "S8_colored_low",
    "S9_colored_high",
    "S_union",
]

CLASS_NAMES = ["2um", "4um", "10um"]

# Tab10 palette (matching PCA / t-SNE)
COLOR_ID = "#1f77b4"       # blue  -- in-distribution
COLOR_NOISE = "#d62728"    # red   -- noise / Mahalanobis
COLOR_2UM = "#1f77b4"      # blue
COLOR_4UM = "#ff7f0e"      # orange
COLOR_10UM = "#2ca02c"     # green

METHOD_COLORS = {
    "MSP": "#1f77b4",
    "Energy": "#ff7f0e",
    "ODIN": "#2ca02c",
    "Mahalanobis": "#d62728",
    "Energy_tuned": "#9467bd",
}

BAR_COLORS = [COLOR_2UM, COLOR_4UM, COLOR_10UM]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_bins(vals, target=50):
    if len(vals) == 0 or np.ptp(vals) == 0:
        return 1
    return target


def get_latest_run_id(dataset_name):
    api = wandb.Api()
    runs = api.runs(
        "julien-leboulch03-laas-cnrs/particle-benchmark",
        filters={"display_name": f"Conv1D-{dataset_name}-run1"},
        order="-created_at",
    )
    run_list = list(runs)
    if not run_list:
        raise ValueError(f"No W&B run found for dataset '{dataset_name}'")
    return run_list[0].id


# ── Plot generation ──────────────────────────────────────────────────────────

def generate_and_log_plots(run, model, test_loader, noise_loader, train_loader,
                           device, class_names):
    """Re-compute OOD scores and log all recolored plots to the W&B run."""

    # ── Score computation ──
    print("    Computing MSP + Energy scores...")
    id_logits, id_probas, _ = compute_ood_scores(model, test_loader, device)
    noise_logits, noise_probas, noise_preds = compute_ood_scores(model, noise_loader, device)

    n_id = len(id_logits)
    n_noise = len(noise_logits)

    msp_id = np.max(id_probas, axis=1)
    msp_noise = np.max(noise_probas, axis=1)
    energy_id = -logsumexp(id_logits, axis=1)
    energy_noise = -logsumexp(noise_logits, axis=1)

    print("    Computing ODIN scores...")
    odin_id = compute_odin_scores(model, test_loader, device)
    odin_noise = compute_odin_scores(model, noise_loader, device)

    print("    Computing Mahalanobis scores...")
    maha_id, maha_noise = compute_mahalanobis_scores(
        model, test_loader, noise_loader, device, train_loader=train_loader
    )

    # OOD labels: 1 = in-dist, 0 = noise
    ood_labels = np.concatenate([np.ones(n_id), np.zeros(n_noise)])

    methods = {}
    for name, id_vals, noise_vals in [
        ("MSP", msp_id, msp_noise),
        ("Energy", -energy_id, -energy_noise),
        ("ODIN", odin_id, odin_noise),
        ("Mahalanobis", maha_id, maha_noise),
    ]:
        scores = np.concatenate([id_vals, noise_vals])
        methods[name] = {
            "scores": scores,
            "auroc": roc_auc_score(ood_labels, scores),
            "fpr95": _compute_fpr_at_tpr(ood_labels, scores),
            "aupr": average_precision_score(ood_labels, scores),
            "avg_id": float(np.mean(id_vals)),
            "avg_noise": float(np.mean(noise_vals)),
        }

    # ── 1) Score histograms (4 methods) ──
    print("    Generating histograms...")
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "standard",
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    hist_configs = [
        ("MSP", msp_id, msp_noise, "Max Softmax Probability", "msp"),
        ("Energy", energy_id, energy_noise, "Energy Score (-logsumexp)", "energy"),
        ("ODIN", odin_id, odin_noise, "ODIN Score", "odin"),
        ("Mahalanobis", maha_id, maha_noise, "Mahalanobis Score", "mahalanobis"),
    ]

    for label, id_vals, noise_vals, xlabel, key in hist_configs:
        m = methods[label]
        fig, ax = plt.subplots(figsize=(3.39, 2.10))
        ax.hist(id_vals, bins=_safe_bins(id_vals), alpha=0.6,
                label=f"In-dist (n={n_id})", color="#0072B2", density=True)
        ax.hist(noise_vals, bins=_safe_bins(noise_vals), alpha=0.6,
                label=f"Noise (n={n_noise})", color="#D55E00", density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
        run.log({f"noise_ood/{key}_histogram": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    # ── 2) Prediction distribution bar chart ──
    noise_class_counts = np.bincount(noise_preds, minlength=len(class_names))
    noise_class_pcts = noise_class_counts / n_noise * 100

    fig, ax = plt.subplots(figsize=(3.39, 2.10))
    bars = ax.bar(class_names, noise_class_pcts,
                  color=["#0072B2", "#E69F00", "#009E73"][:len(class_names)],
                  edgecolor="white")
    for bar, pct in zip(bars, noise_class_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("% of noise samples")
    ax.set_ylim(0, max(noise_class_pcts) * 1.2 if max(noise_class_pcts) > 0 else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
    run.log({"noise_ood/prediction_distribution": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # ── 3) Temperature sweep ──
    print("    Temperature sweep...")
    temperatures = [1, 2, 5, 10, 50, 100, 500, 1000]
    msp_aurocs_t, energy_aurocs_t = [], []

    for T in temperatures:
        msp_t_id = np.max(np.exp(id_logits / T) /
                          np.sum(np.exp(id_logits / T), axis=1, keepdims=True), axis=1)
        msp_t_noise = np.max(np.exp(noise_logits / T) /
                             np.sum(np.exp(noise_logits / T), axis=1, keepdims=True), axis=1)
        msp_t_scores = np.concatenate([msp_t_id, msp_t_noise])
        msp_aurocs_t.append(roc_auc_score(ood_labels, msp_t_scores))

        energy_t_id = -T * logsumexp(id_logits / T, axis=1)
        energy_t_noise = -T * logsumexp(noise_logits / T, axis=1)
        energy_t_scores = np.concatenate([-energy_t_id, -energy_t_noise])
        energy_aurocs_t.append(roc_auc_score(ood_labels, energy_t_scores))

    best_msp_idx = int(np.argmax(msp_aurocs_t))
    best_energy_idx = int(np.argmax(energy_aurocs_t))
    best_T_msp = temperatures[best_msp_idx]
    best_T_energy = temperatures[best_energy_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperatures, msp_aurocs_t, "o-", label="MSP", color=METHOD_COLORS["MSP"])
    ax.plot(temperatures, energy_aurocs_t, "s-", label="Energy", color=METHOD_COLORS["Energy"])
    ax.axvline(best_T_msp, color=METHOD_COLORS["MSP"], linestyle="--", alpha=0.4)
    ax.axvline(best_T_energy, color=METHOD_COLORS["Energy"], linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("AUROC")
    ax.set_title("Temperature Scaling Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    run.log({"noise_ood/temperature_sweep": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # ── 4) Energy tuned histogram ──
    if best_T_energy != 1:
        energy_tuned_id = -best_T_energy * logsumexp(id_logits / best_T_energy, axis=1)
        energy_tuned_noise = -best_T_energy * logsumexp(noise_logits / best_T_energy, axis=1)
        neg_energy_tuned = np.concatenate([-energy_tuned_id, -energy_tuned_noise])
        methods["Energy_tuned"] = {
            "scores": neg_energy_tuned,
            "auroc": roc_auc_score(ood_labels, neg_energy_tuned),
            "fpr95": _compute_fpr_at_tpr(ood_labels, neg_energy_tuned),
            "aupr": average_precision_score(ood_labels, neg_energy_tuned),
            "avg_id": float(np.mean(energy_tuned_id)),
            "avg_noise": float(np.mean(energy_tuned_noise)),
        }
        et = methods["Energy_tuned"]

        fig, ax = plt.subplots(figsize=(3.39, 2.10))
        ax.hist(energy_tuned_id, bins=_safe_bins(energy_tuned_id), alpha=0.6,
                label=f"In-dist (n={n_id})", color="#0072B2", density=True)
        ax.hist(energy_tuned_noise, bins=_safe_bins(energy_tuned_noise), alpha=0.6,
                label=f"Noise (n={n_noise})", color="#D55E00", density=True)
        ax.set_xlabel(f"Energy Score (T={best_T_energy})")
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
        run.log({"noise_ood/energy_tuned_histogram": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    # ── 5) ROC curves comparison ──
    print("    Generating ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in methods.items():
        fpr_m, tpr_m, _ = roc_curve(ood_labels, m["scores"])
        ax.plot(fpr_m, tpr_m, label=f"{name} (AUROC={m['auroc']:.3f})",
                color=METHOD_COLORS.get(name, "#333333"), linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection \u2014 ROC Curves Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    run.log({"noise_ood/roc_comparison": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # ── 6) Threshold analysis ──
    best_method = max(methods, key=lambda k: methods[k]["auroc"])
    best_scores = methods[best_method]["scores"]
    fpr_curve, tpr_curve, thresholds = roc_curve(ood_labels, best_scores)
    noise_rejection = 1.0 - fpr_curve

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, tpr_curve[:-1] if len(tpr_curve) > len(thresholds) else tpr_curve,
             label="TPR (ID recall)", color=COLOR_ID)
    nr_plot = noise_rejection[:-1] if len(noise_rejection) > len(thresholds) else noise_rejection
    ax1.plot(thresholds, nr_plot,
             label="Noise rejected", color=COLOR_NOISE)
    ax1.set_xlabel(f"Threshold ({best_method} score)")
    ax1.set_ylabel("Rate")
    ax1.set_title(f"Threshold Analysis ({best_method}, AUROC={methods[best_method]['auroc']:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    run.log({"noise_ood/threshold_analysis": wandb.Image(fig)}, commit=True)
    plt.close(fig)

    print("    All plots logged.")


# ── Per-dataset processing ───────────────────────────────────────────────────

def process_dataset(dataset_name, output_dir, noise_dir, device,
                    batch_size=32, decimate=4):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    model_path = output_dir / f"Conv1D-{dataset_name}-run1" / "best_model.pth"
    if not model_path.exists():
        print(f"  WARNING: model not found at {model_path}, skipping.")
        return

    input_length = RAW_SIGNAL_LENGTH // decimate

    # Transforms
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0, sample_rate_mhz=2.0)
    decimate_t = Decimate(decimate=decimate)
    truncate = Truncate(RAW_SIGNAL_LENGTH)

    # Model
    model = Conv1DClassifier(input_length=input_length, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.train(False)
    print(f"  Loaded model from {model_path}")

    # Data loaders
    data_dir = Path(dataset_name)
    test_dataset = ParticleDataset(
        data_dir / "test", CLASS_NAMES, transforms=[bandpass, decimate_t]
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_dataset = ParticleDataset(
        data_dir / "train", CLASS_NAMES, transforms=[bandpass, decimate_t]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    noise_dataset = ParticleDataset(
        noise_dir.parent, [noise_dir.name],
        transforms=[truncate, bandpass, decimate_t],
    )
    noise_loader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Resume W&B run and log recolored plots
    print(f"  Fetching W&B run ID...")
    run_id = get_latest_run_id(dataset_name)
    print(f"  Resuming run {run_id}...")

    with wandb.init(
        settings=wandb.Settings(init_timeout=180),
        project="particle-benchmark",
        id=run_id,
        resume="must",
    ) as run:
        generate_and_log_plots(
            run, model, test_loader, noise_loader, train_loader,
            device, CLASS_NAMES,
        )

    print(f"  Done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-generate OOD plots with updated color palette."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to process (default: all 12)",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--noise-dir", type=str, default="Noise")
    parser.add_argument("--decimate", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    noise_dir = Path(args.noise_dir)

    for dataset_name in args.datasets:
        process_dataset(
            dataset_name,
            output_dir=output_dir,
            noise_dir=noise_dir,
            device=device,
            batch_size=args.batch_size,
            decimate=args.decimate,
        )

    print(f"\n{'='*60}")
    print("  All OOD plots recolored.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
