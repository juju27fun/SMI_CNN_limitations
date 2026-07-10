"""Training pipeline for 4-class particle classification (2um, 4um, 10um, Noise).

Supports all models from the model zoo via --model flag and logs
structured metrics to W&B.

Usage:
    python train4classes.py --data-dir S1_white_4c --epochs 150 --wandb-offline
    python train4classes.py --model ResNet1D --data-dir S1_white_4c --epochs 150
    python train4classes.py --model InceptionTime1D --data-dir S2_colored_4c --epochs 150
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from torch.utils.data import DataLoader, Subset
from scipy.special import logsumexp
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, silhouette_score, average_precision_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from p0.data import (
    RAW_SIGNAL_LENGTH,
    AmplitudeScale,
    BandpassFilter,
    Decimate,
    GaussianNoise,
    ParticleDataset,
    TimeShift,
    Truncate,
)
from p0.training import evaluate
from p0.models import create_model
from p0.training_utils import (
    compute_model_macs,
    measure_cpu_latency,
    run_training_loop,
    run_post_testing,
    create_optimizer,
    create_scheduler,
    add_common_training_args,
)

CLASS_NAMES = ["2um", "4um", "10um", "Noise"]
PARTICLE_NAMES = ["2um", "4um", "10um"]  # For 3-class comparison


# ──────────────────────────────────────────────
# Feature extraction and visualization
# ──────────────────────────────────────────────
def extract_features(model, loader, device):
    """Extract penultimate-layer features from the model for all samples.

    Hooks into model.feature_layer (all zoo models expose this attribute).
    """
    model.eval()
    all_labels = []
    activations = []

    def hook_fn(m, inp, out):
        activations.append(out.detach().cpu())

    hook = model.feature_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            model(signals)
            all_labels.extend(labels.numpy())

    hook.remove()
    features = torch.cat(activations, dim=0).numpy()
    return features, np.array(all_labels)


def plot_dimensionality_reduction(features, labels, class_names, prefix):
    """Generate PCA and t-SNE scatter plots colored by class. Returns (pca_fig, tsne_fig)."""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    pca_fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1], label=cls, alpha=0.6, s=15)
    ax.set_title(f"PCA - {prefix}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features)

    tsne_fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        mask = labels == i
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=cls, alpha=0.6, s=15)
    ax.set_title(f"t-SNE - {prefix}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()

    return pca_fig, tsne_fig


# ──────────────────────────────────────────────
# OOD scoring functions
# ──────────────────────────────────────────────
def compute_ood_scores(model, loader, device):
    """Forward pass without ground truth. Returns (logits, probas, preds)."""
    model.eval()
    all_logits, all_probas, all_preds = [], [], []
    with torch.no_grad():
        for signals, _ in loader:
            signals = signals.to(device)
            outputs = model(signals)
            probas = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_logits.append(outputs.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    return (np.concatenate(all_logits),
            np.concatenate(all_probas),
            np.concatenate(all_preds))


def compute_odin_scores(model, loader, device, temperature=1000.0, epsilon=0.0012):
    """ODIN OOD detector: temperature scaling + input perturbation (Liang et al., 2018).

    Returns per-sample ODIN scores (higher = more in-distribution).
    """
    model.eval()
    all_scores = []
    for signals, _ in loader:
        signals = signals.to(device).requires_grad_(True)

        outputs = model(signals)
        scaled = outputs / temperature
        log_soft = torch.log_softmax(scaled, dim=1)
        max_log_soft, _ = log_soft.max(dim=1)
        loss = max_log_soft.sum()

        loss.backward()
        gradient = signals.grad.data

        perturbed = signals.data - epsilon * gradient.sign()

        with torch.no_grad():
            outputs_p = model(perturbed)
            scores = torch.softmax(outputs_p / temperature, dim=1).max(dim=1)[0]
            all_scores.append(scores.cpu().numpy())

        signals.requires_grad_(False)

    return np.concatenate(all_scores)


def compute_mahalanobis_scores(model, id_loader, noise_loader, device, train_loader=None):
    """Single-layer Mahalanobis distance OOD detector using model.feature_layer.

    Uses the penultimate feature layer (model.feature_layer) for all zoo models.
    Returns (id_scores, noise_scores) where higher = more in-distribution.
    """
    ref_loader = train_loader if train_loader is not None else id_loader

    ref_feats, ref_labels = extract_features(model, ref_loader, device)
    id_feats, _ = extract_features(model, id_loader, device)
    noise_feats, _ = extract_features(model, noise_loader, device)

    num_classes = int(ref_labels.max()) + 1

    # Class-conditional means
    class_means = []
    for c in range(num_classes):
        mask = ref_labels == c
        class_means.append(ref_feats[mask].mean(axis=0))
    class_means_arr = np.stack(class_means)

    # Tied (shared) covariance matrix with regularization
    centered = ref_feats.astype(np.float64) - class_means_arr[ref_labels.astype(int)]
    cov = np.cov(centered, rowvar=False)
    reg = max(1e-6, 1e-6 * np.trace(cov) / cov.shape[0])
    cov += reg * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)

    # Negative Mahalanobis distance to nearest class centroid
    id_scores = np.full(len(id_feats), -np.inf)
    noise_scores = np.full(len(noise_feats), -np.inf)
    for c in range(num_classes):
        for feats, scores in [(id_feats.astype(np.float64), id_scores),
                              (noise_feats.astype(np.float64), noise_scores)]:
            diff = feats - class_means_arr[c]
            maha = -np.sum(diff @ cov_inv * diff, axis=1)
            np.maximum(scores, maha, out=scores)

    return id_scores, noise_scores


def _compute_fpr_at_tpr(labels, scores, tpr_target=0.95):
    """Compute FPR at a given TPR threshold using the ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return fpr[-1]
    return float(fpr[idx])


def _safe_bins(vals, target=50):
    """Return bin count that won't fail on constant-valued arrays."""
    if len(vals) == 0 or np.ptp(vals) < 1e-10:
        return 1
    return min(target, max(1, int(np.sqrt(len(vals)))))


def run_ood_evaluation(run, model, id_loader, noise_loader, device, class_names,
                       train_loader=None):
    """Quantitative OOD evaluation with 5 methods: MSP, Energy, ODIN,
    Mahalanobis (single-layer via feature_layer), and Energy_tuned.

    Compares in-distribution test samples vs noise samples and logs
    all metrics and visualizations to W&B under the noise_ood prefix.
    """
    print("\n  [noise_ood] Computing OOD scores...")

    # --- Score computation (MSP + Energy) ---
    id_logits, id_probas, _ = compute_ood_scores(model, id_loader, device)
    noise_logits, noise_probas, noise_preds = compute_ood_scores(model, noise_loader, device)

    n_id = len(id_logits)
    n_noise = len(noise_logits)

    # Max Softmax Probability
    msp_id = np.max(id_probas, axis=1)
    msp_noise = np.max(noise_probas, axis=1)

    # Energy score: -logsumexp(logits)
    energy_id = -logsumexp(id_logits, axis=1)
    energy_noise = -logsumexp(noise_logits, axis=1)

    # Entropy
    eps = 1e-8
    entropy_id = -np.sum(id_probas * np.log(id_probas + eps), axis=1)
    entropy_noise = -np.sum(noise_probas * np.log(noise_probas + eps), axis=1)

    # --- ODIN scores ---
    print("  [noise_ood] Computing ODIN scores...")
    odin_id = compute_odin_scores(model, id_loader, device)
    odin_noise = compute_odin_scores(model, noise_loader, device)

    # --- Mahalanobis scores ---
    print("  [noise_ood] Computing Mahalanobis scores...")
    maha_id, maha_noise = compute_mahalanobis_scores(
        model, id_loader, noise_loader, device, train_loader=train_loader
    )

    # --- AUROC, FPR@95, AUPR for all 4 methods ---
    # Labels: 1 = in-distribution, 0 = OOD (noise)
    ood_labels = np.concatenate([np.ones(n_id), np.zeros(n_noise)])

    methods = {}

    # MSP
    msp_scores = np.concatenate([msp_id, msp_noise])
    methods["MSP"] = {
        "scores": msp_scores,
        "auroc": roc_auc_score(ood_labels, msp_scores),
        "fpr95": _compute_fpr_at_tpr(ood_labels, msp_scores),
        "aupr": average_precision_score(ood_labels, msp_scores),
        "avg_id": float(np.mean(msp_id)),
        "avg_noise": float(np.mean(msp_noise)),
    }

    # Energy (negate: lower = more in-distribution)
    energy_scores = np.concatenate([energy_id, energy_noise])
    neg_energy = -energy_scores
    methods["Energy"] = {
        "scores": neg_energy,
        "auroc": roc_auc_score(ood_labels, neg_energy),
        "fpr95": _compute_fpr_at_tpr(ood_labels, neg_energy),
        "aupr": average_precision_score(ood_labels, neg_energy),
        "avg_id": float(np.mean(energy_id)),
        "avg_noise": float(np.mean(energy_noise)),
    }

    # ODIN
    odin_scores = np.concatenate([odin_id, odin_noise])
    methods["ODIN"] = {
        "scores": odin_scores,
        "auroc": roc_auc_score(ood_labels, odin_scores),
        "fpr95": _compute_fpr_at_tpr(ood_labels, odin_scores),
        "aupr": average_precision_score(ood_labels, odin_scores),
        "avg_id": float(np.mean(odin_id)),
        "avg_noise": float(np.mean(odin_noise)),
    }

    # Mahalanobis
    maha_scores = np.concatenate([maha_id, maha_noise])
    methods["Mahalanobis"] = {
        "scores": maha_scores,
        "auroc": roc_auc_score(ood_labels, maha_scores),
        "fpr95": _compute_fpr_at_tpr(ood_labels, maha_scores),
        "aupr": average_precision_score(ood_labels, maha_scores),
        "avg_id": float(np.mean(maha_id)),
        "avg_noise": float(np.mean(maha_noise)),
    }

    # --- Averages ---
    avg_entropy_id = float(np.mean(entropy_id))
    avg_entropy_noise = float(np.mean(entropy_noise))

    # --- Prediction distribution on noise ---
    noise_class_counts = [int(np.sum(noise_preds == i)) for i in range(len(class_names))]
    noise_class_pcts = [c / n_noise * 100 for c in noise_class_counts]

    # --- Print results ---
    print(f"  [noise_ood] Samples: {n_id} in-dist, {n_noise} noise")
    for name, m in methods.items():
        print(f"  [noise_ood] {name:12s} AUROC: {m['auroc']:.4f} | "
              f"FPR@95: {m['fpr95']:.4f} | AUPR: {m['aupr']:.4f}")
    print(f"  [noise_ood] Avg Entropy: ID={avg_entropy_id:.4f}, Noise={avg_entropy_noise:.4f}")
    print(f"  [noise_ood] Noise prediction distribution:")
    for cls, cnt, pct in zip(class_names, noise_class_counts, noise_class_pcts):
        print(f"    {cls}: {cnt} ({pct:.1f}%)")

    # --- W&B summary scalars ---
    for name, m in methods.items():
        key = name.lower()
        run.summary[f"noise_ood/auroc_{key}"] = m["auroc"]
        run.summary[f"noise_ood/fpr95_{key}"] = m["fpr95"]
        run.summary[f"noise_ood/aupr_{key}"] = m["aupr"]

    run.summary["noise_ood/avg_max_softmax_id"] = methods["MSP"]["avg_id"]
    run.summary["noise_ood/avg_max_softmax_noise"] = methods["MSP"]["avg_noise"]
    run.summary["noise_ood/avg_entropy_id"] = avg_entropy_id
    run.summary["noise_ood/avg_entropy_noise"] = avg_entropy_noise
    run.summary["noise_ood/num_noise_samples"] = n_noise

    # --- W&B plots ---

    # Score histograms for all 4 methods
    hist_configs = [
        ("MSP", msp_id, msp_noise, "Max Softmax Probability", "msp"),
        ("Energy", energy_id, energy_noise, "Energy Score (-logsumexp)", "energy"),
        ("ODIN", odin_id, odin_noise, "ODIN Score", "odin"),
        ("Mahalanobis", maha_id, maha_noise, "Mahalanobis Score", "mahalanobis"),
    ]

    for label, id_vals, noise_vals, xlabel, key in hist_configs:
        m = methods[label]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(id_vals, bins=_safe_bins(id_vals), alpha=0.6,
                label=f"In-dist (n={n_id})", color="#1f77b4", density=True)
        ax.hist(noise_vals, bins=_safe_bins(noise_vals), alpha=0.6,
                label=f"Noise (n={n_noise})", color="#d62728", density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution (AUROC={m['auroc']:.3f})")
        ax.legend()
        run.log({f"noise_ood/{key}_histogram": wandb.Image(fig)})
        plt.close(fig)

    # Prediction distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(class_names, noise_class_pcts,
                  color=bar_colors[:len(class_names)], edgecolor="white")
    for bar, pct in zip(bars, noise_class_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("% of noise samples")
    ax.set_title(f"Model predictions on noise (n={n_noise})")
    ax.set_ylim(0, max(noise_class_pcts) * 1.2 if max(noise_class_pcts) > 0 else 100)
    run.log({"noise_ood/prediction_distribution": wandb.Image(fig)})
    plt.close(fig)

    # --- Temperature scaling sweep ---
    print("  [noise_ood] Running temperature scaling sweep...")
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

    print(f"  [noise_ood] Best T (MSP): T={best_T_msp} "
          f"(AUROC={msp_aurocs_t[best_msp_idx]:.4f})")
    print(f"  [noise_ood] Best T (Energy): T={best_T_energy} "
          f"(AUROC={energy_aurocs_t[best_energy_idx]:.4f})")

    run.summary["noise_ood/best_temperature_msp"] = best_T_msp
    run.summary["noise_ood/best_temperature_msp_auroc"] = msp_aurocs_t[best_msp_idx]
    run.summary["noise_ood/best_temperature_energy"] = best_T_energy
    run.summary["noise_ood/best_temperature_energy_auroc"] = energy_aurocs_t[best_energy_idx]

    # Temperature sweep plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperatures, msp_aurocs_t, "o-", label="MSP", color="#1f77b4")
    ax.plot(temperatures, energy_aurocs_t, "s-", label="Energy", color="#ff7f0e")
    ax.axvline(best_T_msp, color="#1f77b4", linestyle="--", alpha=0.4)
    ax.axvline(best_T_energy, color="#ff7f0e", linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("AUROC")
    ax.set_title("Temperature Scaling Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    run.log({"noise_ood/temperature_sweep": wandb.Image(fig)})
    plt.close(fig)

    # --- Re-evaluate Energy with optimal temperature ---
    if best_T_energy != 1:
        print(f"  [noise_ood] Re-evaluating Energy with optimal T={best_T_energy}...")
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
        print(f"  [noise_ood] {'Energy_tuned':12s} AUROC: {et['auroc']:.4f} | "
              f"FPR@95: {et['fpr95']:.4f} | AUPR: {et['aupr']:.4f}")

        run.summary["noise_ood/auroc_energy_tuned"] = et["auroc"]
        run.summary["noise_ood/fpr95_energy_tuned"] = et["fpr95"]
        run.summary["noise_ood/aupr_energy_tuned"] = et["aupr"]

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(energy_tuned_id, bins=_safe_bins(energy_tuned_id), alpha=0.6,
                label=f"In-dist (n={n_id})", color="#1f77b4", density=True)
        ax.hist(energy_tuned_noise, bins=_safe_bins(energy_tuned_noise), alpha=0.6,
                label=f"Noise (n={n_noise})", color="#d62728", density=True)
        ax.set_xlabel(f"Energy Score (T={best_T_energy})")
        ax.set_ylabel("Density")
        ax.set_title(f"Energy Tuned Distribution (AUROC={et['auroc']:.3f})")
        ax.legend()
        run.log({"noise_ood/energy_tuned_histogram": wandb.Image(fig)})
        plt.close(fig)

    # --- Overlaid ROC curves for all methods ---
    print("  [noise_ood] Plotting ROC curves comparison...")
    method_colors = {
        "MSP": "#1f77b4", "Energy": "#ff7f0e", "ODIN": "#2ca02c",
        "Mahalanobis": "#d62728", "Energy_tuned": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, m in methods.items():
        fpr_m, tpr_m, _ = roc_curve(ood_labels, m["scores"])
        ax.plot(fpr_m, tpr_m, label=f"{name} (AUROC={m['auroc']:.3f})",
                color=method_colors.get(name, "#333333"), linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection \u2014 ROC Curves Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    run.log({"noise_ood/roc_comparison": wandb.Image(fig)})
    plt.close(fig)

    # --- Threshold sweep analysis (using best method) ---
    print("  [noise_ood] Running threshold analysis...")
    best_method = max(methods, key=lambda k: methods[k]["auroc"])
    best_scores = methods[best_method]["scores"]

    fpr_curve, tpr_curve, thresholds = roc_curve(ood_labels, best_scores)
    noise_rejection = 1.0 - fpr_curve

    # Operating points table
    target_tprs = [0.90, 0.95, 0.99]
    op_rows = []
    for target in target_tprs:
        idx = np.searchsorted(tpr_curve, target)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1
        op_rows.append([
            f"TPR={target:.0%}",
            round(float(thresholds[idx]) if idx < len(thresholds) else float("nan"), 4),
            round(float(tpr_curve[idx]), 4),
            round(float(fpr_curve[idx]), 4),
            round(float(noise_rejection[idx]) * 100, 1),
        ])

    run.log({
        "noise_ood/operating_points": wandb.Table(
            columns=["Target", "Threshold", "TPR", "FPR", "Noise_Rejected_Pct"],
            data=op_rows,
        )
    })

    # Threshold analysis plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, tpr_curve[:-1] if len(tpr_curve) > len(thresholds) else tpr_curve,
             label="TPR (ID recall)", color="#1f77b4")
    nr_plot = noise_rejection[:-1] if len(noise_rejection) > len(thresholds) else noise_rejection
    ax1.plot(thresholds, nr_plot,
             label="Noise rejected", color="#d62728")
    ax1.set_xlabel(f"Threshold ({best_method} score)")
    ax1.set_ylabel("Rate")
    ax1.set_title(f"Threshold Analysis ({best_method}, AUROC={methods[best_method]['auroc']:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    run.log({"noise_ood/threshold_analysis": wandb.Image(fig)})
    plt.close(fig)

    print(f"  [noise_ood] Best method: {best_method} (AUROC={methods[best_method]['auroc']:.4f})")
    for row in op_rows:
        print(f"    {row[0]}: threshold={row[1]}, FPR={row[3]}, noise rejected={row[4]}%")

    # --- Per-class OOD analysis ---
    print("  [noise_ood] Per-class OOD analysis...")
    _, id_labels_raw = extract_features(model, id_loader, device)
    per_class_rows = []
    for c, cls in enumerate(class_names):
        mask = id_labels_raw == c
        class_msp = msp_id[mask]
        class_labels = np.concatenate([np.ones(mask.sum()), np.zeros(n_noise)])
        class_scores = np.concatenate([class_msp, msp_noise])
        if mask.sum() > 0:
            class_auroc = roc_auc_score(class_labels, class_scores)
        else:
            class_auroc = float("nan")
        per_class_rows.append([
            cls, int(mask.sum()),
            round(float(np.mean(class_msp)), 4),
            round(float(np.std(class_msp)), 4),
            round(class_auroc, 4),
        ])
        print(f"    {cls}: n={mask.sum()}, avg_msp={np.mean(class_msp):.4f}, "
              f"auroc_vs_noise={class_auroc:.4f}")

    run.log({
        "noise_ood/per_class_analysis": wandb.Table(
            columns=["Class", "N_samples", "Avg_MSP", "Std_MSP", "AUROC_vs_Noise"],
            data=per_class_rows,
        )
    })

    # --- W&B summary table (all methods) ---
    run.log({
        "noise_ood/summary_table": wandb.Table(
            columns=["Method", "AUROC", "AUPR", "FPR_at_95TPR",
                     "Avg_Score_ID", "Avg_Score_Noise"],
            data=[
                [name, round(m["auroc"], 4), round(m["aupr"], 4),
                 round(m["fpr95"], 4), round(m["avg_id"], 4), round(m["avg_noise"], 4)]
                for name, m in methods.items()
            ]
        )
    })

    # --- Latent space separability (silhouette score) ---
    print("  [noise_ood] Computing latent space separability...")
    id_features, _ = extract_features(model, id_loader, device)
    noise_features, _ = extract_features(model, noise_loader, device)
    combined_features = np.concatenate([id_features, noise_features])
    combined_labels = np.concatenate([np.zeros(len(id_features)),
                                      np.ones(len(noise_features))])
    sil_score = silhouette_score(combined_features, combined_labels)
    run.summary["noise_ood/silhouette_score"] = sil_score
    print(f"  [noise_ood] Silhouette score: {sil_score:.4f}")


# ──────────────────────────────────────────────
# Cluster distance evaluation
# ──────────────────────────────────────────────
def cosine_distance(a, b):
    """Cosine distance between two vectors: 1 - cos(theta) in [0, 2]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def compute_distance_matrix(centroids):
    """Compute the full pairwise cosine distance matrix between centroids."""
    n = len(centroids)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            d = cosine_distance(centroids[i], centroids[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


def run_cluster_distance_evaluation(run, test_feats, test_labels, noise_feats,
                                    class_names, dataset_name):
    """Compute pairwise cosine distances between cluster centroids and log to W&B.

    Uses pre-extracted feature_layer features to compute centroids for each particle
    class plus noise, then logs a lower-triangular table, scalar summaries, and a heatmap.
    """
    cluster_names = class_names + ["Noise"]
    n = len(cluster_names)

    # Compute per-cluster centroids
    centroids = []
    for c, cls in enumerate(class_names):
        mask = test_labels == c
        n_samples = int(mask.sum())
        if n_samples == 0:
            print(f"  WARNING: no samples for class {cls}, using zero centroid")
            centroids.append(np.zeros(test_feats.shape[1]))
        else:
            centroids.append(test_feats[mask].mean(axis=0))
        print(f"  [cluster_distances] Centroid {cls:5s}: {n_samples} samples")

    noise_centroid = noise_feats.mean(axis=0)
    centroids.append(noise_centroid)
    print(f"  [cluster_distances] Centroid Noise: {len(noise_feats)} samples")

    # Compute distance matrix
    mat = compute_distance_matrix(centroids)

    # Pretty-print lower triangle
    print(f"\n  Cosine distance matrix (lower triangle):")
    header = f"  {'':10s}" + "".join(f"{name:10s}" for name in cluster_names)
    print(header)
    for i, row_name in enumerate(cluster_names):
        row_str = f"  {row_name:10s}"
        for j in range(i + 1):
            row_str += f"{mat[i, j]:10.4f}"
        print(row_str)

    # W&B Table (lower-triangular)
    columns = ["cluster"] + cluster_names
    rows = []
    for i, row_name in enumerate(cluster_names):
        row = [row_name]
        for j in range(n):
            if j < i:
                row.append(round(float(mat[i, j]), 4))
            elif j == i:
                row.append(0.0)
            else:
                row.append(None)
        rows.append(row)

    run.log({
        "cluster_distances/cosine_distance_table": wandb.Table(
            columns=columns, data=rows
        )
    })

    # Scalar summary entries for each pairwise distance
    for i in range(n):
        for j in range(i):
            key = f"cluster_distances/cosine_{cluster_names[i].lower()}_vs_{cluster_names[j].lower()}"
            run.summary[key] = round(float(mat[i, j]), 4)

    # Heatmap
    display = mat.copy()
    masked = np.ma.array(display, mask=np.isnan(display))

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine distance")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cluster_names, rotation=45, ha="right")
    ax.set_yticklabels(cluster_names)
    ax.set_title(f"Cluster Cosine Distances \u2014 {dataset_name}")

    for i in range(n):
        for j in range(n):
            val = display[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color)

    fig.tight_layout()
    run.log({"cluster_distances/cosine_distance_heatmap": wandb.Image(fig)})
    plt.close(fig)
    print("  [cluster_distances] Table, scalars, and heatmap logged to W&B")


# ──────────────────────────────────────────────
# 3-class comparison evaluation
# ──────────────────────────────────────────────
def run_3class_evaluation(run, model, loader, criterion, device):
    """Evaluate 4-class model on particle classes only (2um, 4um, 10um).

    Filters the test set to exclude Noise samples, then computes confusion
    matrix and F1 per class for direct comparison with a 3-class model.
    """
    prefix = "Charts_3class"
    print(f"\n  [{prefix}] Evaluating on 3 particle classes only...")

    _, _, y_pred, y_true, _ = evaluate(model, loader, criterion, device)

    # Filter to particle classes only (0=2um, 1=4um, 2=10um)
    mask = np.isin(y_true, [0, 1, 2])
    y_true_3c = y_true[mask]
    y_pred_3c = y_pred[mask]
    n_as_noise = int(np.sum(y_pred_3c == 3))
    acc_3c = float(np.mean(y_pred_3c == y_true_3c))

    print(f"  [{prefix}] {len(y_true_3c)} particle samples, "
          f"{n_as_noise} predicted as Noise")
    print(f"  [{prefix}] Accuracy (particle classes): {acc_3c:.4f}")

    run.summary[f"{prefix}/accuracy"] = acc_3c

    # Confusion matrix (3x3, predictions of class Noise excluded from columns)
    from p0.plotting import plot_confusion_matrix
    cm = confusion_matrix(y_true_3c, y_pred_3c, labels=[0, 1, 2])
    fig_cm, _ = plot_confusion_matrix(cm, PARTICLE_NAMES)
    run.log({f"{prefix}/confusion_matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)

    # Interactive confusion matrix (Charts tab)
    run.log({f"{prefix}/confusion_matrix_chart": wandb.plot.confusion_matrix(
        y_true=y_true_3c.tolist(), preds=y_pred_3c.tolist(),
        class_names=PARTICLE_NAMES
    )})

    # Classification report — F1 properly penalizes Noise predictions in recall
    report = classification_report(
        y_true_3c, y_pred_3c, labels=[0, 1, 2],
        target_names=PARTICLE_NAMES, output_dict=True,
    )
    print(classification_report(
        y_true_3c, y_pred_3c, labels=[0, 1, 2],
        target_names=PARTICLE_NAMES, digits=4,
    ))

    # F1 per class table
    rows = []
    for cls in PARTICLE_NAMES:
        rows.append([
            cls,
            round(report[cls]["precision"], 4),
            round(report[cls]["recall"], 4),
            round(report[cls]["f1-score"], 4),
            int(report[cls]["support"]),
        ])
    run.log({
        f"{prefix}/f1_per_class": wandb.Table(
            columns=["Class", "Precision", "Recall", "F1", "Support"],
            data=rows,
        )
    })

    # F1 bar chart
    f1_data = [[cls, report[cls]["f1-score"]] for cls in PARTICLE_NAMES]
    f1_table = wandb.Table(data=f1_data, columns=["class", "f1"])
    run.log({
        f"{prefix}/f1_bar_chart": wandb.plot.bar(
            f1_table, "class", "f1", title="F1 per Class (3-class comparison)"
        )
    })

    if n_as_noise > 0:
        print(f"  [{prefix}] Note: {n_as_noise} particle samples predicted "
              f"as Noise (excluded from CM columns)")

    return acc_3c


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train 4-class classifier (2um, 4um, 10um, Noise)"
    )
    add_common_training_args(parser, data_dir_default="data/S1_white_4c")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Path to noise samples for OOD evaluation (default: None)")
    parser.add_argument("--real-test-dir", type=str, default=None,
                        help="Path to real test set for generalization gap (default: None)")
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = Path(args.data_dir).name

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    run_name = f"{args.model}-{args.dataset_name}-{args.run_id}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_length = RAW_SIGNAL_LENGTH // args.decimate

    # Transforms
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0, sample_rate_mhz=2.0)
    decimate_transform = Decimate(decimate=args.decimate)
    base_transforms = [bandpass, decimate_transform]

    # Build augmentation transforms (training only)
    aug_transforms = []
    if args.augment:
        aug_transforms = [
            GaussianNoise(snr_db=args.aug_snr),
            TimeShift(max_shift_frac=args.aug_shift),
            AmplitudeScale(scale_min=args.aug_scale_min, scale_max=args.aug_scale_max),
        ]
        print(f"Augmentation enabled: GaussianNoise(snr={args.aug_snr}dB), "
              f"TimeShift(max={args.aug_shift}), "
              f"AmplitudeScale([{args.aug_scale_min}, {args.aug_scale_max}])")

    # Datasets (4 classes)
    # Val dataset uses base transforms only; train dataset adds augmentation if enabled
    val_dataset = ParticleDataset(
        data_dir / "train", CLASS_NAMES, transforms=base_transforms
    )
    train_dataset = ParticleDataset(
        data_dir / "train", CLASS_NAMES, transforms=base_transforms + aug_transforms
    )
    test_dataset = ParticleDataset(
        data_dir / "test", CLASS_NAMES, transforms=base_transforms
    )

    # Split indices once, then apply to both datasets
    total_size = len(val_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(args.seed))
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"Dataset: {args.dataset_name} (4 classes)")
    print(f"  Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Real test set (optional, for generalization gap)
    real_test_loader = None
    if args.real_test_dir is not None:
        real_test_dir = Path(args.real_test_dir)
        if real_test_dir.exists():
            real_test_dataset = ParticleDataset(
                real_test_dir, CLASS_NAMES, transforms=[bandpass, decimate_transform]
            )
            real_test_loader = DataLoader(
                real_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            print(f"Real test set loaded: {len(real_test_dataset)} samples")
        else:
            print(f"WARNING: real test dir not found: {real_test_dir}")

    # Noise samples (optional, for OOD evaluation and cluster distances)
    noise_loader = None
    if args.noise_dir is not None:
        noise_dir = Path(args.noise_dir)
        if noise_dir.exists():
            truncate = Truncate(RAW_SIGNAL_LENGTH)
            noise_dataset = ParticleDataset(
                noise_dir.parent, [noise_dir.name],
                transforms=[truncate, bandpass, decimate_transform],
            )
            noise_loader = DataLoader(
                noise_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            print(f"Noise samples loaded: {len(noise_dataset)} samples")
        else:
            print(f"WARNING: noise dir not found: {noise_dir}")

    # Model (4 classes)
    model = create_model(
        args.model, input_length=input_length, num_classes=len(CLASS_NAMES)
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_macs = compute_model_macs(model, (1, 1, input_length), device)
    macs_str = f", {model_macs:,} MACs" if model_macs is not None else ""
    print(f"Model: {args.model} ({num_params:,} params{macs_str}, {len(CLASS_NAMES)} classes)")

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)

    # ── W&B Init ──
    wandb_mode = "offline" if args.wandb_offline else "online"
    config = {
        "model_name": args.model,
        "model_size_params": num_params,
        "dataset": args.dataset_name,
        "dataset_size": train_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "patience": args.patience,
        "num_classes": len(CLASS_NAMES),
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "decimate": args.decimate,
        "input_length": input_length,
        "model_macs": model_macs,
        "val_split": args.val_split,
        "convergence_threshold": args.convergence_threshold,
        "augment": args.augment,
    }

    run = wandb.init(
        project="particle-benchmark",
        config=config,
        group=args.model,
        tags=[args.dataset_name, "4class"],
        name=run_name,
        job_type="training",
        mode=wandb_mode,
    )

    try:
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("val/accuracy", summary="max", goal="maximize")
        run.define_metric("val/loss", summary="min", goal="minimize")

        run.summary["model_size_params"] = num_params
        run.summary["dataset_size"] = train_size
        if model_macs is not None:
            run.summary["model_macs"] = model_macs

        # ── Training ──
        best_val_acc, best_epoch, total_time, convergence_time = run_training_loop(
            run, model, train_loader, val_loader, criterion, optimizer, device, args,
            output_dir=output_dir, scheduler=scheduler,
        )

        # ── Post-training testing (Charts) ──
        print("\n" + "=" * 60)
        print("Post-training testing (best model, 4 classes)")
        print("=" * 60)

        model.load_state_dict(
            torch.load(output_dir / "best_model.pth", weights_only=True)
        )

        synth_acc, _ = run_post_testing(run, model, test_loader, criterion, device, CLASS_NAMES)

        # ── 3-class comparison (particle classes only) ──
        print("\n" + "=" * 60)
        print("3-class comparison (particle classes only)")
        print("=" * 60)
        run_3class_evaluation(run, model, test_loader, criterion, device)

        # ── Inference latency (CPU canonical, see docs/metrics_conventions.md) ──
        latency = measure_cpu_latency(model, (1, 1, input_length))
        run.summary["inference_latency_median_ms"] = latency["median_ms"]
        run.summary["inference_latency_p95_ms"] = latency["p95_ms"]
        run.summary["latency_device"] = latency["latency_device"]
        print(f"  Inference latency (CPU): {latency['median_ms']:.2f} ms/sample")

        # Real test set evaluation (generalization gap)
        if real_test_loader is not None:
            print("\n" + "=" * 60)
            print("Real test set evaluation")
            print("=" * 60)
            real_acc, real_loss = run_post_testing(
                run, model, real_test_loader, criterion, device, CLASS_NAMES
            )
            # Override prefix to test_real for W&B
            run.summary["test_real/accuracy"] = real_acc
            run.summary["test_real/loss"] = real_loss
            gap = synth_acc - real_acc
            run.summary["generalization_gap"] = gap
            print(f"\n  Generalization gap (synthetic - real): {gap:+.4f}")

        # Save model as W&B artifact
        run.log_model(
            path=str(output_dir / "best_model.pth"),
            name=f"{args.model}-{args.dataset_name}-4class",
        )

        # ── Noise separation visualization (latent space) ──
        print("\n" + "=" * 60)
        print("Latent space visualization (4 classes)")
        print("=" * 60)

        noise_test_feats = None
        noise_test_labels = None
        noise_feats = None

        features, feat_labels = extract_features(model, test_loader, device)
        pca_fig, tsne_fig = plot_dimensionality_reduction(
            features, feat_labels, CLASS_NAMES, "noise_separation"
        )
        run.log({
            "noise_separation/pca": wandb.Image(pca_fig),
            "noise_separation/tsne": wandb.Image(tsne_fig),
        })
        plt.close(pca_fig)
        plt.close(tsne_fig)
        print("  [noise_separation] PCA and t-SNE logged to W&B")

        # Extra visualization: test + noise in shared latent space
        if noise_loader is not None:
            print("\n  Noise OOD separation analysis...")
            # Use 3-class test subset (exclude Noise class from test set)
            id_class_names = [c for c in CLASS_NAMES if c != "Noise"]
            noise_test_feats, noise_test_labels = extract_features(model, test_loader, device)
            noise_feats, _ = extract_features(model, noise_loader, device)

            combined_features = np.concatenate([noise_test_feats, noise_feats])
            noise_labels = np.full(len(noise_feats), len(CLASS_NAMES))
            combined_labels = np.concatenate([noise_test_labels, noise_labels])
            combined_names = CLASS_NAMES + ["OOD_Noise"]

            pca_fig, tsne_fig = plot_dimensionality_reduction(
                combined_features, combined_labels, combined_names, "ood_separation"
            )
            run.log({
                "ood_separation/pca": wandb.Image(pca_fig),
                "ood_separation/tsne": wandb.Image(tsne_fig),
            })
            plt.close(pca_fig)
            plt.close(tsne_fig)
            print("  [ood_separation] PCA and t-SNE logged to W&B")

        # ── OOD Noise Evaluation ──
        if noise_loader is not None:
            print("\n" + "=" * 60)
            print("OOD Noise Evaluation")
            print("=" * 60)
            run_ood_evaluation(
                run, model, test_loader, noise_loader, device, CLASS_NAMES,
                train_loader=train_loader,
            )

        # ── Cluster Distances ──
        if noise_loader is not None:
            print("\n" + "=" * 60)
            print("Cluster Distances")
            print("=" * 60)
            if noise_test_feats is None:
                noise_test_feats, noise_test_labels = extract_features(
                    model, test_loader, device
                )
                noise_feats, _ = extract_features(model, noise_loader, device)
            run_cluster_distance_evaluation(
                run, noise_test_feats, noise_test_labels, noise_feats,
                CLASS_NAMES, dataset_name=args.dataset_name,
            )

        print("\n" + "=" * 60)
        print("4-class training complete.")
        print("=" * 60)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
