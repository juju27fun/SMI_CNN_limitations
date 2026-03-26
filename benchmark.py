"""Benchmark pipeline for particle classification.

Wraps the training loop with structured W&B metric logging,
evaluates on both synthetic and real test sets.

Usage:
    python benchmark.py --data-dir dataset --real-test-dir dataset_real/test
    python benchmark.py --data-dir dataset  # without real test set
"""

import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from scipy.special import logsumexp
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             silhouette_score, average_precision_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train import (
    RAW_SIGNAL_LENGTH,
    ParticleDataset,
    Conv1DClassifier,
    BandpassFilter,
    Decimate,
    Truncate,
    train_one_epoch,
    evaluate,
)



# ──────────────────────────────────────────────
# Phase 1 : Pre-training logging
# ──────────────────────────────────────────────
def log_pre_training(run, num_params, args, train_size, val_size, class_names):
    """Log all pre-training parameters to W&B and stdout."""
    print("=" * 60)
    print("PHASE 1 : Pre-training parameters")
    print("=" * 60)
    print(f"  Model:            Conv1DClassifier")
    print(f"  Model size:       {num_params:,} trainable params")
    print(f"  Dataset:          {args.dataset_name}")
    print(f"  Dataset size:     {train_size} train + {val_size} val")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Optimizer:        Adam (weight_decay=1e-4)")
    print(f"  LR scheduler:     {args.scheduler}")
    print(f"  Decimation:       {args.decimate}x")
    print(f"  Input length:     {RAW_SIGNAL_LENGTH // args.decimate}")
    print(f"  Classes:          {class_names}")
    print(f"  Convergence thr:  {args.convergence_threshold:.0%}")
    print("=" * 60)

    run.summary["model_size_params"] = num_params
    run.summary["dataset_size"] = train_size


# ──────────────────────────────────────────────
# Phase 2 : Training loop with metrics
# ──────────────────────────────────────────────
def run_training_loop(run, model, train_loader, val_loader, criterion, optimizer,
                      device, args, output_dir, scheduler=None):
    """Full training loop with per-epoch W&B logging.

    Returns: (best_val_acc, best_epoch, total_time, convergence_time)
    """
    best_val_acc = 0.0
    best_epoch = 0
    convergence_time = None
    epochs_without_improvement = 0
    val_acc = 0.0
    val_loss = float("nan")
    total_start = time.time()

    print("\n" + "=" * 60)
    print("PHASE 2 : Training")
    print("=" * 60)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]

        # W&B per-epoch log
        run.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch_time_sec": epoch_time,
            "learning_rate": current_lr,
        })

        # Step the learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            run.summary["best_val_accuracy"] = best_val_acc
            run.summary["best_epoch"] = best_epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_without_improvement += 1

        # Track convergence
        if convergence_time is None and val_acc >= args.convergence_threshold:
            convergence_time = time.time() - total_start
            run.summary["convergence_time_sec"] = convergence_time

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"  Early stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
            run.summary["early_stopped_epoch"] = epoch
            break

    total_time = time.time() - total_start

    # Summary
    run.summary["total_training_time_sec"] = total_time
    run.summary["final_val_accuracy"] = val_acc
    run.summary["final_val_loss"] = val_loss
    if convergence_time is None:
        run.summary["convergence_time_sec"] = float("nan")  # never converged

    print("-" * 60)
    print(f"  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
    print(f"  Total training time: {total_time:.1f}s")
    if convergence_time is not None:
        print(f"  Convergence time ({args.convergence_threshold:.0%}): {convergence_time:.1f}s")
    else:
        print(f"  Convergence ({args.convergence_threshold:.0%}): NOT REACHED")

    return best_val_acc, best_epoch, total_time, convergence_time


# ──────────────────────────────────────────────
# Phase 3 : Post-training full evaluation
# ──────────────────────────────────────────────
def run_post_evaluation(run, model, loader, criterion, device, class_names, prefix):
    """Run full evaluation on a test set and log everything to W&B.

    Args:
        prefix: metric namespace, e.g. "test_synthetic" or "test_real"
    """
    print(f"\n  [{prefix}] Running evaluation...")
    loss, acc, y_pred, y_true, y_proba = evaluate(model, loader, criterion, device)

    print(f"  [{prefix}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # Summary scalars
    run.summary[f"{prefix}/accuracy"] = acc
    run.summary[f"{prefix}/loss"] = loss

    # Confusion matrix
    cm = wandb.plot.confusion_matrix(
        y_true=y_true.tolist(),
        preds=y_pred.tolist(),
        class_names=class_names,
    )
    run.log({f"{prefix}/confusion_matrix": cm})

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    print(f"  [{prefix}] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # F1 per class table
    rows = []
    for cls in class_names:
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
    f1_data = [[cls, report[cls]["f1-score"]] for cls in class_names]
    f1_table = wandb.Table(data=f1_data, columns=["class", "f1"])
    run.log({
        f"{prefix}/f1_bar_chart": wandb.plot.bar(
            f1_table, "class", "f1", title=f"F1 per Class ({prefix})"
        )
    })

    # PR curve
    run.log({
        f"{prefix}/pr_curve": wandb.plot.pr_curve(
            y_true.tolist(), y_proba.tolist(), labels=class_names
        )
    })

    # ROC curve
    run.log({
        f"{prefix}/roc_curve": wandb.plot.roc_curve(
            y_true.tolist(), y_proba.tolist(), labels=class_names
        )
    })

    return acc, loss


# ──────────────────────────────────────────────
# OOD Noise Evaluation
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

        # Forward pass with gradients
        outputs = model(signals)
        scaled = outputs / temperature
        log_soft = torch.log_softmax(scaled, dim=1)
        max_log_soft, _ = log_soft.max(dim=1)
        loss = max_log_soft.sum()

        # Gradient w.r.t. input
        loss.backward()
        gradient = signals.grad.data

        # Perturb input in the direction that increases confidence
        perturbed = signals.data - epsilon * gradient.sign()

        # Second forward pass on perturbed input (no grad)
        with torch.no_grad():
            outputs_p = model(perturbed)
            scores = torch.softmax(outputs_p / temperature, dim=1).max(dim=1)[0]
            all_scores.append(scores.cpu().numpy())

        signals.requires_grad_(False)

    return np.concatenate(all_scores)


def compute_mahalanobis_scores(model, id_loader, noise_loader, device, train_loader=None):
    """Multi-layer Mahalanobis distance OOD detector (Lee et al., 2018).

    Computes class-conditional Gaussian at each layer (pool1, pool2, pool3, fc1)
    using global average pooling for conv layers. Final score is the sum across
    all 4 layers, following the ensemble approach from the original paper.
    Returns (id_scores, noise_scores) where higher = more in-distribution.
    """
    ref_loader = train_loader if train_loader is not None else id_loader

    ref_layers, ref_labels = extract_multilayer_features(model, ref_loader, device)
    id_layers, _ = extract_multilayer_features(model, id_loader, device)
    noise_layers, _ = extract_multilayer_features(model, noise_loader, device)

    num_classes = int(ref_labels.max()) + 1
    id_total = np.zeros(len(id_layers[0]))
    noise_total = np.zeros(len(noise_layers[0]))

    for ref_feats, id_feats, noise_feats in zip(ref_layers, id_layers, noise_layers):
        # Class-conditional means
        class_means = []
        for c in range(num_classes):
            mask = ref_labels == c
            class_means.append(ref_feats[mask].mean(axis=0))
        class_means_arr = np.stack(class_means)

        # Tied (shared) covariance matrix
        centered = ref_feats - class_means_arr[ref_labels.astype(int)]
        cov = np.cov(centered, rowvar=False)
        cov_inv = np.linalg.pinv(cov)

        # Negative Mahalanobis distance to nearest class centroid
        for feats, accum in [(id_feats, "id"), (noise_feats, "noise")]:
            scores = np.full(len(feats), -np.inf)
            for c in range(num_classes):
                diff = feats - class_means_arr[c]
                maha = -np.sum(diff @ cov_inv * diff, axis=1)  # -d^2
                scores = np.maximum(scores, maha)
            if accum == "id":
                id_total += scores
            else:
                noise_total += scores

    return id_total, noise_total


def _compute_fpr_at_tpr(labels, scores, tpr_target=0.95):
    """Compute FPR at a given TPR threshold using the ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return fpr[-1]
    return float(fpr[idx])


def run_ood_evaluation(run, model, id_loader, noise_loader, device, class_names,
                       train_loader=None):
    """Quantitative OOD evaluation with 5 methods: MSP, Energy, ODIN,
    Mahalanobis (multi-layer), and Energy_tuned (optimal temperature from sweep).

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

    # Entropy: -sum(p * log(p)) for both ID and noise
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
    def _safe_bins(vals, target=50):
        """Return bin count that won't fail on constant-valued arrays."""
        if len(vals) == 0 or np.ptp(vals) == 0:
            return 1
        return target

    for label, id_vals, noise_vals, xlabel, key in hist_configs:
        m = methods[label]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(id_vals, bins=_safe_bins(id_vals), alpha=0.6,
                label=f"In-dist (n={n_id})", color="#4C72B0", density=True)
        ax.hist(noise_vals, bins=_safe_bins(noise_vals), alpha=0.6,
                label=f"Noise (n={n_noise})", color="#C44E52", density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution (AUROC={m['auroc']:.3f})")
        ax.legend()
        run.log({f"noise_ood/{key}_histogram": wandb.Image(fig)})
        plt.close(fig)

    # Prediction distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#4C72B0", "#55A868", "#DD8452"]
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
        # MSP with temperature
        msp_t_id = np.max(np.exp(id_logits / T) /
                          np.sum(np.exp(id_logits / T), axis=1, keepdims=True), axis=1)
        msp_t_noise = np.max(np.exp(noise_logits / T) /
                             np.sum(np.exp(noise_logits / T), axis=1, keepdims=True), axis=1)
        msp_t_scores = np.concatenate([msp_t_id, msp_t_noise])
        msp_aurocs_t.append(roc_auc_score(ood_labels, msp_t_scores))

        # Energy with temperature: -T * logsumexp(logits / T)
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
    ax.plot(temperatures, msp_aurocs_t, "o-", label="MSP", color="#4C72B0")
    ax.plot(temperatures, energy_aurocs_t, "s-", label="Energy", color="#55A868")
    ax.axvline(best_T_msp, color="#4C72B0", linestyle="--", alpha=0.4)
    ax.axvline(best_T_energy, color="#55A868", linestyle="--", alpha=0.4)
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

        # W&B scalars
        run.summary["noise_ood/auroc_energy_tuned"] = et["auroc"]
        run.summary["noise_ood/fpr95_energy_tuned"] = et["fpr95"]
        run.summary["noise_ood/aupr_energy_tuned"] = et["aupr"]

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(energy_tuned_id, bins=_safe_bins(energy_tuned_id), alpha=0.6,
                label=f"In-dist (n={n_id})", color="#4C72B0", density=True)
        ax.hist(energy_tuned_noise, bins=_safe_bins(energy_tuned_noise), alpha=0.6,
                label=f"Noise (n={n_noise})", color="#C44E52", density=True)
        ax.set_xlabel(f"Energy Score (T={best_T_energy})")
        ax.set_ylabel("Density")
        ax.set_title(f"Energy Tuned Distribution (AUROC={et['auroc']:.3f})")
        ax.legend()
        run.log({"noise_ood/energy_tuned_histogram": wandb.Image(fig)})
        plt.close(fig)

    # --- Overlaid ROC curves for all methods ---
    print("  [noise_ood] Plotting ROC curves comparison...")
    method_colors = {
        "MSP": "#4C72B0", "Energy": "#55A868", "ODIN": "#DD8452",
        "Mahalanobis": "#C44E52", "Energy_tuned": "#8172B2",
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
    # Noise rejection rate = 1 - FPR (fraction of noise correctly rejected)
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
             label="TPR (ID recall)", color="#4C72B0")
    nr_plot = noise_rejection[:-1] if len(noise_rejection) > len(thresholds) else noise_rejection
    ax1.plot(thresholds, nr_plot,
             label="Noise rejected", color="#C44E52")
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
        # AUROC: this class alone vs all noise
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
# Phase 4 : Dimensionality reduction (PCA / t-SNE)
# ──────────────────────────────────────────────
def extract_features(model, loader, device):
    """Extract fc1 features (256-dim) from the model for all samples."""
    model.eval()
    all_labels = []
    activations = []

    def hook_fn(m, inp, out):
        activations.append(out.detach().cpu())

    hook = model.fc1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            model(signals)
            all_labels.extend(labels.numpy())

    hook.remove()
    features = torch.cat(activations, dim=0).numpy()
    return features, np.array(all_labels)


def extract_multilayer_features(model, loader, device):
    """Extract features from pool1, pool2, pool3 (with GAP) and fc1.

    Conv layer outputs are global-average-pooled along the temporal axis
    to produce fixed-size vectors (64, 128, 256, 256 dims respectively).

    Returns:
        (layer_features_list, labels): list of 4 numpy arrays + label array.
    """
    model.eval()
    all_labels = []
    layer_names = ["pool1", "pool2", "pool3", "fc1"]
    layer_activations = {name: [] for name in layer_names}

    hooks = []

    def make_hook(name):
        def hook_fn(m, inp, out):
            layer_activations[name].append(out.detach().cpu())
        return hook_fn

    hooks.append(model.pool1.register_forward_hook(make_hook("pool1")))
    hooks.append(model.pool2.register_forward_hook(make_hook("pool2")))
    hooks.append(model.pool3.register_forward_hook(make_hook("pool3")))
    hooks.append(model.fc1.register_forward_hook(make_hook("fc1")))

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            model(signals)
            all_labels.extend(labels.numpy())

    for h in hooks:
        h.remove()

    labels_array = np.array(all_labels)
    result = []
    for name in layer_names:
        feats = torch.cat(layer_activations[name], dim=0)
        if feats.dim() == 3:  # Conv output: (N, C, L) -> GAP -> (N, C)
            feats = feats.mean(dim=2)
        result.append(feats.numpy())

    return result, labels_array


def plot_dimensionality_reduction(features, labels, class_names, prefix):
    """Generate PCA and t-SNE scatter plots colored by class. Returns (pca_fig, tsne_fig)."""
    # PCA
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

    # t-SNE
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
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline for particle classification")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset root (train/test)")
    parser.add_argument("--real-test-dir", type=str, default="dataset/test", help="Path to real test data directory")
    parser.add_argument("--noise-dir", type=str, default="Noise",
                        help="Path to noise samples directory for OOD visualization")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save model and logs")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--decimate", type=int, default=4, help="Decimation factor")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--convergence-threshold", type=float, default=0.95,
                        help="Val accuracy threshold to measure convergence time")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Descriptive name for the dataset (default: auto from --data-dir)")
    parser.add_argument("--run-id", type=str, default="run1", help="Run identifier for W&B naming")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--wandb-offline", action="store_true", help="Run W&B in offline mode")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine",
                        help="LR scheduler: none, cosine (CosineAnnealingLR), plateau (ReduceLROnPlateau)")
    args = parser.parse_args()

    if args.dataset_name is None:
        args.dataset_name = Path(args.data_dir).name
        if args.real_test_dir:
            args.dataset_name += "-" + Path(args.real_test_dir).name
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    run_name = f"Conv1D-{args.dataset_name}-{args.run_id}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["2um", "4um", "10um"]
    input_length = RAW_SIGNAL_LENGTH // args.decimate

    # Transforms
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0, sample_rate_mhz=2.0)
    decimate_transform = Decimate(decimate=args.decimate)

    # Datasets
    train_dataset = ParticleDataset(
        data_dir / "train", class_names, transforms=[bandpass, decimate_transform]
    )
    test_dataset = ParticleDataset(
        data_dir / "test", class_names, transforms=[bandpass, decimate_transform]
    )

    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Real test set (optional)
    real_test_loader = None
    if args.real_test_dir is not None:
        real_test_dir = Path(args.real_test_dir)
        if real_test_dir.exists():
            real_test_dataset = ParticleDataset(
                real_test_dir, class_names, transforms=[bandpass, decimate_transform]
            )
            real_test_loader = DataLoader(
                real_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            print(f"Real test set loaded: {len(real_test_dataset)} samples")
        else:
            print(f"WARNING: real test dir not found: {real_test_dir}")

    # Noise samples (optional, for OOD evaluation and latent space visualization)
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

    # Model
    model = Conv1DClassifier(
        input_length=input_length, num_classes=len(class_names)
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    # LR scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

    # ── W&B Init ──
    wandb_mode = "offline" if args.wandb_offline else "online"
    config = {
        "model_name": "Conv1DClassifier",
        "model_size_params": num_params,
        "dataset": args.dataset_name,
        "dataset_size": train_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "decimate": args.decimate,
        "input_length": input_length,
        "num_classes": len(class_names),
        "dropout_conv": model.drop1.p,
        "dropout_fc": model.drop_fc.p,
        "val_split": args.val_split,
        "convergence_threshold": args.convergence_threshold,
        "has_real_test": real_test_loader is not None,
        "patience": args.patience,
        "seed": args.seed,
        "scheduler": args.scheduler,
    }

    run = wandb.init(
        project="particle-benchmark",
        config=config,
        group="Conv1DClassifier",
        tags=[args.dataset_name, "benchmark"],
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

        # ── Phase 1 ──
        log_pre_training(run, num_params, args, train_size, val_size, class_names)

        # ── Phase 2 ──
        best_val_acc, best_epoch, total_time, convergence_time = run_training_loop(
            run, model, train_loader, val_loader, criterion, optimizer, device, args,
            output_dir=output_dir, scheduler=scheduler,
        )

        # ── Phase 3 ──
        print("\n" + "=" * 60)
        print("PHASE 3 : Post-training (best model)")
        print("=" * 60)

        model.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))

        # Test on synthetic data (same distribution as training)
        synth_acc, synth_loss = run_post_evaluation(
            run, model, test_loader, criterion, device, class_names,
            prefix="test_synthetic",
        )

        # Test on real data (generalization)
        if real_test_loader is not None:
            real_acc, real_loss = run_post_evaluation(
                run, model, real_test_loader, criterion, device, class_names,
                prefix="test_real",
            )
            gap = synth_acc - real_acc
            run.summary["generalization_gap"] = gap
            print(f"\n  Generalization gap (synthetic - real): {gap:+.4f}")

        # Save model as W&B artifact
        run.log_model(
            path=str(output_dir / "best_model.pth"),
            name=f"Conv1D-{args.dataset_name}",
        )

        # ── Phase 4 : Dimensionality reduction ──
        print("\n" + "=" * 60)
        print("PHASE 4 : Dimensionality reduction (PCA / t-SNE)")
        print("=" * 60)

        features, feat_labels = extract_features(model, test_loader, device)
        pca_fig, tsne_fig = plot_dimensionality_reduction(
            features, feat_labels, class_names, "test_synthetic"
        )
        run.log({
            "test_synthetic/pca": wandb.Image(pca_fig),
            "test_synthetic/tsne": wandb.Image(tsne_fig),
        })
        plt.close(pca_fig)
        plt.close(tsne_fig)
        print("  [test_synthetic] PCA and t-SNE logged to W&B")

        if real_test_loader is not None:
            features, feat_labels = extract_features(model, real_test_loader, device)
            pca_fig, tsne_fig = plot_dimensionality_reduction(
                features, feat_labels, class_names, "test_real"
            )
            run.log({
                "test_real/pca": wandb.Image(pca_fig),
                "test_real/tsne": wandb.Image(tsne_fig),
            })
            plt.close(pca_fig)
            plt.close(tsne_fig)
            print("  [test_real] PCA and t-SNE logged to W&B")

        # Noise separation visualization
        if noise_loader is not None:
            print("\n  Noise OOD separation analysis...")
            test_features, test_labels = extract_features(model, test_loader, device)
            noise_features, _ = extract_features(model, noise_loader, device)

            combined_features = np.concatenate([test_features, noise_features])
            noise_labels = np.full(len(noise_features), len(class_names))
            combined_labels = np.concatenate([test_labels, noise_labels])
            combined_names = class_names + ["Noise"]

            pca_fig, tsne_fig = plot_dimensionality_reduction(
                combined_features, combined_labels, combined_names, "noise_separation"
            )
            run.log({
                "noise_separation/pca": wandb.Image(pca_fig),
                "noise_separation/tsne": wandb.Image(tsne_fig),
            })
            plt.close(pca_fig)
            plt.close(tsne_fig)
            print("  [noise_separation] PCA and t-SNE logged to W&B")

        # ── Phase 5 : OOD Noise Evaluation ──
        if noise_loader is not None:
            print("\n" + "=" * 60)
            print("PHASE 5 : OOD Noise Evaluation")
            print("=" * 60)
            run_ood_evaluation(
                run, model, test_loader, noise_loader, device, class_names,
                train_loader=train_loader,
            )

        print("\n" + "=" * 60)
        print("Benchmark complete.")
        print("=" * 60)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
