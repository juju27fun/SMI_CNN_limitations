"""Shared training utilities for particle classification pipelines.

Provides common functions used by both train3classes_proof.py and train4classes.py:
- Model profiling (MACs, latency)
- Training loop with W&B logging and early stopping
- Post-training evaluation with confusion matrix and F1 logging
- Optimizer and scheduler creation
- Common argparse arguments
"""

import time

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from sklearn.metrics import classification_report, confusion_matrix

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

from train import train_one_epoch, evaluate
from models import list_models


# ──────────────────────────────────────────────
# MACs and latency
# ──────────────────────────────────────────────
def compute_model_macs(model, input_shape, device):
    """Compute MACs using thop if available. Returns int or None."""
    if thop_profile is None:
        return None
    dummy = torch.randn(*input_shape).to(device)
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    # Clean up buffers thop adds (total_ops, total_params) to avoid polluting state_dict
    for m in model.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(m, attr):
                delattr(m, attr)
    return int(macs)


def measure_inference_latency(model, input_shape, device, n_runs=100, warmup=10):
    """Measure average per-sample inference latency in milliseconds."""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - start) / n_runs * 1000


# ──────────────────────────────────────────────
# Training loop
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

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

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
            print(f"  Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {args.patience} epochs)")
            run.summary["early_stopped_epoch"] = epoch
            break

    total_time = time.time() - total_start

    # Summary
    run.summary["total_training_time_sec"] = total_time
    run.summary["final_val_accuracy"] = val_acc
    run.summary["final_val_loss"] = val_loss
    if convergence_time is None:
        run.summary["convergence_time_sec"] = float("nan")

    print("-" * 60)
    print(f"  Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
    print(f"  Total training time: {total_time:.1f}s")

    return best_val_acc, best_epoch, total_time, convergence_time


# ──────────────────────────────────────────────
# Post-training testing
# ──────────────────────────────────────────────
def run_post_testing(run, model, loader, criterion, device, class_names, prefix="Charts"):
    """Run testing and log confusion matrix + F1 per class to W&B."""
    print(f"\n  [{prefix}] Running testing...")
    loss, acc, y_pred, y_true, y_proba = evaluate(model, loader, criterion, device)

    print(f"  [{prefix}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # Summary scalars
    run.summary[f"{prefix}/accuracy"] = acc
    run.summary[f"{prefix}/loss"] = loss

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title(f"{prefix} Confusion Matrix (Accuracy: {acc:.4f})")
    plt.tight_layout()
    run.log({f"{prefix}/confusion_matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)

    # Interactive confusion matrix (Charts tab)
    run.log({f"{prefix}/confusion_matrix_chart": wandb.plot.confusion_matrix(
        y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=class_names
    )})

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
            f1_table, "class", "f1", title=f"F1 per Class ({len(class_names)}-class)"
        )
    })

    return acc, loss


# ──────────────────────────────────────────────
# Optimizer and scheduler creation
# ──────────────────────────────────────────────
def create_optimizer(model, args):
    """Create optimizer based on args.optimizer. Returns optimizer instance."""
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    print(f"Optimizer: {args.optimizer} (lr={args.lr}, weight_decay={args.weight_decay})")
    return optimizer


def create_scheduler(optimizer, args):
    """Create LR scheduler based on args.scheduler. Returns scheduler or None."""
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )
    return None


# ──────────────────────────────────────────────
# Common argparse arguments
# ──────────────────────────────────────────────
def add_common_training_args(parser, data_dir_default="data/dataset"):
    """Add training arguments shared across all training scripts.

    Args:
        parser: argparse.ArgumentParser instance
        data_dir_default: Default value for --data-dir
    """
    parser.add_argument("--model", type=str, default="Conv1D",
                        choices=list_models(),
                        help=f"Model architecture ({', '.join(list_models())})")
    parser.add_argument("--data-dir", type=str, default=data_dir_default,
                        help="Path to dataset root (train/test)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save model and logs")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--decimate", type=int, default=4, help="Decimation factor")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--convergence-threshold", type=float, default=0.95,
                        help="Val accuracy threshold for convergence time")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Descriptive name for the dataset (default: auto from --data-dir)")
    parser.add_argument("--run-id", type=str, default="run1",
                        help="Run identifier for W&B naming")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--wandb-offline", action="store_true",
                        help="Run W&B in offline mode")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"],
                        default="cosine",
                        help="LR scheduler: none, cosine, plateau")
    # Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (training set only)")
    parser.add_argument("--aug-snr", type=float, default=20.0,
                        help="Gaussian noise SNR in dB (default: 20.0)")
    parser.add_argument("--aug-shift", type=float, default=0.1,
                        help="Max time-shift fraction (default: 0.1)")
    parser.add_argument("--aug-scale-min", type=float, default=0.8,
                        help="Min amplitude scale factor (default: 0.8)")
    parser.add_argument("--aug-scale-max", type=float, default=1.2,
                        help="Max amplitude scale factor (default: 1.2)")
    # Optimizer
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"],
                        default="adam", help="Optimizer (default: adam)")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="Weight decay (default: 0.0001)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (default: 0.9, ignored for Adam/AdamW)")
