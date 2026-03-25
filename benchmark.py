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
import wandb

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from train import (
    RAW_SIGNAL_LENGTH,
    ParticleDataset,
    Conv1DClassifier,
    BandpassFilter,
    Decimate,
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
                      device, args, scheduler=None):
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
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pth")
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
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline for particle classification")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset root (train/test)")
    parser.add_argument("--real-test-dir", type=str, default=None, help="Path to real test data directory")
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

    output_dir = Path(args.output_dir)
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
        name=f"Conv1D-{args.dataset_name}-{args.run_id}",
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
            scheduler=scheduler,
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

        print("\n" + "=" * 60)
        print("Benchmark complete.")
        print("=" * 60)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
