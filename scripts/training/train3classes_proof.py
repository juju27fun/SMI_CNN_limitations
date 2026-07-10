"""Training pipeline for 3-class particle classification (2um, 4um, 10um).

Mirrors train4classes.py structure and W&B logging for direct comparison.

Usage:
    python train3classes_proof.py --data-dir datasets/processed/p0-baseline-3class/v1 --epochs 150 --wandb-offline
    python train3classes_proof.py --model ResNet1D --data-dir datasets/processed/p0-baseline-3class/v1 --epochs 150
    python train3classes_proof.py --model InceptionTime1D --data-dir datasets/processed/p0-baseline-3class/v1 --epochs 150
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.utils.data import DataLoader, Subset

from p0.data import (
    RAW_SIGNAL_LENGTH,
    AmplitudeScale,
    BandpassFilter,
    Decimate,
    GaussianNoise,
    ParticleDataset,
    TimeShift,
)
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

CLASS_NAMES = ["2um", "4um", "10um"]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train 3-class classifier (2um, 4um, 10um)"
    )
    add_common_training_args(parser)
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

    # Datasets (3 classes)
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

    print(f"Dataset: {args.dataset_name} (3 classes)")
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

    # Model (3 classes)
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
        tags=[args.dataset_name, "3class"],
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

        # ── Post-training testing ──
        print("\n" + "=" * 60)
        print("Post-training testing (best model, 3 classes)")
        print("=" * 60)

        model.load_state_dict(
            torch.load(output_dir / "best_model.pth", weights_only=True)
        )

        run_post_testing(run, model, test_loader, criterion, device, CLASS_NAMES)

        # ── Inference latency (CPU canonical, see docs/metrics_conventions.md) ──
        latency = measure_cpu_latency(model, (1, 1, input_length))
        run.summary["inference_latency_median_ms"] = latency["median_ms"]
        run.summary["inference_latency_p95_ms"] = latency["p95_ms"]
        run.summary["latency_device"] = latency["latency_device"]
        print(f"  Inference latency (CPU): {latency['median_ms']:.2f} ms/sample")

        # Save model as W&B artifact
        run.log_model(
            path=str(output_dir / "best_model.pth"),
            name=f"{args.model}-{args.dataset_name}-3class",
        )

        print("\n" + "=" * 60)
        print("3-class training complete.")
        print("=" * 60)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
