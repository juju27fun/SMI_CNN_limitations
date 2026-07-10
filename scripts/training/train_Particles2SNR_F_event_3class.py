"""Train a 3-class classifier on clean-filt particles2SNR YOLO event crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from p0.models import create_model
from p0.plotting import plot_confusion_matrix
from p0.data import (
    AdaptiveBandpassDecimate,
    AmplitudeScale,
    GaussianNoise,
    ParticleDataset,
    TimeShift,
)
from p0.training import evaluate
from p0.training_utils import (
    add_common_training_args,
    compute_model_macs,
    create_optimizer,
    create_scheduler,
    measure_cpu_latency,
    run_training_loop,
)


DEFAULT_CLASS_NAMES = ("2um", "4um", "10um")


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def dataset_counts(dataset: ParticleDataset, class_names: tuple[str, ...]) -> dict[str, int]:
    labels = np.asarray(dataset.labels, dtype=int)
    return {
        class_name: int(np.sum(labels == class_idx))
        for class_idx, class_name in enumerate(class_names)
    }


def balanced_weights_from_counts(counts_by_class: dict[str, int], class_names: tuple[str, ...]) -> torch.Tensor:
    counts = np.asarray([counts_by_class[class_name] for class_name in class_names], dtype=float)
    if np.any(counts == 0):
        missing = [class_names[idx] for idx, count in enumerate(counts) if count == 0]
        raise ValueError(f"Cannot compute balanced class weights, missing classes: {missing}")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_transforms(args, train: bool):
    transforms = [
        AdaptiveBandpassDecimate(
            target_length=args.input_length,
            native_length=args.native_length,
            native_fs_hz=args.sample_rate_mhz * 1_000_000.0,
            low_khz=args.bandpass_low_khz,
            high_khz_max=args.bandpass_high_khz,
        )
    ]
    if train and args.augment:
        transforms.extend([
            GaussianNoise(snr_db=args.aug_snr),
            TimeShift(max_shift_frac=args.aug_shift),
            AmplitudeScale(scale_min=args.aug_scale_min, scale_max=args.aug_scale_max),
        ])
    return transforms


def make_loaders(args, class_names: tuple[str, ...]):
    data_dir = Path(args.data_dir)
    train_dataset = ParticleDataset(data_dir / "train", list(class_names), transforms=make_transforms(args, train=True))
    val_dataset = ParticleDataset(data_dir / "val", list(class_names), transforms=make_transforms(args, train=False))
    test_dataset = ParticleDataset(data_dir / "test", list(class_names), transforms=make_transforms(args, train=False))
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def json_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def write_predictions(path: Path, test_dataset: ParticleDataset, y_true, y_pred, y_proba, class_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "filename", "path", "y_true", "y_pred", "correct",
            *[f"prob_{class_name}" for class_name in class_names],
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample_path in enumerate(test_dataset.samples):
            row = {
                "filename": sample_path.name,
                "path": str(sample_path),
                "y_true": class_names[int(y_true[idx])],
                "y_pred": class_names[int(y_pred[idx])],
                "correct": bool(int(y_true[idx]) == int(y_pred[idx])),
            }
            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(y_proba[idx, class_idx])
            writer.writerow(row)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def save_confusion_matrix(path: Path, cm: np.ndarray, class_names: tuple[str, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_confusion_matrix(cm, list(class_names))
    fig.savefig(path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 3-class classifier on Particles2SNR_F event crops.")
    add_common_training_args(parser, data_dir_default="datasets/processed/particles2snr-f-c1-events/v1")
    parser.set_defaults(
        model="Conv1DGAP-L",
        output_dir="artifacts/SMI_CNN_limitations/Particles2SNR_F_c1_event_3class",
        optimizer="adamw",
        weight_decay=0.01,
        dataset_name="dataset_Particles2SNR_F_c1_events",
        run_id="conv1dgap_l_seed42",
        wandb_offline=True,
    )
    parser.add_argument("--class-names", type=parse_csv_arg, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--native-length", type=int, default=16384)
    parser.add_argument("--bandpass-low-khz", type=float, default=5.0)
    parser.add_argument("--bandpass-high-khz", type=float, default=100.0)
    parser.add_argument("--sample-rate-mhz", type=float, default=2.0)
    parser.add_argument("--class-weights", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--profile", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    class_names = tuple(args.class_names)
    if len(class_names) != 3:
        raise ValueError("This script expects exactly 3 particle classes")
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = make_loaders(args, class_names)
    train_counts = dataset_counts(train_dataset, class_names)
    val_counts = dataset_counts(val_dataset, class_names)
    test_counts = dataset_counts(test_dataset, class_names)

    run_tag = f"{args.model}-{args.dataset_name}-{args.run_id}"
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(args.model, input_length=args.input_length, num_classes=len(class_names)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_macs = compute_model_macs(model, (1, 1, args.input_length), device) if args.profile else None
    latency = measure_cpu_latency(model, (1, 1, args.input_length), warmup=10, n_runs=50) if args.profile else None
    print(f"Model: {args.model} ({num_params:,} params)")
    print(f"Dataset train counts: {train_counts}")
    print(f"Dataset val counts: {val_counts}")
    print(f"Dataset test counts: {test_counts}")

    if args.class_weights == "balanced":
        class_weight_tensor = balanced_weights_from_counts(train_counts, class_names).to(device)
        class_weights = {class_name: float(class_weight_tensor[idx].detach().cpu()) for idx, class_name in enumerate(class_names)}
    else:
        class_weight_tensor = None
        class_weights = {class_name: 1.0 for class_name in class_names}
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)

    run = wandb.init(
        project="particle-Particles2SNR_F-event-3class",
        name=run_tag,
        config={
            **vars(args),
            "class_names": list(class_names),
            "train_counts": train_counts,
            "val_counts": val_counts,
            "test_counts": test_counts,
            "class_weights_used": args.class_weights,
            "class_weights": class_weights,
            "num_params": num_params,
            "macs": model_macs,
            "cpu_latency": latency,
        },
    )
    best_val_acc, best_epoch, total_time, convergence_time = run_training_loop(
        run, model, train_loader, val_loader, criterion, optimizer,
        device, args, output_dir, scheduler=scheduler,
    )
    best_model_path = output_dir / "best_model.pth"
    if best_model_path.is_file():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_pred, y_true, y_proba = evaluate(model, test_loader, criterion, device)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=list(class_names), output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    save_confusion_matrix(output_dir / "confusion_matrix.pdf", cm, class_names)
    write_predictions(output_dir / "test_predictions.csv", test_dataset, y_true, y_pred, y_proba, class_names)
    metrics = {
        "run_tag": run_tag,
        "model": args.model,
        "dataset_name": args.dataset_name,
        "data_dir": str(Path(args.data_dir).resolve()),
        "class_names": list(class_names),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "class_weights_used": args.class_weights,
        "class_weights": class_weights,
        "best_val_accuracy": json_float(best_val_acc),
        "best_epoch": int(best_epoch),
        "total_training_time_sec": json_float(total_time),
        "convergence_time_sec": json_float(convergence_time),
        "test_loss": json_float(test_loss),
        "test_accuracy": json_float(test_acc),
        "macro_f1": json_float(report["macro avg"]["f1-score"]),
        "weighted_f1": json_float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_params": num_params,
        "macs": model_macs,
        "cpu_latency": latency,
        "preprocessing": {
            "native_length": args.native_length,
            "input_length": args.input_length,
            "bandpass_low_khz": args.bandpass_low_khz,
            "bandpass_high_khz": args.bandpass_high_khz,
            "sample_rate_mhz": args.sample_rate_mhz,
        },
    }
    write_json(output_dir / "test_metrics.json", metrics)
    run.summary["test_accuracy"] = test_acc
    run.summary["test_macro_f1"] = metrics["macro_f1"]
    run.finish()
    print(f"Output: {output_dir}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
