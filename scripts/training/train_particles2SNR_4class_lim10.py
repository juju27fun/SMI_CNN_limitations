"""Train a 4-class classifier on particles2SNR_4_class_lim10.

Classes are: 2um, 4um, 10um, unclear.  The script is intentionally separate
from train4classes.py because this experiment uses an SNR/quality class rather
than the historical Noise class.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

from p0.models import create_model
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
from p0.plotting import plot_confusion_matrix


DEFAULT_CLASS_NAMES = ("2um", "4um", "10um", "unclear")


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def dataset_counts(dataset: ParticleDataset, class_names: tuple[str, ...]) -> dict[str, int]:
    labels = np.asarray(dataset.labels, dtype=int)
    return {
        class_name: int(np.sum(labels == class_idx))
        for class_idx, class_name in enumerate(class_names)
    }


def subset_counts(dataset: ParticleDataset, indices: list[int], class_names: tuple[str, ...]) -> dict[str, int]:
    labels = np.asarray(dataset.labels, dtype=int)[indices]
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
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    val_dataset = ParticleDataset(data_dir / "train", list(class_names), transforms=make_transforms(args, train=False))
    train_dataset = ParticleDataset(data_dir / "train", list(class_names), transforms=make_transforms(args, train=True))
    test_dataset = ParticleDataset(data_dir / "test", list(class_names), transforms=make_transforms(args, train=False))

    total_size = len(val_dataset)
    if total_size == 0:
        raise RuntimeError(f"No training samples found under {data_dir / 'train'}")
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Bad split sizes: train={train_size}, val={val_size}")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_size, generator=generator)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_indices, val_indices


def compute_unclear_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]) -> dict:
    unclear_idx = class_names.index("unclear")
    particle_mask = y_true != unclear_idx
    unclear_true = y_true == unclear_idx
    unclear_pred = y_pred == unclear_idx

    particle_only_accuracy = (
        float(np.mean(y_pred[particle_mask] == y_true[particle_mask]))
        if np.any(particle_mask) else float("nan")
    )
    false_unclear_rate = (
        float(np.mean(unclear_pred[particle_mask]))
        if np.any(particle_mask) else float("nan")
    )
    unclear_recall = (
        float(np.mean(unclear_pred[unclear_true]))
        if np.any(unclear_true) else float("nan")
    )
    unclear_precision = (
        float(np.mean(unclear_true[unclear_pred]))
        if np.any(unclear_pred) else float("nan")
    )
    unclear_purity = unclear_precision
    majority_baseline_accuracy = float(np.max(np.bincount(y_true, minlength=len(class_names))) / len(y_true))

    return {
        "particle_only_accuracy": json_float(particle_only_accuracy),
        "false_unclear_rate": json_float(false_unclear_rate),
        "unclear_recall": json_float(unclear_recall),
        "unclear_precision": json_float(unclear_precision),
        "unclear_purity": json_float(unclear_purity),
        "majority_baseline_accuracy": json_float(majority_baseline_accuracy),
    }


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


def json_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def save_confusion_matrix(path: Path, cm: np.ndarray, class_names: tuple[str, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_confusion_matrix(cm, list(class_names))
    fig.savefig(path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Conv1DGAP/zoo models on particles2SNR_4_class_lim10."
    )
    add_common_training_args(parser, data_dir_default="datasets/processed/particles2snr-4class-lim10/v1")
    parser.set_defaults(
        model="Conv1DGAP-L",
        output_dir="artifacts/SMI_CNN_limitations/particles2SNR_4_class_lim10_conv1dgap",
        optimizer="adamw",
        weight_decay=0.01,
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
    parser.add_argument("--profile", action="store_true",
                        help="Measure MACs and CPU latency before training")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    class_names = tuple(args.class_names)
    if "unclear" not in class_names:
        raise ValueError("--class-names must include unclear")
    if args.dataset_name is None:
        args.dataset_name = Path(args.data_dir).name

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb_offline:
        import os
        os.environ["WANDB_MODE"] = "offline"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    (
        train_loader, val_loader, test_loader, train_dataset, val_dataset,
        test_dataset, train_indices, val_indices,
    ) = make_loaders(args, class_names)
    train_counts = subset_counts(train_dataset, train_indices, class_names)
    val_counts = subset_counts(val_dataset, val_indices, class_names)
    test_counts = dataset_counts(test_dataset, class_names)

    run_tag = f"{args.model}-{args.dataset_name}-{args.run_id}"
    output_dir = Path(args.output_dir) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(args.model, input_length=args.input_length, num_classes=len(class_names)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.profile:
        model_macs = compute_model_macs(model, (1, 1, args.input_length), device)
        latency = measure_cpu_latency(model, (1, 1, args.input_length), warmup=10, n_runs=50)
    else:
        model_macs = None
        latency = None
    print(f"Model: {args.model} ({num_params:,} params, {len(class_names)} classes)")
    print(f"Dataset train counts: {train_counts}")
    print(f"Dataset val counts: {val_counts}")
    print(f"Dataset test counts: {test_counts}")

    if args.class_weights == "balanced":
        class_weight_tensor = balanced_weights_from_counts(train_counts, class_names).to(device)
        class_weights = {
            class_name: float(class_weight_tensor[idx].detach().cpu())
            for idx, class_name in enumerate(class_names)
        }
    else:
        class_weight_tensor = None
        class_weights = {class_name: 1.0 for class_name in class_names}
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)

    run = wandb.init(
        project="particle-particles2SNR-4class-lim10",
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
    unclear_metrics = compute_unclear_metrics(y_true, y_pred, class_names)
    save_confusion_matrix(output_dir / "confusion_matrix.pdf", cm, class_names)

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
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "total_training_time_sec": total_time,
        "convergence_time_sec": convergence_time,
        "test_loss": json_float(test_loss),
        "test_accuracy": json_float(test_acc),
        "macro_f1": json_float(report["macro avg"]["f1-score"]),
        "weighted_f1": json_float(report["weighted avg"]["f1-score"]),
        "unclear_metrics": unclear_metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_params": num_params,
        "macs": model_macs,
        "cpu_latency": latency,
    }
    write_json(output_dir / "test_metrics.json", metrics)
    write_predictions(output_dir / "test_predictions.csv", test_dataset, y_true, y_pred, y_proba, class_names)
    write_json(output_dir / "run_metadata.json", {
        "args": vars(args),
        "class_names": list(class_names),
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "class_weights": class_weights,
    })

    run.summary["test/accuracy"] = test_acc
    run.summary["test/macro_f1"] = report["macro avg"]["f1-score"]
    for key, value in unclear_metrics.items():
        run.summary[f"unclear/{key}"] = value
    run.finish()

    print("\nTest metrics")
    print(f"  accuracy: {test_acc:.4f}")
    print(f"  macro_f1: {report['macro avg']['f1-score']:.4f}")
    for key, value in unclear_metrics.items():
        if value is None:
            print(f"  {key}: n/a")
        else:
            print(f"  {key}: {value:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
