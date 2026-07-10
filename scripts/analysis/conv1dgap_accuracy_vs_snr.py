"""Build Conv1DGAP accuracy-vs-SNR curves from offline predictions.

Expected input is a prediction CSV with at least:
  - filename/path/sample_id
  - y_true/true_class/label
  - y_pred/pred_class/prediction

The CSV may already contain ``snr_db``. If not, pass ``--snr-csv`` from the
particles2SNR pipeline; rows are joined by filename and optionally class.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KEY_COLUMNS = ("filename", "path", "sample_id")
TRUE_COLUMNS = ("y_true", "true_class", "label", "class")
PRED_COLUMNS = ("y_pred", "pred_class", "prediction")


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def first_present(row, names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def as_float(value):
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def canonical_key(row):
    value = first_present(row, KEY_COLUMNS)
    if value is None:
        return None
    return os.path.basename(value)


def build_snr_lookup(rows):
    by_file = defaultdict(list)
    for row in rows:
        key = canonical_key(row)
        snr = as_float(row.get("snr_db"))
        if key is not None and snr is not None:
            by_file[key].append(snr)
    return {key: float(np.median(vals)) for key, vals in by_file.items() if vals}


def normalize_predictions(pred_rows, snr_lookup=None):
    out = []
    for row in pred_rows:
        key = canonical_key(row)
        y_true = first_present(row, TRUE_COLUMNS)
        y_pred = first_present(row, PRED_COLUMNS)
        snr = as_float(row.get("snr_db"))
        if snr is None and snr_lookup is not None and key is not None:
            snr = snr_lookup.get(key)
        if key is None or y_true is None or y_pred is None or snr is None:
            continue
        out.append({
            "filename": key,
            "y_true": str(y_true),
            "y_pred": str(y_pred),
            "snr_db": float(snr),
            "correct": str(y_true) == str(y_pred),
        })
    return out


def make_bins(values, n_bins, fixed_width=None):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    if fixed_width is not None and fixed_width > 0:
        lo = math.floor(float(np.min(values)) / fixed_width) * fixed_width
        hi = math.ceil(float(np.max(values)) / fixed_width) * fixed_width
        edges = np.arange(lo, hi + fixed_width, fixed_width)
    else:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        eps = 0.5 if fixed_width is None else fixed_width / 2.0
        edges = np.asarray([float(values[0]) - eps, float(values[0]) + eps])
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def macro_f1(rows):
    labels = sorted({row["y_true"] for row in rows} | {row["y_pred"] for row in rows})
    scores = []
    for label in labels:
        tp = sum(row["y_true"] == label and row["y_pred"] == label for row in rows)
        fp = sum(row["y_true"] != label and row["y_pred"] == label for row in rows)
        fn = sum(row["y_true"] == label and row["y_pred"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def bin_rows(rows, bins):
    result = []
    for idx, (left, right) in enumerate(bins):
        if idx == len(bins) - 1:
            subset = [row for row in rows if left <= row["snr_db"] <= right]
        else:
            subset = [row for row in rows if left <= row["snr_db"] < right]
        if not subset:
            continue
        labels = sorted({row["y_true"] for row in rows})
        per_class_recall = {}
        for label in labels:
            cls_rows = [row for row in subset if row["y_true"] == label]
            per_class_recall[f"recall_{label}"] = (
                sum(row["correct"] for row in cls_rows) / len(cls_rows)
                if cls_rows else None
            )
        result.append({
            "bin_idx": idx,
            "snr_left": left,
            "snr_right": right,
            "snr_center": float(np.mean([left, right])),
            "n": len(subset),
            "accuracy": float(sum(row["correct"] for row in subset) / len(subset)),
            "macro_f1": macro_f1(subset),
            **per_class_recall,
        })
    return result


def estimate_threshold(bin_stats, derivative_frac=0.2):
    if len(bin_stats) < 3:
        return {
            "unknown_snr_threshold_db": None,
            "method": "not_enough_bins",
            "derivatives": [],
        }
    x = np.asarray([row["snr_center"] for row in bin_stats], dtype=float)
    y = np.asarray([row["accuracy"] for row in bin_stats], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(y) >= 5:
        kernel = np.ones(3) / 3.0
        y_smooth = np.convolve(np.pad(y, (1, 1), mode="edge"), kernel, mode="valid")
    else:
        y_smooth = y
    deriv = np.diff(y_smooth) / np.maximum(np.diff(x), 1e-9)
    positive = deriv[deriv > 0]
    if len(positive) == 0:
        return {
            "unknown_snr_threshold_db": float(x[0]),
            "method": "flat_or_decreasing_curve",
            "derivatives": deriv.tolist(),
        }
    max_deriv = float(np.max(positive))
    cutoff = max_deriv * derivative_frac
    peak_idx = int(np.argmax(deriv))
    threshold = None
    for i in range(peak_idx + 1, len(deriv)):
        if deriv[i] <= cutoff:
            threshold = float(x[i + 1])
            break
    if threshold is None:
        # Fallback: choose the first bin within 95% of best observed accuracy.
        best = float(np.max(y_smooth))
        candidates = x[y_smooth >= 0.95 * best]
        threshold = float(candidates[0]) if len(candidates) else float(x[-1])
    return {
        "unknown_snr_threshold_db": threshold,
        "method": "post_peak_derivative_slowdown",
        "derivative_fraction": derivative_frac,
        "max_derivative": max_deriv,
        "derivative_cutoff": cutoff,
        "derivatives": deriv.tolist(),
    }


def threshold_at_target_accuracy(bin_stats, target_accuracy):
    """Return the SNR where the binned accuracy first reaches target_accuracy."""
    if len(bin_stats) < 2:
        return None

    x = np.asarray([row["snr_center"] for row in bin_stats], dtype=float)
    y = np.asarray([row["accuracy"] for row in bin_stats], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.all(y >= target_accuracy):
        return float(x[0])
    if np.all(y < target_accuracy):
        return None

    for i in range(len(y) - 1):
        y0 = y[i]
        y1 = y[i + 1]
        x0 = x[i]
        x1 = x[i + 1]
        if y0 >= target_accuracy:
            return float(x0)
        if y0 < target_accuracy <= y1:
            if y1 == y0:
                return float(x1)
            frac = (target_accuracy - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return float(x[-1])


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(bin_stats, threshold_info, output_path):
    x = [row["snr_center"] for row in bin_stats]
    y = [row["accuracy"] for row in bin_stats]
    n = [row["n"] for row in bin_stats]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, y, marker="o", label="accuracy")
    for xi, yi, ni in zip(x, y, n):
        ax.text(xi, yi, str(ni), fontsize=8, ha="center", va="bottom")
    threshold = threshold_info.get("unknown_snr_threshold_db")
    if threshold is not None:
        ax.axvline(threshold, linestyle="--", color="tab:red",
                   label=f"unknown threshold {threshold:.2f} dB")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def center_crop_or_pad(signal, length):
    if length <= 0:
        return signal
    if len(signal) == length:
        return signal
    if len(signal) > length:
        start = (len(signal) - length) // 2
        return signal[start:start + length]
    pad = length - len(signal)
    left = pad // 2
    right = pad - left
    return np.pad(signal, (left, right), mode="constant")


def preprocess_signal(signal, args):
    if args.preprocess == "none":
        return center_crop_or_pad(signal, args.input_length)

    import torch
    from p0.data import AdaptiveBandpassDecimate, BandpassFilter, CenterCrop, Decimate

    tensor = torch.from_numpy(signal.astype(np.float32)[None, :])
    if args.preprocess == "bandpass-decimate":
        transforms = [
            BandpassFilter(args.bandpass_low_khz, args.bandpass_high_khz, args.sample_rate_mhz),
            Decimate(args.decimate),
        ]
        if args.input_length > 0:
            transforms.append(CenterCrop(args.input_length))
    elif args.preprocess == "adaptive-bandpass":
        if args.input_length <= 0:
            raise ValueError("--input-length must be > 0 for adaptive-bandpass")
        transforms = [
            AdaptiveBandpassDecimate(
                target_length=args.input_length,
                native_length=len(signal),
                native_fs_hz=args.sample_rate_mhz * 1_000_000.0,
                low_khz=args.bandpass_low_khz,
                high_khz_max=args.bandpass_high_khz,
            )
        ]
    else:
        raise ValueError(f"Unknown preprocess mode: {args.preprocess}")

    for transform in transforms:
        tensor = transform(tensor)
    return tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)


def load_checkpoint_state(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_model_weights(model, checkpoint_path, strict=True):
    state = load_checkpoint_state(checkpoint_path)
    if strict:
        model.load_state_dict(state, strict=True)
        return {"loaded_keys": len(state), "skipped_keys": []}

    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(compatible, strict=False)
    return {"loaded_keys": len(compatible), "skipped_keys": skipped}


def generate_predictions_from_checkpoint(args):
    import torch
    from p0.models import create_model

    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    if not class_names:
        raise ValueError("--class-names must contain at least one class")
    device = torch.device(args.device)
    model = create_model(
        args.model_name,
        input_length=args.input_length,
        num_classes=len(class_names),
        **({"kernel_size": args.kernel_size} if args.kernel_size else {}),
    )
    load_info = load_model_weights(model, args.checkpoint, strict=args.strict_checkpoint)
    model.to(device)
    model.eval()

    rows = []
    seen = 0
    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(args.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.endswith(".npy"):
                    continue
                path = os.path.join(class_dir, fname)
                signal = np.load(path).astype(np.float32)
                signal = preprocess_signal(signal, args)
                tensor = torch.from_numpy(signal[None, None, :]).to(device)
                logits = model(tensor)
                pred_idx = int(torch.argmax(logits, dim=1).item())
                rows.append({
                    "filename": fname,
                    "path": path,
                    "y_true": class_name,
                    "y_pred": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
                    "checkpoint_loaded_keys": load_info["loaded_keys"],
                    "checkpoint_skipped_keys": len(load_info["skipped_keys"]),
                })
                seen += 1
                if args.max_samples and seen >= args.max_samples:
                    return rows
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute Conv1DGAP accuracy as a function of SNR."
    )
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional Conv1DGAP checkpoint used to generate predictions")
    parser.add_argument("--data-dir", default=None,
                        help="Class-folder dataset used with --checkpoint")
    parser.add_argument("--generated-predictions-csv", default=None,
                        help="Where to save generated predictions when --checkpoint is used")
    parser.add_argument("--model-name", default="Conv1DGAP")
    parser.add_argument("--class-names", default="2um,4um,10um")
    parser.add_argument("--input-length", type=int, default=16384)
    parser.add_argument("--kernel-size", type=int, default=None,
                        help="Optional Conv1DGAP kernel size for kernel-length sweep checkpoints")
    parser.add_argument("--preprocess",
                        choices=("none", "bandpass-decimate", "adaptive-bandpass"),
                        default="none",
                        help="Optional P0-style preprocessing before inference")
    parser.add_argument("--decimate", type=int, default=4)
    parser.add_argument("--bandpass-low-khz", type=float, default=5.0)
    parser.add_argument("--bandpass-high-khz", type=float, default=100.0)
    parser.add_argument("--sample-rate-mhz", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit checkpoint inference rows for smoke runs (0 = all)")
    parser.add_argument("--strict-checkpoint", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--snr-csv", default=None,
                        help="Optional particles2SNR snr_particles.csv for joining snr_db by filename")
    parser.add_argument("--output-dir", default="artifacts/SMI_CNN_limitations/snr_accuracy")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--bin-width", type=float, default=None,
                        help="Use fixed-width SNR bins instead of quantile bins")
    parser.add_argument("--derivative-frac", type=float, default=0.2)
    parser.add_argument("--target-accuracy", type=float, default=None,
                        help="Plot a threshold when accuracy reaches this value (0..1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    snr_lookup = build_snr_lookup(read_rows(args.snr_csv)) if args.snr_csv else None
    if args.checkpoint:
        if not args.data_dir:
            raise ValueError("--data-dir is required with --checkpoint")
        pred_rows = generate_predictions_from_checkpoint(args)
        generated_path = args.generated_predictions_csv or os.path.join(
            args.output_dir, "conv1dgap_generated_predictions.csv"
        )
        write_csv(generated_path, pred_rows)
        prediction_source = generated_path
    else:
        if not args.predictions_csv:
            raise ValueError("Provide --predictions-csv or --checkpoint + --data-dir")
        pred_rows = read_rows(args.predictions_csv)
        prediction_source = args.predictions_csv

    rows = normalize_predictions(pred_rows, snr_lookup)
    if not rows:
        raise RuntimeError("No usable prediction rows with y_true, y_pred and snr_db")

    bins = make_bins([row["snr_db"] for row in rows], args.bins, args.bin_width)
    bin_stats = bin_rows(rows, bins)
    if args.target_accuracy is not None:
        threshold = {
            "unknown_snr_threshold_db": threshold_at_target_accuracy(bin_stats, args.target_accuracy),
            "method": "target_accuracy",
            "target_accuracy": args.target_accuracy,
        }
    else:
        threshold = estimate_threshold(bin_stats, args.derivative_frac)

    csv_path = os.path.join(args.output_dir, "conv1dgap_accuracy_by_snr.csv")
    json_path = os.path.join(args.output_dir, "conv1dgap_accuracy_by_snr.json")
    pdf_path = os.path.join(args.output_dir, "conv1dgap_accuracy_by_snr.pdf")
    write_csv(csv_path, bin_stats)
    plot_curve(bin_stats, threshold, pdf_path)

    overall = {
        "n": len(rows),
        "accuracy": float(sum(row["correct"] for row in rows) / len(rows)),
        "macro_f1": macro_f1(rows),
        "class_counts": dict(Counter(row["y_true"] for row in rows)),
    }
    with open(json_path, "w") as f:
        json.dump({
            "model_family": "Conv1DGAP",
            "overall": overall,
            "threshold": threshold,
            "bins": bin_stats,
            "source_predictions_csv": os.path.abspath(prediction_source),
            "source_snr_csv": os.path.abspath(args.snr_csv) if args.snr_csv else None,
            "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
            "model_name": args.model_name,
            "input_length": args.input_length,
            "kernel_size": args.kernel_size,
            "preprocess": args.preprocess,
        }, f, indent=2)

    print(f"Rows used: {len(rows)}")
    print(f"Overall accuracy: {overall['accuracy']:.4f}")
    if args.target_accuracy is not None:
        print(f"Target-accuracy SNR threshold: {threshold.get('unknown_snr_threshold_db')}")
    else:
        print(f"Unknown SNR threshold: {threshold.get('unknown_snr_threshold_db')}")
    print(f"CSV: {csv_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
