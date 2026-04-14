"""Inference on full-length doublet signals using zoo-trained models.

Tests whether models trained on short 2500-sample windows (decimated to 625)
can correctly classify longer 16384-sample raw acquisitions that may contain
multiple events (doublets, triplets, etc.).

Strategy:
  1. Slide a 2500-sample window across each 16384-sample signal (non-overlapping).
  2. Apply the same preprocessing (BandpassFilter + Decimate) as training.
  3. Run inference on each window, aggregate via mean softmax.
  4. For 4-class models, take argmax over particle classes only (ignore Noise).
  5. Report per-model accuracy + confusion matrix on 3 particle classes.

Usage:
    # All available zoo checkpoints
    python infer_doublets.py

    # Specific models only
    python infer_doublets.py --models Conv1D ResNet1D

    # Custom checkpoint directory
    python infer_doublets.py --output-dir output

    # Save publication-quality PDF figures
    python infer_doublets.py --save-figures
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from models import create_model
from train import BandpassFilter, Decimate

# ── Constants ────────────────────────────────────────────────────────────────

RAW_SIGNAL_LENGTH = 2500
DECIMATE_FACTOR = 4
INPUT_LENGTH = RAW_SIGNAL_LENGTH // DECIMATE_FACTOR  # 625

DOUBLET_DIRS = {
    "2um": "C1_HF_5_10_2um_doublet2",
    "4um": "C1_HF_5_10_4um_doublet",
    "10um": "C1_HF_5_10_10um_doublet",
}

PARTICLE_CLASSES = ["2um", "4um", "10um"]
FOUR_CLASSES = ["2um", "4um", "10um", "Noise"]

# Checkpoint auto-discovery patterns: (dir_name_regex -> model_name, num_classes)
# Priority order: zoo 4c > S1_white_4c > dataset 3c > dataset_3c rerun
ZOO_CHECKPOINT_PATTERNS = [
    (r"^(Conv1D|Conv1DGAP|LeNet1D|VGG1D|ResNet1D|InceptionTime1D|MobileNet1D|EfficientNet1D|DenseNet1D)-dataset_4c-zoo-\w+$", 4),
    (r"^(Conv1D|Conv1DGAP|LeNet1D|VGG1D|ResNet1D|InceptionTime1D|MobileNet1D|EfficientNet1D|DenseNet1D)-S1_white_4c-run\d+$", 4),
    (r"^(Conv1D|Conv1DGAP|LeNet1D|VGG1D|ResNet1D|InceptionTime1D|MobileNet1D|EfficientNet1D|DenseNet1D)-dataset-run\d+$", 3),
    (r"^(Conv1D|Conv1DGAP|LeNet1D|VGG1D|ResNet1D|InceptionTime1D|MobileNet1D|EfficientNet1D|DenseNet1D)-dataset_3c-\w+$", 3),
]

# ── Publication style (delegated to pub_utils) ─────────────────────────────
from pub_utils import apply_publication_style, plot_confusion_matrix, COL_W


# ── Data loading ─────────────────────────────────────────────────────────────

def load_doublet_signals(base_dir):
    """Load all doublet signals and return (signals, labels, filenames).

    Each signal is a raw 16384-sample numpy array.
    Labels are integer class indices into PARTICLE_CLASSES.
    """
    signals, labels, fnames = [], [], []
    for class_idx, class_name in enumerate(PARTICLE_CLASSES):
        dir_name = DOUBLET_DIRS[class_name]
        class_dir = base_dir / dir_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping {class_name}")
            continue
        npy_files = sorted(class_dir.glob("*.npy"))
        print(f"  {class_name}: {len(npy_files)} signals from {dir_name}/")
        for f in npy_files:
            arr = np.load(f).astype(np.float32)
            signals.append(arr)
            labels.append(class_idx)
            fnames.append(f.name)
    return signals, np.array(labels), fnames


def extract_windows(signal, window_size=RAW_SIGNAL_LENGTH):
    """Extract non-overlapping windows from a raw signal."""
    n_windows = len(signal) // window_size
    windows = []
    for i in range(n_windows):
        start = i * window_size
        windows.append(signal[start:start + window_size])
    return windows


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def discover_checkpoints(output_dir, model_filter=None):
    """Auto-discover available checkpoints.

    Returns list of (model_name, num_classes, checkpoint_path, run_tag).
    Only the best checkpoint per model is kept (zoo > S1_white > dataset).
    """
    found = {}  # model_name -> (num_classes, path, run_tag, priority)

    if not output_dir.exists():
        return []

    for sub in sorted(output_dir.iterdir()):
        if not sub.is_dir():
            continue
        ckpt = sub / "best_model.pth"
        if not ckpt.exists():
            continue

        for priority, (pattern, n_classes) in enumerate(ZOO_CHECKPOINT_PATTERNS):
            m = re.match(pattern, sub.name)
            if m:
                model_name = m.group(1)
                if model_name not in found or priority < found[model_name][3]:
                    found[model_name] = (n_classes, ckpt, sub.name, priority)
                break

    results = []
    for model_name in sorted(found.keys()):
        n_classes, path, run_tag, _ = found[model_name]
        if model_filter and model_name not in model_filter:
            continue
        results.append((model_name, n_classes, path, run_tag))

    return results


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_signal(model, windows_tensor, num_classes, device):
    """Run inference on all windows of a signal and return predicted class.

    Aggregation: mean softmax over all windows, argmax over particle classes.
    """
    windows_tensor = windows_tensor.to(device)
    logits = model(windows_tensor)  # (n_windows, num_classes)
    probs = F.softmax(logits, dim=1)  # (n_windows, num_classes)
    mean_probs = probs.mean(dim=0)  # (num_classes,)

    # Take argmax over particle classes only (first 3)
    particle_probs = mean_probs[:len(PARTICLE_CLASSES)]
    return particle_probs.argmax().item()


def run_inference(model, signals, num_classes, device, bandpass, decimate_fn):
    """Run sliding-window inference on all signals.

    Returns predictions as numpy array.
    """
    model.eval()
    predictions = []

    for sig in signals:
        windows = extract_windows(sig, RAW_SIGNAL_LENGTH)
        if not windows:
            predictions.append(-1)
            continue

        processed = []
        for w in windows:
            t = torch.from_numpy(w[np.newaxis, :])  # (1, 2500)
            t = bandpass(t)
            t = decimate_fn(t)
            processed.append(t)

        windows_tensor = torch.stack(processed, dim=0)  # (n_windows, 1, 625)
        pred = infer_signal(model, windows_tensor, num_classes, device)
        predictions.append(pred)

    return np.array(predictions)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix_pub(cm, class_names, model_name, figures_dir=None):
    """Plot a publication-quality confusion matrix (PDF vector output)."""
    fig, ax = plot_confusion_matrix(cm, class_names)

    if figures_dir is not None:
        slug = model_name.replace(" ", "_")
        out_path = figures_dir / f"doublet_cm_{slug}.pdf"
        fig.savefig(out_path)
        print(f"  Saved {out_path}")

    plt.close(fig)


def plot_accuracy_summary(results, figures_dir=None):
    """Horizontal bar chart of per-model accuracy on doublets (PDF)."""
    apply_publication_style()

    models = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    sorted_idx = np.argsort(accs)
    models = [models[i] for i in sorted_idx]
    accs = [accs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))
    bars = ax.barh(range(len(models)), accs, color="#0072B2", edgecolor="white",
                   linewidth=0.4, height=0.7)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}", va="center", fontsize=7)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.32, right=0.95, top=0.96, bottom=0.12)

    if figures_dir is not None:
        out_path = figures_dir / "doublet_accuracy_summary.pdf"
        fig.savefig(out_path)
        print(f"  Saved {out_path}")

    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference on full-length doublet signals with zoo models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to test (default: all discovered)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory containing model checkpoints")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Base dir containing C1_HF_* doublet folders")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save PDF confusion matrices and accuracy bar chart")
    parser.add_argument("--figures-dir", type=str,
                        default="results/doublet_inference",
                        help="Output directory for figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load doublet data
    print("\nLoading doublet signals...")
    signals, labels, fnames = load_doublet_signals(base_dir)
    n_total = len(signals)
    print(f"  Total: {n_total} signals across {len(PARTICLE_CLASSES)} classes")

    if n_total == 0:
        print("ERROR: No signals found. Check --data-dir.")
        return

    # Preprocessing (same as training)
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0,
                              sample_rate_mhz=2.0)
    decimate_fn = Decimate(decimate=DECIMATE_FACTOR)

    # Discover checkpoints
    checkpoints = discover_checkpoints(output_dir, args.models)
    if not checkpoints:
        print(f"\nERROR: No checkpoints found in {output_dir}/")
        if output_dir.exists():
            print("Available directories:")
            for d in sorted(output_dir.iterdir()):
                if d.is_dir() and (d / "best_model.pth").exists():
                    print(f"  {d.name}")
        return

    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for model_name, n_classes, path, run_tag in checkpoints:
        print(f"  {model_name} ({n_classes}c) <- {run_tag}")

    figures_dir = None
    if args.save_figures:
        figures_dir = Path(args.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    # Run inference for each model
    results = []
    n_windows = 16384 // RAW_SIGNAL_LENGTH
    class_counts = ", ".join(
        f"{c}: {int((labels == i).sum())}"
        for i, c in enumerate(PARTICLE_CLASSES)
    )
    print(f"\n{'=' * 72}")
    print(f"  Doublet inference -- {n_total} signals ({class_counts})")
    print(f"  Windows per signal: {n_windows} "
          f"(non-overlapping, {RAW_SIGNAL_LENGTH} samples each)")
    print(f"{'=' * 72}")

    for model_name, n_classes, ckpt_path, run_tag in checkpoints:
        print(f"\n--- {model_name} ({n_classes}-class, from {run_tag}) ---")

        model = create_model(model_name, input_length=INPUT_LENGTH,
                             num_classes=n_classes).to(device)
        state_dict = torch.load(ckpt_path, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        preds = run_inference(model, signals, n_classes, device,
                              bandpass, decimate_fn)
        acc = float((preds == labels).mean())
        cm = confusion_matrix(labels, preds,
                              labels=list(range(len(PARTICLE_CLASSES))))

        correct = int((preds == labels).sum())
        print(f"  Accuracy: {acc:.4f} ({correct}/{n_total})")
        print(f"  Confusion matrix:\n{cm}")
        print(classification_report(labels, preds,
                                    target_names=PARTICLE_CLASSES, digits=4))

        results.append({
            "model": model_name,
            "n_classes": n_classes,
            "run_tag": run_tag,
            "accuracy": acc,
            "cm": cm,
        })

        if figures_dir is not None:
            plot_confusion_matrix_pub(cm, PARTICLE_CLASSES, model_name,
                                     figures_dir)

    # Summary table
    print(f"\n{'=' * 72}")
    print("  SUMMARY -- Doublet Inference Accuracy")
    print(f"{'=' * 72}")
    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    for i, r in enumerate(results_sorted):
        marker = " <-- best" if i == 0 else ""
        print(f"  {r['model']:20s}  {r['accuracy']:.4f}  "
              f"({r['n_classes']}c, {r['run_tag']}){marker}")

    if figures_dir is not None:
        plot_accuracy_summary(results, figures_dir)
        print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
