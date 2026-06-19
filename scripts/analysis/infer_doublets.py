"""Inference on full-length doublet signals using zoo-trained models.

Tests whether models trained on short segments can correctly classify longer
16384-sample raw acquisitions that may contain multiple events (doublets,
triplets, etc.).

Strategy (direct full-signal inference -- no sliding window):
  1. Apply BandpassFilter + Decimate(4) to the full 16384-sample signal -> 4096.
  2. Run a single forward pass through the GAP model.
  3. For 4-class models, take argmax over particle classes only (ignore Noise).
  4. Report per-model accuracy + confusion matrix on 3 particle classes.

All 6 supported architectures use AdaptiveAvgPool1d(1) and accept any input
length, so the 4096-sample decimated signal feeds directly into the model.

Usage:
    # All available zoo checkpoints
    python infer_doublets.py

    # Specific models only
    python infer_doublets.py --models Conv1DGAP ResNet1D

    # Custom checkpoint directory
    python infer_doublets.py --output-dir output

    # Save publication-quality PDF figures
    python infer_doublets.py --save-figures
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from p0.models import create_model, get_family
from p0.data import AdaptiveBandpassDecimate, BandpassFilter, CenterCrop, Decimate

# -- Constants ----------------------------------------------------------------

DECIMATE_FACTOR = 4

PARTICLE_CLASSES = ["2um", "4um", "10um"]
FOUR_CLASSES = ["2um", "4um", "10um", "Noise"]

# Checkpoint auto-discovery patterns: (dir_name_regex -> model_name, num_classes)
# Priority order: zoo 4c > S1_white_4c > dataset 3c > dataset_3c rerun
_GAP_MODELS = r"((?:Conv1DGAP|ResNet1D|InceptionTime1D|MobileNet1D|EfficientNet1D|DenseNet1D|Swin1D|PatchTST)(?:-\w+)?)"
ZOO_CHECKPOINT_PATTERNS = [
    # Doublet-trained (highest priority for doublet inference)
    (rf"^{_GAP_MODELS}-L\d+-dataset_doublet-tier\d+-seed\d+$", 3),
    (rf"^{_GAP_MODELS}-dataset_doublet-\w+$", 3),
    # Zoo / benchmark checkpoints (fallback)
    (rf"^{_GAP_MODELS}-dataset_4c-zoo-\w+$", 4),
    (rf"^{_GAP_MODELS}-S1_white_4c-run\d+$", 4),
    (rf"^{_GAP_MODELS}-L\d+-dataset-tier\d+-seed\d+$", 3),
    (rf"^{_GAP_MODELS}-dataset-tier\d+-seed\d+$", 3),
    (rf"^{_GAP_MODELS}-dataset-run\d+$", 3),
    (rf"^{_GAP_MODELS}-dataset_3c-\w+$", 3),
]

# Kernel-length sweep pattern: captures variant + kernel_size (k) + input_length (L).
# Variant suffix is optional and restricted to the known size tags to keep the
# boundary unambiguous with the trailing -k\d+ marker.
_FAMILY_ALT = r"Conv1DGAP|ResNet1D|InceptionTime1D|MobileNet1D|EfficientNet1D|DenseNet1D|Swin1D|PatchTST"
_VARIANT_OPT = r"(?:-(?:Pico|Nano|XXS|XS|S|L))?"
SWEEP_PATTERN = re.compile(
    rf"^(?P<model>(?:{_FAMILY_ALT}){_VARIANT_OPT})"
    rf"-k(?P<k>\d+)-L(?P<L>\d+)"
    rf"(?P<decim>-decim)?"
    rf"-(?P<dataset>\w+?)-tier(?P<tier>\d+)-seed(?P<seed>\d+)$"
)

# -- Publication style (delegated to pub_utils) --------------------------------
from p0.plotting import apply_publication_style, plot_confusion_matrix, COL_W


# -- Data loading --------------------------------------------------------------

def load_doublet_signals(data_dir):
    """Load doublet signals from a standard dataset directory.

    Expects the layout ``data_dir/{class_name}/*.npy`` (e.g. the test split
    of ``data/dataset_doublet/test/``).

    Returns (signals, labels, filenames).
    """
    signals, labels, fnames = [], [], []
    for class_idx, class_name in enumerate(PARTICLE_CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping {class_name}")
            continue
        npy_files = sorted(class_dir.glob("*.npy"))
        print(f"  {class_name}: {len(npy_files)} signals from {class_dir.name}/")
        for f in npy_files:
            arr = np.load(f).astype(np.float32)
            signals.append(arr)
            labels.append(class_idx)
            fnames.append(f.name)
    return signals, np.array(labels), fnames


# -- Checkpoint discovery ------------------------------------------------------

def discover_checkpoints(output_dir, model_filter=None, tag_filter=None):
    """Auto-discover available checkpoints.

    Returns list of (model_name, num_classes, checkpoint_path, run_tag).
    Only the best checkpoint per model is kept (priority: doublet > zoo > …).

    If *tag_filter* is given, only directories whose name contains the
    substring are considered (e.g. ``tag_filter="zoo"``).
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
        if tag_filter and tag_filter not in sub.name:
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


def discover_sweep_checkpoints(output_dir, model_filter=None):
    """Discover kernel-length sweep checkpoints.

    Each directory name is expected to match :data:`SWEEP_PATTERN`, which carries
    the kernel size and input length as part of the run tag. Returns a list of
    dicts with keys: ``model_name``, ``family``, ``kernel_size``,
    ``input_length``, ``num_classes``, ``ckpt_path``, ``run_tag``, ``seed``.
    """
    results = []
    if not output_dir.exists():
        return results

    for sub in sorted(output_dir.iterdir()):
        if not sub.is_dir():
            continue
        ckpt = sub / "best_model.pth"
        if not ckpt.exists():
            continue
        m = SWEEP_PATTERN.match(sub.name)
        if not m:
            continue
        model_name = m.group("model")
        if model_filter and model_name not in model_filter:
            continue
        results.append({
            "model_name": model_name,
            "family": get_family(model_name),
            "kernel_size": int(m.group("k")),
            "input_length": int(m.group("L")),
            "decim_mode": bool(m.group("decim")),
            "num_classes": 3,  # sweep is 3-class (2um/4um/10um)
            "ckpt_path": ckpt,
            "run_tag": sub.name,
            "seed": int(m.group("seed")),
        })
    return results


# -- Inference -----------------------------------------------------------------

@torch.no_grad()
def infer_signal(model, signal_tensor, device):
    """Run a single forward pass and return predicted particle class index."""
    signal_tensor = signal_tensor.to(device)
    logits = model(signal_tensor)              # (1, num_classes)
    probs = F.softmax(logits, dim=1)[0]        # (num_classes,)
    # Argmax over particle classes only (first 3), ignore Noise if 4-class
    particle_probs = probs[:len(PARTICLE_CLASSES)]
    return particle_probs.argmax().item()


def run_inference(model, signals, device, bandpass, decimate_fn):
    """Run direct full-signal inference on all signals.

    Returns predictions as numpy array.
    """
    model.eval()
    predictions = []

    for sig in signals:
        t = torch.from_numpy(sig[np.newaxis, :])  # (1, raw_length)
        t = bandpass(t)
        t = decimate_fn(t)                         # (1, raw_length // 4)
        t = t.unsqueeze(0)                         # (1, 1, decimated_length)
        pred = infer_signal(model, t, device)
        predictions.append(pred)

    return np.array(predictions)


# -- Plotting ------------------------------------------------------------------

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


# -- Sweep inference -----------------------------------------------------------

def _run_sweep_inference(output_dir, signals, labels, n_total,
                         bandpass, decimate_fn, device, args):
    """Infer every kernel-length sweep checkpoint and emit a JSON + figures.

    For each discovered checkpoint, the pre-processing chain matches training:
    bandpass -> decimate(4) -> CenterCrop(input_length). The per-model
    kernel_size is forwarded to :func:`create_model`.
    """
    entries = discover_sweep_checkpoints(output_dir, args.models)
    if not entries:
        print(f"\nERROR: No sweep checkpoints found in {output_dir}/")
        if output_dir.exists():
            print("Available directories (must match -k{K}-L{L}-):")
            for d in sorted(output_dir.iterdir()):
                if d.is_dir() and (d / "best_model.pth").exists():
                    print(f"  {d.name}")
        return

    # Signal length is the native acquisition size (e.g. 16384 for dataset_doublet).
    native_length = int(signals[0].shape[-1])

    print(f"\nSweep mode: {len(entries)} checkpoint(s) discovered")
    for e in entries:
        tag = "decim" if e["decim_mode"] else "crop"
        print(f"  {e['model_name']:20s}  k={e['kernel_size']:2d}  "
              f"L={e['input_length']:5d}  [{tag}]  <-  {e['run_tag']}")

    figures_dir = None
    if args.save_figures:
        figures_dir = Path(args.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for e in entries:
        print(f"\n--- {e['run_tag']} ---")
        L = e["input_length"]
        # Build the pre-processing chain to match training exactly.
        if e["decim_mode"]:
            transform = AdaptiveBandpassDecimate(
                target_length=L, native_length=native_length,
            )
        else:
            crop = CenterCrop(L)
            transform = lambda t, _bp=bandpass, _dec=decimate_fn, _crop=crop: \
                _crop(_dec(_bp(t)))
        model = create_model(
            e["model_name"], input_length=L,
            num_classes=e["num_classes"], kernel_size=e["kernel_size"],
        ).to(device)
        state = torch.load(e["ckpt_path"], map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        preds = []
        with torch.no_grad():
            for sig in signals:
                t = torch.from_numpy(sig[np.newaxis, :])
                t = transform(t)
                t = t.unsqueeze(0).to(device)
                logits = model(t)
                p = F.softmax(logits, dim=1)[0, :len(PARTICLE_CLASSES)].argmax().item()
                preds.append(p)
        preds = np.array(preds)

        acc = float((preds == labels).mean())
        cm = confusion_matrix(labels, preds, labels=list(range(len(PARTICLE_CLASSES))))
        print(f"  Accuracy: {acc:.4f} ({int((preds == labels).sum())}/{n_total})")

        results.append({
            "model_name": e["model_name"],
            "family": e["family"],
            "kernel_size": e["kernel_size"],
            "input_length": L,
            "decim_mode": e["decim_mode"],
            "run_tag": e["run_tag"],
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "n_total": n_total,
        })

        if figures_dir is not None:
            slug = f"{e['model_name']}_k{e['kernel_size']}_L{L}"
            plot_confusion_matrix_pub(cm, PARTICLE_CLASSES, slug, figures_dir)

    out_path = Path(args.sweep_out) if args.sweep_out else output_dir.parent / "doublet_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep results written to {out_path}")

    # Summary table (sorted by accuracy)
    print(f"\n{'=' * 72}")
    print("  SUMMARY -- Sweep Doublet Accuracy")
    print(f"{'=' * 72}")
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"  {r['model_name']:20s}  k={r['kernel_size']:2d}  "
              f"L={r['input_length']:5d}  acc={r['accuracy']:.4f}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference on full-length doublet signals with zoo models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to test (default: all discovered)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory containing model checkpoints")
    parser.add_argument("--data-dir", type=str,
                        default="data/dataset_doublet/test",
                        help="Directory with {class}/*.npy layout (default: test split)")
    parser.add_argument("--checkpoint-tag", type=str, default=None,
                        help="Only use checkpoints whose run tag contains this "
                             "substring (e.g. 'zoo' or 'doublet')")
    parser.add_argument("--save-figures", action="store_true",
                        help="Save PDF confusion matrices and accuracy bar chart")
    parser.add_argument("--figures-dir", type=str,
                        default="results/doublet_inference",
                        help="Output directory for figures")
    parser.add_argument("--sweep", action="store_true",
                        help="Discover kernel-length sweep checkpoints (dir "
                             "names carrying -k{K}-L{L}-) and apply matching "
                             "CenterCrop + kernel_size at inference")
    parser.add_argument("--sweep-out", type=str,
                        default=None,
                        help="Path to write sweep JSON results (default: "
                             "<output-dir>/../doublet_accuracy.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load doublet data
    print(f"\nLoading doublet signals from {data_dir} ...")
    signals, labels, fnames = load_doublet_signals(data_dir)
    n_total = len(signals)
    print(f"  Total: {n_total} signals across {len(PARTICLE_CLASSES)} classes")

    if n_total == 0:
        print("ERROR: No signals found. Check --data-dir.")
        return

    # Preprocessing (same as training)
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0,
                              sample_rate_mhz=2.0)
    decimate_fn = Decimate(decimate=DECIMATE_FACTOR)

    # Kernel-length sweep flow — dedicated loop with per-run CenterCrop + kernel.
    if args.sweep:
        _run_sweep_inference(
            output_dir, signals, labels, n_total,
            bandpass, decimate_fn, device, args,
        )
        return

    # Discover checkpoints
    checkpoints = discover_checkpoints(output_dir, args.models, args.checkpoint_tag)
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

    # Determine decimated signal length from first signal
    sig_len = len(signals[0])
    dec_len = sig_len // DECIMATE_FACTOR
    class_counts = ", ".join(
        f"{c}: {int((labels == i).sum())}"
        for i, c in enumerate(PARTICLE_CLASSES)
    )
    print(f"\n{'=' * 72}")
    print(f"  Doublet inference -- {n_total} signals ({class_counts})")
    print(f"  Full-signal mode: {sig_len} raw -> {dec_len} decimated samples")
    print(f"{'=' * 72}")

    # Run inference for each model
    results = []
    for model_name, n_classes, ckpt_path, run_tag in checkpoints:
        print(f"\n--- {model_name} ({n_classes}-class, from {run_tag}) ---")

        model = create_model(model_name, input_length=dec_len,
                             num_classes=n_classes).to(device)
        state_dict = torch.load(ckpt_path, map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        preds = run_inference(model, signals, device, bandpass, decimate_fn)
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
