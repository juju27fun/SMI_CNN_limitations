"""Re-log 4 publication-quality plots to existing W&B runs.

For each target dataset, this script:
  1. Loads the Conv1D model checkpoint
  2. Re-computes only what the 4 target plots need
  3. Generates publication-quality figures (Okabe-Ito, PUB_RC, etc.)
  4. Resumes the existing W&B run and logs at a new step

Target plots:
  - noise_ood/mahalanobis_histogram
  - noise_ood/prediction_distribution
  - noise_separation/tsne  (+ noise_separation/pca)
  - cluster_distances/cosine_distance_heatmap

Usage:
    python archive/relog_pub_plots.py                              # all 3 datasets
    python archive/relog_pub_plots.py --datasets S1_white          # single dataset
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `train` and `models` are importable
# regardless of which directory the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train import (
    RAW_SIGNAL_LENGTH,
    ParticleDataset,
    BandpassFilter,
    Decimate,
    Truncate,
)
from models import Conv1DClassifier


# ── Constants ────────────────────────────────────────────────────────────────

TARGET_DATASETS = ["S1_white", "S2_colored", "dataset"]

CLASS_NAMES = ["2um", "4um", "10um"]
CLUSTER_NAMES = CLASS_NAMES + ["Noise"]

PUB_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
}

# Okabe-Ito CVD-safe colors
COLOR_ID = "#0072B2"       # blue  — in-distribution
COLOR_NOISE = "#D55E00"    # vermillion — noise
BAR_COLORS = ["#0072B2", "#E69F00", "#009E73"]  # blue, orange, bluish green

# Scatter plot colors/markers for class distinction (§5 three-channel)
_SCATTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
_SCATTER_MARKERS = ["o", "s", "^", "D", "v"]

FIG_SINGLE = (3.39, 2.10)
COL_W = 3.39


# ── Computation helpers (inlined from archive/benchmark.py) ──────────────────

def _safe_bins(vals, target=50):
    """Return bin count that won't fail on constant-valued arrays."""
    if len(vals) == 0 or np.ptp(vals) < 1e-10:
        return 1
    return min(target, max(1, int(np.sqrt(len(vals)))))


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


def extract_features(model, loader, device):
    """Extract fc1 features (256-dim) from the model for all samples."""
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


def _extract_multilayer_features(model, loader, device):
    """Extract features from pool1, pool2, pool3 (with GAP) and fc1."""
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


def compute_mahalanobis_scores(model, id_loader, noise_loader, device,
                               train_loader=None):
    """Multi-layer Mahalanobis distance OOD detector (Lee et al., 2018).

    Returns (id_scores, noise_scores) where higher = more in-distribution.
    """
    ref_loader = train_loader if train_loader is not None else id_loader

    ref_layers, ref_labels = _extract_multilayer_features(model, ref_loader, device)
    id_layers, _ = _extract_multilayer_features(model, id_loader, device)
    noise_layers, _ = _extract_multilayer_features(model, noise_loader, device)

    num_classes = int(ref_labels.max()) + 1
    id_total = np.zeros(len(id_layers[0]))
    noise_total = np.zeros(len(noise_layers[0]))

    for ref_feats, id_feats, noise_feats in zip(ref_layers, id_layers, noise_layers):
        class_means = []
        for c in range(num_classes):
            mask = ref_labels == c
            class_means.append(ref_feats[mask].mean(axis=0))
        class_means_arr = np.stack(class_means)

        centered = ref_feats.astype(np.float64) - class_means_arr[ref_labels.astype(int)]
        cov = np.cov(centered, rowvar=False)
        reg = max(1e-6, 1e-6 * np.trace(cov) / cov.shape[0])
        cov += reg * np.eye(cov.shape[0])
        cov_inv = np.linalg.inv(cov)

        for feats, accum in [(id_feats.astype(np.float64), "id"),
                             (noise_feats.astype(np.float64), "noise")]:
            scores = np.full(len(feats), -np.inf)
            for c in range(num_classes):
                diff = feats - class_means_arr[c]
                maha = -np.sum(diff @ cov_inv * diff, axis=1)
                scores = np.maximum(scores, maha)
            if accum == "id":
                id_total += scores
            else:
                noise_total += scores

    return id_total, noise_total


# ── W&B helpers ──────────────────────────────────────────────────────────────

def get_latest_run_id(dataset_name):
    """Return the W&B run ID of the most recent run named Conv1D-{dataset}-run1."""
    api = wandb.Api()
    runs = api.runs(
        "julien-leboulch03-laas-cnrs/particle-benchmark",
        filters={"display_name": f"Conv1D-{dataset_name}-run1"},
        order="-created_at",
    )
    run_list = list(runs)
    if not run_list:
        raise ValueError(f"No W&B run found for dataset '{dataset_name}'")
    return run_list[0].id


# ── Distance helpers ─────────────────────────────────────────────────────────

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


# ── Plot generation ──────────────────────────────────────────────────────────

def generate_plots(run, model, test_loader, noise_loader, train_loader, device):
    """Generate and log the 4 publication-quality plots to the W&B run."""

    plt.rcParams.update(PUB_RC)

    # ── Shared computation: feature extraction ──
    print("    Extracting fc1 features (test)...")
    test_feats, test_labels = extract_features(model, test_loader, device)
    print(f"    Test features: {test_feats.shape}")

    print("    Extracting fc1 features (noise)...")
    noise_feats, _ = extract_features(model, noise_loader, device)
    print(f"    Noise features: {noise_feats.shape}")

    n_id = len(test_feats)
    n_noise = len(noise_feats)

    # ── 1) Mahalanobis histogram ─────────────────────────────────────────
    print("    Computing Mahalanobis scores...")
    maha_id, maha_noise = compute_mahalanobis_scores(
        model, test_loader, noise_loader, device, train_loader=train_loader
    )

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.hist(maha_id, bins=_safe_bins(maha_id), alpha=0.6,
            label=f"In-dist (n={n_id})", color=COLOR_ID, density=True)
    ax.hist(maha_noise, bins=_safe_bins(maha_noise), alpha=0.6,
            label=f"Noise (n={n_noise})", color=COLOR_NOISE, density=True)
    ax.set_xlabel("Mahalanobis Score")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
    run.log({"noise_ood/mahalanobis_histogram": wandb.Image(fig)}, commit=False)
    plt.close(fig)
    print("    [1/4] Mahalanobis histogram logged.")

    # ── 2) Prediction distribution bar chart ─────────────────────────────
    print("    Computing noise predictions...")
    _, _, noise_preds = compute_ood_scores(model, noise_loader, device)

    noise_class_counts = np.bincount(noise_preds, minlength=len(CLASS_NAMES))
    noise_class_pcts = noise_class_counts / n_noise * 100

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.bar(CLASS_NAMES, noise_class_pcts,
                  color=BAR_COLORS[:len(CLASS_NAMES)], edgecolor="white")
    for bar, pct in zip(bars, noise_class_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("% of noise samples")
    ax.set_ylim(0, max(noise_class_pcts) * 1.2 if max(noise_class_pcts) > 0 else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
    run.log({"noise_ood/prediction_distribution": wandb.Image(fig)}, commit=False)
    plt.close(fig)
    print("    [2/4] Prediction distribution logged.")

    # ── 3) Noise-separation t-SNE (+ PCA) ────────────────────────────────
    print("    Computing PCA + t-SNE (this may take a moment)...")
    combined_feats = np.concatenate([test_feats, noise_feats], axis=0)
    combined_labels = np.concatenate([
        test_labels,
        np.full(n_noise, len(CLASS_NAMES)),  # Noise label = 3
    ])

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_feats)

    pca_fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for i, cls in enumerate(CLUSTER_NAMES):
        mask = combined_labels == i
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   label=cls, alpha=0.6, s=20,
                   color=_SCATTER_COLORS[i % len(_SCATTER_COLORS)],
                   marker=_SCATTER_MARKERS[i % len(_SCATTER_MARKERS)],
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    pca_fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
    run.log({"noise_separation/pca": wandb.Image(pca_fig)}, commit=False)
    plt.close(pca_fig)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(combined_feats)

    tsne_fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for i, cls in enumerate(CLUSTER_NAMES):
        mask = combined_labels == i
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   label=cls, alpha=0.6, s=20,
                   color=_SCATTER_COLORS[i % len(_SCATTER_COLORS)],
                   marker=_SCATTER_MARKERS[i % len(_SCATTER_MARKERS)],
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    tsne_fig.subplots_adjust(left=0.18, right=0.96, top=0.96, bottom=0.22)
    run.log({"noise_separation/tsne": wandb.Image(tsne_fig)}, commit=False)
    plt.close(tsne_fig)
    print("    [3/4] PCA + t-SNE logged.")

    # ── 4) Cosine distance heatmap ───────────────────────────────────────
    print("    Computing cluster centroids and cosine distances...")
    centroids = []
    for c, cls in enumerate(CLASS_NAMES):
        mask = test_labels == c
        n_samples = int(mask.sum())
        if n_samples == 0:
            print(f"    WARNING: no samples for class {cls}, using zero centroid")
            centroids.append(np.zeros(test_feats.shape[1]))
        else:
            centroids.append(test_feats[mask].mean(axis=0))
        print(f"    Centroid {cls:5s}: {n_samples} samples")

    noise_centroid = noise_feats.mean(axis=0)
    centroids.append(noise_centroid)
    print(f"    Centroid Noise: {n_noise} samples")

    mat = compute_distance_matrix(centroids)
    n = len(CLUSTER_NAMES)

    # Strict lower triangle (no diagonal): rows 1..n-1, cols 0..n-2
    tri = mat[1:, :-1].copy()                          # (n-1) x (n-1)
    tri_mask = np.triu(np.ones_like(tri, dtype=bool), k=1)
    tri[tri_mask] = np.nan

    row_labels = CLUSTER_NAMES[1:]   # 4um, 10um, Noise
    col_labels = CLUSTER_NAMES[:-1]  # 2um, 4um, 10um

    # Power norm (gamma < 1) expands low values → small distances show
    # visibly more blue than with a linear scale, while keeping good
    # separation across the full range.  Seaborn "Blues" palette.
    from matplotlib.colors import PowerNorm
    vmax = np.nanmax(tri)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    im = ax.imshow(tri, cmap=cmap, norm=norm, aspect="equal")

    # Horizontal colorbar below the heatmap
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.18,
                      shrink=0.85)
    cb.set_label("Cosine distance", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    nr, nc = tri.shape
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate visible cells (1 decimal place)
    for i in range(nr):
        for j in range(nc):
            if tri_mask[i, j]:
                continue
            val = tri[i, j]
            rgba = cmap(norm(val))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "white" if lum < 0.45 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_color)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.96, bottom=0.26)
    run.log({"cluster_distances/cosine_distance_heatmap": wandb.Image(fig)},
            commit=True)
    plt.close(fig)
    print("    [4/4] Cosine distance heatmap logged (commit=True).")


# ── Per-dataset processing ───────────────────────────────────────────────────

def process_dataset(dataset_name, output_dir, data_root, noise_dir, device,
                    batch_size=32, decimate=4):
    """Load model + data, resume W&B run, and log the 4 updated plots."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    model_path = output_dir / f"Conv1D-{dataset_name}-run1" / "best_model.pth"
    if not model_path.exists():
        print(f"  WARNING: model not found at {model_path}, skipping.")
        return

    input_length = RAW_SIGNAL_LENGTH // decimate

    # Transforms
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0,
                              sample_rate_mhz=2.0)
    decimate_t = Decimate(decimate=decimate)
    truncate = Truncate(RAW_SIGNAL_LENGTH)

    # Model
    model = Conv1DClassifier(input_length=input_length,
                             num_classes=len(CLASS_NAMES)).to(device)
    # strict=False: old checkpoints lack the feature_layer alias keys
    # (feature_layer is just an alias for fc1, so missing keys are harmless)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True), strict=False)
    model.train(False)
    print(f"  Loaded model from {model_path}")

    # Data loaders
    data_dir = data_root / dataset_name
    test_dataset = ParticleDataset(
        data_dir / "test", CLASS_NAMES, transforms=[bandpass, decimate_t]
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    train_dataset = ParticleDataset(
        data_dir / "train", CLASS_NAMES, transforms=[bandpass, decimate_t]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4)

    noise_dataset = ParticleDataset(
        noise_dir.parent, [noise_dir.name],
        transforms=[truncate, bandpass, decimate_t],
    )
    noise_loader = DataLoader(noise_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # Resume W&B run and log updated plots
    print("  Fetching W&B run ID...")
    run_id = get_latest_run_id(dataset_name)
    print(f"  Resuming run {run_id}...")

    with wandb.init(
        settings=wandb.Settings(init_timeout=180),
        project="particle-benchmark",
        id=run_id,
        resume="must",
    ) as run:
        generate_plots(run, model, test_loader, noise_loader, train_loader,
                       device)

    print("  Done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-log 4 publication-quality plots to existing W&B runs."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=TARGET_DATASETS,
        help="Datasets to process (default: S1_white S2_colored dataset)",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory containing dataset folders and Noise/")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Path to noise folder (default: <data-root>/Noise)")
    parser.add_argument("--decimate", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    noise_dir = Path(args.noise_dir) if args.noise_dir else data_root / "Noise"

    for dataset_name in args.datasets:
        process_dataset(
            dataset_name,
            output_dir=output_dir,
            data_root=data_root,
            noise_dir=noise_dir,
            device=device,
            batch_size=args.batch_size,
            decimate=args.decimate,
        )

    print(f"\n{'='*60}")
    print("  All 4 plots re-logged for all datasets.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
