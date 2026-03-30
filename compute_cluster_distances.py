"""Compute pairwise cosine distances between cluster centroids in the fc1 feature space.

For each benchmark run, this script:
  1. Loads the saved best_model.pth from output/
  2. Extracts 256-dim fc1 features for test samples (2um, 4um, 10um) + noise
  3. Computes the centroid of each cluster (mean feature vector)
  4. Computes the 4x4 pairwise cosine distance matrix
  5. Resumes the existing W&B run and logs a lower-triangular table + heatmap

Cosine distance = 1 - cosine_similarity = 1 - (A·B) / (||A||*||B||)
Computed between centroids, the natural representative point of a cluster.

Usage:
    python compute_cluster_distances.py                          # all 12 datasets
    python compute_cluster_distances.py --datasets S0_baseline  # single dataset
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader

from train import (
    RAW_SIGNAL_LENGTH,
    ParticleDataset,
    Conv1DClassifier,
    BandpassFilter,
    Decimate,
    Truncate,
)


# ── Constants ────────────────────────────────────────────────────────────────

ALL_DATASETS = [
    "dataset",
    "S0_baseline",
    "S1_white",
    "S2_colored",
    "S3_realistic",
    "S4_real_noise",
    "S5_signal_realism",
    "S6_noise_realism",
    "S7_pure_real",
    "S8_colored_low",
    "S9_colored_high",
    "S_union",
]

CLASS_NAMES = ["2um", "4um", "10um"]
CLUSTER_NAMES = CLASS_NAMES + ["Noise"]  # 4 clusters total


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_fc1_features(model, loader, device):
    """Extract fc1 output (256-dim) for all samples via forward hook.

    Returns:
        features: np.ndarray, shape (N, 256)
        labels:   np.ndarray, shape (N,) -- class indices from the loader
    """
    model.train(False)
    activations = []
    all_labels = []

    def hook_fn(m, inp, out):
        activations.append(out.detach().cpu())

    hook = model.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            model(signals)
            all_labels.extend(labels.numpy())
    hook.remove()

    return torch.cat(activations, dim=0).numpy(), np.array(all_labels)


# ── Distance computation ─────────────────────────────────────────────────────

def cosine_distance(a, b):
    """Cosine distance between two vectors: 1 - cos(theta) in [0, 2]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def compute_distance_matrix(centroids):
    """Compute the full pairwise cosine distance matrix between centroids.

    Returns an (n, n) symmetric matrix where entry [i,j] = cosine_distance(c_i, c_j).
    Diagonal is 0.
    """
    n = len(centroids)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            d = cosine_distance(centroids[i], centroids[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ── W&B logging ──────────────────────────────────────────────────────────────

def log_distance_matrix(run, mat, dataset_name):
    """Log the cosine distance matrix to W&B as a lower-triangular table and heatmap.

    Args:
        run:          active wandb Run (already initialised with resume)
        mat:          (4, 4) numpy array -- full symmetric distance matrix
        dataset_name: string used in plot titles
    """
    n = len(CLUSTER_NAMES)

    # ── Lower-triangular W&B Table ──
    # Upper triangle is left blank; diagonal shows 0
    columns = ["cluster"] + CLUSTER_NAMES
    rows = []
    for i, row_name in enumerate(CLUSTER_NAMES):
        row = [row_name]
        for j in range(n):
            if j < i:
                row.append(round(float(mat[i, j]), 4))
            elif j == i:
                row.append(0.0)
            else:
                row.append(None)
        rows.append(row)

    run.log({
        "cluster_distances/cosine_distance_table": wandb.Table(
            columns=columns, data=rows
        )
    }, commit=False)

    # ── Scalar summary entries for each pairwise distance ──
    for i in range(n):
        for j in range(i):
            key = f"cluster_distances/cosine_{CLUSTER_NAMES[i].lower()}_vs_{CLUSTER_NAMES[j].lower()}"
            run.summary[key] = round(float(mat[i, j]), 4)

    # ── Heatmap (full symmetric, annotated) ──
    display = mat.copy()
    masked = np.ma.array(display, mask=np.isnan(display))

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine distance")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLUSTER_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLUSTER_NAMES)
    ax.set_title(f"Cluster Cosine Distances — {dataset_name}")

    for i in range(n):
        for j in range(n):
            val = display[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color)

    fig.tight_layout()
    run.log({"cluster_distances/cosine_distance_heatmap": wandb.Image(fig)}, commit=False)
    plt.close(fig)
    print("  Logged table and heatmap to W&B")


# ── Per-dataset processing ────────────────────────────────────────────────────

def get_latest_run_id(dataset_name):
    """Return the W&B run ID of the most recent finished run named Conv1D-{dataset}-run1."""
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


def process_dataset(dataset_name, output_dir, noise_dir, device, batch_size=32, decimate=4):
    """Full pipeline for one dataset: extract features, compute distances, log to W&B."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    # ── Locate model checkpoint ──
    model_path = output_dir / f"Conv1D-{dataset_name}-run1" / "best_model.pth"
    if not model_path.exists():
        print(f"  WARNING: model not found at {model_path}, skipping.")
        return

    input_length = RAW_SIGNAL_LENGTH // decimate

    # ── Build transforms ──
    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0, sample_rate_mhz=2.0)
    decimate_t = Decimate(decimate=decimate)
    truncate = Truncate(RAW_SIGNAL_LENGTH)

    # ── Load model ──
    model = Conv1DClassifier(input_length=input_length, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.train(False)
    print(f"  Loaded model from {model_path}")

    # ── Test set loader (2um / 4um / 10um) ──
    test_dataset = ParticleDataset(
        Path(dataset_name) / "test",
        CLASS_NAMES,
        transforms=[bandpass, decimate_t],
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ── Noise loader ──
    noise_dataset = ParticleDataset(
        noise_dir.parent,
        [noise_dir.name],
        transforms=[truncate, bandpass, decimate_t],
    )
    noise_loader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ── Extract features ──
    print("  Extracting fc1 features...")
    test_feats, test_labels = extract_fc1_features(model, test_loader, device)
    noise_feats, _ = extract_fc1_features(model, noise_loader, device)
    print(f"  Test:  {test_feats.shape}  |  Noise: {noise_feats.shape}")

    # ── Compute per-cluster centroids (mean feature vector) ──
    centroids = []
    for c, cls in enumerate(CLASS_NAMES):
        mask = test_labels == c
        n_samples = int(mask.sum())
        if n_samples == 0:
            print(f"  WARNING: no samples for class {cls}, using zero centroid")
            centroids.append(np.zeros(test_feats.shape[1]))
        else:
            centroids.append(test_feats[mask].mean(axis=0))
        print(f"  Centroid {cls:5s}: {n_samples} samples")

    noise_centroid = noise_feats.mean(axis=0)
    centroids.append(noise_centroid)
    print(f"  Centroid Noise: {len(noise_feats)} samples")

    # ── Compute pairwise cosine distance matrix ──
    mat = compute_distance_matrix(centroids)

    # Pretty-print lower triangle
    print(f"\n  Cosine distance matrix (lower triangle):")
    header = f"  {'':10s}" + "".join(f"{n:10s}" for n in CLUSTER_NAMES)
    print(header)
    for i, row_name in enumerate(CLUSTER_NAMES):
        row_str = f"  {row_name:10s}"
        for j in range(i + 1):
            row_str += f"{mat[i, j]:10.4f}"
        print(row_str)

    # ── Resume W&B run and log ──
    print(f"\n  Fetching W&B run ID for '{dataset_name}'...")
    run_id = get_latest_run_id(dataset_name)
    print(f"  Resuming run {run_id}...")

    with wandb.init(
        settings=wandb.Settings(init_timeout=180),
        project="particle-benchmark",
        id=run_id,
        resume="must",
    ) as run:
        log_distance_matrix(run, mat, dataset_name)

    print(f"  Done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine distances between cluster centroids "
                    "and log results to existing W&B runs."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to process (default: all 12)",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--noise-dir", type=str, default="Noise")
    parser.add_argument("--decimate", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    noise_dir = Path(args.noise_dir)

    for dataset_name in args.datasets:
        process_dataset(
            dataset_name,
            output_dir=output_dir,
            noise_dir=noise_dir,
            device=device,
            batch_size=args.batch_size,
            decimate=args.decimate,
        )

    print(f"\n{'='*60}")
    print("  All cluster distance computations complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
