"""Re-log confusion matrices with updated styling to existing W&B runs.

Targets:
  - Conv1D-dataset_3c-rerun_report (id=zau4nmcg): 3-class, Charts/confusion_matrix
  - Conv1D-dataset_4c-rerun_report (id=aqdat939): 4-class, Charts/confusion_matrix + Charts_3class/confusion_matrix

New styling: Blues colormap, LogNorm, horizontal colorbar, 9pt fonts, (COL_W, COL_W) square.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from train import (
    RAW_SIGNAL_LENGTH,
    ParticleDataset,
    BandpassFilter,
    Decimate,
    Truncate,
    evaluate,
)
from models import Conv1DClassifier

# ── Constants ────────────────────────────────────────────────────────────────

from pub_utils import plot_confusion_matrix

RUNS = {
    "3c": {
        "run_id": "zau4nmcg",
        "run_name": "Conv1D-dataset_3c-rerun_report",
        "model_path": _PROJECT_ROOT / "output" / "Conv1D-dataset_3c-rerun_report" / "best_model.pth",
        "data_dir": _PROJECT_ROOT / "data" / "dataset" / "test",
        "n_classes": 3,
        "class_names": ["2um", "4um", "10um"],
        "keys": ["Charts/confusion_matrix"],
    },
    "4c": {
        "run_id": "aqdat939",
        "run_name": "Conv1D-dataset_4c-rerun_report",
        "model_path": _PROJECT_ROOT / "artifacts" / "Conv1D-dataset_4c-4class:v3" / "best_model.pth",
        "data_dir": _PROJECT_ROOT / "data" / "dataset_4c" / "test",
        "n_classes": 4,
        "class_names": ["2um", "4um", "10um", "Noise"],
        "keys": ["Charts/confusion_matrix", "Charts_3class/confusion_matrix"],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_test_loader(data_dir, class_names, decimate=4, batch_size=64):
    """Build a test DataLoader from test directory."""
    seq_len = RAW_SIGNAL_LENGTH // decimate
    transforms = [BandpassFilter(), Decimate(decimate), Truncate(seq_len)]

    dataset = ParticleDataset(Path(data_dir), class_names=class_names, transforms=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def plot_confusion_matrix_fig(cm, class_names):
    """Create a publication-quality confusion matrix figure.

    Returns the matplotlib figure.
    """
    fig, _ = plot_confusion_matrix(cm, class_names)
    return fig


def process_run(run_key, device):
    """Load model, run inference, and re-log confusion matrices for one run."""
    cfg = RUNS[run_key]
    print(f"\n{'='*60}")
    print(f"Processing: {cfg['run_name']} ({run_key})")
    print(f"{'='*60}")

    # Load model
    n_classes = cfg["n_classes"]
    seq_len = RAW_SIGNAL_LENGTH // 4  # decimate=4
    model = Conv1DClassifier(input_length=seq_len, num_classes=n_classes)
    state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"  Model loaded: {cfg['model_path']}")

    # Build test loader
    loader = build_test_loader(cfg["data_dir"], cfg["class_names"])
    print(f"  Test data: {cfg['data_dir']} ({len(loader.dataset)} samples)")

    # Run inference
    criterion = torch.nn.CrossEntropyLoss()
    _, _, y_pred, y_true, _ = evaluate(model, loader, criterion, device)
    print(f"  Inference complete.")

    # Resume W&B run
    run = wandb.init(
        project="particle-benchmark",
        id=cfg["run_id"],
        resume="must",
    )
    print(f"  W&B run resumed: {cfg['run_id']}")

    # Generate and log confusion matrices
    for i, key in enumerate(cfg["keys"]):
        is_last = (i == len(cfg["keys"]) - 1)

        if key == "Charts_3class/confusion_matrix":
            # 3-class subset: filter to particle classes only (exclude Noise = class 3)
            particle_mask = y_true < 3
            y_true_3c = y_true[particle_mask]
            y_pred_3c = y_pred[particle_mask]
            cm = confusion_matrix(y_true_3c, y_pred_3c, labels=[0, 1, 2])
            class_names = ["2um", "4um", "10um"]
            print(f"  3-class CM (particle-only):\n{cm}")
        else:
            cm = confusion_matrix(y_true, y_pred)
            class_names = cfg["class_names"]
            print(f"  {n_classes}-class CM:\n{cm}")

        fig = plot_confusion_matrix_fig(cm, class_names)
        run.log({key: wandb.Image(fig)}, commit=is_last)
        plt.close(fig)
        print(f"  Logged: {key} (commit={is_last})")

    run.finish()
    print(f"  Done: {cfg['run_name']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update(PUB_RC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for run_key in ["3c", "4c"]:
        process_run(run_key, device)

    print("\nAll confusion matrices re-logged successfully.")


if __name__ == "__main__":
    main()
