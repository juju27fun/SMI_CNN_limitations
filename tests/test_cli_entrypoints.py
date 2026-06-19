"""Root command smoke tests."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RETAINED_HELP_COMMANDS = (
    "train.py",
    "train4classes.py",
    "benchmark_zoo.py",
    "benchmark_base.py",
    "generate_dataset.py",
    "analyze_noise.py",
    "dataset_leaks.py",
    "fix_leaks.py",
    "run_dataset_audit.py",
)


REMOVED_ROOT_WRAPPERS = (
    "train3classes_proof.py",
    "train_Particles2SNR_F_event_3class.py",
    "train_particles2SNR_4class_lim10.py",
    "infer_doublets.py",
    "build_union_dataset.py",
    "generate_4class_dataset.py",
)


def run_help(script: str):
    return subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def test_retained_cli_help_commands():
    for script in RETAINED_HELP_COMMANDS:
        result = run_help(script)
        assert result.returncode == 0, (
            f"{script} --help failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_niche_root_wrappers_removed():
    for script in REMOVED_ROOT_WRAPPERS:
        assert not (ROOT / script).exists(), f"unexpected niche root wrapper remains: {script}"


class _FakeRun:
    def __init__(self):
        self.summary = {}

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def log_model(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def _write_tiny_4class_dataset(root: Path):
    rng = np.random.default_rng(0)
    classes = ("2um", "4um", "10um", "Noise")
    x = np.linspace(0.0, 1.0, 2500, dtype=np.float32)
    for split in ("train", "test"):
        for class_idx, class_name in enumerate(classes):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(4):
                frequency = float(class_idx + 1)
                signal = np.sin(2 * np.pi * frequency * x)
                signal += 0.01 * rng.normal(size=x.shape)
                np.save(class_dir / f"sample_{sample_idx}.npy", signal.astype(np.float32))


def test_train4classes_tiny_offline_smoke(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.training.train4classes")
    data_dir = tmp_path / "tiny_4class"
    output_dir = tmp_path / "out"
    _write_tiny_4class_dataset(data_dir)

    fake_run = _FakeRun()
    monkeypatch.setattr(module.wandb, "init", lambda *args, **kwargs: fake_run)
    monkeypatch.setattr(module, "compute_model_macs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "measure_cpu_latency",
        lambda *args, **kwargs: {
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "latency_device": "cpu",
        },
    )
    monkeypatch.setattr(module, "run_post_testing", lambda *args, **kwargs: (0.0, 0.0))
    monkeypatch.setattr(module, "run_3class_evaluation", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        module,
        "extract_features",
        lambda model, loader, device: (
            np.zeros((len(loader.dataset), 2), dtype=np.float32),
            np.zeros(len(loader.dataset), dtype=np.int64),
        ),
    )

    def fake_dimensionality_plot(*args, **kwargs):
        import matplotlib.pyplot as plt

        return plt.figure(), plt.figure()

    monkeypatch.setattr(module, "plot_dimensionality_reduction", fake_dimensionality_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train4classes.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--model",
            "Conv1DGAP-Pico",
            "--decimate",
            "4",
            "--val-split",
            "0.25",
            "--scheduler",
            "none",
            "--wandb-offline",
            "--dataset-name",
            "smoke",
            "--run-id",
            "pytest",
        ],
    )

    module.main()

    assert (output_dir / "Conv1DGAP-Pico-smoke-pytest" / "best_model.pth").exists()
