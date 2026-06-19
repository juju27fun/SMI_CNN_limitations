"""Re-measure CPU latency for the 22 retrained doublet models.

Phases:
  1. Patch _rt_data.json with fresh CPU latency (input_length=4096, num_classes=2).
  2. Generate _lat_comparison.json (LAT_BEFORE@625, LAT_AFTER@4096) for the 3 families
     consumed by plot_doublet_comparison.py.
  3. Re-execute the two plot scripts via subprocess.

The checkpoints in output/<model>-dataset_doublet-train/ are zoo baseline models
(3 classes, trained with architecture-specific kwargs such as kernel_size=7 for
ResNet1D). They are loaded via model.pth.tar (full serialised model object) to
recover the exact architecture without guessing non-default constructor kwargs.
Latency is then measured by feeding a dummy tensor of shape (1, 1, input_length)
through the loaded backbone; the 3-class head contributes negligible compute
relative to the backbone.

CLI flags:
  --dry-run     Validate mapping and instantiate models; no measurement, no writes.
  --no-replot   Skip Phase 3 (subprocess plot regeneration).
  --rt-data     Override path to _rt_data.json
                (default: results/doublet_3fam_retrained/_rt_data.json).
"""

import argparse
import json
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any

import torch

# ── Ensure project root is on sys.path ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from p0.training_utils import measure_cpu_latency  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
SIGNAL_DURATION_MS: float = 8.192   # 4096 samples / 500 kHz × 1000 ms
INPUT_LENGTH_AFTER: int = 4096      # retrained doublet input length
INPUT_LENGTH_BEFORE: int = 625      # zoo baseline input length
NUM_CLASSES_DOUBLET: int = 2        # binary: doublet / non-doublet (unused for load)

COMPARISON_FAMILIES: list[str] = ["Conv1DGAP", "ResNet1D", "EfficientNet1D"]

DEFAULT_RT_DATA = Path("results/doublet_3fam_retrained/_rt_data.json")
LAT_COMPARISON_PATH = Path("results/doublet_3fam_retrained/_lat_comparison.json")

WARMUP: int = 20
N_RUNS: int = 200


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tar_path(model_name: str) -> Path:
    """Return the full-model archive path for a given model_name."""
    return Path(f"output/{model_name}-dataset_doublet-train/model.pth.tar")


def _load_model_from_tar(model_name: str) -> "torch.nn.Module | None":
    """Load the full serialised model from model.pth.tar.

    Uses weights_only=False because the file was saved with torch.save(model, ...)
    rather than torch.save(model.state_dict(), ...).

    Returns the loaded nn.Module on CPU, or None on failure.
    """
    tar = _tar_path(model_name)
    if not tar.exists():
        print(f"[SKIP] {model_name}: model.pth.tar not found at {tar}")
        return None
    try:
        model = torch.load(tar, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            print(f"[ERROR] {model_name}: model.pth.tar does not contain an nn.Module "
                  f"(got {type(model).__name__})")
            return None
        return model
    except Exception as exc:
        print(f"[ERROR] {model_name}: failed to load model.pth.tar — {exc}")
        return None


# ── Phase 1: patch _rt_data.json ──────────────────────────────────────────────

def phase1_patch_rt_data(rt_data_path: Path, dry_run: bool) -> None:
    """Re-measure CPU latency for all 22 entries and patch _rt_data.json in-place."""
    with open(rt_data_path) as f:
        rt_data: list[dict[str, Any]] = json.load(f)

    # Backup original (GPU latency) if not already done
    if not dry_run:
        backup_path = rt_data_path.parent / "_rt_data_gpu_backup.json"
        if not backup_path.exists():
            shutil.copy2(rt_data_path, backup_path)
            print(f"[INFO] Backup written to {backup_path}")
        else:
            print(f"[INFO] Backup already exists at {backup_path} — skipping.")

    ok_count = 0
    skip_count = 0

    for entry in rt_data:
        model_name: str = entry["model_name"]

        model = _load_model_from_tar(model_name)
        if model is None:
            skip_count += 1
            continue

        if dry_run:
            print(f"[INFO] [DRY-RUN] {model_name}: OK "
                  f"(model.pth.tar loaded, {type(model).__name__})")
            ok_count += 1
            continue

        # Measure latency at the new input length (4096 samples)
        print(f"[INFO] Measuring {model_name} ...")
        latency = measure_cpu_latency(model, (1, 1, INPUT_LENGTH_AFTER),
                                      warmup=WARMUP, n_runs=N_RUNS)

        entry["latency_ms"] = round(latency["median_ms"], 4)
        entry["latency_p95_ms"] = round(latency["p95_ms"], 4)
        entry["rt_factor"] = round(SIGNAL_DURATION_MS / latency["median_ms"], 1)
        entry["latency_device"] = "cpu"
        ok_count += 1
        print(f"[INFO] {model_name}: {entry['latency_ms']} ms  "
              f"(rt={entry['rt_factor']:.1f}x)")

    # Write back
    if not dry_run:
        with open(rt_data_path, "w") as f:
            json.dump(rt_data, f, indent=2)
            f.write("\n")
        print(f"[INFO] Patched {rt_data_path} ({ok_count} entries updated, "
              f"{skip_count} skipped).")
    else:
        total = len(rt_data)
        print(f"\n[DRY-RUN] Phase 1 summary: {ok_count}/{total} OK, {skip_count} SKIP")


# ── Phase 2: generate _lat_comparison.json ────────────────────────────────────

def phase2_generate_lat_comparison(dry_run: bool) -> None:
    """Measure latency at 625 and 4096 for the 3 base families."""
    lat_before: dict[str, float] = {}
    lat_after: dict[str, float] = {}
    ok_count = 0

    for family in COMPARISON_FAMILIES:
        model = _load_model_from_tar(family)
        if model is None:
            # _load_model_from_tar already printed the reason
            continue

        if dry_run:
            print(f"[INFO] [DRY-RUN] {family}: OK "
                  f"(model.pth.tar loaded, {type(model).__name__})")
            ok_count += 1
            continue

        print(f"[INFO] Measuring {family} @ {INPUT_LENGTH_BEFORE} samples ...")
        result_before = measure_cpu_latency(model, (1, 1, INPUT_LENGTH_BEFORE),
                                            warmup=WARMUP, n_runs=N_RUNS)
        lat_before[family] = round(result_before["median_ms"], 4)

        print(f"[INFO] Measuring {family} @ {INPUT_LENGTH_AFTER} samples ...")
        result_after = measure_cpu_latency(model, (1, 1, INPUT_LENGTH_AFTER),
                                           warmup=WARMUP, n_runs=N_RUNS)
        lat_after[family] = round(result_after["median_ms"], 4)

        ok_count += 1
        print(f"[INFO] {family}: before={lat_before[family]} ms  "
              f"after={lat_after[family]} ms")

    if dry_run:
        print(f"\n[DRY-RUN] Phase 2 summary: {ok_count}/3 OK")
        return

    comparison = {
        "LAT_BEFORE": lat_before,
        "LAT_AFTER": lat_after,
        "device": "cpu",
        "warmup": WARMUP,
        "n_runs": N_RUNS,
        "input_length_before": INPUT_LENGTH_BEFORE,
        "input_length_after": INPUT_LENGTH_AFTER,
    }
    LAT_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAT_COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
        f.write("\n")
    print(f"[INFO] Wrote {LAT_COMPARISON_PATH}")


# ── Phase 3: re-run plot scripts ──────────────────────────────────────────────

def phase3_replot() -> None:
    """Re-execute the two plot scripts via subprocess."""
    scripts = [
        "scripts/plot_realtime_factor_doublet.py",
        "scripts/plot_doublet_comparison.py",
    ]
    for script in scripts:
        print(f"[INFO] Running {script} ...")
        r = subprocess.run(
            [sys.executable, script],
            cwd=_PROJECT_ROOT,
        )
        if r.returncode != 0:
            print(f"[WARN] {script} returned exit code {r.returncode}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-measure CPU latency for doublet 3-family retrained models."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate mapping and instantiate models; no measurement, no writes.",
    )
    parser.add_argument(
        "--no-replot", action="store_true",
        help="Skip Phase 3 (subplot regeneration).",
    )
    parser.add_argument(
        "--rt-data", type=Path, default=DEFAULT_RT_DATA,
        dest="rt_data",
        help=f"Path to _rt_data.json (default: {DEFAULT_RT_DATA}).",
    )
    args = parser.parse_args()

    rt_data_path: Path = args.rt_data
    dry_run: bool = args.dry_run

    if not rt_data_path.exists():
        print(f"[ERROR] {rt_data_path} not found.", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Phase 1: {'[DRY-RUN] ' if dry_run else ''}Patch _rt_data.json")
    print("=" * 60)
    phase1_patch_rt_data(rt_data_path, dry_run=dry_run)

    print()
    print("=" * 60)
    print(f"Phase 2: {'[DRY-RUN] ' if dry_run else ''}Generate _lat_comparison.json")
    print("=" * 60)
    phase2_generate_lat_comparison(dry_run=dry_run)

    if not dry_run and not args.no_replot:
        print()
        print("=" * 60)
        print("Phase 3: Regenerate figures")
        print("=" * 60)
        phase3_replot()

    if dry_run:
        print()
        print("[DRY-RUN] No files were modified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
