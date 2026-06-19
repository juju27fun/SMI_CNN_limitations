"""Re-measure CPU latency for the 432 benchmark2 runs from saved checkpoints.

Phases:
  1. For each JSON in results/benchmark2/runs/:
     - Instantiate the model (registry first, legacy fallback).
     - Load checkpoint weights.
     - Measure CPU latency with measure_cpu_latency().
     - Patch latency_median_ms / latency_p95_ms / latency_device in-place.
  2. Re-aggregate: aggregate_results() -> summary.csv.
  3. Re-generate publication figures via regenerate_and_publish_figures().

CLI flags:
  --dry-run       Validate mapping + instantiate; no measurement, no writes.
  --limit N       Process only the first N JSONs (debug).
  --runs-dir PATH Override runs directory (default: results/benchmark2/runs).
  --no-figures    Skip Phase 3 figure regeneration.
"""

import argparse
import importlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# ── Ensure project root is on sys.path ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from p0.models import create_model  # noqa: E402
from p0.data import RAW_SIGNAL_LENGTH  # noqa: E402
from p0.training_utils import measure_cpu_latency  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
# benchmark_zoo.py inherits --decimate default=4 from add_common_training_args(),
# so all benchmark2 checkpoints were trained with input_length = 2500 // 4 = 625.
_DECIMATE: int = 4
INPUT_LENGTH: int = RAW_SIGNAL_LENGTH // _DECIMATE  # 625
NUM_CLASSES: int = 3
WARMUP: int = 20
N_RUNS: int = 200

DEFAULT_RUNS_DIR = Path("results/benchmark2/runs")
CHECKPOINTS_DIR = Path("results/benchmark2/checkpoints")
SUMMARY_CSV = Path("results/benchmark2/summary.csv")
SUMMARY_BACKUP = Path("results/benchmark2/summary_gpu_backup.csv")

# Legacy models no longer in MODEL_REGISTRY / MODEL_VARIANTS.
# Format: name -> (module, class_name, extra_kwargs)
LEGACY_MODELS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "Conv1D":       ("models.conv1d", "Conv1DClassifier", {}),
    "Conv1D-Pico":  ("models.conv1d", "Conv1DClassifier", {"width_mult": 0.0125}),
    "Conv1D-Nano":  ("models.conv1d", "Conv1DClassifier", {"width_mult": 0.025}),
    "Conv1D-XXS":   ("models.conv1d", "Conv1DClassifier", {"width_mult": 0.1}),
    "Conv1D-XS":    ("models.conv1d", "Conv1DClassifier", {"width_mult": 0.25}),
    "Conv1D-S":     ("models.conv1d", "Conv1DClassifier", {"width_mult": 0.5}),
    "Conv1D-L":     ("models.conv1d", "Conv1DClassifier", {"width_mult": 2.0}),
    "LeNet1D":      ("models.lenet1d", "LeNet1D", {}),
    "LeNet1D-Pico": ("models.lenet1d", "LeNet1D", {"width_mult": 0.01}),
    "LeNet1D-Nano": ("models.lenet1d", "LeNet1D", {"width_mult": 0.025}),
    "LeNet1D-XXS":  ("models.lenet1d", "LeNet1D", {"width_mult": 0.1}),
    "LeNet1D-XS":   ("models.lenet1d", "LeNet1D", {"width_mult": 0.25}),
    "LeNet1D-S":    ("models.lenet1d", "LeNet1D", {"width_mult": 0.5}),
    "LeNet1D-L":    ("models.lenet1d", "LeNet1D", {"width_mult": 2.0}),
    "VGG1D":        ("models.vgg1d", "VGG1D", {}),
    "VGG1D-Pico":   ("models.vgg1d", "VGG1D", {"width_mult": 0.0125}),
    "VGG1D-Nano":   ("models.vgg1d", "VGG1D", {"width_mult": 0.025}),
    "VGG1D-XXS":    ("models.vgg1d", "VGG1D", {"width_mult": 0.1}),
    "VGG1D-XS":     ("models.vgg1d", "VGG1D", {"width_mult": 0.25}),
    "VGG1D-S":      ("models.vgg1d", "VGG1D", {"width_mult": 0.5}),
    "VGG1D-L":      ("models.vgg1d", "VGG1D", {"width_mult": 2.0}),
}


# ── Model instantiation ────────────────────────────────────────────────────────

# ResNet1D variants whose checkpoints were saved against the v0 architecture
# (stem kernel_size=7 hardcoded, blocks kernel_size=3 hardcoded). The current
# `models/resnet1d.py` exposes a unified `kernel_size` parameter that swaps
# both stem and blocks together, which cannot reproduce the v0 layout. We
# instead instantiate the frozen snapshot in `models/_resnet1d_legacy_v0.py`.
RESNET1D_V0_VARIANTS: dict[str, dict[str, Any]] = {
    "ResNet1D":      {},                    # base_width=74 (default)
    "ResNet1D-Nano": {"base_width": 2},
    "ResNet1D-XXS":  {"base_width": 7},
    "ResNet1D-XS":   {"base_width": 18},
    "ResNet1D-S":    {"base_width": 37},
    "ResNet1D-L":    {"base_width": 148},
}


def instantiate_model(model_name: str) -> tuple[Any, str]:
    """Instantiate a model by name, with legacy fallback.

    Returns (model, source) where source is "registry", "legacy", or
    "legacy-v0". Raises RuntimeError if instantiation fails from all
    sources.
    """
    # 1. Special-case: ResNet1D variants — checkpoints use the v0 stem kernel
    if model_name in RESNET1D_V0_VARIANTS:
        from models._resnet1d_legacy_v0 import ResNet1DLegacyV0
        kwargs = RESNET1D_V0_VARIANTS[model_name]
        model = ResNet1DLegacyV0(input_length=INPUT_LENGTH,
                                 num_classes=NUM_CLASSES, **kwargs)
        return model, "legacy-v0"

    # 2. Try current registry / variant table
    try:
        model = create_model(model_name, input_length=INPUT_LENGTH,
                             num_classes=NUM_CLASSES)
        return model, "registry"
    except ValueError:
        pass  # Not in registry — try legacy

    # 2. Legacy fallback via dynamic import
    if model_name in LEGACY_MODELS:
        module_path, class_name, kwargs = LEGACY_MODELS[model_name]
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            model = cls(input_length=INPUT_LENGTH, num_classes=NUM_CLASSES, **kwargs)
            return model, "legacy"
        except Exception as exc:
            raise RuntimeError(
                f"Legacy instantiation of '{model_name}' failed: {exc}"
            ) from exc

    raise RuntimeError(
        f"Model '{model_name}' not found in registry/variants and not in LEGACY_MODELS."
    )


# ── Phase 1: patch JSONs ───────────────────────────────────────────────────────

def phase1_patch_jsons(
    runs_dir: Path,
    dry_run: bool,
    limit: int | None,
) -> dict[str, int]:
    """Iterate over JSONs, measure CPU latency, patch in-place.

    Returns stats dict: {model_name: {"ok": int, "skip": int, "error": int}}.
    """
    json_files = sorted(runs_dir.glob("*.json"))
    if limit is not None:
        json_files = json_files[:limit]

    total = len(json_files)
    print(f"[INFO] Processing {total} JSON files from {runs_dir}")
    print()

    # Per-model stats
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "skip": 0, "error": 0})
    global_ok = global_skip = global_error = 0

    # Dry-run table header
    if dry_run:
        print(f"{'JSON stem':<55} {'Status'}")
        print("-" * 75)

    for jf in json_files:
        stem = jf.stem

        # Read JSON
        try:
            with open(jf) as f:
                data: dict[str, Any] = json.load(f)
        except Exception as exc:
            print(f"[ERROR] {stem}: cannot read JSON — {exc}")
            global_error += 1
            continue

        model_name: str = data.get("model_name", "")
        expected_params: int = data.get("params", -1)

        # Locate checkpoint
        ckpt_path = CHECKPOINTS_DIR / stem / "best_model.pth"
        if not ckpt_path.exists():
            reason = f"checkpoint not found at {ckpt_path}"
            print(f"[SKIP] {stem}: {reason}")
            if dry_run:
                print(f"  {'  ' + stem:<53} SKIP ({reason})")
            stats[model_name]["skip"] += 1
            global_skip += 1
            continue

        # Instantiate model
        try:
            model, source = instantiate_model(model_name)
        except RuntimeError as exc:
            print(f"[ERROR] {stem}: {exc}")
            if dry_run:
                print(f"  {'  ' + stem:<53} ERROR (instantiation)")
            stats[model_name]["error"] += 1
            global_error += 1
            continue

        # Verify param count
        actual_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        if expected_params >= 0 and actual_params != expected_params:
            print(
                f"[WARN] {stem}: param mismatch — JSON={expected_params}, "
                f"model={actual_params} (source={source}). Continuing anyway."
            )

        # Load checkpoint weights
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            print(f"[SKIP] {stem}: state_dict mismatch — {exc}")
            if dry_run:
                print(f"  {'  ' + stem:<53} SKIP (state_dict mismatch)")
            stats[model_name]["skip"] += 1
            global_skip += 1
            continue
        except Exception as exc:
            print(f"[ERROR] {stem}: checkpoint load error — {exc}")
            if dry_run:
                print(f"  {'  ' + stem:<53} ERROR (checkpoint load)")
            stats[model_name]["error"] += 1
            global_error += 1
            continue

        # Dry-run: just report OK
        if dry_run:
            print(f"  {stem:<55} OK ({source})")
            stats[model_name]["ok"] += 1
            global_ok += 1
            continue

        # Measure CPU latency
        print(f"[INFO] Measuring {stem} ({source}) ...", end=" ", flush=True)
        try:
            latency = measure_cpu_latency(
                model, (1, 1, INPUT_LENGTH), warmup=WARMUP, n_runs=N_RUNS
            )
        except Exception as exc:
            print(f"\n[ERROR] {stem}: latency measurement failed — {exc}")
            stats[model_name]["error"] += 1
            global_error += 1
            continue

        print(
            f"{latency['median_ms']:.4f} ms  "
            f"(p95={latency['p95_ms']:.4f} ms)"
        )

        # Patch JSON in-place
        data["latency_median_ms"] = round(latency["median_ms"], 4)
        data["latency_p95_ms"] = round(latency["p95_ms"], 4)
        data["latency_device"] = "cpu"

        with open(jf, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

        stats[model_name]["ok"] += 1
        global_ok += 1

    # Summary
    print()
    print("=" * 60)
    if dry_run:
        print("[DRY-RUN] Phase 1 summary (no files modified):")
    else:
        print("Phase 1 summary:")
    print(f"  Total:   {total}")
    print(f"  OK:      {global_ok}")
    print(f"  Skipped: {global_skip}")
    print(f"  Errors:  {global_error}")
    print()
    print(f"  {'Model':<30} {'OK':>4} {'SKIP':>5} {'ERR':>5}")
    print(f"  {'-'*30} {'-'*4} {'-'*5} {'-'*5}")
    for mname in sorted(stats):
        s = stats[mname]
        print(f"  {mname:<30} {s['ok']:>4} {s['skip']:>5} {s['error']:>5}")
    print("=" * 60)

    return dict(stats)


# ── Phase 2: re-aggregate summary.csv ─────────────────────────────────────────

def phase2_aggregate() -> None:
    """Re-aggregate all JSONs into summary.csv via benchmark_zoo.aggregate_results."""
    from benchmark_zoo import aggregate_results  # local import to defer heavy deps

    print("[INFO] Re-aggregating results -> summary.csv ...")
    aggregate_results(Path("results/benchmark2"))
    print("[INFO] summary.csv updated.")


# ── Phase 3: re-generate publication figures ───────────────────────────────────

def phase3_figures() -> None:
    """Re-generate publication figures via benchmark_zoo.regenerate_and_publish_figures."""
    import argparse as _argparse
    from benchmark_zoo import regenerate_and_publish_figures  # local import

    # Minimal Namespace: only output_dir and scaling are accessed when
    # wandb_publish=False (the wandb.init block is skipped entirely).
    fig_args = _argparse.Namespace(
        output_dir="results/benchmark2",
        scaling=True,   # enables pareto_latency + scaling_grid
        project="particle-benchmark",
        no_wandb=True,
    )
    print("[INFO] Re-generating figures ...")
    regenerate_and_publish_figures(fig_args, summary_df=None, wandb_publish=False)
    print("[INFO] Figures regenerated in results/benchmark2/figures/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-measure CPU latency for the 432 benchmark2 runs "
            "from saved checkpoints and patch the result JSONs."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Validate JSON<->checkpoint mapping and instantiate models; "
            "no measurement, no writes."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Process only the first N JSONs (for quick debugging).",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, dest="runs_dir",
        help=f"Override runs directory (default: {DEFAULT_RUNS_DIR}).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip Phase 3 figure regeneration (only patch JSONs + summary.csv).",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    dry_run: bool = args.dry_run

    if not runs_dir.exists():
        print(f"[ERROR] Runs directory not found: {runs_dir}", file=sys.stderr)
        return 1

    # ── Backup summary.csv (only on real runs) ─────────────────────────────────
    if not dry_run:
        if SUMMARY_CSV.exists() and not SUMMARY_BACKUP.exists():
            shutil.copy2(SUMMARY_CSV, SUMMARY_BACKUP)
            print(f"[INFO] Backup written to {SUMMARY_BACKUP}")
        elif SUMMARY_BACKUP.exists():
            print(f"[INFO] Backup already exists at {SUMMARY_BACKUP} — skipping.")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Phase 1: {'[DRY-RUN] ' if dry_run else ''}Patch JSONs")
    print("=" * 60)
    phase1_patch_jsons(runs_dir, dry_run=dry_run, limit=args.limit)

    # Phases 2 & 3 are skipped in dry-run mode or when --limit is active
    # (partial data would corrupt the summary).
    if dry_run:
        print("[DRY-RUN] No files were modified. Phases 2 & 3 skipped.")
        return 0

    if args.limit is not None:
        print(
            f"[INFO] --limit {args.limit} active: skipping Phase 2 & 3 "
            "(summary would be partial)."
        )
        return 0

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Phase 2: Re-aggregate -> summary.csv")
    print("=" * 60)
    phase2_aggregate()

    # ── Phase 3 ────────────────────────────────────────────────────────────────
    if not args.no_figures:
        print()
        print("=" * 60)
        print("Phase 3: Re-generate figures")
        print("=" * 60)
        phase3_figures()
    else:
        print("[INFO] --no-figures: skipping Phase 3.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
