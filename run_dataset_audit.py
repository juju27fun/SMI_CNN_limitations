#!/usr/bin/env python3
"""Batch dataset audit: leak detection + noise analysis on all datasets.

Auto-discovers dataset directories (containing train/ and test/ splits) and
standalone .npy folders, runs leak detection and noise analysis on each, and
skips datasets that have already been processed.

Completion state is tracked in a JSON manifest so re-runs only process new or
modified datasets.

Usage:
    python run_dataset_audit.py
    python run_dataset_audit.py --force                # re-process everything
    python run_dataset_audit.py --datasets dataset_f1  # process only specific ones
    python run_dataset_audit.py --dry-run              # show what would be done
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_FILE = SCRIPT_DIR / "audit_manifest.json"
OUTPUT_ROOT = SCRIPT_DIR / "audit_results"

# Use venv python if available, otherwise fall back to current interpreter
VENV_PYTHON = SCRIPT_DIR / "venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable


# ── Dataset discovery ─────────────────────────────────────────────────────────

def is_split_dataset(path: Path) -> bool:
    """True if directory contains both train/ and test/ subdirectories."""
    return (path / "train").is_dir() and (path / "test").is_dir()


def is_npy_folder(path: Path) -> bool:
    """True if directory directly contains .npy files (standalone noise folder)."""
    return any(path.glob("*.npy"))


def discover_datasets(root: Path, explicit: list[str] | None = None) -> dict:
    """Find all processable directories under root.

    Returns dict with keys:
        "split_datasets": list of Paths (have train/test structure)
        "noise_folders":  list of Paths (standalone .npy folders)
    """
    split_datasets = []
    noise_folders = []

    if explicit:
        candidates = [root / name for name in explicit]
    else:
        candidates = sorted(
            p for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    for p in candidates:
        if not p.is_dir():
            continue
        if p.name in ("venv", "__pycache__", "wandb", "output", "archive",
                       "audit_results"):
            continue
        # Skip analysis output directories
        if p.name.endswith("_full_analysis"):
            continue

        if is_split_dataset(p):
            split_datasets.append(p)
        elif is_npy_folder(p):
            noise_folders.append(p)

    return {"split_datasets": split_datasets, "noise_folders": noise_folders}


# ── Fingerprint (for change detection) ────────────────────────────────────────

def dataset_fingerprint(path: Path) -> str:
    """Lightweight fingerprint: file count + newest mtime.

    Doesn't hash contents (too slow for large datasets) but detects
    added/removed/modified files reliably.
    """
    newest = 0.0
    count = 0
    for f in path.rglob("*.npy"):
        count += 1
        mt = f.stat().st_mtime
        if mt > newest:
            newest = mt
    return f"{count}:{newest:.0f}"


# ── Manifest (skip-if-done tracking) ──────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def should_skip(name: str, fingerprint: str, manifest: dict) -> bool:
    """True if this dataset was already processed with the same fingerprint."""
    entry = manifest.get(name)
    if entry is None:
        return False
    return entry.get("fingerprint") == fingerprint and entry.get("status") == "done"


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_step(cmd: list[str], label: str) -> subprocess.CompletedProcess:
    """Run a subprocess, printing header and streaming output."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result


# ── Analysis steps ────────────────────────────────────────────────────────────

def run_leak_detection(dataset_path: Path, output_dir: Path) -> subprocess.CompletedProcess:
    """Run dataset_leaks.py on a split dataset."""
    report_dir = output_dir / "leak_reports"
    cmd = [
        PYTHON, str(SCRIPT_DIR / "dataset_leaks.py"),
        str(dataset_path),
        "--report-dir", str(report_dir),
    ]
    return run_step(cmd, f"Leak Detection: {dataset_path.name}")


def run_noise_analysis(folder: Path, output_pdf: Path, fs: float, segment: int
                       ) -> subprocess.CompletedProcess:
    """Run analyze_noise.py on a single folder of .npy files."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(SCRIPT_DIR / "analyze_noise.py"),
        str(folder),
        "--output", str(output_pdf),
        "--fs", str(fs),
        "--segment", str(segment),
    ]
    return run_step(cmd, f"Noise Analysis: {folder}")


def run_noise_for_dataset(dataset_path: Path, output_dir: Path,
                          fs: float, segment: int) -> dict[str, subprocess.CompletedProcess]:
    """Run noise analysis on every split/class combination in a dataset."""
    noise_dir = output_dir / "noise_analysis"
    results = {}

    for split in ("train", "test"):
        split_dir = dataset_path / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if not any(class_dir.glob("*.npy")):
                continue
            pdf_name = f"{split}_{class_dir.name}_Noise_Analysis.pdf"
            pdf_path = noise_dir / pdf_name
            key = f"{split}/{class_dir.name}"
            results[key] = run_noise_analysis(class_dir, pdf_path, fs, segment)
    return results


# ── Summary report ────────────────────────────────────────────────────────────

def write_dataset_summary(output_dir: Path, dataset_name: str,
                          leak_result: subprocess.CompletedProcess | None,
                          noise_results: dict[str, subprocess.CompletedProcess]):
    """Write a per-dataset summary markdown file."""
    md_path = output_dir / "audit_summary.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(md_path, "w") as f:
        f.write(f"# Audit Summary — `{dataset_name}`\n\n")
        f.write(f"> **Generated:** {now}\n\n")

        # Leak detection
        if leak_result is not None:
            f.write("## Leak Detection\n\n")
            if leak_result.returncode == 0:
                f.write("**Result: PASSED** (no critical leaks)\n\n")
            else:
                f.write("**Result: CRITICAL LEAKS DETECTED**\n\n")
            f.write("See [leak_reports/summary.md](leak_reports/summary.md) for details.\n\n")

        # Noise analysis
        if noise_results:
            f.write("## Noise Analysis\n\n")
            # Check if keys contain "/" (split dataset) or not (standalone folder)
            has_splits = any("/" in k for k in noise_results)
            if has_splits:
                f.write("| Split | Class | Status | Report |\n")
                f.write("|---|---|---|---|\n")
                for key in sorted(noise_results):
                    result = noise_results[key]
                    split, cls = key.split("/")
                    pdf_name = f"{split}_{cls}_Noise_Analysis.pdf"
                    status = "Done" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
                    link = f"[PDF](noise_analysis/{pdf_name})" if result.returncode == 0 else "—"
                    f.write(f"| {split} | {cls} | {status} | {link} |\n")
            else:
                f.write("| Folder | Status | Report |\n")
                f.write("|---|---|---|\n")
                for key in sorted(noise_results):
                    result = noise_results[key]
                    pdf_name = f"{key}_Noise_Analysis.pdf"
                    status = "Done" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
                    link = f"[PDF]({pdf_name})" if result.returncode == 0 else "—"
                    f.write(f"| {key} | {status} | {link} |\n")
            f.write("\n")

        f.write("---\n\n*Generated by `run_dataset_audit.py`*\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch audit: leak detection + noise analysis on all datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_dataset_audit.py\n"
            "  python run_dataset_audit.py --force\n"
            "  python run_dataset_audit.py --datasets dataset_f1 dataset_nof1\n"
            "  python run_dataset_audit.py --dry-run\n"
        ),
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Process only these dataset directory names (default: auto-discover)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if already done")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running anything")
    parser.add_argument("--fs", type=float, default=2_000_000,
                        help="Sampling frequency in Hz (default: 2000000)")
    parser.add_argument("--segment", type=int, default=2500,
                        help="Segment length for noise analysis (default: 2500)")

    args = parser.parse_args()

    # Discover datasets
    discovered = discover_datasets(SCRIPT_DIR, args.datasets)
    split_datasets = discovered["split_datasets"]
    noise_folders = discovered["noise_folders"]

    if not split_datasets and not noise_folders:
        print("No datasets found to process.")
        sys.exit(0)

    # Load manifest
    manifest = load_manifest()

    print("=" * 60)
    print("  DATASET AUDIT")
    print("=" * 60)
    print(f"  Split datasets:    {[d.name for d in split_datasets]}")
    print(f"  Noise folders:     {[d.name for d in noise_folders]}")
    print(f"  Force:             {args.force}")
    print()

    # Build work list
    work = []  # list of (name, path, kind, fingerprint)

    for ds in split_datasets:
        fp = dataset_fingerprint(ds)
        if not args.force and should_skip(ds.name, fp, manifest):
            print(f"  SKIP (already done): {ds.name}")
        else:
            work.append((ds.name, ds, "split_dataset", fp))

    for nf in noise_folders:
        fp = dataset_fingerprint(nf)
        if not args.force and should_skip(nf.name, fp, manifest):
            print(f"  SKIP (already done): {nf.name}")
        else:
            work.append((nf.name, nf, "noise_folder", fp))

    if not work:
        print("\nAll datasets already processed. Use --force to re-run.")
        sys.exit(0)

    if args.dry_run:
        print(f"\nWould process {len(work)} item(s):")
        for name, path, kind, fp in work:
            print(f"  [{kind}] {name}  (fingerprint: {fp})")
        sys.exit(0)

    print(f"\nProcessing {len(work)} item(s)...\n")
    overall_start = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for i, (name, path, kind, fp) in enumerate(work, 1):
        ds_start = time.time()
        output_dir = OUTPUT_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"  [{i}/{len(work)}] {name}  ({kind})")
        print("=" * 60)

        leak_result = None
        noise_results = {}
        all_ok = True

        if kind == "split_dataset":
            # Step 1: Leak detection
            leak_result = run_leak_detection(path, output_dir)
            if leak_result.returncode != 0:
                print(f"\n  WARNING: Leak detection found critical issues for {name}")
                all_ok = False
                # Continue with noise analysis anyway — it's still useful

            # Step 2: Noise analysis on all split/class combos
            noise_results = run_noise_for_dataset(path, output_dir, args.fs, args.segment)
            for key, result in noise_results.items():
                if result.returncode != 0:
                    print(f"  WARNING: Noise analysis failed for {name}/{key}")
                    all_ok = False

        elif kind == "noise_folder":
            # Just run noise analysis on the folder itself
            pdf_path = output_dir / f"{name}_Noise_Analysis.pdf"
            result = run_noise_analysis(path, pdf_path, args.fs, args.segment)
            noise_results[name] = result
            if result.returncode != 0:
                all_ok = False

        # Write per-dataset summary
        write_dataset_summary(output_dir, name, leak_result, noise_results)

        ds_time = time.time() - ds_start

        # Update manifest
        manifest[name] = {
            "fingerprint": fp,
            "status": "done",
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(ds_time, 1),
            "kind": kind,
            "leak_ok": leak_result.returncode == 0 if leak_result else None,
            "noise_ok": all(r.returncode == 0 for r in noise_results.values()),
        }
        save_manifest(manifest)

        status = "DONE" if all_ok else "DONE (with warnings)"
        print(f"\n  {name}: {status} ({ds_time:.1f}s)")

    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 60)
    print("  AUDIT COMPLETE")
    print("=" * 60)
    print(f"  Processed:  {len(work)} dataset(s)")
    print(f"  Results:    {OUTPUT_ROOT.resolve()}/")
    print(f"  Manifest:   {MANIFEST_FILE.resolve()}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
