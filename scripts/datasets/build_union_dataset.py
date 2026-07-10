"""Rebuild data/S_union from source datasets S0–S9 (train) and datasets/processed/p0-baseline-3class/v1 (test).

Train set: copies .npy files from S0–S9 train dirs into S_union/train/{class}/
Test set:  copies files from datasets/processed/p0-baseline-3class/v1/test/{class}/ (real laser data)

Usage:
    python build_union_dataset.py          # only build if S_union doesn't exist
    python build_union_dataset.py --force  # wipe and rebuild
"""

import argparse
import shutil
from pathlib import Path

SOURCE_DATASETS = [
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
]

CLASSES = ["2um", "4um", "10um"]
DATA_ROOT = Path("data")
UNION_DIR = DATA_ROOT / "S_union"
REAL_TEST_DIR = DATA_ROOT / "dataset" / "test"


def build_union(force: bool = False) -> None:
    if UNION_DIR.exists():
        if not force:
            print(f"{UNION_DIR} already exists. Use --force to rebuild.")
            return
        print(f"Removing existing {UNION_DIR}...")
        shutil.rmtree(UNION_DIR)

    # Validate sources exist
    for ds in SOURCE_DATASETS:
        ds_path = DATA_ROOT / ds / "train"
        if not ds_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {ds_path}")
    if not REAL_TEST_DIR.exists():
        raise FileNotFoundError(f"Real test set not found: {REAL_TEST_DIR}")

    # Build train set
    print("Building train set from S0–S9...")
    for cls in CLASSES:
        out_dir = UNION_DIR / "train" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for ds in SOURCE_DATASETS:
            src_dir = DATA_ROOT / ds / "train" / cls
            prefix = ds.lower()
            for src_file in sorted(src_dir.glob("*.npy")):
                idx = int(src_file.stem.split("_")[-1])
                dst_name = f"{prefix}_sample_{idx:04d}.npy"
                shutil.copy2(src_file, out_dir / dst_name)
                count += 1
        print(f"  {cls}: {count} files")

    # Build test set (copy real laser data, keep original filenames)
    print("Building test set from datasets/processed/p0-baseline-3class/v1/test/...")
    for cls in CLASSES:
        src_dir = REAL_TEST_DIR / cls
        out_dir = UNION_DIR / "test" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for src_file in sorted(src_dir.glob("*.npy")):
            shutil.copy2(src_file, out_dir / src_file.name)
            count += 1
        print(f"  {cls}: {count} files")

    print(f"\nDone. S_union built at {UNION_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild S_union dataset")
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild")
    args = parser.parse_args()
    build_union(force=args.force)
