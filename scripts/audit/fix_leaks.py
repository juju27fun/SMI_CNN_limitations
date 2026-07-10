#!/usr/bin/env python3
"""Fix dataset leaks by source-level re-splitting.

Instead of deleting leaked files (which shrinks the test set), this script
reassigns files between train and test so that all crops from a given source
recording stay in the same split. It also removes exact and intra-split
duplicates.

Steps:
  1. Parse all filenames to extract source recording IDs
  2. Assign each source to exactly one split (train or test), targeting ~80/20
  3. Move files that are in the wrong split
  4. Remove exact duplicates (intra-split, keeping one copy)
  5. Verify final counts and zero source leaks

No external replacement DB is needed — all existing files are preserved.
"""

import argparse
import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["2um", "4um", "10um"]
SPLITS = ["train", "test"]
TARGET_TEST_RATIO = 0.20

SOURCE_RE = re.compile(r"^(.+?)(\d+)\.npy(\d+)\.npy$")


def parse_source(fname: str) -> str | None:
    m = SOURCE_RE.match(fname)
    if m:
        return f"{m.group(1)}{m.group(2)}.npy"
    return None


def hash_npy(path: Path) -> str:
    return hashlib.md5(np.load(path).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Fix dataset leaks by source-level re-splitting."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/interim/particles2SNR-pipeline/leak-repair-candidate"),
        help="Writable candidate dataset; registered datasets are immutable",
    )
    parser.add_argument("--execute", action="store_true", help="Apply the repair (default: dry run)")
    args = parser.parse_args()

    root = args.dataset_root
    dry_run = not args.execute

    if dry_run:
        print("*** DRY RUN — no files will be modified ***\n")

    # ── 1. Inventory: map source -> split -> [files] per class ────────────
    print("=== Step 1: Inventory ===")
    # class -> source -> split -> [filename]
    inventory = {}
    for cls in CLASSES:
        sources = defaultdict(lambda: defaultdict(list))
        for split in SPLITS:
            cls_dir = root / split / cls
            if not cls_dir.is_dir():
                continue
            for f in sorted(os.listdir(cls_dir)):
                if not f.endswith(".npy"):
                    continue
                src = parse_source(f)
                if src:
                    sources[src][split].append(f)
        inventory[cls] = dict(sources)

    for cls in CLASSES:
        sources = inventory[cls]
        shared = sum(1 for s in sources.values()
                     if "train" in s and "test" in s)
        total = sum(len(sp.get("train", [])) + len(sp.get("test", []))
                    for sp in sources.values())
        print(f"  {cls}: {len(sources)} sources, {total} files, "
              f"{shared} shared (leaked)")

    # ── 2. Source-level split assignment ───────────────────────────────────
    print("\n=== Step 2: Source-level split assignment ===")
    # For each class, assign each source to train or test.
    # Strategy: keep non-shared sources where they are, then distribute
    # shared sources to reach the target test ratio.
    moves = defaultdict(list)  # (src_split, dst_split, cls) -> [filename]

    for cls in CLASSES:
        sources = inventory[cls]
        total_files = sum(len(sp.get("train", [])) + len(sp.get("test", []))
                         for sp in sources.values())
        target_test = round(total_files * TARGET_TEST_RATIO)

        # Categorize sources
        test_only = {s: sp for s, sp in sources.items()
                     if "test" in sp and "train" not in sp}
        train_only = {s: sp for s, sp in sources.items()
                      if "train" in sp and "test" not in sp}
        shared = {s: sp for s, sp in sources.items()
                  if "train" in sp and "test" in sp}

        # Start with non-shared counts
        test_count = sum(len(sp["test"]) for sp in test_only.values())
        train_count = sum(len(sp["train"]) for sp in train_only.values())

        # Assign shared sources
        assigned_test = set()
        assigned_train = set()

        # Sort shared sources by total crop count (smallest first for
        # fine-grained filling of test quota)
        shared_sorted = sorted(
            shared.items(),
            key=lambda x: len(x[1].get("train", [])) + len(x[1].get("test", []))
        )

        test_deficit = target_test - test_count
        for src, sp in shared_sorted:
            n = len(sp.get("train", [])) + len(sp.get("test", []))
            if test_deficit > 0 and n <= test_deficit + 2:
                assigned_test.add(src)
                test_count += n
                test_deficit -= n
            else:
                assigned_train.add(src)
                train_count += n

        # Compute file moves
        cls_moves_to_train = []
        cls_moves_to_test = []

        for src in assigned_train:
            # Move test files of this source to train
            for f in shared[src].get("test", []):
                cls_moves_to_train.append(f)

        for src in assigned_test:
            # Move train files of this source to test
            for f in shared[src].get("train", []):
                cls_moves_to_test.append(f)

        print(f"  {cls}: final train={train_count}, test={test_count} "
              f"(moves: {len(cls_moves_to_train)} test->train, "
              f"{len(cls_moves_to_test)} train->test)")

        for f in cls_moves_to_train:
            moves[("test", "train", cls)].append(f)
        for f in cls_moves_to_test:
            moves[("train", "test", cls)].append(f)

    # ── 3. Execute moves ──────────────────────────────────────────────────
    print("\n=== Step 3: Moving files ===")
    total_moved = 0
    for (src_split, dst_split, cls), files in sorted(moves.items()):
        if not files:
            continue
        src_dir = root / src_split / cls
        dst_dir = root / dst_split / cls
        print(f"  {src_split}/{cls} -> {dst_split}/{cls}: {len(files)} file(s)")
        for f in files:
            src_path = src_dir / f
            dst_path = dst_dir / f
            if not dry_run:
                shutil.move(str(src_path), str(dst_path))
            total_moved += 1
    print(f"  Total moved: {total_moved}")

    # ── 4. Remove exact intra-split duplicates ────────────────────────────
    print("\n=== Step 4: Removing intra-split duplicates ===")
    total_removed = 0
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = root / split / cls
            if not cls_dir.is_dir():
                continue
            seen = {}
            dupes = []
            for f in sorted(os.listdir(cls_dir)):
                if not f.endswith(".npy"):
                    continue
                h = hash_npy(cls_dir / f)
                if h in seen:
                    dupes.append(f)
                else:
                    seen[h] = f
            if dupes:
                print(f"  {split}/{cls}: removing {len(dupes)} duplicate(s)")
                for f in dupes:
                    if not dry_run:
                        os.remove(cls_dir / f)
                    total_removed += 1
    print(f"  Total removed: {total_removed}")

    # ── 5. Verify ─────────────────────────────────────────────────────────
    print("\n=== Final dataset counts ===")
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = root / split / cls
            count = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")]) \
                if cls_dir.is_dir() else 0
            print(f"  {split}/{cls}: {count}")

    # Verify no source leaks remain
    print("\n=== Verification: source leaks ===")
    any_leak = False
    for cls in CLASSES:
        train_sources = set()
        test_sources = set()
        for f in os.listdir(root / "train" / cls):
            src = parse_source(f)
            if src:
                train_sources.add(src)
        for f in os.listdir(root / "test" / cls):
            src = parse_source(f)
            if src:
                test_sources.add(src)
        shared = train_sources & test_sources
        if shared:
            print(f"  FAIL {cls}: {len(shared)} shared source(s) remain!")
            any_leak = True
        else:
            print(f"  OK {cls}: 0 shared sources")

    if any_leak:
        print("\nERROR: Source leaks still present!")
        return 1
    else:
        print("\nAll source leaks resolved.")
        return 0


if __name__ == "__main__":
    exit(main())
