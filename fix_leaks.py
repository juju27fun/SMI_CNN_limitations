#!/usr/bin/env python3
"""
Fix dataset leaks:
1. Remove leaked test files (source-level leaks + exact duplicates)
2. Remove intra-split duplicates (one copy from each pair, respecting split)
3. Replace removed files with fresh data from Downloads/<size>_DB
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

DATASET_ROOT = Path("/home/intern/Documents/particlesSebas/dataset")
REPORTS_DIR = Path("/home/intern/Documents/particlesSebas/leak_reports")
DB_ROOT = Path("/home/intern/Downloads")

CLASSES = ["10um", "2um", "4um"]
SPLITS = ["train", "test"]


# ---------- helpers ----------

def load_report(name):
    """Load a JSON leak report, returning [] if the file does not exist."""
    path = REPORTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"  (no {name}.json found – skipping)")
        return []
    with open(path) as f:
        return json.load(f)


def extract_class(filepath_str):
    """Extract the class name from a report path (robust to ./ and absolute paths).

    Works for paths like:
        dataset/test/10um/file.npy
        ./dataset/test/10um/file.npy
        /home/.../dataset/test/10um/file.npy
    """
    return Path(filepath_str).parent.name


def extract_split(filepath_str):
    """Extract the split name (train/test) from a report path."""
    return Path(filepath_str).parent.parent.name


def extract_source(filename):
    """Extract original source name from a dataset filename.

    e.g. 'HFocusing_5_10_10um_0_1126.npy48.npy' -> 'HFocusing_5_10_10um_0_1126.npy'
    """
    idx = filename.find(".npy")
    if idx >= 0:
        return filename[:idx + 4]
    return filename


def resolve_path(relative_path):
    """Resolve a report path to an absolute path on disk."""
    return DATASET_ROOT.parent / relative_path


def hash_npy(path):
    """Compute MD5 of .npy file content."""
    return hashlib.md5(np.load(path).tobytes()).hexdigest()


# ---------- 1. Load leak reports ----------
print("=== Loading reports ===")
source_leaks = load_report("source_leaks")
exact_duplicates = load_report("exact_duplicates")
intra_duplicates = load_report("intra_duplicates")

# ---------- 2. Collect files to remove ----------

# Per-split sets of file paths to remove
files_to_remove = {"train": set(), "test": set()}

# Source leaks & exact duplicates → remove from test
for entry in source_leaks:
    for tf in entry["test_files"]:
        files_to_remove["test"].add(tf)

for entry in exact_duplicates:
    files_to_remove["test"].add(entry["test_file"])

# Intra-split duplicates → remove file_b from the CORRECT split
for entry in intra_duplicates:
    split = entry["split"]
    files_to_remove[split].add(entry["file_b"])

# ---------- 3. Count removals per class per split ----------
removal_counts = {s: {} for s in SPLITS}

for split in SPLITS:
    for f in files_to_remove[split]:
        cls = extract_class(f)
        removal_counts[split][cls] = removal_counts[split].get(cls, 0) + 1

print("\n=== Files to remove ===")
for split in SPLITS:
    total = len(files_to_remove[split])
    print(f"{split}: {total} file(s)")
    for cls, count in sorted(removal_counts[split].items()):
        print(f"  {split}/{cls}: {count}")

# ---------- 4. Build content-hash set of files that will remain ----------
print("\n=== Building content-hash index of remaining dataset ===")

# Basenames to remove (for quick lookup)
basenames_to_remove = set()
for split in SPLITS:
    for f in files_to_remove[split]:
        basenames_to_remove.add((split, extract_class(f), Path(f).name))

# Hash every file that will REMAIN in the dataset after removals
remaining_hashes = set()
for split in SPLITS:
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / split / cls
        if not cls_dir.exists():
            continue
        for fname in os.listdir(cls_dir):
            if (split, cls, fname) in basenames_to_remove:
                continue
            remaining_hashes.add(hash_npy(cls_dir / fname))

print(f"  {len(remaining_hashes)} unique content hashes in remaining dataset")

# ---------- 5. Compute remaining sources per split/class ----------
remaining_sources = {s: {} for s in SPLITS}
for split in SPLITS:
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / split / cls
        if not cls_dir.exists():
            continue
        all_files = set(os.listdir(cls_dir))
        removed = {Path(f).name for f in files_to_remove[split] if extract_class(f) == cls}
        remaining = all_files - removed
        remaining_sources[split][cls] = {extract_source(f) for f in remaining}

# Basenames still in the dataset (after subtracting removals) — for filename-collision check
all_remaining_basenames = set()
for split in SPLITS:
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / split / cls
        if not cls_dir.exists():
            continue
        all_files = set(os.listdir(cls_dir))
        removed = {Path(f).name for f in files_to_remove[split] if extract_class(f) == cls}
        all_remaining_basenames.update(all_files - removed)

# ---------- 6. Find replacement files from DB ----------
class_to_db = {
    "10um": DB_ROOT / "10um_DB",
    "2um": DB_ROOT / "2um_DB",
    "4um": DB_ROOT / "4um_DB",
}

# Track hashes of already-selected replacements (avoid picking duplicates of each other)
selected_replacement_hashes = set()


def find_replacements(cls, split, count_needed):
    """Find replacement files from the DB that won't introduce new leaks."""
    db_dir = class_to_db[cls]
    available = sorted(os.listdir(db_dir))

    # Sources to avoid: opposite split (would create cross-split leak)
    opposite = "test" if split == "train" else "train"
    forbidden_sources = remaining_sources[opposite].get(cls, set())

    # Also avoid sources already in the same split (would create intra-split risk)
    same_split_sources = remaining_sources[split].get(cls, set())

    selected = []
    for fname in available:
        if len(selected) >= count_needed:
            break
        # Skip if filename already in dataset
        if fname in all_remaining_basenames:
            continue
        # Skip if source overlaps with opposite split
        src = extract_source(fname)
        if src in forbidden_sources:
            continue
        # Skip if source overlaps with same split
        if src in same_split_sources:
            continue
        # Skip if content is identical to any remaining dataset file or already-selected replacement
        fpath = db_dir / fname
        h = hash_npy(fpath)
        if h in remaining_hashes or h in selected_replacement_hashes:
            continue
        selected.append(fname)
        selected_replacement_hashes.add(h)

    if len(selected) < count_needed:
        print(f"  WARNING: Only found {len(selected)}/{count_needed} replacements for {split}/{cls}")

    return selected


print("\n=== Finding replacements ===")
replacements = {s: {} for s in SPLITS}
for split in SPLITS:
    for cls in sorted(removal_counts[split].keys()):
        count = removal_counts[split][cls]
        repl = find_replacements(cls, split, count)
        replacements[split][cls] = repl
        print(f"  {split}/{cls}: need {count}, found {len(repl)}")

# ---------- 7. Execute: remove files and copy replacements ----------
print("\n=== Executing removals ===")

# Track actual removals per split/class so we only copy that many replacements
actual_removals = {s: {} for s in SPLITS}

for split in SPLITS:
    removed = 0
    for f in sorted(files_to_remove[split]):
        full_path = resolve_path(f)
        if full_path.exists():
            os.remove(full_path)
            removed += 1
            cls = extract_class(f)
            actual_removals[split][cls] = actual_removals[split].get(cls, 0) + 1
        else:
            print(f"  WARNING: {f} not found (already removed?)")
    print(f"Removed {removed} {split} files")

print("\n=== Copying replacements ===")
for split in SPLITS:
    for cls, files in replacements[split].items():
        # Only copy as many replacements as files actually removed
        n_needed = actual_removals[split].get(cls, 0)
        to_copy = files[:n_needed]
        if not to_copy:
            print(f"  {split}/{cls}: 0 to copy (files were already removed)")
            continue
        dest_dir = DATASET_ROOT / split / cls
        db_dir = class_to_db[cls]
        for fname in to_copy:
            shutil.copy2(db_dir / fname, dest_dir / fname)
        print(f"  Copied {len(to_copy)} files to {split}/{cls}")

# ---------- 8. Verify final counts ----------
print("\n=== Final dataset counts ===")
for split in SPLITS:
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / split / cls
        count = len(os.listdir(cls_dir)) if cls_dir.exists() else 0
        print(f"  {split}/{cls}: {count}")
