"""Generate 4-class dataset directories (2um, 4um, 10um, Noise).

Symlinks existing particle class folders from S1_white, S2_colored, and dataset,
then generates Noise class samples by slicing real noise files (Noise/*.npy, 16384 samples)
into 2500-length segments.

Usage:
    python generate_4class_dataset.py
    python generate_4class_dataset.py --noise-dir ./Noise --seed 42
    python generate_4class_dataset.py --datasets S1_white  # single dataset only
"""

import argparse
import os

import numpy as np

SEGMENT_LENGTH = 2500
NOISE_FILE_LENGTH = 16384
TRAIN_COUNT = 403
TEST_COUNT = 101

DATASETS = ["data/S1_white", "data/S2_colored", "data/dataset"]
PARTICLE_CLASSES = ["2um", "4um", "10um"]


def load_noise_files(noise_dir):
    """Load all .npy noise files from directory. Returns list of np arrays."""
    import glob
    paths = sorted(glob.glob(os.path.join(noise_dir, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in '{noise_dir}'")
    arrays = [np.load(p) for p in paths]
    print(f"Loaded {len(arrays)} noise files from '{noise_dir}'")
    return arrays


def generate_noise_segments(noise_arrays, num_segments, rng):
    """Generate non-overlapping, content-unique 2500-sample segments.

    Each segment is drawn from a random noise file at a random non-overlapping
    offset. Segments with duplicate content (identical bytes) are skipped to
    prevent intra-split and cross-split exact duplicates.
    """
    import hashlib

    # Build a pool of available (file_idx, offset) slots
    slots = []
    for file_idx, arr in enumerate(noise_arrays):
        n_slots = len(arr) // SEGMENT_LENGTH
        for slot_idx in range(n_slots):
            slots.append((file_idx, slot_idx * SEGMENT_LENGTH))

    if len(slots) < num_segments:
        raise ValueError(
            f"Need {num_segments} segments but only {len(slots)} non-overlapping "
            f"slots available from {len(noise_arrays)} noise files"
        )

    # Shuffle all slots and pick content-unique segments
    order = rng.permutation(len(slots))
    segments = []
    seen_hashes = set()
    for idx in order:
        if len(segments) >= num_segments:
            break
        file_idx, offset = slots[idx]
        segment = noise_arrays[file_idx][offset:offset + SEGMENT_LENGTH].copy()
        segment = segment.astype(np.float64)
        h = hashlib.md5(segment.tobytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        segments.append(segment)

    if len(segments) < num_segments:
        raise ValueError(
            f"Need {num_segments} unique segments but only found "
            f"{len(segments)} after deduplication"
        )
    return segments


def create_4class_dataset(source_dir, output_dir, noise_segments_train,
                          noise_segments_test):
    """Create a 4-class dataset directory with symlinks for particle classes
    and real noise .npy files for the Noise class."""
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Symlink existing particle class folders
        for cls in PARTICLE_CLASSES:
            link_path = os.path.join(split_dir, cls)
            target = os.path.relpath(
                os.path.join(source_dir, split, cls),
                start=split_dir,
            )
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path)
            os.symlink(target, link_path)

        # Create Noise class folder with .npy files
        noise_dir = os.path.join(split_dir, "Noise")
        os.makedirs(noise_dir, exist_ok=True)

        segments = noise_segments_train if split == "train" else noise_segments_test
        for i, segment in enumerate(segments):
            np.save(os.path.join(noise_dir, f"noise_{i:04d}.npy"), segment)

    # Count and report
    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, split)
        for cls in PARTICLE_CLASSES + ["Noise"]:
            cls_dir = os.path.join(split_dir, cls)
            count = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
            print(f"  {split}/{cls}: {count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-class datasets (2um, 4um, 10um, Noise)"
    )
    parser.add_argument("--noise-dir", type=str, default="data/Noise",
                        help="Directory containing real noise .npy files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Source dataset directories to process")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load noise files once
    noise_arrays = load_noise_files(args.noise_dir)

    total_needed = (TRAIN_COUNT + TEST_COUNT) * len(args.datasets)
    total_slots = sum(len(arr) // SEGMENT_LENGTH for arr in noise_arrays)
    print(f"Need {total_needed} total segments, {total_slots} slots available")

    for source_dir in args.datasets:
        if not os.path.isdir(source_dir):
            print(f"WARNING: '{source_dir}' not found, skipping")
            continue

        output_dir = f"{source_dir}_4c"
        print(f"\n{'='*60}")
        print(f"Creating {output_dir} from {source_dir}")
        print(f"{'='*60}")

        # Generate noise segments for this dataset (single draw, then split
        # to guarantee no overlap between train and test)
        all_segments = generate_noise_segments(
            noise_arrays, TRAIN_COUNT + TEST_COUNT, rng
        )
        noise_train = all_segments[:TRAIN_COUNT]
        noise_test = all_segments[TRAIN_COUNT:]

        create_4class_dataset(source_dir, output_dir, noise_train, noise_test)
        print(f"Done: {output_dir}")

    print(f"\nAll 4-class datasets generated.")


if __name__ == "__main__":
    main()
