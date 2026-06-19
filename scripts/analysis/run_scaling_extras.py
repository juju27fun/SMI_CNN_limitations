"""Run only the new XS/XXS/Nano scaling variants on tier 1.

Avoids re-training the existing S/M/L variants. Reuses run_single from benchmark_zoo.

Usage:
    python scripts/run_scaling_extras.py --seeds 42 --epochs 150
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_zoo import run_single
from p0.models import MODEL_VARIANTS, get_family
from p0.training_utils import add_common_training_args


NEW_SUFFIXES = ("-Pico", "-Nano", "-XXS", "-XS")


def main():
    parser = argparse.ArgumentParser(description="Run only Pico/Nano/XXS/XS scaling variants")
    add_common_training_args(parser, data_dir_default="data/dataset")
    parser.set_defaults(patience=20, epochs=150, scheduler="cosine",
                        output_dir="results/benchmark2")
    parser.add_argument("--tier", type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="42")
    # Tier-5/6-specific args (must mirror benchmark_zoo.py defaults so that
    # run_single can read them regardless of tier).
    parser.add_argument("--noise-dir", type=str, default="data/Noise",
                        help="Directory of real noise .npy files (tier 5)")
    parser.add_argument("--real-test-dir", type=str, default="data/S7_pure_real",
                        help="Directory of real measurements for tier 6")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Collect new variants from all families
    new_variants = sorted(
        v for v in MODEL_VARIANTS if any(v.endswith(s) for s in NEW_SUFFIXES)
    )

    print(f"New variants to run ({len(new_variants)}):")
    for v in new_variants:
        print(f"  {v}  (family={get_family(v)})")

    print(f"\nTier: {args.tier} | Seeds: {seeds} | Epochs: {args.epochs}")
    print(f"Total runs: {len(new_variants) * len(seeds)}")

    successes = 0
    for i, model_name in enumerate(new_variants, 1):
        for seed in seeds:
            print(f"\n>>> [{i}/{len(new_variants)}] {model_name} seed={seed}")
            try:
                run_single(model_name, args.tier, seed, args)
                successes += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone. {successes}/{len(new_variants) * len(seeds)} runs succeeded.")


if __name__ == "__main__":
    main()
