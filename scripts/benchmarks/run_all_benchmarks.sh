#!/usr/bin/env bash
set -e

DATASETS=(
    data/dataset
    data/S0_baseline
    data/S1_white
    data/S2_colored
    data/S3_realistic
    data/S4_real_noise
    data/S5_signal_realism
    data/S6_noise_realism
    data/S7_pure_real
    data/S8_colored_low
    data/S9_colored_high
    data/S_union
)

for ds in "${DATASETS[@]}"; do
    echo "========================================"
    echo "  Benchmarking: $ds"
    echo "========================================"
    python benchmark.py \
        --data-dir "$ds" \
        --dataset-name "$ds" \
        --real-test-dir data/dataset/test \
        --noise-dir data/Noise
done

echo "All benchmarks complete."
