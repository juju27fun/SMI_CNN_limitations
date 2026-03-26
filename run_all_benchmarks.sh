#!/usr/bin/env bash
set -e

DATASETS=(
    dataset
    S0_baseline
    S1_white
    S2_colored
    S3_realistic
    S4_real_noise
    S5_signal_realism
    S6_noise_realism
    S7_pure_real
    S8_colored_low
    S9_colored_high
    S_union
)

for ds in "${DATASETS[@]}"; do
    echo "========================================"
    echo "  Benchmarking: $ds"
    echo "========================================"
    python benchmark.py \
        --data-dir "$ds" \
        --dataset-name "$ds" \
        --real-test-dir dataset/test \
        --noise-dir Noise
done

echo "All benchmarks complete."
