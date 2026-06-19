#!/usr/bin/env bash
# Phase 2 — Kernel × length sweep with GRADUATED DECIMATION (not center-crop).
#
# Uses AdaptiveBandpassDecimate: bandpass with dynamic upper cutoff
# (min(100 kHz, 0.9 * Nyquist_post)) then stride-decimate from the native
# 16384-sample signal down to the target input length.
#
# 3 families × 3 kernel sizes × 6 input lengths = 54 training runs (seed 42).
# Budget: 6-15 h on GPU (the slowest run is L=16384 native, no stride).
#
# Outputs under results/kernel_length_decim_sweep_3fam/:
#   checkpoints/<run_tag>/best_model.pth       (54 checkpoints, run_tag carries -decim)
#   runs/<run_tag>.json                         (54 per-run JSONs)
#   summary.csv                                 (aggregated)
#   doublet_accuracy.json                       (from infer_doublets.py --sweep)
#   doublet_cms/doublet_cm_*.pdf                (54 per-model confusion matrices)
set -euo pipefail

OUT=results/kernel_length_decim_sweep_3fam
DATA=data/dataset_doublet
NATIVE=16384
MODELS=("Conv1DGAP-L" "EfficientNet1D-S" "ResNet1D-S")
KERNELS=(3 9 49)
LENGTHS=(512 1024 2048 4096 8192 16384)

mkdir -p "$OUT"

echo "=================================================="
echo "Phase 2 — Kernel × length sweep (graduated decimation)"
echo "  Native  : $NATIVE samples @ 2 MHz"
echo "  Models  : ${MODELS[*]}"
echo "  Kernels : ${KERNELS[*]}"
echo "  Lengths : ${LENGTHS[*]}"
echo "  Output  : $OUT"
echo "=================================================="

idx=0
total=$((${#MODELS[@]} * ${#KERNELS[@]} * ${#LENGTHS[@]}))
for MODEL in "${MODELS[@]}"; do
  for K in "${KERNELS[@]}"; do
    for L in "${LENGTHS[@]}"; do
      idx=$((idx + 1))
      echo ""
      echo ">>> [$idx/$total]  $MODEL  k=$K  L=$L  (factor=$((NATIVE / L)))"
      python benchmark_zoo.py \
        --model "$MODEL" \
        --tier 1 \
        --seeds 42 \
        --kernel-size "$K" \
        --input-length "$L" \
        --native-length "$NATIVE" \
        --data-dir "$DATA" \
        --output-dir "$OUT"
    done
  done
done

# Aggregate per-run JSONs into summary.csv.
python benchmark_zoo.py --aggregate-only --output-dir "$OUT"

# Doublet inference (AdaptiveBandpassDecimate auto-selected via -decim tag).
python infer_doublets.py \
  --sweep \
  --output-dir "$OUT/checkpoints" \
  --data-dir "$DATA/test" \
  --figures-dir "$OUT/doublet_cms" \
  --save-figures \
  --sweep-out "$OUT/doublet_accuracy.json"

# Regenerate publication plots.
python scripts/plot_acc_length.py \
  --json "$OUT/doublet_accuracy.json" \
  --summary-csv "$OUT/summary.csv" \
  --out-dir "$OUT"

echo ""
echo "=================================================="
echo "Phase 2 sweep complete."
echo "PDFs in $OUT/:"
echo "  acc_length_kernel_3fam.pdf"
echo "  acc_kernel_length_3fam.pdf"
echo "  delta_acc_length_3fam.pdf"
echo "  latency_kernel_3fam.pdf"
echo "=================================================="
