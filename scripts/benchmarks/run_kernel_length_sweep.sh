#!/usr/bin/env bash
# Kernel × input-length sweep on the 3 best-accuracy models from the
# doublet_3fam_retrained experiment.
#
# 3 families × 3 kernel sizes × 3 input lengths = 27 training runs (seed 42).
# Outputs:
#   results/kernel_length_sweep_3fam/checkpoints/<run_tag>/best_model.pth
#   results/kernel_length_sweep_3fam/runs/<run_tag>.json
#   results/kernel_length_sweep_3fam/summary.csv
#   results/kernel_length_sweep_3fam/doublet_accuracy.json (via infer_doublets.py --sweep)
#
# Each run takes ~3-10 minutes on GPU. Budget: ~3-6 h total.
set -euo pipefail

OUT=results/kernel_length_sweep_3fam
DATA=data/dataset_doublet
MODELS=("Conv1DGAP-L" "EfficientNet1D-S" "ResNet1D-S")
KERNELS=(3 9 49)
LENGTHS=(512 2048 4096)

mkdir -p "$OUT"

echo "=================================================="
echo "Kernel × length sweep on 3 best models"
echo "  Models : ${MODELS[*]}"
echo "  Kernels: ${KERNELS[*]}"
echo "  Lengths: ${LENGTHS[*]}"
echo "  Output : $OUT"
echo "=================================================="

idx=0
total=$((${#MODELS[@]} * ${#KERNELS[@]} * ${#LENGTHS[@]}))
for MODEL in "${MODELS[@]}"; do
  for K in "${KERNELS[@]}"; do
    for L in "${LENGTHS[@]}"; do
      idx=$((idx + 1))
      echo ""
      echo ">>> [$idx/$total]  $MODEL  k=$K  L=$L"
      python benchmark_zoo.py \
        --model "$MODEL" \
        --tier 1 \
        --seeds 42 \
        --kernel-size "$K" \
        --input-length "$L" \
        --data-dir "$DATA" \
        --output-dir "$OUT"
    done
  done
done

# Aggregate all per-run JSONs into summary.csv (no benchmark2 figures since sweep mode).
python benchmark_zoo.py --aggregate-only --output-dir "$OUT"

# Doublet inference on the full 16384-sample test set.
python infer_doublets.py \
  --sweep \
  --output-dir "$OUT/checkpoints" \
  --data-dir "$DATA/test" \
  --figures-dir "$OUT/doublet_cms" \
  --save-figures \
  --sweep-out "$OUT/doublet_accuracy.json"

echo ""
echo "=================================================="
echo "Sweep complete. Next step:"
echo "  python scripts/plot_acc_length.py"
echo "=================================================="
