#!/usr/bin/env bash
# Full benchmark2 sweep: tier pass + variant pass, both online to W&B.
#   Pass A: 8 base models  x 6 tiers x 3 seeds = 144 runs (tier figures)
#   Pass B: 51 variants    x 1 tier  x 3 seeds = 153 runs (scaling figures)
# Output:  results/benchmark2/{runs,figures,leaderboard.md,summary.csv}
# Logs:    logs/benchmark2_full_<timestamp>.log
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/benchmark2_full_${TS}.log"
SEEDS="42,123,7"
EPOCHS=150

{
  echo "============================================================"
  echo "  BENCHMARK 2 — FULL SWEEP   $(date -Iseconds)"
  echo "  Seeds=${SEEDS}  Epochs=${EPOCHS}"
  echo "  W&B: online (project=particle-benchmark)"
  echo "============================================================"

  echo
  echo ">>> PASS A: tier benchmark (8 base x 6 tiers x 3 seeds = 144 runs)"
  python benchmark_zoo.py --all --tier all --seeds "${SEEDS}" --epochs "${EPOCHS}"
  RC_A=$?
  echo "Pass A finished with rc=${RC_A} at $(date -Iseconds)"

  echo
  echo ">>> PASS B: variant benchmark (51 variants x tier 1 x 3 seeds = 153 runs)"
  python benchmark_zoo.py --all --tier 1 --scaling --seeds "${SEEDS}" --epochs "${EPOCHS}"
  RC_B=$?
  echo "Pass B finished with rc=${RC_B} at $(date -Iseconds)"

  echo
  echo ">>> Final aggregation + figure publication (scaling enabled)"
  python benchmark_zoo.py --all --tier all --scaling --seeds "${SEEDS}" --epochs "${EPOCHS}" --aggregate-only
  RC_AGG=$?
  echo "Aggregation finished with rc=${RC_AGG} at $(date -Iseconds)"

  echo
  echo "============================================================"
  echo "  DONE   passA=${RC_A}  passB=${RC_B}  agg=${RC_AGG}"
  echo "============================================================"
} >"${LOG}" 2>&1

echo "${LOG}"
