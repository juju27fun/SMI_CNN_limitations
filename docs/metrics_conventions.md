# Metrics conventions — latency, RTF, embeddability

Convention adopted in P0 since 2026-04 to match P1
(`P1/.claude/rules/metrics-conventions.md`). Prior runs measured on
GPU are still on disk under `*_gpu_backup.csv` / `_rt_data_gpu_backup.json`.

## Why CPU is the canonical reference

GPU latency at batch=1 is dominated by the per-call CUDA kernel-launch
floor (~0.13 ms on RTX A1000). Below this floor, model latency is
indistinguishable from launch overhead, and the small/medium variants
of the zoo cluster at the leftmost x-edge regardless of compute. CPU
latency, single-thread, batch=1, is the closest available proxy for
the embedded deployment target (Cortex-A / FPGA softcore) and
discriminates between architectures across the full parameter range.

## Measurement procedure

`training_utils.measure_cpu_latency(model, input_shape, *, warmup=20, n_runs=200)`:

1. `copy.deepcopy(model).cpu()` — never mutates the trained model.
2. `model_cpu.train(False)`.
3. 20 warmup forward passes (untimed).
4. 200 timed forward passes; per-pass duration via `time.perf_counter()`.
5. `del model_cpu`.

Returns `{median_ms, p95_ms, mean_ms, latency_device="cpu"}`.

## JSON / CSV keys

| Key | Type | Source | Semantics |
|---|---|---|---|
| `latency_median_ms` | float | runs JSON, `summary.csv:Latency_Median_ms` | Median CPU latency (ms), batch=1 |
| `latency_p95_ms` | float | runs JSON | p95 CPU latency (ms) |
| `latency_device` | str | runs JSON | Always `"cpu"` since 2026-04 |

## W&B summary keys

| Key | Type | Notes |
|---|---|---|
| `inference_latency_median_ms` | float | Mirror of `latency_median_ms` |
| `inference_latency_p95_ms` | float | Mirror of `latency_p95_ms` |
| `latency_device` | str | Always `"cpu"` |

The legacy key `inference_latency_ms` (without `_median_`) is removed
as of 2026-04 — `train4classes.py` and `train3classes_proof.py` now
emit `inference_latency_median_ms` for parity with `benchmark_zoo.py`.

## Real-time factor (RTF)

For doublet inference (`artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained/_rt_data.json`):

```
ρ = signal_duration_ms / latency_median_ms
signal_duration_ms = 8.192   # 4096 samples @ 500 kHz × 1000
```

ρ > 1 means inference is faster than acquisition (real-time capable).
ρ < 1 means saturation (faster acquisition than the model can
process). The figure
`artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained/realtime_factor_doublet_3fam.pdf`
shades the ρ < 1 zone for visual clarity, with annotation
`"CPU torch, batch = 1"` for device provenance.

## Cross-project parity

P1 documents the same convention at
`P1/.claude/rules/metrics-conventions.md`. Both projects use identical
warmup (20), n_runs (200), batch (1), and helper logic. Figures from
the two projects can be cross-referenced because the latency metric is
device-comparable.

## Re-measuring legacy runs

When existing runs need to be retrofitted from GPU → CPU:

```bash
# benchmark2 (~30 min CPU on 432 runs)
venv/bin/python scripts/remeasure_cpu_latency_benchmark2.py --dry-run
venv/bin/python scripts/remeasure_cpu_latency_benchmark2.py

# doublet_3fam_retrained (~2 min CPU on 22 + 3 models)
venv/bin/python scripts/remeasure_cpu_latency_doublet_3fam.py --dry-run
venv/bin/python scripts/remeasure_cpu_latency_doublet_3fam.py
```

Both scripts back up the prior GPU summary before patching in place.
