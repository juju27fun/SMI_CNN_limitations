# Benchmark Construction Checklist
## Particle Classification in 1D Signals — Benchmark 2: Model Zoo on Real Data

---

## Phase 0 — Scoping & Motivation

- [x] **Why does this benchmark need to exist?** Write a 3–5 sentence justification. What gap does it fill? Who is the target audience (yourself, your lab, the community)?
In the context of research to put transformer models on FPGA, I need to prove the limitations of CNNs in 1D SMI signal classification. Benchmark 1 evaluated a single architecture (Conv1D) across 12 synthetic datasets to study sim-to-real transfer. **Benchmark 2 flips the axis**: it evaluates all 8 architectures from the model zoo on the **real dataset** (S7_pure_real, 3 classes: 2um, 4um, 10um) to compare models on the data that actually matters for deployment. The goal is to build an accuracy-vs-efficiency Pareto front that answers: "which architecture gives the best classification on real OFI signals for the smallest hardware footprint?" This directly informs which model to target for FPGA synthesis.

- [x] **What claim should the benchmark support?**
"Among CNN architectures of comparable parameter count (~5M), lightweight designs (MobileNet1D, EfficientNet1D) match or approach the accuracy of heavier architectures (ResNet1D, DenseNet1D) on real OFI particle signals while requiring significantly fewer MACs and lower peak RAM — making them better candidates for FPGA deployment." Additionally: "All tested CNN architectures plateau within a narrow accuracy band on this 3-class real task, motivating the exploration of transformer-based alternatives."

- [x] **What is the scope?**
1D signals only. 3 particle classes (2um, 4um, 10um) from the OFI sensor. No noise class in this benchmark (noise rejection was covered in Benchmark 1 OOD evaluation). Only the real dataset S7_pure_real is the primary evaluation target. Models trained on a synthetic dataset (S1_white_4c or S3_realistic) are also tested on S7_pure_real to measure the generalization gap, but the core ranking uses models **trained and tested on real data**.

- [x] **What are the success criteria for the benchmark itself?**
  1. At least 5 percentage points of spread between the weakest and strongest baseline model on the real test set — if all 8 models cluster within 2%, the task is too easy and a harder real dataset variant is needed.
  2. A clear Pareto front emerges when plotting accuracy vs. MACs (not all models on the same point).
  3. Latency and RAM measurements are stable across repeated runs (coefficient of variation < 5%).

- [x] **Literature review:**
No standard benchmark exists for 1D OFI particle classification — this is a niche domain. Relevant references:
  - **UCR Time Series Archive**: the standard benchmark for 1D classification, but focuses on general time-series, not sensor-specific physics-based signals. We borrow their evaluation protocol (fixed splits, multi-seed runs, critical difference diagrams).
  - **MLPerf Tiny**: benchmarks for edge ML inference (latency, model size, accuracy). We borrow their efficiency metric methodology (MACs, peak RAM, latency protocol).
  - **InceptionTime (Fawaz et al., 2020)**: established that InceptionTime is competitive with ResNet on UCR. We include both in our zoo.
  - Gap: none of these benchmarks consider the sim-to-real transfer dimension or FPGA-portability constraints specific to OFI sensors.

---

## Phase 1 — Metric Definition

### 1.1 — Primary Metrics
- [x] **Accuracy:** Top-1 accuracy, macro-averaged across 3 classes (2um, 4um, 10um). Per-class accuracy and F1 also reported. Classes are roughly balanced (~100 test samples each) so no special imbalance handling is needed.
- [x] **Latency:** Wall-clock inference time per single sample (batch_size=1) on CPU. Measured as median over 1000 runs after 50 warmup passes. CPU-only because the FPGA deployment target has no GPU. Preprocessing (bandpass + decimate) is **excluded** from latency — it is fixed across all models.
- [x] **MACs (Multiply-Accumulate Operations):** Counted using `thop` (already integrated in `training_utils.py`). Forward pass only. Input shape: `(1, 1, 625)` (raw) or `(1, 1, 156)` (after decimation=4). Document which input shape is used.
- [x] **Peak RAM:** Measured using `tracemalloc` on CPU (since target is FPGA/edge, CPU memory is the relevant proxy). Measured at batch_size=1 during a single forward pass. Reset peak before each model.
- [x] **Model size:** Both parameter count (`sum(p.numel() for p in model.parameters())`) and on-disk size of saved `state_dict` (`.pth` file size in MB). The on-disk size captures quantization effects if later applied.

### 1.2 — Composite / Derived Metrics
- [x] **Accuracy vs. latency curve:** Scatter plot with one point per model. Pareto front highlighted. Models above the Pareto front are dominated.
- [x] **Accuracy vs. MACs:** Same scatter + Pareto front. MACs on log scale if range exceeds 10x.
- [x] **Efficiency score:** Define as `accuracy / log10(MACs)`. This rewards models that achieve high accuracy with exponentially fewer operations. Document this formula in the results table header.
- [x] **Accuracy vs. difficulty tier:** Not applicable for Benchmark 2 in its primary form (single real dataset). If a "hard" real dataset is created (see Phase 2.2), report accuracy for both the standard and hard tiers per model.
- [ ] **Architecture scaling curves (v1.1):** Plot accuracy versus latency (and/or MACs) with points grouped by architecture family and connected across size variants (e.g., ResNet1D-S/M/L, MobileNet1D-0.5x/1.0x/1.5x). Each family becomes a colored line; each variant is a point on that line. This reveals not just "which model is best?" but "how does each architecture family scale as compute increases?" — a much stronger argument for FPGA deployment decisions. Inspired by YOLO-generation comparison plots. **v1.0 approximation:** use the same visual logic (family-colored labeled points, clean Pareto frontier) but with one point per architecture since current models are single-scale. **v1.1 upgrade:** add 2–3 size variants per family to enable true scaling curves.

### 1.3 — Measurement Protocol
- [x] **Hardware specification:** All measurements on the same machine:
  - CPU: Intel Core Ultra 7 265
  - RAM: 32 GB
  - No GPU used for inference benchmarking (CPU-only to reflect edge deployment)
  - OS: Linux 6.17.0-1012-oem, x86_64
- [x] **Software specification:** PyTorch 2.10.0, Python 3.12.3, Ubuntu Linux, thop for MAC counting, scikit-learn 1.8.0 for metrics.
- [x] **Warmup:** 50 warmup inference passes before timing (consistent with `measure_inference_latency` in `training_utils.py` which uses warmup=10; increase to 50 for the benchmark).
- [x] **Repetitions:** 3 full training runs per model with seeds {42, 123, 7}. For inference metrics (latency, RAM), 1000 timed forward passes per model.
- [x] **Batch size policy:** Fixed batch_size=1 for all inference measurements (latency, RAM, MACs). Training uses batch_size=32 for all models.
- [x] **Precision:** FP32 for all measurements. FP16 and INT8 quantized inference are out of scope for v1 but flagged as future work (directly relevant to FPGA deployment).

---

## Phase 2 — Dataset Design

### 2.1 — Existing Data Audit
- [x] **Document your current dataset:**
  - **S7_pure_real** (primary): 3 classes (2um, 4um, 10um), train: 403/403/403 = 1209 samples, test: 100/101/101 = 302 samples. Signal length: 2500 points, float64. Sampling rate: from OFI sensor (document exact value). These are real signals — no synthetic generation involved.
  - **dataset** (secondary real): 3 classes, train: 401/393/398 = 1192 samples, test: 100/101/100 = 301 samples. Same physical sensor, potentially different acquisition sessions.
  - **S1_white_4c** (synthetic reference): 4 classes (includes Noise), train: 403/class, test: ~101/class. For cross-domain evaluation only.
- [x] **Why is it saturated?** The 3-class real task may be too easy for large CNNs (~5M params) because the particle size classes (2, 4, 10 um) produce distinct Doppler frequency signatures. If all 8 models achieve >95% on S7_pure_real, the benchmark fails to discriminate. This is the main risk.
- [x] **Can you reuse it as the "easy" tier?** Yes — S7_pure_real becomes Tier 1 (standard). A harder variant (Tier 2) can be constructed by: (a) reducing training set size (few-shot: 50 samples/class), (b) adding real noise corruption, or (c) mixing in ambiguous boundary signals.

### 2.2 — Difficulty Engineering
- [x] **Define difficulty axes:**
  1. **Training set size** — reduce from 403 to {200, 100, 50, 25} samples per class to test data efficiency. Smaller training sets penalize overparameterized models.
  2. **Signal corruption** — add real noise (from `Noise/` directory) at controlled SNR levels to the real test signals, creating a harder evaluation.
  3. **Class granularity** — if a second real dataset with closer particle sizes (e.g., 2um vs 3um) can be acquired, this would be the hardest tier.
- [x] **Design difficulty tiers:** All tiers derived from the same `datasets/processed/p0-baseline-3class/v1` source:
  - **Tier 1 (Standard):** Full train (~400/class) + clean test (~100/class). The primary benchmark.
  - **Tier 2 (Data-starved):** Same test set, but models trained on only 50 samples/class. Tests data efficiency — critical for real-world deployments where labelled real data is scarce.
  - **Tier 3 (Noisy):** Full train, but test signals corrupted with additive Gaussian noise at SNR=10 dB (always-on, p=1.0). Tests robustness to noise degradation.
  - **Tier 4 (Combined):** 50 samples/class train + noisy test (SNR=10 dB). The hardest tier — combines data scarcity and noise degradation.
- [x] **Validate that difficulty tiers actually modulate difficulty:** Run Conv1D (the simplest baseline) on all 4 tiers. Expected: Tier 1 > Tier 2 > Tier 3 > Tier 4 accuracy. If not monotonic, adjust tier parameters.
- [x] **Is the difficulty parametric and reproducible?** Yes — Tier 2 uses a fixed random seed for stratified subsampling. Tier 3 uses Gaussian noise at a fixed SNR=10 dB (always-on). Tier 4 combines both. All tiers derive from the same `datasets/processed/p0-baseline-3class/v1` — no external data dependencies.

### 2.3 — Dataset Construction
- [x] **Data source:** `datasets/processed/p0-baseline-3class/v1` — real data acquired from the OFI sensor. All 4 tiers derive from this single source.
- [x] **Dataset size per tier:**
  - Tier 1: ~301 test samples (100+101+100). Below the 1000/class ideal but this is the real-world constraint — all available labelled real data is used.
  - Tier 2: Same test samples, reduced train (50/class).
  - Tier 3: Same test samples with Gaussian noise, full train.
  - Tier 4: Same noisy test samples, reduced train (50/class).
  - **Mitigation for small test set**: report 95% confidence intervals via bootstrap (1000 resamples of the test set).
- [x] **Class balance:** Approximately balanced. Tier 1 test: 100/101/101. No rebalancing needed.
- [x] **Train / Validation / Test split:**
  - [x] Fixed splits: test set from `datasets/processed/p0-baseline-3class/v1/test` is frozen and never touched during development.
  - [x] Test set held out: the ~301 real test signals are the ground truth. No model selection decisions use this set.
  - [x] Validation: 20% of `datasets/processed/p0-baseline-3class/v1/train` (stratified via `StratifiedShuffleSplit`), used for early stopping and hyperparameter tuning.
  - [x] Split strategy: stratified random split with fixed seed=42. No grouped split needed since real data acquisition sessions are not tracked per-sample.
- [x] **Data leakage check:** Already performed via `dataset_leaks.py`. S7_pure_real audit results exist in `audit_artifacts/SMI_CNN_limitations/benchmarks/S7_pure_real/`. Verify no exact or near-duplicate contamination between train/test.
- [x] **Signal preprocessing:** Fixed preprocessing pipeline applied identically to all models: `BandpassFilter(5-100 kHz) -> Decimate(4x)`. Input to models: `(1, 625)` tensor. Preprocessing is NOT part of the benchmark (not varied across models).
- [x] **Data format:** `.npy` files (float64, shape `(2500,)`), one file per signal. Directory structure: `datasets/processed/p0-baseline-3class/v1/{train,test}/{2um,4um,10um}/*.npy`.

### 2.4 — Documentation
- [x] **Datasheet for the dataset:** Extend the existing `docs/generate_dataset_datasheet.md` with a section on S7_pure_real covering: sensor model, acquisition conditions, particle preparation method, labelling process (how are ground-truth sizes known?).
- [x] **License:** Internal / lab use. Not publicly released. Document this explicitly.
- [x] **Versioning:** v1.0 = current S7_pure_real. v1.1 if corrupted noise tiers are added. v2.0 if a new real dataset with more/harder classes is acquired.

---

## Phase 3 — Evaluation Harness

### 3.1 — Architecture
- [x] **Single entry point:** A new script `benchmark_zoo.py` that takes `--model <name>` (or `--all` to run all 8) and produces all metrics in a single JSON output. Internally calls the existing `training_utils.py` functions.
- [x] **Model interface contract:** Must follow the model zoo interface: `model = create_model(name, input_length=625, num_classes=3)`. Input: `(batch, 1, 625)` float32 tensor. Output: `(batch, 3)` logits. Must expose `model.feature_layer` (penultimate Linear) for feature extraction.
- [x] **Reproducibility:** Fixed seeds (42, 123, 7) for all randomness (data loading, weight initialization, train/val split). `torch.use_deterministic_algorithms(True)` where possible. Document any non-deterministic operations (e.g., certain CUDA kernels).

### 3.2 — Implementation
- [x] **Accuracy computation:** Top-1 accuracy (macro), per-class accuracy, full confusion matrix, per-class precision/recall/F1 via `classification_report`. All already implemented in `run_post_testing()` in `training_utils.py`.
- [x] **Latency measurement:** Use `time.perf_counter` (higher resolution than `time.time()`). CPU-only: no `torch.cuda.synchronize()` needed. 50 warmup, 1000 timed runs, report median and p95. Existing `measure_inference_latency()` in `training_utils.py` uses `time.time()` — upgrade to `time.perf_counter` and add p95 reporting.
- [x] **MAC counting:** `thop` already integrated via `compute_model_macs()`. Validate by manually computing MACs for Conv1D's first layer: `Conv1d(1, 64, kernel_size=5)` on input length 625 → expected MACs = 1 * 64 * 5 * 625 = 200,000. Cross-check against thop output.
- [x] **Peak RAM measurement:** Use `tracemalloc.start()` / `tracemalloc.get_traced_memory()` around a single forward pass at batch_size=1. Reset between models. Report peak in MB.
- [x] **Model size measurement:** Parameter count via `sum(p.numel() for p in model.parameters())`. On-disk size: `torch.save(model.state_dict(), tmp_path)` then `os.path.getsize(tmp_path)` in MB. Both already partially implemented.
- [x] **Output format:** JSON file per run: `{model_name, seed, tier, params, macs, size_mb, peak_ram_mb, latency_median_ms, latency_p95_ms, accuracy, per_class_f1, confusion_matrix, timestamp, hardware_info}`. Also a summary CSV aggregating all models x tiers for easy plotting.

### 3.3 — Sanity Checks
- [x] **Random baseline:** 3-class random classifier → 33.3% accuracy. Reported as floor in results table.
- [x] **Majority baseline:** Predict the most common class → ~33.6% (classes are near-balanced, so majority is barely above random). Reported alongside random baseline.
- [x] **Known model check:** Run Conv1D with seed=42 on datasets/processed/p0-baseline-3class/v1. Record accuracy. Re-run the harness and verify identical accuracy output.
- [x] **Determinism check:** Run Conv1D twice with same seed. Verify: identical accuracy, latency within 5% relative difference.

---

## Phase 4 — Baseline Population

### 4.1 — Model Selection
- [x] **Conv1D** — The original homemade model. 3 conv layers + 2 FC. ~5.3M params. The reference baseline.
- [x] **LeNet1D** — Classic shallow CNN. Smallest architecture in the zoo. Should be fastest but possibly least accurate.
- [x] **VGG1D** — Deep sequential convolutions, no skip connections. Heavy on MACs due to large FC layers.
- [x] **ResNet1D** — Skip connections. Strong baseline, likely among the most accurate.
- [x] **InceptionTime1D** — Multi-scale parallel convolutions. Good at capturing features at different temporal resolutions.
- [x] **MobileNet1D** — Depthwise separable convolutions. Designed for efficiency — expected low MACs relative to accuracy.
- [x] **EfficientNet1D** — Compound scaling. Another efficiency-oriented architecture.
- [x] **DenseNet1D** — Dense connections. High feature reuse, potentially good accuracy but higher memory.
- [ ] **Transformer-based (future):** A small 1D Vision Transformer (ViT-1D) or hybrid CNN-Transformer. This is the architecture the FPGA work aims to deploy — it should eventually join this benchmark.
- [ ] **Simple baselines (optional):** Logistic regression on FFT features, SVM on spectral features. Not strictly needed if all 8 CNNs provide enough spread, but useful as a "no deep learning" reference point.
- [x] **At least 5 models:** 8 models in the zoo, well above the minimum of 5.
- [ ] **Architecture size variants (v1.1 — for scaling curves):** Add small/medium/large variants within 2–4 core families to enable YOLO-style scaling curve analysis. Priority families and proposed width multipliers:
  - ResNet1D: S (0.5x width) / M (1.0x, current) / L (2.0x width)
  - MobileNet1D: 0.5x / 1.0x (current) / 1.5x width multiplier
  - EfficientNet1D: B0 (0.5x) / B1 (1.0x, current) / B2 (1.5x) compound scaling
  - Conv1D: small (32 filters) / base (64, current) / large (128 filters)
  Each variant must follow the same `create_model()` interface with an additional `width_multiplier` kwarg. Target: 3 variants x 4 families = 12 additional models, covering a param range from ~500K to ~20M. This gives enough spread for meaningful scaling curves.

### 4.2 — Fair Training Protocol
- [x] **Hyperparameter tuning budget:** Same for all models — no per-model tuning. Fixed hyperparameters: Adam optimizer, lr=6e-4, cosine annealing, batch_size=32, 150 epochs, patience=20 (early stopping), weight_decay=1e-4. This ensures the benchmark measures architecture quality, not tuning effort.
- [x] **Training configuration:**
  - Optimizer: Adam (lr=6e-4, weight_decay=1e-4)
  - Scheduler: Cosine annealing over 150 epochs
  - Augmentation: None for v1 (to isolate architecture effects). Augmentation ablation is future work.
  - Early stopping: patience=20 on validation accuracy
  - Validation split: 20% of training set, stratified, seed=42
  - Loss: CrossEntropyLoss
- [x] **Is training part of the benchmark?** Yes — each model is trained from scratch on the same data with the same hyperparameters. The benchmark measures the full pipeline (trainability + final accuracy + efficiency). Pre-trained weights are not used. This is deliberate: for FPGA deployment, the model must be trainable from scratch on domain-specific data.

### 4.3 — Results Table
- [x] **Build the results table:**

| Model | Params | MACs | Size (MB) | Peak RAM (MB) | Latency (ms) | Acc Tier 1 | Acc Tier 2 | Acc Tier 3 | Acc Tier 4 | Efficiency Score |
|-------|--------|------|-----------|---------------|---------------|------------|------------|------------|------------|-----------------|
| Conv1D | | | | | | | | | | |
| LeNet1D | | | | | | | | | | |
| VGG1D | | | | | | | | | | |
| ResNet1D | | | | | | | | | | |
| InceptionTime1D | | | | | | | | | | |
| MobileNet1D | | | | | | | | | | |
| EfficientNet1D | | | | | | | | | | |
| DenseNet1D | | | | | | | | | | |
| *Random baseline* | — | — | — | — | — | *33.3%* | *33.3%* | *33.3%* | *33.3%* | — |

- [x] **Verify discrimination:** After filling the table, check that max accuracy - min accuracy > 5%. If not, difficulty engineering (Tier 2/3/4) should provide the necessary spread.
- [x] **Pareto analysis:** Generate 3 plots: (1) Accuracy vs. MACs, (2) Accuracy vs. Latency, (3) Accuracy vs. Peak RAM. Pareto-optimal models highlighted. Expect MobileNet1D and EfficientNet1D to dominate the efficiency frontier. **v1.0 styling upgrade:** use YOLO-style visual conventions — family-colored labeled points, clean Pareto frontier line, optional param-count annotations, consistent color mapping across all Pareto plots.
- [ ] **Scaling frontier analysis (v1.1):** Once multiple size variants are available per architecture family, connect them into family curves and compare: (1) slope — which family gains the most accuracy per additional MAC? (2) saturation — where does added compute stop helping? (3) dominance — does one family's curve sit above all others? This turns Phase 4 from "which point is best?" into "which family scales best?" and supports the stronger claim: "Across increasing compute budgets, lightweight families maintain a better accuracy-efficiency slope than heavier families, suggesting they are better candidates for FPGA-oriented deployment."
- [x] **Error analysis:** On the hardest tier (Tier 4 = combined), examine: which class degrades most across architectures (likely 2um due to low-amplitude signals)? Do all models fail on the same signals? Compute inter-model agreement on errors.

---

## Phase 5 — Robustness & Validation

- [x] **Seed stability:** 3 seeds per model (42, 123, 7). Report mean +/- std for all metrics. 3 seeds is the minimum; expand to 5 if budget allows.
- [x] **Ranking stability:** For each seed, rank models by Tier 1 accuracy. Compute Kendall's tau between rankings across seeds. If tau < 0.8, flag the benchmark as having weak discrimination at the current dataset size.
- [ ] **Cross-hardware check (optional but recommended):** Run inference benchmarks on a different CPU (e.g., lab server) to verify that MACs-based rankings align with latency-based rankings. Latency can flip between Intel and ARM — relevant for FPGA portability analysis. *Defer to v1.1.*
- [x] **Ablation on difficulty tiers:** Verify Tier 1 > Tier 2 > Tier 3 > Tier 4 for at least 6 out of 8 models. If a tier breaks monotonicity for most models, revise that tier's parameters.
- [x] **Overfitting to benchmark check:** The test set is small (302 samples). Risk: a model could overfit by chance. Mitigations: (1) bootstrap confidence intervals, (2) multi-seed averaging, (3) the test set is never used for any hyperparameter or architecture decision — only the validation split is used during training.

---

## Phase 6 — Packaging & Release

### 6.1 — Code
- [x] **Clean repository:** Structure:
  - `models/` — model zoo (already exists)
  - `datasets/processed/p0-baseline-3class/v1/` — primary real dataset
  - `benchmark_zoo.py` — the evaluation harness (to be created)
  - `artifacts/SMI_CNN_limitations/benchmarks/benchmark2/` — JSON/CSV results, figures
  - `docs/` — reports and datasheets
- [x] **README:** Update `README.md` with Benchmark 2 quickstart: `.venv/bin/python SMI_CNN_limitations/scripts/benchmarks/benchmark_zoo.py --all --tier 1` to reproduce all results. Document how to add a new model to the zoo.
- [x] **Requirements file:** Already exists (`requirements.txt`) with pinned versions. Add `thop` if not present.
- [ ] **CI/tests (optional):** A smoke test that runs `benchmark_zoo.py --model Conv1D --epochs 1 --tier 1` on a 10-sample toy dataset and verifies JSON output format. *Nice to have for v1.1.*

### 6.2 — Paper / Report
- [x] **Benchmark paper:** Extend `docs/temporary_report.md` with a "Benchmark 2: Model Zoo Comparison" section covering: motivation (FPGA deployment), design, results, Pareto analysis, key findings.
- [x] **Figures:** (full design rationale in [`docs/variant_plotting.md`](variant_plotting.md))
  1. Accuracy vs. MACs Pareto front (`pareto.pdf`) — single column, colorblind-safe, numbered badges + boxed key listing every front member
  2. Accuracy vs. Latency Pareto front (`pareto_latency.pdf`) — same layout as above with log-latency on the x-axis (replaces the old `scaling_latency.pdf`, which became unreadable in the variant zoo because of kernel-launch clumping)
  3. Per-family scaling envelope (`scaling_macs.pdf`) — accuracy vs MACs with cumulative-max envelope per family + dominated variants as faint hollow markers
  4. Family small-multiples grid (`scaling_grid.pdf`) — 2×4 panels, shared axes, one family per panel
  5. Tier accuracy breakdown / heatmap (`tier_heatmap.pdf`, `tier_grid.pdf`, `tier_robustness.pdf`)
  6. Confusion matrices for best and worst model on Tier 1
  7. Per-class F1 heatmap (models x classes)
  8. Seed stability boxplots per model
- [x] **Limitations section:**
  - Small real test set (~301 samples) limits statistical power. Bootstrap CIs partially mitigate.
  - All models have ~5.3M params — the zoo does not yet include very small (<100K) or very large (>50M) models. Discrimination may improve with wider parameter range. This also means v1.0 Pareto plots show individual architectures, not scaling behavior within families (addressed in v1.1 with S/M/L variants).
  - CPU latency is a proxy for FPGA latency, not a direct measurement. Actual FPGA synthesis results would be the definitive metric.
  - No transformer baseline yet — the architecture the research aims to deploy on FPGA is missing from v1.
  - Fixed hyperparameters mean some models may be undertrained (e.g., SGD might suit ResNet better than Adam). This is a deliberate trade-off for fairness.
  - v1.0 cannot answer "which family scales best?" — only "which single model is best at ~5M params." The v1.1 scaling curves with size variants will address this gap.

### 6.3 — Maintenance
- [x] **Leaderboard:** A markdown table in `artifacts/SMI_CNN_limitations/benchmarks/benchmark2/leaderboard.md` with the latest results. Updated whenever a new model is added to the zoo.
- [x] **Versioning policy:**
  - v1.0: 8 CNN models, datasets/processed/p0-baseline-3class/v1, 4 tiers, CPU inference. Pareto plots with family-colored labeled points (YOLO-style visual conventions).
  - v1.1: Add S/M/L size variants for 2–4 core families (ResNet1D, MobileNet1D, EfficientNet1D, Conv1D) to enable true architecture scaling curves. Add transformer baseline. Cross-hardware latency checks. Quantized inference (INT8).
  - v2.0: New harder real dataset (more particle sizes or lower SNR acquisition), FPGA synthesis metrics.
- [x] **Deprecation plan:** If all models score >98% on Tier 1 after v1.0, escalate to Tier 2/3 as the primary benchmark. If all tiers are saturated, acquire a harder real dataset (v2.0) or introduce a new evaluation axis (e.g., online learning, continual adaptation).

---

## Quick Decision Log

Use this section to record key design decisions as you go:

| # | Decision | Chosen Option | Rationale | Date |
|---|----------|---------------|-----------|------|
| 1 | Primary evaluation target | datasets/processed/p0-baseline-3class/v1 (3-class real data) | Benchmark 1 covered synthetic; Benchmark 2 must reflect deployment reality | 2026-04-03 |
| 2 | Inference hardware for latency | CPU-only (Intel Core Ultra 7 265) | FPGA target has no GPU; CPU is the best available proxy | 2026-04-03 |
| 3 | Hyperparameter strategy | Fixed across all models (no per-model tuning) | Ensures benchmark measures architecture, not tuning effort | 2026-04-03 |
| 4 | Number of difficulty tiers | 4 (Standard, Data-starved, Noisy, Combined) | Covers the key failure modes: overfitting, noise robustness, and their combination | 2026-04-03 |
| 5 | Precision for v1 | FP32 only | Simplicity for v1; INT8/FP16 deferred to v1.1 for FPGA relevance | 2026-04-03 |
| 6 | Training included in benchmark | Yes — train from scratch | FPGA deployment requires domain-specific training, not transfer learning | 2026-04-03 |
| 7 | Efficiency score formula | accuracy / log10(MACs) | Rewards exponential MAC reduction; simple and interpretable | 2026-04-03 |
| 8 | Test set confidence intervals | Bootstrap (1000 resamples) | Mitigates small test set (302 samples) statistical limitations | 2026-04-03 |
| 9 | Architecture scaling curves | v1.0: YOLO-style Pareto styling; v1.1: S/M/L variants + connected curves | Single-scale models can't show scaling behavior; need width variants to answer "which family scales best?" — stronger paper argument | 2026-04-03 |
| 10 | Primary dataset | datasets/processed/p0-baseline-3class/v1 (instead of S7_pure_real) | Unified source for all 4 tiers; all difficulty derived from same dataset | 2026-04-03 |
| 11 | Tier 4 redesign | Combined (data-starved + noisy) instead of cross-domain | All tiers now derive from a single dataset; no external dependencies | 2026-04-03 |
