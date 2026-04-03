# Benchmark Construction Checklist
## Particle Classification in 1D Signals

---

## Phase 0 — Scoping & Motivation

- [ ] **Why does this benchmark need to exist?** Write a 3–5 sentence justification. What gap does it fill? Who is the target audience (yourself, your lab, the community)?
In the context of research to put transformer models on fpga, I need to prove the limitation of CNN's in 1D SMI signal classification. For this I need to show precisely where they fail in terms of classification (software) and in terms of hardware as well (ram usage, size, latency etc).
- [ ] **What claim should the benchmark support?** e.g., "Model X is better than Model Y for particle classification under resource constraints."
- [ ] **What is the scope?** 1D signals only? Specific particle types? Specific sensor modalities?
- [ ] **What are the success criteria for the benchmark itself?** e.g., "At least 10 percentage points of spread between the weakest and strongest baseline model."
- [ ] **Literature review:** Are there existing benchmarks for similar tasks? What do they get right / wrong? What can you borrow?

---

## Phase 1 — Metric Definition

### 1.1 — Primary Metrics
- [ ] **Accuracy:** Top-1? Top-k? Per-class? Macro/micro average? How do you handle class imbalance?
- [ ] **Latency:** Wall-clock inference time per sample? Per batch? Median or p95? Include preprocessing or not?
- [ ] **MACs (Multiply-Accumulate Operations):** Which tool will you use to count them? (e.g., `ptflops`, `fvcore`, `thop`) Are you counting forward pass only?
- [ ] **Peak RAM:** How will you measure it? (`torch.cuda.max_memory_allocated` for GPU? `tracemalloc` for CPU?) At what batch size?
- [ ] **Model size:** Parameter count? On-disk size of saved weights? Both?

### 1.2 — Composite / Derived Metrics
- [ ] **Accuracy vs. latency curve:** How will you plot and compare? Pareto front?
- [ ] **Accuracy vs. MACs:** Same question.
- [ ] **Efficiency score:** Will you define a single composite score (e.g., accuracy per MegaMAC)? If so, document the formula.
- [ ] **Accuracy vs. difficulty tier:** If using multiple difficulty levels, how do you aggregate across tiers?

### 1.3 — Measurement Protocol
- [ ] **Hardware specification:** Exact GPU/CPU model, driver versions, CUDA version. All models must be measured on the same hardware.
- [ ] **Software specification:** Framework version (PyTorch X.Y.Z), Python version, OS.
- [ ] **Warmup:** How many warmup inference passes before timing? (Recommend ≥ 50.)
- [ ] **Repetitions:** How many timed runs per model? (Recommend ≥ 3 full runs with different seeds.)
- [ ] **Batch size policy:** Fixed batch size across all models? Or each model at its optimal batch size? Document the choice.
- [ ] **Precision:** FP32? FP16? Mixed? Is quantized inference in scope?

---

## Phase 2 — Dataset Design

### 2.1 — Existing Data Audit
- [ ] **Document your current dataset:** Number of samples, number of classes, class distribution, signal length, sampling rate, SNR characteristics.
- [ ] **Why is it saturated?** Is the task too easy (clean signals, well-separated classes)? Too small (models memorize)? Too few classes?
- [ ] **Can you reuse it as the "easy" tier?**

### 2.2 — Difficulty Engineering
- [ ] **Define difficulty axes:** What makes particle classification harder in your domain? (SNR, temporal overlap, background complexity, class similarity, sample scarcity, signal length variation…)
- [ ] **Design difficulty tiers:** Recommend 3–5 levels. For each tier, specify exact parameter ranges (e.g., Tier 1: SNR > 20 dB, Tier 2: 10–20 dB, Tier 3: 0–10 dB, Tier 4: < 0 dB).
- [ ] **Validate that difficulty tiers actually modulate difficulty:** Run a simple model across tiers, check that accuracy degrades monotonically.
- [ ] **Is the difficulty parametric and reproducible?** Someone else should be able to regenerate equivalent data from your parameters.

### 2.3 — Dataset Construction
- [ ] **Data source:** Real data? Simulated? Semi-synthetic (real signals + synthetic degradation)? Document the generation/collection process.
- [ ] **Dataset size per tier:** Enough samples for statistical significance. Rule of thumb: ≥ 1000 test samples per class, more if classes are imbalanced.
- [ ] **Class balance:** Is each class equally represented? If not, is the imbalance intentional and documented?
- [ ] **Train / Validation / Test split:**
  - [ ] Fixed splits (no random re-splitting allowed).
  - [ ] Test set is held out and never used for any design decision.
  - [ ] Validation set is used for hyperparameter tuning.
  - [ ] Split strategy: random? stratified? grouped by acquisition session to avoid leakage?
- [ ] **Data leakage check:** Can a model cheat by memorizing acquisition artifacts, temporal ordering, or session-specific noise patterns?
- [ ] **Signal preprocessing:** Is preprocessing part of the benchmark (models receive raw signals) or fixed (benchmark provides preprocessed inputs)? Document either way.
- [ ] **Data format:** File format (.npy, .h5, .csv, .pt)? Directory structure? Metadata file?

### 2.4 — Documentation
- [ ] **Datasheet for the dataset:** Following the "Datasheets for Datasets" template (Gebru et al., 2021). Covers motivation, composition, collection process, preprocessing, distribution, maintenance.
- [ ] **License:** Under what license is the dataset released?
- [ ] **Versioning:** How will you handle updates? Semantic versioning (v1.0, v1.1…)?

---

## Phase 3 — Evaluation Harness

### 3.1 — Architecture
- [ ] **Single entry point:** One script/command that takes a model and produces all metrics.
- [ ] **Model interface contract:** What must a submitted model expose? (e.g., a `forward(x) -> logits` method, input shape, etc.)
- [ ] **Reproducibility:** Fixed random seeds for data loading order. Deterministic operations where possible.

### 3.2 — Implementation
- [ ] **Accuracy computation:** Implement all accuracy variants (top-1, per-class, confusion matrix).
- [ ] **Latency measurement:** Use `torch.cuda.synchronize()` before timing on GPU. Use high-resolution timers (`time.perf_counter`).
- [ ] **MAC counting:** Integrate a profiling tool. Validate on a known architecture (e.g., check that your tool gives the expected MACs for a standard Conv1d layer).
- [ ] **Peak RAM measurement:** Profile GPU and/or CPU memory. Reset peak stats before each run.
- [ ] **Model size measurement:** Count parameters (`sum(p.numel() for p in model.parameters())`), save and measure file size.
- [ ] **Output format:** Structured results file (JSON/CSV) with all metrics, model name, timestamp, hardware info.

### 3.3 — Sanity Checks
- [ ] **Random baseline:** What accuracy does a random classifier get? Your harness should report this as a floor.
- [ ] **Majority baseline:** What accuracy does always-predict-majority-class get?
- [ ] **Known model check:** Run a model with known performance and verify the harness reproduces it.
- [ ] **Determinism check:** Run the same model twice, verify identical accuracy and near-identical latency.

---

## Phase 4 — Baseline Population

### 4.1 — Model Selection
- [ ] **Your homemade model** (the one hitting 94% on the current dataset).
- [ ] **Classic CNNs adapted to 1D:** LeNet-1D, small ResNet-1D, VGG-style 1D.
- [ ] **Lightweight models:** MobileNet-1D, SqueezeNet-1D, or any architecture targeting efficiency.
- [ ] **Transformer-based:** Small 1D transformer or hybrid CNN-Transformer.
- [ ] **Simple baselines:** Logistic regression on raw features, SVM, random forest on hand-crafted features.
- [ ] **At least 5 models** spanning a wide range of complexity.

### 4.2 — Fair Training Protocol
- [ ] **Hyperparameter tuning budget:** Same budget for all models (e.g., same number of trials on validation set).
- [ ] **Training configuration:** Document optimizer, learning rate schedule, augmentation, epochs, early stopping criterion.
- [ ] **Is training part of the benchmark, or do you only benchmark inference on pretrained models?** This is a critical design choice — document it.

### 4.3 — Results Table
- [ ] **Build the results table:** Model name | Params | MACs | Size (MB) | Peak RAM | Latency (ms) | Acc Tier 1 | Acc Tier 2 | … | Acc Overall
- [ ] **Verify discrimination:** Is there meaningful spread across models? If all models cluster within 1–2%, revisit Phase 2.
- [ ] **Pareto analysis:** Plot accuracy vs. each efficiency metric. Identify Pareto-optimal models.
- [ ] **Error analysis:** For the hardest tier, what types of signals do models fail on? Are failures correlated across models?

---

## Phase 5 — Robustness & Validation

- [ ] **Seed stability:** Run each baseline with 3–5 different random seeds. Report mean ± std for all metrics.
- [ ] **Ranking stability:** Does the ranking of models change across seeds? If it does for adjacent models, the benchmark may not discriminate them reliably — flag this.
- [ ] **Cross-hardware check (optional but recommended):** Do rankings hold if you switch GPU or use CPU-only? Latency rankings can flip across hardware.
- [ ] **Ablation on difficulty tiers:** Confirm that difficulty tiers produce monotonically decreasing accuracy for most models. If a tier breaks the pattern, investigate.
- [ ] **Overfitting to benchmark check:** If someone trains specifically to maximize your benchmark, can they game it without actually building a better model? (e.g., by memorizing test samples, exploiting data artifacts.)

---

## Phase 6 — Packaging & Release

### 6.1 — Code
- [ ] **Clean repository:** Separate directories for dataset, harness, baselines, results.
- [ ] **README:** Installation, quickstart, how to submit a model, how to reproduce baselines.
- [ ] **Requirements file:** Pinned dependencies (`requirements.txt` or `environment.yml`).
- [ ] **CI/tests (optional but impressive):** Automated tests that verify the harness runs on a toy dataset.

### 6.2 — Paper / Report
- [ ] **Benchmark paper:** Even if informal, write up the motivation, design choices, dataset description, baseline results, and known limitations.
- [ ] **Figures:** Accuracy vs. difficulty tier plot, Pareto front plots (accuracy vs. MACs, accuracy vs. latency), confusion matrices for best/worst models on hardest tier.
- [ ] **Limitations section:** What the benchmark doesn't measure. Caveats about generalization.

### 6.3 — Maintenance
- [ ] **Leaderboard (optional):** A simple table (even a Markdown file in the repo) where results can be added.
- [ ] **Versioning policy:** When do you release v2? (New difficulty tiers, new data, corrected bugs.)
- [ ] **Deprecation plan:** How do you communicate if the benchmark becomes saturated again?

---

## Quick Decision Log

Use this section to record key design decisions as you go:

| # | Decision | Chosen Option | Rationale | Date |
|---|----------|---------------|-----------|------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
