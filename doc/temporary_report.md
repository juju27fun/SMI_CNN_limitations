# Benchmark Strategy Report — OFI Particle Classification

## 1. The Problem

An Optical Feedback Interferometry (OFI) sensor detects micro-particles (2, 4, 10 um) crossing a laser beam. Each particle produces a 1D signal: a cosine at the Doppler frequency, modulated by a Gaussian envelope. A Conv1D neural network classifies these signals by particle size.

The core challenge is **sim-to-real transfer**. Acquiring large labelled datasets of real particles at controlled sizes is expensive and slow. The practical approach is to train on synthetic signals generated from the OFI physical model and deploy on real sensor data. But synthetic signals are idealized — real signals contain hardware noise, DC drift, envelope distortions, and multi-particle events that the simulator does not fully capture. This creates a **generalization gap**: a model that reaches 98% accuracy on synthetic data may drop to 55% on real data.

The benchmark is designed to systematically measure, understand, and reduce this gap.

---

## 2. Three Pillars of the Evaluation Protocol

The benchmark rests on three complementary evaluation axes:

### Pillar 1 — Dual test sets (generalization gap measurement)

Every trained model is evaluated on **two** test sets:

| Test set | Source | What it measures |
|----------|--------|-----------------|
| **Synthetic test** (self) | Same distribution as training data | Nominal performance — can the model learn the task? |
| **Real test** (`dataset/test`) | 302 signals from the physical OFI sensor | Generalization — does performance survive the sim-to-real domain shift? |

The **generalization gap** = synthetic accuracy - real accuracy. This single number quantifies how much of the model's "knowledge" is distribution-specific rather than genuinely class-discriminative.

By keeping `dataset/test` as the **fixed** real reference across all experiments, every model is compared on the same ground truth. A model trained on S0 (clean signals, no noise) might score 99% on its own test set but 55% on real data (gap = 44%). A model trained on S3 (realistic distortions + noise) might score 97% synthetic but 57% real (gap = 40%). The dual-test protocol makes these comparisons meaningful.

### Pillar 2 — Controlled multi-dataset training (factorial design)

Rather than training on a single dataset and hoping it generalizes, we train the **same model architecture** on **12 different datasets**, each with a specific combination of signal realism and noise type:

```
Signal axis:  pure (ideal cosine + Gaussian envelope)
              realistic (+ DC offset, multiburst, envelope skew)

Noise axis:   none → white → colored → realistic → real
              (increasing fidelity to hardware noise)
```

This produces a **signal x noise grid**:

| | none | white | colored | realistic | real |
|---|---|---|---|---|---|
| **pure** | S0 | S1 | S2 (+ S8, S9) | S6 | S7 |
| **realistic** | S5 | — | — | S3 | S4 |

Plus **S_union** (all S0-S9 combined) and **dataset** (real data).

The datasets are organized in tiers:

- **Tier 1 (Core)**: S0 → S1 → S2 → S3 → S4 form a gradient of increasing difficulty. Each step adds one layer of realism. Comparing their performance reveals which factors degrade classification most.

- **Tier 2 (Ablation)**: S5 (realistic signal, no noise) and S6 (pure signal, realistic noise) isolate signal vs. noise effects. If accuracy(S0) - accuracy(S5) measures the signal-only impact, and accuracy(S0) - accuracy(S6) measures the noise-only impact, we can test whether these effects are additive, synergistic, or compensatory. S7 (pure signal, real noise) completes the noise axis.

- **Tier 2b (SNR sweep)**: S8 (sigma=0.03), S2 (sigma=0.058), S9 (sigma=0.1) trace a 3-point curve of accuracy vs. noise level, revealing the model's SNR sensitivity and the critical noise threshold.

- **Tier 3 (Validation)**: `dataset` is the real-world ground truth. S_union combines all synthetic datasets via domain randomization — the hypothesis being that training on diverse conditions forces the model to learn invariant features that transfer better to reality.

### Pillar 3 — OOD noise evaluation (deployment safety)

A classifier deployed on a real sensor will inevitably encounter **noise-only** windows — acquisitions where no particle crossed the beam. A well-behaved model should recognize these as "none of the above" rather than confidently assigning them to a particle class. This is the **Out-of-Distribution (OOD) detection** problem.

The benchmark evaluates OOD detection using 305 real noise recordings from the sensor (the `Noise/` directory). Five complementary methods are applied:

| Method | Principle | What it reveals |
|--------|-----------|-----------------|
| **MSP** | Max softmax probability | Baseline: does the model output lower confidence on noise? |
| **Energy** | -logsumexp(logits) | Theoretically grounded alternative to softmax confidence |
| **ODIN** | Temperature scaling + input perturbation | Can gradient-based perturbation amplify the ID/OOD gap? |
| **Mahalanobis** | Distance to class centroids in feature space (4 layers) | Does noise land far from all class clusters in the learned representation? |
| **Energy_tuned** | Energy with optimal temperature from sweep | Does temperature tuning improve separation? |

For each method, AUROC, FPR@95%TPR, and AUPR are computed. An AUROC near 1.0 means the model can reliably separate particle signals from noise; an AUROC near 0.5 means it cannot — the model would classify noise as particles with high confidence, making it unreliable in deployment.

The OOD evaluation also includes:
- **Silhouette score** in the 256-dim latent space (fc1) — measures cluster separation between ID signals and noise
- **Per-class analysis** — which particle classes are hardest to distinguish from noise (typically 2 um, whose low-amplitude signals resemble noise)
- **Prediction distribution on noise** — reveals which class the model defaults to when presented with pure noise

---

## 3. How the Three Pillars Connect

The three evaluation axes answer different but complementary questions:

```
                    "Can it learn?"          "Does it generalize?"         "Is it safe?"
                         │                          │                          │
                    Synthetic test              Real test                 OOD evaluation
                    (Pillar 1a)               (Pillar 1b)                (Pillar 3)
                         │                          │                          │
                    Self-accuracy            Generalization gap         AUROC / FPR@95
                         │                          │                          │
                         └──────────┬───────────────┘                          │
                                    │                                          │
                              Measured across                           Measured across
                              12 datasets                               12 datasets
                              (Pillar 2)                                (Pillar 2)
```

**Together, they answer**: for a given model architecture, which training conditions produce the best trade-off between classification accuracy on real data and the ability to reject noise?

A model trained on S0 (no noise) may have perfect self-accuracy but poor OOD detection — it has never seen noise, so noise activates learned features the same way signals do. A model trained on S4 (realistic signals + real noise) may have lower self-accuracy but excellent OOD detection — it has learned to distinguish signal structure from noise patterns.

---

## 4. Key Hypotheses Under Test

The 12-dataset design enables testing specific, falsifiable hypotheses:

| # | Hypothesis | How it is tested |
|---|-----------|-----------------|
| H1 | Difficulty increases with realism: acc(S0) > acc(S1) > acc(S2) > acc(S3) >= acc(S4) | Compare self-accuracy across Tier 1 |
| H2 | Noise is the dominant degradation factor (more than signal distortions) | Compare delta(S5) vs delta(S6) from S0 baseline |
| H3 | Signal and noise effects are approximately additive | Check if delta(S3) approx delta(S5) + delta(S6) |
| H4 | The realistic noise model is a good proxy for real noise (gap < 3%) | Compare S6 vs S7, and S3 vs S4 |
| H5 | Accuracy degrades non-linearly with noise level (has a "knee") | SNR curve from S8, S2, S9 |
| H6 | The 2 um class degrades first as noise increases | Per-class F1 across S8, S2, S9 |
| H7 | S_union (domain randomization) transfers better than any individual S_i | Compare generalization gaps |
| H8 | Transfer is asymmetric: hard-to-easy works better than easy-to-hard | Cross-testing matrix |

---

## 5. The Benchmark Pipeline

Each of the 12 benchmark runs executes the same 5-phase pipeline:

| Phase | What happens |
|-------|-------------|
| **1. Pre-training** | Log model config (2.2M params Conv1D), dataset info, hyperparameters to W&B |
| **2. Training** | 150 epochs, Adam optimizer, cosine LR scheduler, early stopping. Per-epoch metrics: train/val loss and accuracy |
| **3. Post-training evaluation** | Load best model. Evaluate on synthetic test set AND real test set. Log confusion matrices, F1 per class, PR/ROC curves. Compute generalization gap |
| **4. Dimensionality reduction** | PCA + t-SNE of fc1 features for synthetic test, real test, and noise. Visualizes cluster structure and ID/noise separation |
| **5. OOD noise evaluation** | Run all 5 OOD methods. Log AUROC, FPR@95, AUPR, score histograms, ROC comparison, temperature sweep, per-class analysis, silhouette score |

All metrics are logged to the W&B project `particle-benchmark`, enabling side-by-side comparison across all 12 runs.

---

## 6. Data Integrity

Before any training, each dataset undergoes leak detection (`dataset_leaks.py`) to verify:
- No source-level leaks (same recording in both train and test)
- No exact content duplicates across splits
- No near-duplicate contamination
- Class balance is maintained

This ensures that reported accuracies reflect genuine learning, not data leakage artifacts.

---

## 7. Summary

The benchmark does not just train a model and report an accuracy number. It systematically answers:

1. **What makes the task hard?** — by isolating signal and noise factors across a controlled dataset grid
2. **How well does it generalize?** — by measuring every model against the same fixed real test set
3. **Is it safe to deploy?** — by testing whether the model can distinguish real particle signals from pure noise

The combination of dual test sets, 12 controlled training conditions, and 5 OOD detection methods creates a comprehensive evaluation framework for OFI particle classification that goes beyond standard train/test accuracy to assess robustness, generalization, and deployment readiness.
