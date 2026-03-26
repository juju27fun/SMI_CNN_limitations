# OOD Detection in the Benchmark Pipeline — Full Data Flow

## Overview

The benchmark (`benchmark.py`) implements **5 OOD detection methods** to distinguish in-distribution (ID) particle signals from noise. The pipeline runs in **Phase 5** after the model is trained and evaluated on test sets.

**Methods**: MSP, Energy, ODIN, Mahalanobis (multi-layer), Energy_tuned (temperature-optimized)

---

## 1. Data Preparation

### In-Distribution (ID) data
The **synthetic test set** serves as the ID reference. Same preprocessing as training:
```
Raw .npy signals → BandpassFilter(5–100 kHz) → Decimate(4x) → (1, 250) tensor
```
Loaded via `test_loader` (line 900).

### Noise (OOD) data
Noise samples from `--noise-dir` (default `Noise/`). Extra step: **truncation to 2500 samples** before bandpass + decimate, because noise files may be longer than particle signals:
```
Raw .npy noise → Truncate(2500) → BandpassFilter(5–100 kHz) → Decimate(4x) → (1, 250) tensor
```
Loaded as a `ParticleDataset` with a single dummy class (lines 919–932). The class label is unused — only the signal tensor matters.

### Training data (for Mahalanobis)
The `train_loader` is passed to the OOD evaluation so Mahalanobis can fit its class-conditional Gaussians on training features rather than test features (line 1091).

---

## 2. Entry Point: `run_ood_evaluation()` (line 357)

Called at line 1089 as:
```python
run_ood_evaluation(run, model, test_loader, noise_loader, device, class_names,
                   train_loader=train_loader)
```

This function orchestrates all 5 methods and logs everything to W&B.

---

## 3. Method-by-Method Data Flow

### 3.1 MSP (Max Softmax Probability)

**Principle**: A well-calibrated model should output lower softmax confidence on OOD samples.

**Score function** (`compute_ood_scores`, line 249):
```
Input signal → model.forward() → logits
                                → softmax(logits) → probas
Score = max(probas)     # per sample
```

**Where computed**: Lines 374–376
```python
msp_id    = np.max(id_probas, axis=1)    # shape: (n_id,)
msp_noise = np.max(noise_probas, axis=1)  # shape: (n_noise,)
```

**Interpretation**: Higher score = more likely in-distribution.

---

### 3.2 Energy Score

**Principle**: The energy function `-logsumexp(logits)` provides a theoretically grounded scoring function that doesn't suffer from the overconfidence problem of softmax (Liu et al., 2020).

**Score function** (lines 378–380):
```
Input signal → model.forward() → logits
Energy = -logsumexp(logits)     # scalar per sample
```

```python
energy_id    = -logsumexp(id_logits, axis=1)
energy_noise = -logsumexp(noise_logits, axis=1)
```

**For AUROC computation**: scores are negated (`neg_energy = -energy_scores`) so that higher = more ID (line 417). This is because lower energy (more negative) = higher density = more likely ID, so we negate to align the polarity with the AUROC convention (1 = ID, 0 = OOD).

---

### 3.3 ODIN (Out-of-Distribution detector for Neural Networks)

**Principle**: Combines **temperature scaling** of logits with **input perturbation** in the gradient direction to widen the gap between ID and OOD softmax scores (Liang et al., 2018).

**Score function** (`compute_odin_scores`, line 267):

```
Step 1: Forward pass WITH gradients enabled
  Input x → model(x) → logits → logits/T → log_softmax → max → loss = sum(max_log_softmax)

Step 2: Compute gradient of loss w.r.t. input
  loss.backward() → gradient = d(loss)/d(x)

Step 3: Perturb input in confidence-increasing direction
  x_perturbed = x - ε * sign(gradient)

Step 4: Second forward pass (no gradients)
  x_perturbed → model(x_perturbed) → logits_p → softmax(logits_p / T) → max
  Score = max(softmax(logits_p / T))
```

**Hyperparameters**: `temperature=1000.0`, `epsilon=0.0012` (line 267)

**Key detail**: The perturbation direction is `- ε * sign(gradient)` (line 289). The gradient of the max log-softmax w.r.t. the input tells us which direction to shift the input to increase the model's confidence. For ID samples, this perturbation significantly boosts confidence; for OOD samples, the boost is smaller — widening the separation.

**Where distances are computed**: There are no explicit "distances" in ODIN. The discrimination comes from the **temperature-scaled softmax confidence** after input perturbation. The key computation is:
- Line 279: `scaled = outputs / temperature` (temperature scaling)
- Line 289: `perturbed = signals.data - epsilon * gradient.sign()` (perturbation)
- Line 294: `scores = softmax(outputs_p / temperature).max(dim=1)` (final score)

---

### 3.4 Mahalanobis Distance (Multi-Layer)

**Principle**: Fit a class-conditional Gaussian at each layer of the network using training data, then measure the Mahalanobis distance of a test sample to the nearest class centroid. OOD samples should be far from all class centroids in the learned feature space (Lee et al., 2018).

This is the most complex method. Here's the full flow:

#### Step 1: Multi-layer feature extraction (`extract_multilayer_features`, line 753)

PyTorch forward hooks are registered on **4 layers**: `pool1`, `pool2`, `pool3`, `fc1` (lines 774–777).

```
Input (1, 250)
  → conv1 → bn1 → relu → pool1 → [HOOK: capture output, shape (N, 64, L1)]  → GAP → (N, 64)
  → conv2 → bn2 → relu → pool2 → [HOOK: capture output, shape (N, 128, L2)] → GAP → (N, 128)
  → conv3 → bn3 → relu → pool3 → [HOOK: capture output, shape (N, 256, L3)] → GAP → (N, 256)
  → flatten → fc1            → [HOOK: capture output, shape (N, 256)]        → (N, 256)
```

**Global Average Pooling (GAP)**: Conv layer outputs are 3D tensors `(N, C, L)`. They are averaged along the temporal axis to get `(N, C)` (line 793: `feats = feats.mean(dim=2)`). This converts variable-length feature maps into fixed-size vectors.

This extraction is done **3 times**:
- `ref_layers, ref_labels` ← from `train_loader` (for fitting the Gaussians)
- `id_layers` ← from `id_loader` (test set, in-distribution)
- `noise_layers` ← from `noise_loader` (OOD samples)

#### Step 2: Class-conditional Gaussian fitting (per layer)

For each of the 4 layers (`compute_mahalanobis_scores`, line 302):

```python
# Class-conditional means (line 323-325)
for each class c:
    class_means[c] = mean(ref_feats[labels == c])   # shape: (D,) where D ∈ {64, 128, 256, 256}

# Tied covariance matrix (lines 329-331)
centered = ref_feats - class_means[ref_labels]       # center each sample around its class mean
cov = np.cov(centered, rowvar=False)                 # shape: (D, D)
cov_inv = np.linalg.pinv(cov)                        # pseudo-inverse for numerical stability
```

#### Step 3: Mahalanobis distance computation (lines 334–343)

For each sample `x` in both ID and noise sets:
```
For each class c:
    diff = x - class_means[c]                        # (D,)
    maha_c = -diff @ cov_inv @ diff^T                # negative squared Mahalanobis distance (scalar)

score = max over all classes (maha_c)                # nearest class centroid
```

The score is **negative Mahalanobis distance** (line 338): `maha = -sum(diff @ cov_inv * diff, axis=1)`, meaning higher score = closer to a class centroid = more ID.

#### Step 4: Multi-layer ensemble (lines 317–343)

The scores from all 4 layers are **summed** (line 341: `id_total += scores`, `noise_total += scores`):
```
final_score = maha_pool1 + maha_pool2 + maha_pool3 + maha_fc1
```

This multi-layer ensemble is what gives Mahalanobis its power — features at different depths capture different aspects of the distribution.

**Where distances are explicitly computed**: Line 338
```python
maha = -np.sum(diff @ cov_inv * diff, axis=1)  # -d² Mahalanobis
```
This is the squared Mahalanobis distance `d² = (x - μ)ᵀ Σ⁻¹ (x - μ)`, negated so that higher = more ID.

---

### 3.5 Energy_tuned (Temperature-Optimized Energy)

**Principle**: Same as Energy but with an optimal temperature found by grid search.

**Temperature sweep** (lines 518–565):
```
For T in [1, 2, 5, 10, 50, 100, 500, 1000]:
    energy_T = -T * logsumexp(logits / T)
    Compute AUROC with this T
Pick T with best AUROC
```

If the optimal T differs from 1, the Energy score is recomputed with this temperature (lines 568–601) and added as a 5th method `Energy_tuned`.

---

## 4. Metric Computation (Common to All Methods)

For each method, scores are concatenated and evaluated against binary labels (lines 399–447):

```python
ood_labels = [1, 1, ..., 1, 0, 0, ..., 0]   # 1 = ID, 0 = noise
scores     = [id_scores..., noise_scores...]
```

Three metrics are computed:
- **AUROC** (`roc_auc_score`): Area under the ROC curve. 1.0 = perfect separation.
- **FPR@95** (`_compute_fpr_at_tpr`, line 348): False Positive Rate when True Positive Rate = 95%. Lower = better.
- **AUPR** (`average_precision_score`): Area under the Precision-Recall curve.

---

## 5. Additional Analyses

### Silhouette Score (lines 717–725)
Extracts `fc1` features for ID and noise, computes silhouette score on the combined set with binary labels (ID=0, noise=1). Measures how well-separated the two groups are in the 256-dim latent space.

### Per-Class OOD Analysis (lines 674–701)
For each particle class (2um, 4um, 10um), computes AUROC of that class alone vs all noise samples using MSP. This reveals which classes are most/least separable from noise.

### Entropy (lines 383–385)
```python
entropy = -sum(p * log(p))   # per sample
```
Avg entropy on ID vs noise is logged. Higher entropy on noise = model is less certain = desirable.

### Prediction Distribution on Noise (lines 454–455)
Shows which classes the model assigns to noise samples. If noise is truly OOD, predictions should be random or biased toward a single class (not meaningful).

---

## 6. Visual Summary of the Complete Data Flow

```
                    ┌─────────────────────────────────────────────────┐
                    │               TRAINED MODEL                     │
                    │  conv1→pool1→conv2→pool2→conv3→pool3→fc1→fc2   │
                    └────────────────────┬────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
        test_loader               noise_loader               train_loader
        (ID samples)             (OOD samples)            (for Mahalanobis)
              │                          │                          │
              ▼                          ▼                          │
     ┌────────────────┐        ┌────────────────┐                   │
     │ Forward Pass   │        │ Forward Pass   │                   │
     │ → logits       │        │ → logits       │                   │
     │ → softmax      │        │ → softmax      │                   │
     └───────┬────────┘        └───────┬────────┘                   │
             │                         │                            │
     ┌───────┴─────────────────────────┴──────────┐                 │
     │                                            │                 │
     │  MSP:    max(softmax)                      │                 │
     │  Energy: -logsumexp(logits)                │                 │
     │                                            │                 │
     └────────────────────────────────────────────┘                 │
                                                                    │
     ┌────────────────────────────────────────────┐                 │
     │  ODIN (requires gradients):                │                 │
     │  1. Forward: logits/T → log_softmax → max  │                 │
     │  2. Backward: gradient w.r.t. input        │                 │
     │  3. Perturb: x - ε·sign(grad)             │                 │
     │  4. Forward again: softmax(model(x')/T)    │                 │
     └────────────────────────────────────────────┘                 │
                                                                    │
     ┌────────────────────────────────────────────────────────────────┐
     │  Mahalanobis (multi-layer):                                   │
     │                                                                │
     │  1. Hook pool1, pool2, pool3, fc1 on train/test/noise         │
     │  2. GAP on conv outputs → fixed vectors per layer             │
     │  3. Per layer:                                                 │
     │     a. Fit class means μ_c from train features                │
     │     b. Fit tied covariance Σ from train features              │
     │     c. For each test/noise sample:                            │
     │        score = max_c( -(x-μ_c)ᵀ Σ⁻¹ (x-μ_c) )              │
     │  4. Sum scores across all 4 layers                            │
     └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Binary classification:        │
              │  label=1 (ID) vs label=0 (OOD)│
              │                               │
              │  → AUROC                      │
              │  → FPR@95%TPR                 │
              │  → AUPR                       │
              │  → Histograms                 │
              │  → ROC curves                 │
              │  → Temperature sweep          │
              └───────────────────────────────┘
```

---

## 7. Key File Locations

| Component | File | Lines |
|-----------|------|-------|
| OOD entry point | `benchmark.py` | 1084–1092 |
| `compute_ood_scores` (MSP/Energy) | `benchmark.py` | 249–264 |
| `compute_odin_scores` | `benchmark.py` | 267–299 |
| `compute_mahalanobis_scores` | `benchmark.py` | 302–345 |
| `extract_multilayer_features` | `benchmark.py` | 753–796 |
| `extract_features` (fc1 only) | `benchmark.py` | 731–750 |
| `_compute_fpr_at_tpr` | `benchmark.py` | 348–354 |
| `run_ood_evaluation` (orchestrator) | `benchmark.py` | 357–726 |
| Noise data loading | `benchmark.py` | 918–932 |
| Temperature sweep | `benchmark.py` | 517–565 |

---

## References

[1] D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," in *Proceedings of the 5th International Conference on Learning Representations (ICLR)*, 2017. arXiv: [1610.02136](https://arxiv.org/abs/1610.02136)

[2] S. Liang, Y. Li, and R. Srikant, "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks," in *Proceedings of the 6th International Conference on Learning Representations (ICLR)*, 2018. arXiv: [1706.02690](https://arxiv.org/abs/1706.02690)

[3] K. Lee, K. Lee, H. Lee, and J. Shin, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," in *Advances in Neural Information Processing Systems 31 (NeurIPS)*, pp. 7167–7177, 2018. arXiv: [1807.03888](https://arxiv.org/abs/1807.03888)

[4] W. Liu, X. Wang, J. Owens, and Y. Li, "Energy-based Out-of-distribution Detection," in *Advances in Neural Information Processing Systems 33 (NeurIPS)*, pp. 21464–21475, 2020. arXiv: [2010.03759](https://arxiv.org/abs/2010.03759)
