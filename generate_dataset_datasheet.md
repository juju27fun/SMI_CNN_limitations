# Generate Dataset — Datasheet

Technical documentation for `generate_dataset.py`: justification, formulas, realism measures, and known limitations for each generation option.

---

## 1. Core Signal Model (`simulated_particle`)

### Why we need it

The OFI (Optical Feedback Interferometry) sensor detects particles crossing a laser beam. Each particle produces a characteristic signal: a cosine oscillation at the Doppler frequency, modulated by a Gaussian envelope corresponding to the beam's spatial profile. Simulating this signal is necessary because acquiring large labelled datasets of real particles at controlled sizes (2, 4, 10 µm) is expensive and time-consuming.

### How it is generated

The signal model follows the standard OFI self-mixing formulation:

```
P(t) = Po × [1 + m0 × cos(2π × f_D × t)] × exp(−(t − T_impact)² / (2τ²))
```

Where:

| Symbol | Formula / Value | Physical meaning |
|---|---|---|
| `f_D` | `2 × V × sin(90° − θ) / λ` | Doppler frequency — encodes particle velocity |
| `τ` | `(D + S_l) / (V × sin(θ))` | Burst duration — time the particle spends in the beam |
| `m0` | drawn from `U[m0_min, m0_max]` per class | Modulation index — scales with particle cross-section |
| `Po` | `0.004134 × 4 = 0.0166 mV` | Baseline laser power at the photodetector |
| `θ` | 80° | Angle between beam axis and particle trajectory |
| `λ` | 1550 nm | Laser wavelength (telecom standard) |
| `S_l` | 7 µm | Laser beam spot diameter |
| `D` | 2, 4, or 10 µm | Particle diameter |

**Theoretical background.** In OFI, the laser cavity receives a fraction of its own light back-scattered by the particle. This causes a modulation of the emitted power at the Doppler frequency. The Gaussian envelope arises because the beam has a near-Gaussian intensity profile (TEM₀₀ mode), so the back-scattered power follows the spatial distribution of the beam as the particle transits through it.

### What is done to match realistic signals

- **Doppler frequency range.** The speed range `[0.05, 0.20] m/s` is chosen so that `f_D` falls within `[11.2, 44.8] kHz`, inside the bandpass filter window `[7, 80] kHz`. This matches the velocity range observed in the experimental micro-fluidic setup.
- **Modulation index per class.** The `m0` ranges are calibrated per particle size to reproduce the amplitude differences observed in real data. Larger particles produce stronger back-scattering, hence higher `m0`:

| Class | m0 range | Measured amplitude ratio vs 2 µm |
|---|---|---|
| 2 µm | [7, 14] | 1.0× (reference) |
| 4 µm | [18, 36] | ~1.6× |
| 10 µm | [20, 95] | ~4.2× |

### Current limitations

1. **Burst width too narrow.** The formula `τ = (D + S_l) / (V × sin(θ))` produces burst widths of ~340–770 samples. Real signals show burst widths of ~900–2200 samples. The discrepancy suggests that the effective interaction zone in the real setup is larger than `D + S_l`, possibly due to beam divergence, particle deceleration, or multiple scattering. Real 10 µm signals frequently span the entire 2500-sample window.
2. **Single-frequency model.** The model assumes a pure cosine at `f_D`. Real signals can exhibit frequency chirp (particle acceleration/deceleration), harmonic content, and speckle effects that broaden the spectrum.
3. **No amplitude modulation due to focusing.** Real OFI signals can show amplitude variations as the particle moves along the beam axis (defocusing), which is not modelled.
4. **10 µm spectral mismatch.** Real 10 µm data has ~70% of its energy below the 7 kHz lowcut. The synthetic model places the dominant frequency at ~25 kHz. This suggests that real 10 µm particles may be slower than the simulated range, or that the effective Doppler geometry differs.

---

## 2. Bandpass Filter

### Why we need it

The real acquisition chain includes an analog bandpass filter to reject DC components and high-frequency noise above the Nyquist-relevant range. Applying the same filter to synthetic signals ensures that the CNN trains on signals with the same spectral characteristics it will encounter at inference.

### How it is generated

A 4th-order Butterworth bandpass filter is applied using `scipy.signal.butter` + `filtfilt` (zero-phase, forward-backward filtering):

```
Passband: [7 000, 80 000] Hz
Order:    4 (4th-order Butterworth → −24 dB/octave rolloff)
Method:   filtfilt (zero-phase, equivalent to 8th-order magnitude response)
Fs:       2 MHz
```

The Butterworth design is chosen for its maximally flat magnitude response in the passband.

### What is done to match realistic signals

- The cutoff frequencies (7 kHz / 80 kHz) are set to match the hardware filter in the Bokeh viewer acquisition software used in the real setup.
- Zero-phase filtering (`filtfilt`) avoids phase distortion that would alter the burst envelope shape.

### Current limitations

1. **Analog vs digital mismatch.** The real system uses an analog filter; the simulation uses a digital IIR filter. At low orders (4), the difference is negligible, but edge effects near the passband boundaries may differ.
2. **Double filtering.** The training pipeline applies an additional FFT-based bandpass at load time (`train.py` uses `[5, 100] kHz` or `[8, 40] kHz`). The interaction between the generation-time Butterworth filter and the inference-time FFT filter is not explicitly characterized.

---

## 3. Noise Injection (`--noise`)

### Why we need it

Real signals always contain noise from the photodetector, amplification chain, and quantization. Training a CNN on clean signals produces a model that fails on real noisy inputs. Noise injection is the primary tool for bridging the sim-to-real gap.

### 3.1 `--noise none`

**Purpose.** Baseline for ablation studies. Isolates the signal model from noise effects to validate that the CNN can learn class-discriminative features from the burst structure alone.

**Generation.** No noise added. The signal goes through the pipeline unchanged.

**Limitations.** Not representative of real acquisition conditions. A model trained on noise-free data will likely fail on real signals.

### 3.2 `--noise white`

**Purpose.** Simplest noise model. Tests robustness to additive noise with flat power spectral density.

**Generation.**
```
noise(t) ~ N(0, σ²)     where σ = 0.058
```
Each sample is independently drawn from a zero-mean Gaussian. The noise is added **after** the bandpass filter.

**Calibration.** The value `σ = 0.058` is the measured standard deviation of noise in the quiet regions (signal edges) of real acquired signals.

**Limitations.** Real photodetector noise is not white — it has a colored spectrum with more power at low frequencies (1/f noise, amplifier noise). White noise overpopulates high-frequency bins and underpopulates low-frequency bins relative to reality.

### 3.3 `--noise colored`

**Purpose.** More realistic noise model that approximates the measured power spectral density (PSD) of real noise.

**Generation.** Two independent white noise signals are bandpass-filtered and mixed:

```
low_band  = BPF(white₁, [1 kHz, 10 kHz], order=4)
high_band = BPF(white₂, [10 kHz, 80 kHz], order=4)
colored   = √0.7 × low_band + √0.3 × high_band
colored   = colored × (σ / std(colored))          # normalize to target σ
```

The 70/30 power split approximates the measured PSD of real noise files:

| Band | Measured (real noise) | Synthetic colored |
|---|---|---|
| 1–10 kHz | ~35% | ~70% (weighted) |
| 10–80 kHz | ~37% | ~30% (weighted) |

**What is done to match realistic signals.** The dual-band approach captures the first-order shape of the real noise PSD: more power at low frequencies, less at high frequencies.

**Limitations.**
1. The 70/30 split is a coarse approximation. The real PSD has a smooth rolloff, not a two-step function.
2. The bandpass boundaries (1 kHz and 10 kHz) are fixed and not calibrated against the measured PSD breakpoints.
3. After the signal's own bandpass filter at `[7, 80] kHz`, the noise component below 7 kHz is present in the final signal (since noise is injected after filtering), which does not match the real system where noise goes through the same analog filter as the signal.

### 3.4 `--noise realistic`

**Purpose.** Most physically faithful synthetic noise model. Simulates noise that enters the system before the analog filter and has variable amplitude across acquisitions.

**Generation.**
```
σ_sample = σ_base × lognormal(0, CV)
noise    = N(0, σ_sample²)
# injected BEFORE the bandpass filter
signal   = BPF(raw_signal + noise)
```

Parameters:
- `σ_base = 0.21` — higher than other presets because the bandpass filter will attenuate most of the noise power
- `CV = 0.19` (coefficient of variation) — the lognormal multiplier ensures that each sample sees a different noise amplitude
- `noise_injection = before` — noise is added before filtering, so it gets shaped by the same bandpass as the signal

**Calibration.**
- `σ_base = 0.21`: measured as the mean standard deviation of the 305 real noise files (measured value: 0.216, rounded).
- `CV = 0.19`: the coefficient of variation of standard deviations across the 305 real noise files is 0.19 (std of stds = 0.041, mean of stds = 0.216).

**What is done to match realistic signals.** This preset reproduces two key properties of real noise: (1) the noise passes through the same bandpass filter as the signal, acquiring the same spectral shape, and (2) the noise amplitude varies from sample to sample following the measured inter-file distribution.

**Limitations.**
1. The base noise is white, not colored. Even after bandpass filtering, the in-band spectrum is flat, whereas real noise has a non-flat in-band spectrum.
2. The lognormal variability model is a first-order approximation. Real noise amplitude may depend on environmental factors (temperature, laser power drift) that have non-lognormal distributions.

### 3.5 `--noise real`

**Purpose.** Highest-fidelity noise option. Uses actual recorded noise segments from the real sensor.

**Generation.**
```
1. Load 305 real noise files from ./Noise/ (each 16384 samples, float64)
2. For each synthetic signal:
   a. Pick a random noise file
   b. Pick a random offset within the file
   c. Extract a 2500-sample segment
   d. Zero-mean and scale to target σ = 0.058
   e. Add to the signal (after bandpass by default)
```

**What is done to match realistic signals.** The noise retains the exact spectral shape, temporal correlations, and statistical properties of real hardware noise — including 1/f components, amplifier resonances, and quantization artifacts that cannot be captured by parametric models.

**Limitations.**
1. **Scaling distortion.** The extracted segment is rescaled to a fixed `σ = 0.058`, which can alter the relative power of spectral components. A segment with unusually high low-frequency content will be attenuated overall, changing its character.
2. **Finite diversity.** Only 305 noise files are available. With large datasets (1511+ signals), noise segments are reused, potentially creating correlated noise patterns across training samples.
3. **Segment boundary effects.** Random cropping can cut across non-stationary events in the noise (e.g., transient interference), though zero-mean subtraction mitigates DC jumps.

---

## 4. Signal Realism Options (`--signal`)

### 4.1 DC Offset (`dc_offset_std`)

**Why we need it.** Real signals exhibit a non-zero DC level due to amplifier bias drift, temperature effects, and imperfect AC coupling. A CNN trained only on zero-mean signals may use the mean as a spurious feature. Introducing random DC offsets forces the model to learn from the oscillatory structure instead.

**How it is generated.**
```
dc_offset ~ N(0, σ_dc²)     where σ_dc = 0.15 mV (realistic preset)
signal    = signal + dc_offset
```
Applied as the last step in the pipeline (after filtering and noise), since the real DC offset is a property of the analog readout chain, not the physical signal.

**What is done to match realistic signals.** The value `σ_dc = 0.15` is estimated from the distribution of per-signal means observed in the real dataset. Real data shows `std(means) ≈ 0.04–0.11` for 2 µm and 4 µm, up to ~1.0 for 10 µm. The value 0.15 is a compromise across classes.

**Current limitations.**
1. **Per-class variation not captured.** The 10 µm class has much larger DC spread (`std ≈ 0.99`) than 2 µm (`std ≈ 0.11`), likely because wide 10 µm bursts shift the signal mean. A single `σ_dc` value cannot capture this class-dependent behavior.
2. **Constant offset model.** Real DC drift can be time-varying (slow ramp), not constant within a window.

### 4.2 Multiburst (`multiburst_pct`)

**Why we need it.** In real flows, two particles occasionally cross the beam in rapid succession within the same acquisition window. The resulting signal has two overlapping bursts, which can confuse a classifier that expects a single burst per window. Training with multiburst samples improves robustness.

**How it is generated.**
```
For each sample, with probability multiburst_pct/100 (default: 10%):
  1. Generate a second particle with independent speed, m0, and particle size (same class)
  2. Place it at a time offset ≥ 0.15 × window from the first burst
  3. If the first burst is in the first half of the window, the second goes to the right half, and vice versa
  4. Sum both signals: P_t = P_t1 + P_t2
```

**Variability management.** The second burst has fully independent random parameters (speed, m0), ensuring it is not a mere copy of the first. The 15%-of-window minimum separation prevents complete overlap (which would just increase amplitude rather than create a distinct double-burst pattern).

**What is done to match realistic signals.** The 10% rate is estimated from visual inspection of real acquisition windows. In a typical micro-fluidic flow, multi-particle events are rare but not negligible.

**Current limitations.**
1. **Same-class constraint.** Both bursts are from the same particle class. In reality, two particles of different sizes could cross simultaneously, but modelling this would require multi-label classification.
2. **No constructive/destructive interference.** The two signals are simply added. Real overlapping particles in a coherent beam could produce interference patterns.
3. **Fixed probability.** The 10% rate is a rough estimate. The actual rate depends on particle concentration, which varies between experiments.

### 4.3 Envelope Skew (`envelope_skew_min/max`)

**Why we need it.** The idealized OFI model produces a symmetric Gaussian envelope. Real signals often show asymmetric envelopes because: the particle accelerates or decelerates as it crosses the beam (inertial effects in the flow), the beam intensity profile is not perfectly Gaussian, and the particle's scattering cross-section changes with angle.

**How it is generated.** The Gaussian envelope is modulated by a skew-normal factor using the error function:

```
z           = (t − T_impact) / τ
skew_factor = 1 + erf(α × z / √2)
P_t         = P_t × skew_factor
```

Where `α` (the skew parameter) is drawn uniformly from `[−0.5, +0.5]` in the `realistic` preset.

**Theoretical background.** The skew-normal distribution is obtained by multiplying a Gaussian density by `Φ(α × z)`, where `Φ` is the standard normal CDF. Using `1 + erf(...)` provides a smooth transition from ~0 (suppressed) on one side to ~2 (amplified) on the other, creating the asymmetry. For `α = 0`, the factor is identically 1 (no skew).

**What is done to match realistic signals.** The range `[−0.5, +0.5]` introduces mild asymmetry. Negative values create a faster rise / slower decay; positive values create the opposite. This diversity helps the CNN generalize to the variety of envelope shapes observed in real data.

**Current limitations.**
1. **First-order approximation.** Real envelope distortions can be more complex than a single skew parameter (e.g., bimodal shapes from beam aberrations).
2. **Independent of physics.** The skew is drawn randomly and not linked to the particle speed or size, whereas in reality the asymmetry has a physical cause tied to flow dynamics.

---

## 5. Noise Injection Point (`noise_injection`)

### Why we need it

The point where noise enters the signal chain affects its spectral characteristics after filtering. This option lets the user choose which effect to model.

### `after` (default for most presets)

Noise is added **after** the bandpass filter. The noise retains its original spectrum (white or colored), regardless of the filter. This is simpler but means the noise can have energy outside the signal's passband.

### `before` (used by `--noise realistic`)

Noise is added **before** the bandpass filter. The filter then shapes the noise spectrum identically to the signal. This more accurately models a system where noise is present at the sensor (photodetector shot noise, Johnson noise) and goes through the same analog filter as the signal.

**Trade-off.** `before` injection requires a higher `σ_base` (0.21 vs 0.058) because the bandpass filter attenuates most of the noise energy. The ratio `0.21 / 0.058 ≈ 3.6` reflects the fact that only ~1/3.6² ≈ 7.7% of the white noise power falls within the `[7, 80] kHz` passband at `f_s = 2 MHz`.

### Current limitations

In reality, there are multiple noise sources: some before the filter (sensor noise), some after (digitization noise, amplifier noise downstream of the filter). A single injection point cannot capture this layered noise structure.

---

## 6. Noise Amplitude Variability (`noise_variability`)

### Why we need it

In real experiments, the noise level varies from one acquisition to another due to environmental changes (temperature, vibration, laser power fluctuations). A model trained with fixed noise amplitude may overfit to a specific SNR.

### How it is generated

For each sample, the base `σ` is multiplied by a random factor drawn from a log-normal distribution:

```
σ_effective = σ_base × exp(N(0, CV²))
```

Where `CV = 0.19` is the coefficient of variation.

**Why log-normal.** Noise amplitude is a positive quantity. The log-normal distribution ensures `σ_effective > 0` and produces a right-skewed distribution that matches the observed pattern: most files cluster near the mean, with occasional high-noise outliers.

**Calibration.** From the 305 real noise files: `mean(std) = 0.216`, `std(std) = 0.041`, giving `CV = 0.041 / 0.216 = 0.19`.

### Current limitations

The variability is independent across samples. In reality, noise level may be correlated within a batch of acquisitions (same environmental conditions), creating structured variation that the i.i.d. model does not capture.

---

## 7. Randomized Parameters

### 7.1 Particle Speed (`p_speed`)

**Range.** `U[0.05, 0.20] m/s`

**Constraint.** The lower bound ensures `f_D = 2V sin(10°) / λ ≥ 11.2 kHz`, safely above the 7 kHz lowcut of the bandpass filter. At `V = 0.05 m/s`: `f_D = 2 × 0.05 × sin(10°) / 1550e-9 ≈ 11.2 kHz`.

**Limitation.** The uniform distribution does not match the actual velocity distribution in the micro-fluidic channel, which may follow a parabolic (Poiseuille) profile.

### 7.2 Impact Time (`t_impact`)

**Range.** `U[0.4, 0.6] × window` where `window = 2500 / 2 MHz = 1.25 ms`

**Purpose.** Centers the burst roughly in the middle of the window while adding positional variability. The CNN should learn to detect bursts regardless of their exact position.

**Limitation.** Real signals show burst centers distributed across a wider range (`std ≈ 186–425 samples` in real data vs `~64–111` in synthetic). The `[0.4, 0.6]` range is conservative.

### 7.3 Modulation Index (`m0`)

**Purpose.** The modulation index is the primary feature that differentiates particle sizes. Larger particles have larger scattering cross-sections, producing stronger modulation.

**Ranges.** Set per class to reproduce the measured amplitude hierarchy:

| Class | m0 range | Resulting peak amplitude (clean) |
|---|---|---|
| 2 µm | [7, 14] | ~0.17–0.23 mV |
| 4 µm | [18, 36] | ~0.32–0.54 mV |
| 10 µm | [20, 95] | ~0.57–1.18 mV |

**Limitation.** The ranges overlap (especially 4 µm and 10 µm at their lower/upper bounds), creating ambiguous samples. This is realistic (real data has similar overlap) but means that perfect classification accuracy is physically impossible from amplitude alone.

---

## 8. Signal Presets (`--signal`)

### 8.1 `--signal pure`

| Parameter | Value | Effect |
|---|---|---|
| `dc_offset_std` | 0 | No DC offset |
| `multiburst_pct` | 0 | No double-burst signals |
| `envelope_skew` | [0, 0] | Perfectly symmetric Gaussian envelope |

**Purpose.** Clean, theory-perfect signals for baseline experiments. Useful for validating that the CNN architecture can separate classes using the Doppler frequency and modulation depth alone.

### 8.2 `--signal realistic`

| Parameter | Value | Effect |
|---|---|---|
| `dc_offset_std` | 0.15 | Random DC offset per sample |
| `multiburst_pct` | 10% | 10% of signals have a second burst |
| `envelope_skew` | [−0.5, +0.5] | Mild envelope asymmetry |

**Purpose.** Introduces imperfections observed in real signals. Combines three independent realism factors that each address a different failure mode of a model trained on pure signals.

---

## 9. Generation Modes

### 9.1 `auto` mode

Generates a full train/test split dataset with default parameters:

| Class | Train | Test | Total |
|---|---|---|---|
| 2 µm | 403 | 101 | 504 |
| 4 µm | 403 | 101 | 504 |
| 10 µm | 403 | 100 | 503 |
| **Total** | **1209** | **302** | **1511** |

The ~80/20 split follows standard machine learning practice. All classes are balanced in the training set (403 each).

### 9.2 `test` mode

Generates 3 samples per class in a flat directory structure (no train/test split). Designed for quick visual inspection of signal shapes before committing to a full generation run.

### 9.3 `manual` mode

Reads all parameters from a `.ini` configuration file, allowing full control. Useful for ablation studies or reproducing specific experimental conditions.

**Parameter resolution order:**
```
DEFAULT_SIM → --signal preset → --noise preset → .ini config file (highest priority)
```

---

## 10. Reproducibility (`seed`)

All random number generation uses `numpy.random.default_rng(seed)` with `seed = 42` by default. The legacy `np.random.seed(42)` is also set for backward compatibility. Given the same seed, parameters, and code version, the output is bit-for-bit identical.

**Limitation.** Reproducibility depends on NumPy/SciPy versions. A different version of `scipy.signal.filtfilt` could produce slightly different filter coefficients.

---

## 11. Existing Generated Datasets

The following datasets exist in the project directory, each generated with a specific combination of `--signal` and `--noise`:

| Directory | Signal | Noise | Injection | Files | Purpose |
|---|---|---|---|---|---|
| `dataset_test` | realistic | none | — | 9 | Quick preview |
| `v_none` | pure | none | — | 9 | Clean baseline |
| `v_pure` | pure | none | — | 9 | Same as v_none |
| `v_white` | realistic | white | after | 9 | White noise test |
| `v_colored` | realistic | colored | after | 9 | Colored noise test |
| `v_real_n` | realistic | real | after | 9 | Real noise test |
| `v_realistic` | realistic | realistic | before | 9 | Full realism test |
| `v_realistic_n` | realistic | realistic | before | 9 | Full realism (variant) |
| `v_pure_noise` | pure | colored | after | 9 | Pure signal + noise |
| `v_dc` | realistic | colored | after | 9 | DC offset emphasis |
| `v_mb` | realistic | colored | after | 9 | Multiburst emphasis |
| `v_custom_n` | realistic | colored | after | 9 | Custom noise params |

The `dataset/` directory contains **real experimental data** (1511 files), not synthetic signals. Its filenames (`HFocusing_5_10_*.npy`) reflect the original acquisition naming convention.

---

## 12. Summary of Known Limitations

| # | Limitation | Impact | Possible Improvement |
|---|---|---|---|
| 1 | Burst width 3–5× too narrow vs real data | CNN may learn incorrect envelope duration features | Introduce a beam-waist scaling factor or calibrate `S_l` against measured burst widths |
| 2 | 10 µm spectral mismatch (dominant frequency too high) | Class confusion if model relies on spectral features | Extend the speed range downward or add a speed distribution per class |
| 3 | Single-frequency Doppler model (no chirp) | Missing frequency-time structure present in real signals | Add linear or quadratic frequency modulation term |
| 4 | Colored noise PSD is a coarse 2-band approximation | Noise spectrum shape differs from real hardware noise | Fit a parametric PSD model to the measured noise or use `--noise real` |
| 5 | No beam-axis defocusing effect | Amplitude modulation pattern missing | Model the confocal parameter and Gouy phase |
| 6 | DC offset is class-independent | Underestimates DC spread for 10 µm (real σ ≈ 1.0 vs synthetic 0.15) | Make `dc_offset_std` a per-class parameter |
| 7 | Fixed uniform speed distribution | Does not reflect Poiseuille flow profile in micro-channel | Use a beta or truncated parabolic distribution |
| 8 | Narrow burst center range ([0.4, 0.6] × window) | Less positional diversity than real data (std 64–111 vs 186–425) | Widen to [0.2, 0.8] or calibrate from real data |
| 9 | Finite real noise file pool (305 files) | Noise reuse in large datasets | Acquire more noise recordings or augment with parametric noise |
| 10 | Noise rescaling alters spectral shape | Real noise segments lose their native amplitude relationships | Use raw noise without rescaling, or rescale per-band |
