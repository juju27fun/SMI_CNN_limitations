# Noise Analysis Tool — `analyze_noise.py`

Standalone script that produces a comprehensive PDF noise report from any folder of `.npy` signal files.

---

## Quick Start

```bash
# Analyze raw noise recordings
python analyze_noise.py ./Noise

# Analyze real acquired signals
python analyze_noise.py dataset/test/10um

# Analyze synthetic signals
python analyze_noise.py v_colored/2um

# Custom output path
python analyze_noise.py ./Noise --output my_report.pdf
```

**Output:** a multi-page PDF named `<folder_name>_Noise_Analysis.pdf` in the current directory.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `folder` (positional) | — | Path to folder containing `.npy` files |
| `--output FILE` | `<folder_name>_Noise_Analysis.pdf` | Override output PDF path |
| `--fs HZ` | `2000000` | Sampling frequency in Hz |
| `--segment N` | `2500` | Segment length for long-file analysis (matches the OFI acquisition window) |

---

## Report Pages

The PDF contains 6 pages (7 if files are longer than `--segment`). Each page is a self-contained 2x2 grid of plots and text summaries.

### Page 1 — File Statistics

Overview of the dataset and aggregate noise amplitude.

| Panel | Content |
|---|---|
| Top-left | Text summary: file count, signal length, dtype, aggregate stats (mean/std/CV of per-file σ, mean kurtosis, DC offset spread, amplitude range) |
| Top-right | Histogram of per-file standard deviations (σ) with mean line |
| Bottom-left | Histogram of per-file means (DC offsets) |
| Bottom-right | Box plots comparing σ, RMS, and |DC offset| distributions |

**What to look for:**
- **CV(σ)** close to 0.19 indicates noise variability consistent with real hardware
- **DC offset spread** near 0 means signals are well-centered
- A wide σ histogram suggests mixed signal conditions or variable noise

### Page 2 — Time-Domain Analysis

Visual inspection of waveforms and temporal properties.

| Panel | Content |
|---|---|
| Top-left | Overlay of 10 random signals |
| Top-right | Single signal detail view |
| Bottom-left | Crest factor (peak/RMS) distribution with √2 Gaussian reference line |
| Bottom-right | Stationarity test: histogram of σ(first half) / σ(second half) |

**What to look for:**
- **Crest factor ≈ √2 (1.41)**: pure Gaussian noise. Higher values indicate bursts or transients
- **Stationarity ratio ≈ 1.0**: noise is stationary. Spread away from 1.0 means the noise level changes within a recording

### Page 3 — Power Spectral Density (PSD)

Frequency-domain characterization using the Welch method.

| Panel | Content |
|---|---|
| Top-left | PSD on log-log scale (mean + min/max envelope across all files) |
| Top-right | PSD with linear frequency axis, log power. Green shading marks the system bandpass [7–80 kHz] |
| Bottom-left | Spectral slope fit: scatter of log₁₀(PSD) vs log₁₀(f) with linear regression line |
| Bottom-right | Text summary: slope β, R², spectral flatness, classification |

**Key metric — Spectral slope β:**

The slope is estimated via linear regression on the log-log PSD within the [1–80 kHz] range:

```
log₁₀(PSD) = β × log₁₀(f) + c
```

| β value | Noise type | Meaning |
|---|---|---|
| ≈ 0 | White | Flat spectrum, equal power at all frequencies |
| ≈ −1 | Pink (1/f) | Equal energy per octave, common in electronics |
| ≈ −2 | Brown (1/f²) | Random walk, strong low-frequency dominance |

**Spectral flatness** is the ratio of the geometric mean to the arithmetic mean of the PSD. A value of 1.0 means perfectly white; lower values indicate colored noise.

### Page 4 — Frequency Band Energy

How noise power distributes across frequency ranges relevant to the OFI system.

| Panel | Content |
|---|---|
| Top-left | Bar chart: average % energy per band |
| Top-right | Box plots: per-file energy distribution per band |
| Bottom-left | Histogram of in-band [7–80 kHz] energy ratio per file |
| Bottom-right | Summary table with mean and std per band |

**Frequency bands:**

| Band | Relevance |
|---|---|
| 0–1 kHz | Below system range, DC neighborhood |
| 1–7 kHz | Below bandpass lowcut (7 kHz) |
| 7–10 kHz | Lower edge of system bandpass |
| 10–40 kHz | Core Doppler frequency range for most particle speeds |
| 40–80 kHz | Upper bandpass region |
| 80 kHz–Nyquist | Above system bandpass |

**What to look for:**
- Raw noise (`./Noise`): ~37% in 1–7 kHz, ~27% in 10–40 kHz, ~51% in-band
- Filtered signals should concentrate energy within [7–80 kHz]
- High energy below 7 kHz in signal folders may indicate insufficient filtering or wide-burst particles

### Page 5 — Noise Type Classification

Automated classification combining spectral and statistical tests.

| Panel | Content |
|---|---|
| Top-left | Classification verdict: spectral color (White/Pink/Brown) + amplitude distribution (Gaussian/Near-Gaussian/Non-Gaussian) |
| Top-right | Per-file kurtosis distribution with Gaussian reference (0) |
| Bottom-left | Average autocorrelation for the first 100 lags with 95% confidence bands |
| Bottom-right | Amplitude histogram (all samples pooled) with Gaussian fit overlay |

**Classification rules:**

Spectral color (from slope β):

| Criterion | Label |
|---|---|
| \|β\| < 0.3 | White |
| 0.3 ≤ \|β\| < 1.5 | Pink (1/f) |
| \|β\| ≥ 1.5 | Brown (1/f²) |

Amplitude distribution (from excess kurtosis):

| Criterion | Label |
|---|---|
| \|kurt\| < 0.5 | Gaussian |
| 0.5 ≤ \|kurt\| < 2.0 | Near-Gaussian |
| \|kurt\| ≥ 2.0 | Non-Gaussian |

**Autocorrelation interpretation:**
- **Drops to ~0 after lag 0**: white noise (no temporal correlation)
- **Decays slowly** (e.g., still at 0.8 at lag 100): colored noise with strong temporal structure
- **Oscillates**: periodic component present (e.g., residual Doppler signal)

### Page 6 — Amplitude Variability (Inter-file)

Characterizes how the noise level varies from one file to another.

| Panel | Content |
|---|---|
| Top-left | Box plot of per-file σ values |
| Top-right | Histogram of per-file σ with fitted lognormal PDF overlay |
| Bottom-left | Q-Q plot: observed σ quantiles vs theoretical lognormal quantiles |
| Bottom-right | Summary: CV, lognormal fit parameters (σ_ln, scale), comparison to reference CV = 0.19 |

**Why lognormal?** Noise amplitude is positive and right-skewed. The lognormal distribution captures this naturally. The Coefficient of Variation (CV = std(σ)/mean(σ)) is the key metric:

| CV value | Interpretation |
|---|---|
| < 0.05 | Fixed noise level (synthetic or controlled conditions) |
| 0.05–0.15 | Moderate variability |
| 0.15–0.25 | Consistent with real hardware (reference: CV = 0.19 from 305 noise files) |
| > 0.25 | High variability — mixed conditions, signal content, or environmental drift |

The **Q-Q plot** visually confirms whether the lognormal model fits. Points on the diagonal = good fit. Deviation at the tails = the lognormal model is approximate.

### Page 7 — Segment-Level Analysis (conditional)

Only generated when files are longer than `--segment` samples (e.g., the 16384-sample noise files when `--segment 2500`).

| Panel | Content |
|---|---|
| Top-left | Per-segment σ evolution within 10 random files (shows how noise level changes over time within a single recording) |
| Top-right | Histogram of within-file CV values |
| Bottom-left | Bar chart comparing between-file CV vs mean within-file CV |
| Bottom-right | Summary and stationarity interpretation |

**Key comparison — Between-file CV vs Within-file CV:**
- **Between > Within**: noise level is stable within recordings but varies across acquisitions (environmental drift)
- **Within > Between**: noise is non-stationary within recordings (transient events, equipment warm-up)
- **Comparable**: both sources of variability are similar

---

## Dependencies

```
numpy
scipy
matplotlib
```

All available in the project's `venv`.

---

## How It Works Internally

The script follows a compute-then-render architecture:

```
load_signals()                  → Load all .npy files from folder
        ↓
compute_file_stats()            → Per-file: mean, std, min, max, RMS, kurtosis
compute_psd()                   → Welch PSD: average, min, max across files
compute_spectral_slope()        → Linear fit on log-log PSD → β, R²
compute_spectral_flatness()     → Geometric/arithmetic mean ratio
compute_band_energy()           → % energy per frequency band
compute_band_energy_per_file()  → Same, but per-file for box plots
compute_autocorrelation()       → FFT-based normalized autocorrelation
compute_stationarity()          → Half-split σ ratio per file
compute_amplitude_variability() → CV + lognormal fit of per-file σ
compute_segment_stats()         → Per-segment σ within long files
classify_noise_type()           → White/Pink/Brown + Gaussian decision
        ↓
render_page_*()                 → One function per PDF page
```

All compute functions are stateless and can be imported independently for use in other scripts.

---

## Typical Results

### Raw noise files (`./Noise`)

| Metric | Expected value |
|---|---|
| Spectral color | Pink (1/f), β ≈ −0.93 |
| Amplitude dist. | Near-Gaussian (kurtosis ≈ −0.73) |
| mean(σ) | ≈ 0.215 |
| CV(σ) | ≈ 0.19 |
| In-band energy | ≈ 51% |
| Spectral flatness | ≈ 0.04 (far from white) |
| Autocorrelation | Slow decay (0.8 at lag 100) |

### Synthetic signals (e.g., `v_colored/2um`)

| Metric | Expected value |
|---|---|
| Spectral color | Varies with preset |
| mean(σ) | Lower than raw noise (signal is bandpass-filtered) |
| CV(σ) | Low (< 0.05) if noise_variability = 0 |
| In-band energy | High (>80%) for filtered signals |
| Kurtosis | Near 0 if Gaussian noise, negative if signal-dominated |
