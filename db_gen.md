# OFI Dataset Generation

Script: `generate_dataset.py`
Generates simulated OFI (Optical Feedback Interferometry) signals for training a particle classification CNN (2 um, 4 um, 10 um).

## Prerequisites

```bash
pip install numpy scipy tqdm
```

The `Noise/` folder (305 real noise `.npy` files) is required for `--noise real`.

---

## Quick Examples

```bash
# Pure signal, no noise (default)
python generate_dataset.py test --output ./preview --force

# Realistic signal + colored noise
python generate_dataset.py auto --signal realistic --noise colored --force

# Full realism (signal + noise before filter + variable amplitude)
python generate_dataset.py auto --signal realistic --noise realistic --force

# Real noise injection
python generate_dataset.py auto --noise real --force

# Override a parameter via config
echo -e "[noise]\nnoise_sigma = 0.1" > custom.ini
python generate_dataset.py auto --config custom.ini --force
```

---

## The 3 Modes

| Mode | Description | Command |
|------|------------|---------|
| `auto` | Full dataset (1511 files, train/test split) | `... auto` |
| `test` | Quick preview (3 signals/class, flat folder) | `... test` |
| `manual` | Requires a `.ini` config file | `... manual --config params.ini` |

---

## Common Options

| Option | Default | Description |
|---|---|---|
| `--signal {pure,realistic}` | `realistic` | Signal preset |
| `--noise {none,white,colored,realistic,real}` | `none` | Noise preset |
| `--noise-dir DIR` | `./Noise` | Directory containing real noise files |
| `--config FILE` | — | `.ini` file to override default parameters |
| `--init-config FILE` | — | Generate a `.ini` template and exit |
| `--output DIR` | `./dataset` or `./dataset_test` | Output directory |
| `--force` | — | Overwrite existing directory |
| `--with-filter` | — | Enable generation-time bandpass filter |
| `--filter-lowcut HZ` | 7000 | Low cutoff frequency (implies `--with-filter`) |
| `--filter-highcut HZ` | 80000 | High cutoff frequency (implies `--with-filter`) |
| `--filter-order N` | 4 | Butterworth filter order (implies `--with-filter`) |

### Parameter Resolution Order

```
DEFAULT_SIM  ->  --signal preset  ->  --noise preset  ->  --config .ini file  ->  --filter-* CLI args (wins)
```

Note: when `noise_injection == "before"`, the bandpass filter is auto-enabled with a `[5, 100] kHz` passband. Explicit `--filter-*` arguments override the auto defaults.

---

## Signal Presets (`--signal`)

| Preset | dc_offset_std | multiburst_pct | envelope_skew |
|--------|--------------|----------------|---------------|
| `pure` | 0 | 0 | [0, 0] |
| `realistic` (default) | 0.15 | 10% | [-0.5, 0.5] |

### Signal Parameter Details (`[signal]` in the `.ini`)

| Parameter | Description |
|---|---|
| `dc_offset_std` | Standard deviation of random DC offset (0 = disabled) |
| `multiburst_pct` | % of signals with a second burst (0 = disabled) |
| `envelope_skew_min` | Lower bound of envelope asymmetry |
| `envelope_skew_max` | Upper bound (0/0 = symmetric Gaussian) |

---

## Noise Presets (`--noise`)

| Preset | noise_type | noise_injection | noise_sigma | noise_variability |
|--------|-----------|-----------------|-------------|-------------------|
| `none` (default) | — | — | 0 | 0 |
| `white` | white Gaussian | after filter | 0.058 | 0 |
| `colored` | dual-band colored (1-80 kHz) | after filter | 0.058 | 0 |
| `realistic` | white | **before filter** | 0.21 | 19% |
| `real` | real noise files | after filter | 0.058 | 0 |

### Noise Parameter Details (`[noise]` in the `.ini`)

| Parameter | Values | Description |
|---|---|---|
| `noise_type` | `none`, `white`, `colored`, `real` | Noise type |
| `noise_injection` | `before`, `after` | Injection point vs bandpass |
| `noise_sigma` | float | Noise standard deviation |
| `noise_variability` | float (CV) | Inter-sample amplitude variability (0 = fixed) |

### `noise_injection = before`

Noise is added BEFORE the bandpass filter — it is filtered naturally as in the real system. The bandpass filter is auto-enabled with a `[5, 100] kHz` passband. Sigma must be higher (~0.21) because the filter removes out-of-band power.

### `noise_variability`

For each sample, sigma is scaled by `lognormal(0, noise_variability)`. The value 0.19 reproduces the 19% coefficient of variation measured on real noise files.

---

## Generation Pipeline

```
1. simulated_particle()       -> raw signal
2. [multiburst]               -> optional second burst
3. [envelope skew]            -> envelope asymmetry
4. DC subtraction             -> remove DC component
5. [noise if before]          -> noise BEFORE bandpass
6. bandpass filter             -> 7-80 kHz bandpass (if enabled)
7. [noise if after]           -> noise AFTER bandpass
8. [DC offset]                -> random offset
```

- Steps 2, 3, 8 controlled by `--signal`
- Steps 5, 7 controlled by `--noise`
- Step 6 disabled by default; auto-enabled when `noise_injection=before`
- Signal and noise are **independent**

---

## Default Parameters

### Physics Simulation

| Parameter | Value | Description |
|---|---|---|
| `laser_lambda` | 1550e-9 m | Wavelength |
| `adq_freq` | 2 MHz | Acquisition frequency |
| `inc_angle` | 80 deg | Incidence angle |
| `po` | 0.016536 mV | Laser power |
| `time_max` | 2500 | Samples per signal |
| `s_l` | 7e-6 m | Laser spot diameter |
| `p_speed` | [0.05, 0.20] m/s | Particle speed |
| `t_impact` | [40%, 60%] of window | Burst position |

### Filter

| Parameter | Value |
|---|---|
| `filter_lowcut` | 7 000 Hz |
| `filter_highcut` | 80 000 Hz |
| `filter_order` | 4 (Butterworth) |

### Modulation Index per Class (m0)

| Class | m0 min | m0 max | Amplitude ratio |
|---|---|---|---|
| 2 um | 7.0 | 14.0 | 1.00x |
| 4 um | 18.0 | 36.0 | ~1.86x |
| 10 um | 20.0 | 95.0 | ~4.33x |

---

## `.ini` Configuration File

Available sections:

| Section | Content |
|---|---|
| `[simulation]` | Physics parameters |
| `[randomization]` | Speed and T_impact ranges |
| `[postprocessing]` | Bandpass filter |
| `[noise]` | Type, injection, sigma, variability |
| `[signal]` | DC offset, multiburst, envelope skew |
| `[class_NAME]` | Particle size, train/test counts, m0 min/max |

Generate a complete template:
```bash
python generate_dataset.py auto --init-config params.ini
```

---

## Graphical Interface (Streamlit)

Visual alternative to the CLI. All parameters are visible and editable before generation.

```bash
streamlit run generate_ui.py
```

Features:
- Dropdowns for signal and noise presets (auto-fill fields)
- All parameters editable in real-time
- Preview: generates 3 samples/class and displays plots + statistics
- Equivalent CLI command shown at the bottom
- Collapsible advanced sections (physics simulation, filter, classes)
- Full generation button with progress bar

---

## Generated File Format

- `.npy` (NumPy binary), shape `(2500,)`, dtype `float64`
- Naming: `sample_0000.npy`, `sample_0001.npy`, ...
