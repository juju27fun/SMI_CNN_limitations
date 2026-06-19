"""Streamlit UI for generate_dataset.py — OFI particle signal generator."""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from generate_dataset import (
    DEFAULT_CLASSES,
    DEFAULT_SIM,
    NOISE_PRESETS,
    SIGNAL_PRESETS,
    generate_sample,
    load_noise_files,
    run_generation,
)

st.set_page_config(page_title="OFI Dataset Generator", layout="wide")
st.title("OFI Dataset Generator")

# ── Sidebar: infrastructure only ──────────────────────────────────────────

st.sidebar.header("Output")
mode = st.sidebar.selectbox(
    "Mode", ["auto", "test"],
    help="**auto**: full dataset with train/test split (1511 files). "
         "**test**: 3 samples per class for quick visual inspection.",
)
output_dir = st.sidebar.text_input(
    "Output directory",
    value="./dataset" if mode == "auto" else "./dataset_test",
)
force = st.sidebar.checkbox("Overwrite if exists", value=True)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
noise_dir = st.sidebar.text_input(
    "Real noise directory",
    value="./Noise",
    help="Path to the folder containing real noise `.npy` files. "
         "Only used when noise type is set to `real`.",
)

# ── Main area: Signal + Noise side by side ────────────────────────────────

col_signal, col_noise = st.columns(2)

# ── Signal column ─────────────────────────────────────────────────────────

with col_signal:
    st.subheader("Signal")
    signal_preset = st.selectbox(
        "Preset", list(SIGNAL_PRESETS.keys()),
        index=list(SIGNAL_PRESETS.keys()).index("realistic"),
        help="**pure**: symmetric Gaussian envelope, single burst, zero DC offset. "
             "**realistic**: asymmetric envelope, random DC offset, 10% multi-burst.",
    )

    sim = dict(DEFAULT_SIM)
    sim.update(SIGNAL_PRESETS[signal_preset])
    sim["seed"] = int(seed)

    sim["dc_offset_std"] = st.number_input(
        "DC offset std (mV)", value=sim["dc_offset_std"], format="%.4f", step=0.01,
        help="Standard deviation of a random constant added to the entire signal. "
             "Simulates electronic drift and incomplete baseline removal. "
             "Set to 0 to disable. Calibrated value: **0.15** (from real data mean distribution).",
    )
    sim["multiburst_pct"] = st.slider(
        "Multi-burst (%)", min_value=0, max_value=100, value=int(sim["multiburst_pct"]),
        help="Percentage of signals containing **two particles** in the same acquisition window. "
             "The second burst has independent speed, m0, and position. "
             "In real experiments, multi-burst events occur when two particles cross the beam simultaneously.",
    )
    skew_col1, skew_col2 = st.columns(2)
    with skew_col1:
        sim["envelope_skew_min"] = st.number_input(
            "Envelope skew min", value=sim["envelope_skew_min"], format="%.2f", step=0.1,
            help="Lower bound of the skew-normal asymmetry parameter applied to the Gaussian envelope. "
                 "**Negative** = left tail extended (particle entering beam slowly). "
                 "**0** = perfectly symmetric Gaussian. "
                 "The skew factor is: `1 + erf(alpha * z / sqrt(2))` where `z = (t - t_impact) / tau`.",
        )
    with skew_col2:
        sim["envelope_skew_max"] = st.number_input(
            "Envelope skew max", value=sim["envelope_skew_max"], format="%.2f", step=0.1,
            help="Upper bound of the skew parameter. "
                 "**Positive** = right tail extended (particle exiting beam slowly). "
                 "Each sample draws a random skew uniformly from [min, max].",
        )

# ── Noise column ──────────────────────────────────────────────────────────

with col_noise:
    st.subheader("Noise")
    noise_preset = st.selectbox(
        "Preset", list(NOISE_PRESETS.keys()),
        index=list(NOISE_PRESETS.keys()).index("none"),
        help="**none**: no noise. "
             "**white**: Gaussian white noise (flat spectrum), injected after bandpass. "
             "**colored**: dual-band colored noise (70% in 1-10 kHz, 30% in 10-80 kHz), after bandpass. "
             "**realistic**: white noise injected *before* bandpass (filtered naturally), with 19% amplitude variability. "
             "**real**: segments from actual noise recordings (`Noise/` folder).",
        key="noise_preset",
    )
    sim.update(NOISE_PRESETS[noise_preset])

    if sim["noise_type"] != "none":
        st.caption({
            "white": "White Gaussian noise — flat spectrum across all frequencies.",
            "colored": "Colored noise — shaped PSD (70% in 1-10 kHz, 30% in 10-80 kHz).",
            "real": "Real noise — crops from actual noise recordings (Noise/ folder).",
        }.get(sim["noise_type"], ""))

    sim["noise_injection"] = st.selectbox(
        "Injection point", ["before", "after"],
        index=["before", "after"].index(sim["noise_injection"]),
        help="**before**: noise is added *before* the bandpass filter. "
             "The filter shapes the noise naturally (like in a real experiment where noise exists in the raw signal). "
             "White noise becomes band-limited; sigma must be higher (~0.21) because the filter removes ~70% of power. "
             "**after**: noise is added *after* filtering. The noise keeps its full spectral content.",
    )
    sim["noise_sigma"] = st.number_input(
        "Sigma (mV)", value=sim["noise_sigma"], format="%.4f", step=0.01,
        help="Standard deviation of the additive noise. "
             "For `injection=after`: this is the final baseline noise level (calibrated: **0.058**). "
             "For `injection=before`: use a higher value (~**0.21**) since the bandpass filter attenuates it.",
    )
    sim["noise_variability"] = st.number_input(
        "Amplitude variability (CV)", value=sim["noise_variability"], format="%.3f", step=0.01,
        help="Coefficient of variation of the noise amplitude across samples. "
             "For each sample, sigma is scaled by `lognormal(0, CV)`. "
             "**0** = constant noise level. **0.19** = realistic (matches the 19% CV measured on real noise files).",
    )

# ── Advanced sections ─────────────────────────────────────────────────────

with st.expander("Physics simulation"):
    st.latex(
        r"P(t) = P_0 \cdot \underbrace{\left[1 + m_0 \cos\!\left(\frac{4\pi V \sin\theta'}{\lambda} t\right)\right]}"
        r"_{\text{modulation}}"
        r"\cdot \underbrace{\exp\!\left(-\frac{(t - T_\text{impact})^2}{2\tau^2}\right)}_{\text{envelope}}"
    )
    st.caption(
        "where tau = (P_size + S_l) / (V * sin(theta)), "
        "theta' = 90 - theta, "
        "and f_Doppler = 2*V*sin(theta') / lambda"
    )

    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        sim["laser_lambda"] = st.number_input(
            "Lambda (m)", value=sim["laser_lambda"], format="%.2e",
            help="Laser wavelength. Appears in the **Doppler frequency**: "
                 "`f_D = 2 * V * sin(90-theta) / lambda`. "
                 "Higher lambda = lower oscillation frequency in the signal.",
        )
        sim["adq_freq"] = st.number_input(
            "Acquisition freq (Hz)", value=sim["adq_freq"], step=100000,
            help="Sampling rate of the acquisition system. "
                 "Determines the time resolution: `dt = 1/f_acq`. "
                 "Must be > 2x the highest signal frequency (Nyquist).",
        )
    with adv2:
        sim["inc_angle"] = st.number_input(
            "Incidence angle (deg)", value=sim["inc_angle"], format="%.1f",
            help="Angle between the laser beam and the particle flow direction. "
                 "Affects **Doppler frequency** (via sin(90-theta)) and **transit time tau** (via sin(theta)). "
                 "At 80 deg: sin(10 deg) ~ 0.17 for Doppler, sin(80 deg) ~ 0.98 for transit.",
        )
        sim["po"] = st.number_input(
            "Laser power P0 (mV)", value=sim["po"], format="%.6f",
            help="Laser output power. Directly scales the **signal amplitude**: `P(t) = P0 * modulation * envelope`. "
                 "Higher P0 = stronger signal relative to noise.",
        )
    with adv3:
        sim["time_max"] = int(st.number_input(
            "Samples per signal", value=sim["time_max"], step=100,
            help="Number of samples in each generated signal (array length). "
                 "Window duration = time_max / adq_freq.",
        ))
        sim["s_l"] = st.number_input(
            "Laser spot diameter (m)", value=sim["s_l"], format="%.2e",
            help="Diameter of the focused laser beam. Affects the **transit time**: "
                 "`tau = (P_size + S_l) / (V * sin(theta))`. "
                 "Larger spot = wider envelope = particle spends more time in the beam.",
        )

with st.expander("Bandpass filter"):
    st.caption(
        "Butterworth zero-phase bandpass filter (scipy.signal.filtfilt). "
        "Applied to the signal after DC subtraction. Removes DC drift and high-frequency noise."
    )
    apply_gen_filter = st.checkbox(
        "Enable bandpass filter during generation", value=False,
        help="When enabled, the Butterworth bandpass is applied to each signal during generation. "
             "When disabled, the raw signal is saved (filtering can be done at training time instead).",
    )
    sim["apply_generation_filter"] = apply_gen_filter

    if sim["noise_injection"] == "before" and not apply_gen_filter and sim["noise_type"] != "none":
        st.warning(
            "Noise injection is set to **before** but the generation bandpass filter is **disabled**. "
            "'before' and 'after' will behave identically (noise is added but never filtered). "
            "Enable the bandpass filter or switch injection to 'after'."
        )

    f1, f2, f3 = st.columns(3)
    with f1:
        sim["filter_lowcut"] = st.number_input(
            "Low cutoff (Hz)", value=sim["filter_lowcut"], step=1000,
            disabled=not apply_gen_filter,
            help="Lower cutoff frequency. Removes slow baseline drift. "
                 "Must be below the Doppler frequency: `f_D = 2*V*sin(10 deg)/lambda`. "
                 "At V=0.05 m/s: f_D ~ 11.2 kHz.",
        )
    with f2:
        sim["filter_highcut"] = st.number_input(
            "High cutoff (Hz)", value=sim["filter_highcut"], step=1000,
            disabled=not apply_gen_filter,
            help="Upper cutoff frequency. Removes high-frequency electronic noise. "
                 "Should be above the maximum Doppler frequency in the dataset.",
        )
    with f3:
        sim["filter_order"] = int(st.number_input(
            "Filter order", value=sim["filter_order"], step=1,
            disabled=not apply_gen_filter,
            help="Order of the Butterworth filter. Higher order = sharper cutoff but more ringing. "
                 "Order 4 is standard for OFI signal processing.",
        ))

    # Validation warnings
    if apply_gen_filter:
        nyquist = sim["adq_freq"] / 2
        if sim["filter_lowcut"] >= sim["filter_highcut"]:
            st.error(
                f"Low cutoff ({sim['filter_lowcut']:.0f} Hz) must be less than "
                f"high cutoff ({sim['filter_highcut']:.0f} Hz)."
            )
        elif sim["filter_highcut"] >= nyquist:
            st.error(
                f"High cutoff ({sim['filter_highcut']:.0f} Hz) must be below "
                f"Nyquist frequency ({nyquist:.0f} Hz = adq_freq / 2)."
            )
        elif sim["filter_lowcut"] <= 0:
            st.error("Low cutoff must be positive.")

with st.expander("Randomization"):
    st.caption("Ranges for random parameters drawn independently for each sample.")
    r1, r2 = st.columns(2)
    with r1:
        sim["p_speed_min"] = st.number_input(
            "Speed min (m/s)", value=sim["p_speed_min"], format="%.3f",
            help="Minimum particle speed. Affects both Doppler frequency and transit time. "
                 "Lower speed = lower f_D and wider envelope.",
        )
        sim["t_impact_factor_min"] = st.number_input(
            "T_impact min (frac)", value=sim["t_impact_factor_min"], format="%.2f",
            help="Minimum position of the burst center as a fraction of the window. "
                 "0.4 = burst centered at 40% of the window. Prevents bursts from being clipped at edges.",
        )
    with r2:
        sim["p_speed_max"] = st.number_input(
            "Speed max (m/s)", value=sim["p_speed_max"], format="%.3f",
            help="Maximum particle speed. Higher speed = higher Doppler frequency, narrower burst.",
        )
        sim["t_impact_factor_max"] = st.number_input(
            "T_impact max (frac)", value=sim["t_impact_factor_max"], format="%.2f",
            help="Maximum position of the burst center as a fraction of the window.",
        )

with st.expander("Classes (m0 / train / test)"):
    st.caption(
        "The modulation index m0 controls signal amplitude. "
        "Larger particles scatter more light, producing higher m0 values. "
        "Calibrated amplitude ratios: 2um : 4um : 10um = 1 : 1.86 : 4.33."
    )
    classes = []
    for i, default_cls in enumerate(DEFAULT_CLASSES):
        st.markdown(f"**{default_cls['name']}** (diameter = {default_cls['p_size']*1e6:.0f} um)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            m0_min = st.number_input(
                f"m0 min##{i}", value=default_cls["m0_min"], format="%.1f", key=f"m0min_{i}",
                help="Minimum modulation index. Controls the weakest signals for this class.",
            )
        with c2:
            m0_max = st.number_input(
                f"m0 max##{i}", value=default_cls["m0_max"], format="%.1f", key=f"m0max_{i}",
                help="Maximum modulation index. A wider range increases intra-class variability.",
            )
        with c3:
            train = int(st.number_input(f"train##{i}", value=default_cls["train"], step=1, key=f"train_{i}"))
        with c4:
            test = int(st.number_input(f"test##{i}", value=default_cls["test"], step=1, key=f"test_{i}"))
        classes.append({
            "name": default_cls["name"],
            "p_size": default_cls["p_size"],
            "m0_min": m0_min, "m0_max": m0_max,
            "train": train, "test": test,
        })

# ── CLI equivalent ────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Equivalent CLI command")
cli_parts = ["python generate_dataset.py", mode, "--output", output_dir,
             "--signal", signal_preset, "--noise", noise_preset]
if force:
    cli_parts.append("--force")
if noise_preset == "real":
    cli_parts.extend(["--noise-dir", noise_dir])
if apply_gen_filter:
    cli_parts.append("--with-filter")
    cli_parts.extend(["--filter-lowcut", str(int(sim["filter_lowcut"]))])
    cli_parts.extend(["--filter-highcut", str(int(sim["filter_highcut"]))])
    cli_parts.extend(["--filter-order", str(sim["filter_order"])])
st.code(" ".join(cli_parts), language="bash")

# ── Action buttons ────────────────────────────────────────────────────────

col_preview, col_generate = st.columns(2)
with col_preview:
    preview_clicked = st.button("Preview (3 samples/class)", use_container_width=True)
with col_generate:
    generate_clicked = st.button("Generate dataset", type="primary", use_container_width=True)

# ── Preview ───────────────────────────────────────────────────────────────

if preview_clicked:
    rng = np.random.default_rng(sim["seed"])

    noise_files = None
    if sim["noise_type"] == "real":
        try:
            noise_files = load_noise_files(noise_dir)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    t_ms = np.linspace(0, sim["time_max"] / sim["adq_freq"] * 1000, sim["time_max"])

    stats_rows = []
    for row, cls in enumerate(classes):
        for col in range(3):
            signal, params = generate_sample(
                cls["p_size"], rng, sim, cls["m0_min"], cls["m0_max"],
                noise_files=noise_files,
            )
            ax = axes[row, col]
            ax.plot(t_ms, signal, linewidth=0.5)
            title = f"{cls['name']} | v={params['p_speed']:.3f} m/s | m0={params['m0']:.1f}"
            if params["multiburst"]:
                title += " [MB]"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("t (ms)" if row == 2 else "", fontsize=8)
            ax.set_ylabel("Amplitude" if col == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            bl = np.concatenate([signal[:200], signal[-200:]])
            stats_rows.append({
                "Class": cls["name"],
                "mean": f"{np.mean(signal):+.4f}",
                "std": f"{np.std(signal):.4f}",
                "baseline_std": f"{np.std(bl):.4f}",
                "multiburst": "yes" if params["multiburst"] else "",
                "skew": f"{params['skew']:.2f}",
            })

    fig.suptitle(f"signal={signal_preset}  |  noise={noise_preset}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Sample statistics:**")
    st.dataframe(stats_rows, use_container_width=True)

# ── Full generation ───────────────────────────────────────────────────────

if generate_clicked:
    if os.path.exists(output_dir) and not force:
        st.error(f"Directory '{output_dir}' already exists. Check 'Overwrite if exists'.")
    else:
        test_mode = mode == "test"
        total = sum(cls["train"] + cls["test"] for cls in classes) if not test_mode else 3 * len(classes)
        progress = st.progress(0, text="Generating...")

        try:
            with st.spinner("Generating..."):
                run_generation(
                    classes, sim, output_dir, force, test_mode=test_mode,
                    noise_dir=noise_dir,
                )
            progress.progress(100, text="Done!")
            st.success(f"{total} files generated in '{output_dir}'")
        except Exception as e:
            st.error(f"Error: {e}")
