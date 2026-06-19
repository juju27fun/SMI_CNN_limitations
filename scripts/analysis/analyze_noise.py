"""
analyze_noise.py
----------------
Noise analysis report generator for OFI signal datasets.

Takes a folder of .npy signal files as input and produces a comprehensive
PDF report characterizing the noise: amplitude statistics, PSD, spectral
slope, frequency-band energy, noise-type classification, inter-file
variability, and stationarity.

Usage:
    python analyze_noise.py ./Noise
    python analyze_noise.py dataset/test/10um
    python analyze_noise.py v_colored/2um --output custom_report.pdf
    python analyze_noise.py ./Noise --fs 2000000 --segment 2500
"""

import argparse
import glob
import os
import warnings

import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis as scipy_kurtosis, lognorm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_signals(folder):
    """Load all .npy files from *folder*. Returns (signals, filenames)."""
    paths = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in '{folder}'")
    signals = [np.load(p) for p in paths]
    filenames = [os.path.basename(p) for p in paths]
    return signals, filenames


# ---------------------------------------------------------------------------
# Per-file statistics
# ---------------------------------------------------------------------------
def compute_file_stats(signals):
    """Return dict of arrays: mean, std, min, max, rms, kurtosis per file."""
    means = np.array([np.mean(s) for s in signals])
    stds = np.array([np.std(s) for s in signals])
    mins = np.array([np.min(s) for s in signals])
    maxs = np.array([np.max(s) for s in signals])
    rms = np.array([np.sqrt(np.mean(s ** 2)) for s in signals])
    kurt = np.array([scipy_kurtosis(s, fisher=True) for s in signals])
    return {
        "mean": means, "std": stds, "min": mins,
        "max": maxs, "rms": rms, "kurtosis": kurt,
    }


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------
def compute_psd(signals, fs, nperseg=None):
    """Welch PSD averaged across files. Returns (freqs, avg_psd, min_psd, max_psd)."""
    if nperseg is None:
        nperseg = min(1024, min(len(s) for s in signals))
    psds = []
    for s in signals:
        f, pxx = welch(s, fs=fs, nperseg=nperseg)
        psds.append(pxx)
    psds = np.array(psds)
    return f, np.mean(psds, axis=0), np.min(psds, axis=0), np.max(psds, axis=0)


# ---------------------------------------------------------------------------
# Frequency-band energy
# ---------------------------------------------------------------------------
DEFAULT_BANDS = [
    (0, 1_000, "0–1 kHz"),
    (1_000, 7_000, "1–7 kHz"),
    (7_000, 10_000, "7–10 kHz"),
    (10_000, 40_000, "10–40 kHz"),
    (40_000, 80_000, "40–80 kHz"),
    (80_000, 1_000_000, "80 kHz–Nyq"),
]


def compute_band_energy(freqs, psd, bands=None):
    """Return list of (label, pct) for each band."""
    if bands is None:
        bands = DEFAULT_BANDS
    total = np.sum(psd[freqs > 0])
    results = []
    for lo, hi, label in bands:
        mask = (freqs >= lo) & (freqs < hi)
        pct = np.sum(psd[mask]) / total * 100 if total > 0 else 0.0
        results.append((label, pct))
    return results


def compute_band_energy_per_file(signals, fs, bands=None, nperseg=None):
    """Return array (n_files, n_bands) of per-file band energy percentages."""
    if bands is None:
        bands = DEFAULT_BANDS
    if nperseg is None:
        nperseg = min(1024, min(len(s) for s in signals))
    all_pcts = []
    for s in signals:
        f, pxx = welch(s, fs=fs, nperseg=nperseg)
        total = np.sum(pxx[f > 0])
        row = []
        for lo, hi, _ in bands:
            mask = (f >= lo) & (f < hi)
            row.append(np.sum(pxx[mask]) / total * 100 if total > 0 else 0.0)
        all_pcts.append(row)
    return np.array(all_pcts)


# ---------------------------------------------------------------------------
# Spectral slope (β) via log-log linear fit
# ---------------------------------------------------------------------------
def compute_spectral_slope(freqs, psd, fmin=1_000, fmax=80_000):
    """Fit log10(PSD) = β * log10(f) + c in [fmin, fmax]. Returns β, r²."""
    mask = (freqs >= fmin) & (freqs <= fmax) & (psd > 0)
    if np.sum(mask) < 3:
        return 0.0, 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    coeffs = np.polyfit(log_f, log_p, 1)
    beta = coeffs[0]
    residuals = log_p - np.polyval(coeffs, log_f)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


# ---------------------------------------------------------------------------
# Spectral flatness
# ---------------------------------------------------------------------------
def compute_spectral_flatness(psd):
    """Spectral flatness = exp(mean(log(PSD))) / mean(PSD). 1.0 = white."""
    psd_pos = psd[psd > 0]
    if len(psd_pos) == 0:
        return 0.0
    log_mean = np.exp(np.mean(np.log(psd_pos)))
    arith_mean = np.mean(psd_pos)
    return log_mean / arith_mean if arith_mean > 0 else 0.0


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------
def compute_autocorrelation(signals, max_lag=100):
    """Average normalized autocorrelation across files using FFT."""
    acfs = []
    for s in signals:
        s_centered = s - np.mean(s)
        var = np.var(s_centered)
        if var == 0:
            continue
        n = len(s_centered)
        # FFT-based autocorrelation (fast, uses full signal)
        fft_s = np.fft.rfft(s_centered, n=2 * n)
        acf_full = np.fft.irfft(np.abs(fft_s) ** 2)[:n]
        acf_full = acf_full / (var * n)  # normalize so lag-0 = 1.0
        acfs.append(acf_full[:max_lag + 1])
    if not acfs:
        return np.zeros(max_lag + 1)
    return np.mean(acfs, axis=0)


# ---------------------------------------------------------------------------
# Stationarity (half-split std ratio)
# ---------------------------------------------------------------------------
def compute_stationarity(signals):
    """Return array of std(first_half) / std(second_half) per file."""
    ratios = []
    for s in signals:
        mid = len(s) // 2
        s1 = np.std(s[:mid])
        s2 = np.std(s[mid:])
        ratios.append(s1 / s2 if s2 > 0 else 1.0)
    return np.array(ratios)


# ---------------------------------------------------------------------------
# Amplitude variability (inter-file)
# ---------------------------------------------------------------------------
def compute_amplitude_variability(stds):
    """CV, lognormal fit parameters. Returns (cv, shape, loc, scale)."""
    cv = np.std(stds) / np.mean(stds) if np.mean(stds) > 0 else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            shape, loc, scale = lognorm.fit(stds, floc=0)
        except Exception:
            shape, loc, scale = 0.0, 0.0, np.mean(stds)
    return cv, shape, loc, scale


# ---------------------------------------------------------------------------
# Noise type classification
# ---------------------------------------------------------------------------
def classify_noise_type(beta, flatness, kurt_mean):
    """Return (color_label, gaussian_label) strings."""
    abs_beta = abs(beta)
    if abs_beta < 0.3:
        color = "White"
    elif abs_beta < 1.5:
        color = "Pink (1/f)"
    else:
        color = "Brown (1/f²)"

    if abs(kurt_mean) < 0.5:
        gauss = "Gaussian"
    elif abs(kurt_mean) < 2.0:
        gauss = "Near-Gaussian"
    else:
        gauss = "Non-Gaussian"

    return color, gauss


# ---------------------------------------------------------------------------
# Segment-level analysis (for long files)
# ---------------------------------------------------------------------------
def compute_segment_stats(signals, segment_len):
    """Chop each signal into non-overlapping segments, return per-segment stds."""
    within_stds = []   # list of arrays, one per file
    for s in signals:
        n_seg = len(s) // segment_len
        if n_seg < 2:
            continue
        segs = s[:n_seg * segment_len].reshape(n_seg, segment_len)
        within_stds.append(np.std(segs, axis=1))
    return within_stds


# ===================================================================
# PDF rendering
# ===================================================================
FIG_W, FIG_H = 11, 8.5  # letter landscape


def _text_block(ax, lines, fontsize=9):
    """Render a list of text lines on a blank axes."""
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment="top", fontfamily="monospace")


def render_page_summary(pdf, folder, signals, filenames, stats):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Noise Analysis — {os.path.basename(folder)}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Text summary (top-left) ---
    ax_txt = fig.add_subplot(gs[0, 0])
    n = len(signals)
    lengths = [len(s) for s in signals]
    lines = [
        "PAGE 1 — FILE STATISTICS",
        "",
        f"Folder:           {folder}",
        f"Files:            {n}",
        f"Signal length:    {min(lengths)}–{max(lengths)} samples",
        f"Dtype:            {signals[0].dtype}",
        "",
        "Aggregate noise statistics:",
        f"  mean(σ):        {np.mean(stats['std']):.6f}",
        f"  std(σ):         {np.std(stats['std']):.6f}",
        f"  CV(σ):          {np.std(stats['std']) / np.mean(stats['std']):.4f}" if np.mean(stats['std']) > 0 else "  CV(σ):          N/A",
        f"  mean(RMS):      {np.mean(stats['rms']):.6f}",
        f"  mean(kurtosis): {np.mean(stats['kurtosis']):.4f}  (0 = Gaussian)",
        "",
        f"  mean(DC):       {np.mean(stats['mean']):.6f}",
        f"  std(DC):        {np.std(stats['mean']):.6f}",
        f"  Amplitude range: [{np.min(stats['min']):.4f}, {np.max(stats['max']):.4f}]",
    ]
    _text_block(ax_txt, lines)

    # --- Histogram of per-file σ (top-right) ---
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(stats["std"], bins=min(40, max(10, n // 5)), color="#4C72B0", edgecolor="white")
    ax_hist.set_xlabel("Per-file σ")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of per-file standard deviations")
    ax_hist.axvline(np.mean(stats["std"]), color="red", ls="--", label=f"mean = {np.mean(stats['std']):.4f}")
    ax_hist.legend(fontsize=8)

    # --- Histogram of per-file means (bottom-left) ---
    ax_dc = fig.add_subplot(gs[1, 0])
    ax_dc.hist(stats["mean"], bins=min(40, max(10, n // 5)), color="#55A868", edgecolor="white")
    ax_dc.set_xlabel("Per-file mean (DC offset)")
    ax_dc.set_ylabel("Count")
    ax_dc.set_title("Distribution of per-file DC offsets")
    ax_dc.axvline(0, color="gray", ls=":", alpha=0.7)

    # --- Box plot of key stats (bottom-right) ---
    ax_box = fig.add_subplot(gs[1, 1])
    box_data = [stats["std"], stats["rms"], np.abs(stats["mean"])]
    bp = ax_box.boxplot(box_data, tick_labels=["σ", "RMS", "|DC offset|"], patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_box.set_title("Distribution of amplitude metrics")
    ax_box.set_ylabel("Amplitude")

    pdf.savefig(fig)
    plt.close(fig)


def render_page_timedomain(pdf, folder, signals, stats, fs):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Time-Domain Analysis — {os.path.basename(folder)}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    rng = np.random.default_rng(42)
    n_show = min(10, len(signals))
    indices = rng.choice(len(signals), n_show, replace=False)

    # --- Overlay of random signals (top-left) ---
    ax_overlay = fig.add_subplot(gs[0, 0])
    for i, idx in enumerate(indices):
        t_ms = np.arange(len(signals[idx])) / fs * 1000
        ax_overlay.plot(t_ms, signals[idx], alpha=0.6, lw=0.5, label=f"#{idx}" if i < 5 else None)
    ax_overlay.set_xlabel("Time [ms]")
    ax_overlay.set_ylabel("Amplitude")
    ax_overlay.set_title(f"Overlay of {n_show} random signals")
    if n_show <= 5:
        ax_overlay.legend(fontsize=7)

    # --- Zoomed first signal (top-right) ---
    ax_zoom = fig.add_subplot(gs[0, 1])
    idx0 = indices[0]
    t_ms = np.arange(len(signals[idx0])) / fs * 1000
    ax_zoom.plot(t_ms, signals[idx0], lw=0.6, color="#4C72B0")
    ax_zoom.set_xlabel("Time [ms]")
    ax_zoom.set_ylabel("Amplitude")
    ax_zoom.set_title(f"Signal #{idx0} (detail)")
    ax_zoom.grid(True, alpha=0.3)

    # --- Crest factor distribution (bottom-left) ---
    ax_crest = fig.add_subplot(gs[1, 0])
    peaks = np.max(np.abs(np.array([s for s in signals])), axis=1) if all(len(s) == len(signals[0]) for s in signals) else np.array([np.max(np.abs(s)) for s in signals])
    crest = peaks / stats["rms"]
    ax_crest.hist(crest, bins=min(30, max(10, len(signals) // 5)), color="#C44E52", edgecolor="white")
    ax_crest.set_xlabel("Crest factor (peak / RMS)")
    ax_crest.set_ylabel("Count")
    ax_crest.set_title("Crest factor distribution")
    ax_crest.axvline(np.sqrt(2), color="blue", ls="--", lw=1, label=f"√2 (Gaussian ref)")
    ax_crest.axvline(np.mean(crest), color="red", ls="--", lw=1, label=f"mean = {np.mean(crest):.2f}")
    ax_crest.legend(fontsize=8)

    # --- Stationarity (bottom-right) ---
    ax_stat = fig.add_subplot(gs[1, 1])
    ratios = compute_stationarity(signals)
    ax_stat.hist(ratios, bins=min(30, max(10, len(signals) // 5)), color="#8172B2", edgecolor="white")
    ax_stat.set_xlabel("σ(first half) / σ(second half)")
    ax_stat.set_ylabel("Count")
    ax_stat.set_title("Stationarity test (half-split σ ratio)")
    ax_stat.axvline(1.0, color="green", ls="--", lw=1.5, label="Stationary (1.0)")
    ax_stat.axvline(np.mean(ratios), color="red", ls="--", lw=1, label=f"mean = {np.mean(ratios):.3f}")
    ax_stat.legend(fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


def render_page_psd(pdf, folder, freqs, avg_psd, min_psd, max_psd, beta, r2):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Power Spectral Density — {os.path.basename(folder)}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- PSD log-log (top-left) ---
    ax_psd = fig.add_subplot(gs[0, 0])
    mask = freqs > 0
    ax_psd.loglog(freqs[mask] / 1000, avg_psd[mask], color="#4C72B0", lw=1.2, label="Mean PSD")
    ax_psd.fill_between(freqs[mask] / 1000, min_psd[mask], max_psd[mask],
                        alpha=0.2, color="#4C72B0", label="Min–Max range")
    ax_psd.set_xlabel("Frequency [kHz]")
    ax_psd.set_ylabel("PSD [V²/Hz]")
    ax_psd.set_title("PSD (log-log)")
    ax_psd.legend(fontsize=8)
    ax_psd.grid(True, which="both", alpha=0.2)

    # --- PSD linear (top-right) ---
    ax_lin = fig.add_subplot(gs[0, 1])
    ax_lin.semilogy(freqs[mask] / 1000, avg_psd[mask], color="#DD8452", lw=0.8)
    ax_lin.set_xlabel("Frequency [kHz]")
    ax_lin.set_ylabel("PSD [V²/Hz]")
    ax_lin.set_title("PSD (linear freq, log power)")
    ax_lin.grid(True, alpha=0.3)
    # Mark bandpass region
    ax_lin.axvspan(7, 80, alpha=0.08, color="green", label="System bandpass [7–80 kHz]")
    ax_lin.legend(fontsize=8)

    # --- Spectral slope fit (bottom-left) ---
    ax_slope = fig.add_subplot(gs[1, 0])
    fit_mask = (freqs >= 1_000) & (freqs <= 80_000) & (avg_psd > 0)
    if np.sum(fit_mask) > 3:
        log_f = np.log10(freqs[fit_mask])
        log_p = np.log10(avg_psd[fit_mask])
        ax_slope.scatter(log_f, log_p, s=3, alpha=0.4, color="#4C72B0")
        fit_line = np.polyval([beta, np.mean(log_p) - beta * np.mean(log_f)], log_f)
        ax_slope.plot(log_f, fit_line, color="red", lw=2,
                      label=f"β = {beta:.3f}  (R² = {r2:.3f})")
        ax_slope.set_xlabel("log₁₀(f)")
        ax_slope.set_ylabel("log₁₀(PSD)")
        ax_slope.set_title("Spectral slope fit (1–80 kHz)")
        ax_slope.legend(fontsize=9)
        ax_slope.grid(True, alpha=0.3)

    # --- Interpretation text (bottom-right) ---
    ax_txt = fig.add_subplot(gs[1, 1])
    flatness = compute_spectral_flatness(avg_psd[mask])
    color_label, _ = classify_noise_type(beta, flatness, 0)
    lines = [
        "SPECTRAL ANALYSIS SUMMARY",
        "",
        f"Spectral slope β:     {beta:.4f}",
        f"Fit quality R²:       {r2:.4f}",
        f"Spectral flatness:    {flatness:.4f}  (1.0 = white)",
        "",
        "Slope interpretation:",
        f"  β ≈  0 → White noise    (flat PSD)",
        f"  β ≈ −1 → Pink / 1/f     (equal energy per octave)",
        f"  β ≈ −2 → Brown / 1/f²   (random walk)",
        "",
        f"Classification:       {color_label}",
    ]
    _text_block(ax_txt, lines, fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def render_page_bands(pdf, folder, freqs, avg_psd, band_energies, band_per_file, signals, fs):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Frequency Band Energy — {os.path.basename(folder)}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    labels = [be[0] for be in band_energies]
    pcts = [be[1] for be in band_energies]

    # --- Bar chart (top-left) ---
    ax_bar = fig.add_subplot(gs[0, 0])
    colors_bar = ["#CCB974", "#C44E52", "#55A868", "#4C72B0", "#8172B2", "#937860"]
    bars = ax_bar.bar(labels, pcts, color=colors_bar[:len(labels)], edgecolor="white")
    ax_bar.set_ylabel("% of total energy")
    ax_bar.set_title("Average energy per frequency band")
    ax_bar.tick_params(axis="x", rotation=30)
    for bar, p in zip(bars, pcts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{p:.1f}%", ha="center", fontsize=8)

    # --- Box plots per band (top-right) ---
    ax_box = fig.add_subplot(gs[0, 1])
    bp = ax_box.boxplot(band_per_file.T.tolist(), tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors_bar[:len(labels)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_box.set_ylabel("% of total energy")
    ax_box.set_title("Per-file band energy distribution")
    ax_box.tick_params(axis="x", rotation=30)

    # --- In-band ratio (bottom-left) ---
    ax_inband = fig.add_subplot(gs[1, 0])
    inband_mask = (freqs >= 7_000) & (freqs <= 80_000)
    inband_ratios = []
    nperseg = min(1024, min(len(s) for s in signals))
    for s in signals:
        f, pxx = welch(s, fs=fs, nperseg=nperseg)
        total = np.sum(pxx[f > 0])
        inband = np.sum(pxx[inband_mask[:len(pxx)] if len(inband_mask) != len(pxx) else inband_mask])
        # Recompute mask for this specific f array
        ib_mask = (f >= 7_000) & (f <= 80_000)
        inband = np.sum(pxx[ib_mask])
        inband_ratios.append(inband / total * 100 if total > 0 else 0)
    inband_ratios = np.array(inband_ratios)
    ax_inband.hist(inband_ratios, bins=min(30, max(10, len(signals) // 5)),
                   color="#55A868", edgecolor="white")
    ax_inband.set_xlabel("In-band energy [7–80 kHz] (%)")
    ax_inband.set_ylabel("Count")
    ax_inband.set_title("System bandpass energy ratio per file")
    ax_inband.axvline(np.mean(inband_ratios), color="red", ls="--",
                      label=f"mean = {np.mean(inband_ratios):.1f}%")
    ax_inband.legend(fontsize=8)

    # --- Text summary (bottom-right) ---
    ax_txt = fig.add_subplot(gs[1, 1])
    lines = [
        "BAND ENERGY SUMMARY",
        "",
        f"{'Band':<15s}  {'Mean %':>8s}  {'Std %':>8s}",
        "─" * 35,
    ]
    for i, (label, pct) in enumerate(band_energies):
        std_pct = np.std(band_per_file[:, i])
        lines.append(f"{label:<15s}  {pct:>7.1f}%  {std_pct:>7.1f}%")
    lines.append("─" * 35)
    lines.append(f"{'In-band [7–80k]':<15s}  {np.mean(inband_ratios):>7.1f}%  {np.std(inband_ratios):>7.1f}%")
    _text_block(ax_txt, lines, fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def render_page_classification_with_acf(pdf, folder, signals, freqs, avg_psd, stats, beta, r2):
    """Full classification page with autocorrelation computed from signals."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Noise Type Classification — {os.path.basename(folder)}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    mask = freqs > 0
    flatness = compute_spectral_flatness(avg_psd[mask])
    kurt_mean = np.mean(stats["kurtosis"])
    color_label, gauss_label = classify_noise_type(beta, flatness, kurt_mean)

    # --- Classification summary (top-left) ---
    ax_txt = fig.add_subplot(gs[0, 0])
    lines = [
        "NOISE TYPE CLASSIFICATION",
        "",
        f"Spectral color:    {color_label}",
        f"  slope β =        {beta:.4f}",
        f"  flatness =       {flatness:.4f}",
        "",
        f"Amplitude dist.:   {gauss_label}",
        f"  kurtosis (mean): {kurt_mean:.4f}",
        f"  (excess kurtosis: 0 = Gaussian,",
        f"   >0 = heavy-tailed, <0 = light-tailed)",
        "",
        "Decision boundaries:",
        f"  |β| < 0.3   → White",
        f"  0.3–1.5     → Pink (1/f)",
        f"  > 1.5       → Brown (1/f²)",
        "",
        f"  |kurt| < 0.5  → Gaussian",
        f"  0.5–2.0       → Near-Gaussian",
        f"  > 2.0         → Non-Gaussian",
    ]
    _text_block(ax_txt, lines, fontsize=10)

    # --- Kurtosis distribution (top-right) ---
    ax_kurt = fig.add_subplot(gs[0, 1])
    ax_kurt.hist(stats["kurtosis"], bins=min(30, max(10, len(stats["kurtosis"]) // 5)),
                 color="#DD8452", edgecolor="white")
    ax_kurt.set_xlabel("Excess kurtosis")
    ax_kurt.set_ylabel("Count")
    ax_kurt.set_title("Per-file kurtosis distribution")
    ax_kurt.axvline(0, color="green", ls="--", lw=1.5, label="Gaussian (0)")
    ax_kurt.axvline(kurt_mean, color="red", ls="--", lw=1, label=f"mean = {kurt_mean:.2f}")
    ax_kurt.legend(fontsize=8)

    # --- Autocorrelation (bottom-left) ---
    ax_acf = fig.add_subplot(gs[1, 0])
    max_lag = 100
    acf = compute_autocorrelation(signals, max_lag=max_lag)
    lags = np.arange(len(acf))
    ax_acf.plot(lags, acf, color="#4C72B0", lw=1.2)
    ax_acf.axhline(0, color="gray", ls="-", lw=0.5)
    n_eff = min(len(s) for s in signals)
    conf = 1.96 / np.sqrt(n_eff)
    ax_acf.axhline(conf, color="red", ls=":", lw=0.8, label=f"95% conf (±{conf:.4f})")
    ax_acf.axhline(-conf, color="red", ls=":", lw=0.8)
    ax_acf.set_xlabel("Lag (samples)")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_title("Average autocorrelation (first 100 lags)")
    ax_acf.legend(fontsize=8)
    ax_acf.grid(True, alpha=0.3)

    # --- Amplitude histogram (bottom-right) ---
    ax_amp = fig.add_subplot(gs[1, 1])
    all_vals = np.concatenate(signals)
    if len(all_vals) > 200_000:
        all_vals = np.random.default_rng(42).choice(all_vals, 200_000, replace=False)
    ax_amp.hist(all_vals, bins=200, density=True, color="#8172B2", edgecolor="none", alpha=0.7)
    mu, sigma = np.mean(all_vals), np.std(all_vals)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax_amp.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                color="red", lw=1.5, label=f"Gaussian fit (σ={sigma:.4f})")
    ax_amp.set_xlabel("Amplitude")
    ax_amp.set_ylabel("Density")
    ax_amp.set_title("Amplitude distribution (all samples pooled)")
    ax_amp.legend(fontsize=8)

    pdf.savefig(fig)
    plt.close(fig)


def render_page_variability(pdf, folder, stats):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Amplitude Variability (Inter-file) — {os.path.basename(folder)}",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    stds = stats["std"]
    cv, shape, loc, scale = compute_amplitude_variability(stds)

    # --- Box plot of σ (top-left) ---
    ax_box = fig.add_subplot(gs[0, 0])
    bp = ax_box.boxplot(stds, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][0].set_alpha(0.7)
    ax_box.set_ylabel("Per-file σ")
    ax_box.set_title("σ distribution (box plot)")
    ax_box.set_xticklabels(["σ"])

    # --- Histogram + lognormal fit (top-right) ---
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(stds, bins=min(40, max(10, len(stds) // 5)), density=True,
                 color="#55A868", edgecolor="white", alpha=0.7, label="Observed")
    if shape > 0:
        x = np.linspace(max(0, np.min(stds) * 0.8), np.max(stds) * 1.2, 200)
        pdf_vals = lognorm.pdf(x, shape, loc=loc, scale=scale)
        ax_hist.plot(x, pdf_vals, color="red", lw=2,
                     label=f"Lognormal fit (σ_ln={shape:.3f})")
    ax_hist.set_xlabel("Per-file σ")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("σ distribution + lognormal fit")
    ax_hist.legend(fontsize=8)

    # --- Q-Q plot (bottom-left) ---
    ax_qq = fig.add_subplot(gs[1, 0])
    sorted_stds = np.sort(stds)
    n = len(sorted_stds)
    quantiles = np.arange(1, n + 1) / (n + 1)
    if shape > 0:
        theoretical = lognorm.ppf(quantiles, shape, loc=loc, scale=scale)
        ax_qq.scatter(theoretical, sorted_stds, s=10, alpha=0.6, color="#4C72B0")
        mn = min(np.min(theoretical), np.min(sorted_stds))
        mx = max(np.max(theoretical), np.max(sorted_stds))
        ax_qq.plot([mn, mx], [mn, mx], color="red", ls="--", lw=1, label="Perfect fit")
        ax_qq.set_xlabel("Theoretical quantiles (lognormal)")
        ax_qq.set_ylabel("Observed σ")
        ax_qq.set_title("Q-Q plot: observed σ vs lognormal")
        ax_qq.legend(fontsize=8)
        ax_qq.grid(True, alpha=0.3)
    else:
        ax_qq.text(0.5, 0.5, "Lognormal fit failed", transform=ax_qq.transAxes, ha="center")
        ax_qq.set_title("Q-Q plot (N/A)")

    # --- Text summary (bottom-right) ---
    ax_txt = fig.add_subplot(gs[1, 1])
    lines = [
        "AMPLITUDE VARIABILITY SUMMARY",
        "",
        f"Number of files:     {len(stds)}",
        f"mean(σ):             {np.mean(stds):.6f}",
        f"std(σ):              {np.std(stds):.6f}",
        f"CV = std(σ)/mean(σ): {cv:.4f}",
        "",
        "Lognormal fit parameters:",
        f"  shape (σ_ln):      {shape:.4f}",
        f"  scale (exp(μ_ln)): {scale:.6f}",
        "",
        "Reference (from real noise files):",
        f"  CV_ref = 0.19",
        f"  Ratio CV/CV_ref:   {cv / 0.19:.2f}" if cv > 0 else "  Ratio: N/A",
        "",
        "Interpretation:",
    ]
    if cv < 0.05:
        lines.append("  Very low variability — fixed noise level")
    elif cv < 0.15:
        lines.append("  Moderate variability")
    elif cv < 0.25:
        lines.append("  Consistent with real hardware noise (CV≈0.19)")
    else:
        lines.append("  High variability — mixed conditions or signal content")
    _text_block(ax_txt, lines, fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


def render_page_segments(pdf, folder, signals, segment_len, fs):
    """Segment-level analysis for files longer than segment_len."""
    within_stds = compute_segment_stats(signals, segment_len)
    if not within_stds:
        return  # Skip page if no files are long enough

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.suptitle(f"Segment-Level Analysis — {os.path.basename(folder)}",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Per-file segment σ evolution (top-left) ---
    ax_evo = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(42)
    n_show = min(10, len(within_stds))
    indices = rng.choice(len(within_stds), n_show, replace=False)
    for idx in indices:
        ax_evo.plot(within_stds[idx], alpha=0.6, lw=0.8)
    ax_evo.set_xlabel(f"Segment index ({segment_len} samples each)")
    ax_evo.set_ylabel("Segment σ")
    ax_evo.set_title(f"σ evolution within files ({n_show} shown)")
    ax_evo.grid(True, alpha=0.3)

    # --- Within-file CV distribution (top-right) ---
    ax_wcv = fig.add_subplot(gs[0, 1])
    within_cvs = []
    for ws in within_stds:
        if np.mean(ws) > 0:
            within_cvs.append(np.std(ws) / np.mean(ws))
    within_cvs = np.array(within_cvs)
    ax_wcv.hist(within_cvs, bins=min(30, max(10, len(within_cvs) // 5)),
                color="#C44E52", edgecolor="white")
    ax_wcv.set_xlabel("Within-file CV of segment σ")
    ax_wcv.set_ylabel("Count")
    ax_wcv.set_title("Within-file noise stationarity")
    ax_wcv.axvline(np.mean(within_cvs), color="red", ls="--",
                   label=f"mean = {np.mean(within_cvs):.4f}")
    ax_wcv.legend(fontsize=8)

    # --- Between-file vs within-file CV comparison (bottom-left) ---
    ax_comp = fig.add_subplot(gs[1, 0])
    between_stds = np.array([np.mean(ws) for ws in within_stds])
    between_cv = np.std(between_stds) / np.mean(between_stds) if np.mean(between_stds) > 0 else 0
    bars = ax_comp.bar(["Between-file CV", "Within-file CV (mean)"],
                       [between_cv, np.mean(within_cvs)],
                       color=["#4C72B0", "#C44E52"], edgecolor="white")
    ax_comp.set_ylabel("Coefficient of Variation")
    ax_comp.set_title("Between-file vs within-file variability")
    for bar in bars:
        ax_comp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.4f}", ha="center", fontsize=9)

    # --- Text summary (bottom-right) ---
    ax_txt = fig.add_subplot(gs[1, 1])
    n_segs = [len(ws) for ws in within_stds]
    lines = [
        "SEGMENT ANALYSIS SUMMARY",
        "",
        f"Segment length:        {segment_len} samples",
        f"Files analyzed:        {len(within_stds)} / {len(signals)}",
        f"Segments per file:     {min(n_segs)}–{max(n_segs)}",
        "",
        f"Between-file CV:       {between_cv:.4f}",
        f"Within-file CV (mean): {np.mean(within_cvs):.4f}",
        f"Within-file CV (std):  {np.std(within_cvs):.4f}",
        "",
        "Interpretation:",
    ]
    if np.mean(within_cvs) < 0.05:
        lines.append("  Noise is stationary within files")
    elif np.mean(within_cvs) < 0.15:
        lines.append("  Mild non-stationarity within files")
    else:
        lines.append("  Significant non-stationarity — noise level")
        lines.append("  varies within individual recordings")
    if between_cv > 2 * np.mean(within_cvs):
        lines.append("  Between-file variability dominates")
    else:
        lines.append("  Within-file variability is comparable to")
        lines.append("  between-file variability")
    _text_block(ax_txt, lines, fontsize=10)

    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Noise analysis report for OFI signal datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analyze_noise.py ./Noise\n"
            "  python analyze_noise.py dataset/test/10um\n"
            "  python analyze_noise.py v_colored/2um --output custom_report.pdf\n"
        ),
    )
    parser.add_argument("folder", help="Path to folder containing .npy files")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: <folder_name>_Noise_Analysis.pdf)")
    parser.add_argument("--fs", type=float, default=2_000_000,
                        help="Sampling frequency in Hz (default: 2000000)")
    parser.add_argument("--segment", type=int, default=2500,
                        help="Segment length for long-file analysis (default: 2500)")

    args = parser.parse_args()

    folder = args.folder.rstrip("/")
    folder_name = os.path.basename(folder) or "root"
    if args.output is None:
        output_path = f"{folder_name}_Noise_Analysis.pdf"
    else:
        output_path = args.output

    print(f"Loading signals from '{folder}' ...")
    signals, filenames = load_signals(folder)
    print(f"  {len(signals)} files loaded (lengths: {min(len(s) for s in signals)}–{max(len(s) for s in signals)})")

    print("Computing file statistics ...")
    stats = compute_file_stats(signals)

    print("Computing PSD (Welch) ...")
    nperseg = min(1024, min(len(s) for s in signals))
    freqs, avg_psd, min_psd, max_psd = compute_psd(signals, args.fs, nperseg=nperseg)

    print("Computing spectral slope ...")
    beta, r2 = compute_spectral_slope(freqs, avg_psd)

    print("Computing band energies ...")
    band_energies = compute_band_energy(freqs, avg_psd)
    band_per_file = compute_band_energy_per_file(signals, args.fs, nperseg=nperseg)

    print(f"Generating PDF report: {output_path} ...")
    with PdfPages(output_path) as pdf:
        render_page_summary(pdf, folder, signals, filenames, stats)
        render_page_timedomain(pdf, folder, signals, stats, args.fs)
        render_page_psd(pdf, folder, freqs, avg_psd, min_psd, max_psd, beta, r2)
        render_page_bands(pdf, folder, freqs, avg_psd, band_energies, band_per_file,
                          signals, args.fs)
        render_page_classification_with_acf(pdf, folder, signals, freqs, avg_psd,
                                            stats, beta, r2)
        render_page_variability(pdf, folder, stats)
        # Segment page only if files are longer than segment size
        if max(len(s) for s in signals) > args.segment:
            render_page_segments(pdf, folder, signals, args.segment, args.fs)

    print(f"Done. Report saved to '{output_path}'")


if __name__ == "__main__":
    main()
