"""
filter.py
---------
Centralised bandpass filters for the particle classification pipeline.

Three filters extracted from across the codebase:
  F1 – Generation filter  : Butterworth 7–80 kHz  (scipy, applied on numpy arrays)
  F2 – Training filter    : FFT bandpass 5–100 kHz (torch, applied on tensors)
  F3 – Notebook filter    : FFT bandpass 8–40 kHz  (torch, applied on tensors)
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt


# ─── F1 : Generation filter (Butterworth, scipy) ────────────────────────────

def generation_bandpass(data: np.ndarray, fs: float,
                        lowcut: float = 7000, highcut: float = 80000,
                        order: int = 4) -> np.ndarray:
    """Butterworth zero-phase bandpass filter (7–80 kHz default).

    Operates on raw numpy arrays during dataset generation.

    Parameters
    ----------
    data : np.ndarray   – 1-D signal
    fs   : float        – sampling frequency in Hz
    lowcut / highcut    – cutoff frequencies in Hz
    order               – Butterworth filter order
    """
    b, a = butter(order, [lowcut, highcut], btype="band", fs=fs)
    return filtfilt(b, a, data)


# ─── F2 : Training filter (FFT, torch) ──────────────────────────────────────

class TrainingBandpass:
    """FFT bandpass filter applied at training time (5–100 kHz default).

    Callable transform for ``ParticleDataset``.
    """

    def __init__(self, low_cutoff_khz: float = 5.0,
                 high_cutoff_khz: float = 100.0,
                 sample_rate_mhz: float = 2.0):
        self.low_cutoff = low_cutoff_khz * 1_000       # → Hz
        self.high_cutoff = high_cutoff_khz * 1_000      # → Hz
        self.sample_rate = sample_rate_mhz * 1_000_000  # → Hz

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        fft_signal = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(signal.size(-1), 1 / self.sample_rate)
        mask = (torch.abs(freqs) >= self.low_cutoff) & (torch.abs(freqs) <= self.high_cutoff)
        return torch.fft.ifft(fft_signal * mask).real


# ─── F3 : Notebook filter (FFT, torch) ──────────────────────────────────────

class NotebookBandpass:
    """FFT bandpass filter originally used in the analysis notebook (8–40 kHz default).

    Callable transform for ``ParticleDataset``.
    """

    def __init__(self, low_cutoff_khz: float = 8.0,
                 high_cutoff_khz: float = 40.0,
                 sample_rate_mhz: float = 2.0):
        self.low_cutoff = low_cutoff_khz * 1_000
        self.high_cutoff = high_cutoff_khz * 1_000
        self.sample_rate = sample_rate_mhz * 1_000_000

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        fft_signal = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(signal.size(-1), 1 / self.sample_rate)
        mask = (torch.abs(freqs) >= self.low_cutoff) & (torch.abs(freqs) <= self.high_cutoff)
        return torch.fft.ifft(fft_signal * mask).real
