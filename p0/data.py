"""Dataset and signal transforms shared by P0 training and benchmark code."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

RAW_SIGNAL_LENGTH = 2500


class ParticleDataset(Dataset):
    """Dataset for loading particle signals from class folders of ``.npy`` files."""

    def __init__(self, root_dir: Path, class_names: list, transforms=None):
        self.samples = []
        self.labels = []
        self.root_dir = root_dir
        self.class_names = class_names
        self.transforms = transforms

        for class_idx, class_name in enumerate(class_names):
            class_dir = root_dir / class_name
            if not class_dir.exists():
                continue
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append(npy_file)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal = np.load(self.samples[idx]).astype(np.float32)
        signal = signal[np.newaxis, :]
        signal_tensor = torch.from_numpy(signal)

        if self.transforms is not None:
            for transform in self.transforms:
                signal_tensor = transform(signal_tensor)

        label = self.labels[idx]
        return signal_tensor, label


class BandpassFilter:
    """FFT bandpass filter (5-100 kHz)."""

    def __init__(
        self,
        low_cutoff_khz: float = 5.0,
        high_cutoff_khz: float = 100.0,
        sample_rate_mhz: float = 2.0,
    ):
        self.low_cutoff = low_cutoff_khz * 1000
        self.high_cutoff = high_cutoff_khz * 1000
        self.sample_rate = sample_rate_mhz * 1_000_000

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        fft_signal = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(signal.size(-1), 1 / self.sample_rate)
        mask = (torch.abs(freqs) >= self.low_cutoff) & (torch.abs(freqs) <= self.high_cutoff)
        return torch.fft.ifft(fft_signal * mask).real


class Decimate:
    """Decimation by slicing every Nth sample."""

    def __init__(self, decimate: int):
        self.decimate = decimate

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if self.decimate > 1:
            return signal[..., ::self.decimate]
        return signal


class AdaptiveBandpassDecimate:
    """Bandpass + stride-decimate with a dynamic anti-alias cutoff."""

    def __init__(
        self,
        target_length: int,
        native_length: int = 16384,
        native_fs_hz: float = 2_000_000.0,
        low_khz: float = 5.0,
        high_khz_max: float = 100.0,
    ):
        if native_length % target_length != 0:
            raise ValueError(
                f"native_length ({native_length}) must be divisible by "
                f"target_length ({target_length}); integer stride required."
            )
        self.target_length = target_length
        self.native_length = native_length
        self.native_fs = float(native_fs_hz)
        self.decimate_factor = native_length // target_length
        new_fs = self.native_fs / self.decimate_factor
        new_nyquist = new_fs / 2.0
        self.low_cutoff_hz = low_khz * 1000.0
        self.high_cutoff_hz = min(high_khz_max * 1000.0, 0.9 * new_nyquist)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        fft_signal = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(signal.size(-1), 1.0 / self.native_fs)
        mask = (torch.abs(freqs) >= self.low_cutoff_hz) & (
            torch.abs(freqs) <= self.high_cutoff_hz
        )
        filtered = torch.fft.ifft(fft_signal * mask).real
        if self.decimate_factor > 1:
            filtered = filtered[..., ::self.decimate_factor]
        return filtered


class Truncate:
    """Truncate signal to a fixed length from the beginning."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        return signal[..., : self.length]


class CenterCrop:
    """Crop a centered fixed-length window, padding symmetrically if needed."""

    def __init__(self, length: int):
        self.length = length

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        n = signal.size(-1)
        if n == self.length:
            return signal
        if n > self.length:
            start = (n - self.length) // 2
            return signal[..., start : start + self.length]
        pad_total = self.length - n
        left = pad_total // 2
        right = pad_total - left
        return torch.nn.functional.pad(signal, (left, right))


class GaussianNoise:
    """Additive Gaussian noise with fixed or uniformly sampled SNR."""

    def __init__(self, snr_db: float = 20.0, snr_range: tuple = None, p: float = 0.5):
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return signal
        if self.snr_range is not None:
            low, high = self.snr_range
            snr_db = low + torch.rand(1).item() * (high - low)
        else:
            snr_db = self.snr_db
        sig_power = signal.pow(2).mean()
        noise_power = sig_power / (10 ** (snr_db / 10))
        return signal + torch.randn_like(signal) * noise_power.sqrt()


class RealNoise:
    """Additive captured noise sampled from ``.npy`` files."""

    def __init__(
        self,
        noise_dir: str,
        snr_range: tuple = (-3.0, 3.0),
        p: float = 1.0,
        seed: int = 42,
    ):
        self.snr_range = snr_range
        self.p = p
        self._rng = np.random.default_rng(seed)

        files = sorted(glob.glob(os.path.join(noise_dir, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No .npy noise files found in {noise_dir!r}")
        self._noise_arrays = [np.load(f).astype(np.float32) for f in files]

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if self._rng.random() > self.p:
            return signal

        length = signal.size(-1)
        valid = [a for a in self._noise_arrays if a.shape[-1] >= length]
        if not valid:
            raise ValueError(f"No noise file in pool has length >= {length} samples")
        noise_arr = valid[self._rng.integers(0, len(valid))]
        max_offset = noise_arr.shape[-1] - length
        offset = int(self._rng.integers(0, max_offset + 1))
        segment = noise_arr[offset : offset + length]
        noise = torch.from_numpy(np.ascontiguousarray(segment))

        sig_power = signal.pow(2).mean()
        noise_power_actual = noise.pow(2).mean()
        if noise_power_actual.item() == 0:
            return signal

        low, high = self.snr_range
        snr_db = low + float(self._rng.random()) * (high - low)
        target_noise_power = sig_power / (10 ** (snr_db / 10))
        scale = (target_noise_power / noise_power_actual).sqrt()
        return signal + noise * scale


class TimeMasking:
    """Zero out a single random temporal block of the signal."""

    def __init__(self, mask_ratio: float = 0.15, p: float = 1.0):
        self.mask_ratio = mask_ratio
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return signal
        length = signal.size(-1)
        block_len = int(length * self.mask_ratio)
        if block_len <= 0:
            return signal
        start = int(torch.randint(0, length - block_len + 1, (1,)).item())
        signal = signal.clone()
        signal[..., start : start + block_len] = 0
        return signal


class TimeShift:
    """Random circular shift along the time axis."""

    def __init__(self, max_shift_frac: float = 0.1, p: float = 0.5):
        self.max_shift_frac = max_shift_frac
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return signal
        length = signal.size(-1)
        max_shift = int(length * self.max_shift_frac)
        if max_shift == 0:
            return signal
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(signal, shifts=shift, dims=-1)


class AmplitudeScale:
    """Random amplitude scaling."""

    def __init__(self, scale_min: float = 0.8, scale_max: float = 1.2, p: float = 0.5):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return signal
        scale = self.scale_min + torch.rand(1).item() * (self.scale_max - self.scale_min)
        return signal * scale
