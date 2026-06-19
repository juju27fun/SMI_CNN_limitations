"""Smoke tests for measure_cpu_latency in training_utils.

Two invariants:
1. Result advertises latency_device == "cpu" with positive timings.
2. The original model's parameters stay on their initial device — the
   helper deepcopies + cpu()'s internally and must never mutate the
   caller's model (especially relevant when the caller's model is on
   CUDA and the helper must not silently pull it back to CPU).

Run directly: ``python tests/test_latency_cpu.py``.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0.training_utils import measure_cpu_latency  # noqa: E402


class _TinyNet(nn.Module):
    """Minimal Conv1D + GAP + Linear — no dataset dependency."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(self.gap(self.conv(x)).squeeze(-1))


def test_returns_cpu_device():
    model = _TinyNet()
    result = measure_cpu_latency(model, (1, 1, 256), warmup=3, n_runs=10)
    assert result["latency_device"] == "cpu", f"expected cpu, got {result['latency_device']}"
    assert result["median_ms"] > 0, "median_ms must be > 0"
    assert result["p95_ms"] >= result["median_ms"], "p95 must be >= median"
    assert "mean_ms" in result, "result missing mean_ms"


def test_original_model_device_preserved():
    model = _TinyNet()
    pre_device = next(model.parameters()).device
    measure_cpu_latency(model, (1, 1, 256), warmup=3, n_runs=10)
    post_device = next(model.parameters()).device
    assert pre_device == post_device, (
        f"helper mutated original model device: pre={pre_device}, post={post_device}"
    )


if __name__ == "__main__":
    test_returns_cpu_device()
    print("PASS: test_returns_cpu_device")
    test_original_model_device_preserved()
    print("PASS: test_original_model_device_preserved")
    print("All tests passed.")
