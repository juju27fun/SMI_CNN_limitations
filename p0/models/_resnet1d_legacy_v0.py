"""Frozen snapshot of ResNet1D as of git HEAD 54e007b (pre-WIP).

This module exists only to support :file:`scripts/remeasure_cpu_latency_benchmark2.py`
when re-loading benchmark2 checkpoints that were trained with the v0
architecture (stem kernel_size=7 hardcoded, blocks kernel_size=3 hardcoded).

The current `models/resnet1d.py` exposes a unified `kernel_size` parameter
that swaps stem and block kernels together — useful for the kernel sweep
work-in-progress, but incompatible with the legacy v0 checkpoints whose
stem is 7 and whose blocks are 3.

Do NOT register this class in `models/__init__.py`. It is imported
explicitly by the retrofit script for ResNet1D variants.
"""

import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock1DV0(nn.Module):
    """Basic residual block: two Conv1d(3) with skip connection. v0."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNet1DLegacyV0(nn.Module):
    """ResNet-18 style 1D classifier — frozen v0 architecture."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2,
                 base_width: int = 74):
        super().__init__()
        w = base_width

        self.conv1 = nn.Conv1d(1, w, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(w)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(w, w, 2)
        self.layer2 = self._make_layer(w, w * 2, 2, stride=2)
        self.layer3 = self._make_layer(w * 2, w * 4, 2, stride=2)
        self.layer4 = self._make_layer(w * 4, w * 8, 2, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        # v0 head: feature_layer (w*8 -> 256) + classifier (256 -> num_classes)
        self.feature_layer = nn.Linear(w * 8, 256)
        self.classifier = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [_BasicBlock1DV0(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(_BasicBlock1DV0(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.feature_layer(x))
        return self.classifier(x)
