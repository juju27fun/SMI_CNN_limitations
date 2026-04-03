"""1D MobileNetV2 for particle signal classification.

MobileNetV2 hallmarks: depthwise separable convolutions, inverted residual blocks
with linear bottleneck. Scaled to ~5.3M params via width multiplier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InvertedResidual1D(nn.Module):
    """MobileNetV2 inverted residual block adapted for 1D."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = int(in_ch * expand_ratio)

        layers = []
        if expand_ratio != 1:
            # Pointwise expand
            layers.extend([
                nn.Conv1d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU6(inplace=True),
            ])
        # Depthwise
        layers.extend([
            nn.Conv1d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU6(inplace=True),
        ])
        # Pointwise project (linear bottleneck — no activation)
        layers.extend([
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


def _make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class MobileNet1D(nn.Module):
    """MobileNetV2-style 1D classifier."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2,
                 width_mult: float = 1.5):
        super().__init__()

        # Inverted residual settings: [expand_ratio, out_channels, num_blocks, stride]
        settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Stem
        first_ch = _make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(1, first_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(first_ch),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks
        blocks = []
        in_ch = first_ch
        for t, c, n, s in settings:
            out_ch = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual1D(in_ch, out_ch, stride, expand_ratio=t))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Head
        last_ch = _make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head_conv = nn.Sequential(
            nn.Conv1d(in_ch, last_ch, 1, bias=False),
            nn.BatchNorm1d(last_ch),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(last_ch, 256)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MobileNet1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MobileNet1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
