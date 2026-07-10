"""1D EfficientNet for particle signal classification.

EfficientNet-B0 style with MBConv blocks, Squeeze-and-Excitation, SiLU activation,
and stochastic depth. Scaled to ~5.3M params via width multiplier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SqueezeExcite1D(nn.Module):
    """Squeeze-and-Excitation block for 1D."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, mid, 1)
        self.fc2 = nn.Conv1d(mid, channels, 1)

    def forward(self, x):
        scale = x.mean(dim=-1, keepdim=True)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConv1D(nn.Module):
    """Mobile inverted bottleneck conv block with SE for 1D."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, expand_ratio=1,
                 se_ratio=0.25, drop_path_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = int(in_ch * expand_ratio)
        self.drop_path_rate = drop_path_rate

        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.SiLU(inplace=True),
            ])
        # Depthwise
        layers.extend([
            nn.Conv1d(hidden, hidden, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        ])
        self.conv = nn.Sequential(*layers)

        # SE
        self.se = SqueezeExcite1D(hidden, reduction=int(1 / se_ratio)) if se_ratio > 0 else nn.Identity()

        # Project
        self.project = nn.Sequential(
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            if self.training and self.drop_path_rate > 0:
                if torch.rand(1).item() < self.drop_path_rate:
                    return x
            return x + out
        return out


def _make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class EfficientNet1D(nn.Module):
    """EfficientNet-B0 style 1D classifier."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2,
                 width_mult: float = 0.85, head_ch: int = None, kernel_size: int = None):
        super().__init__()

        # Block configs: [expand_ratio, out_channels, num_blocks, stride, kernel_size]
        configs = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        # Uniform kernel override: replace every per-block `k` by `kernel_size`
        # AND the stem kernel below. `None` preserves default B0 kernels (3/5).
        if kernel_size is not None:
            for c in configs:
                c[4] = kernel_size
            stem_k = kernel_size
        else:
            stem_k = 3

        # Stem
        stem_ch = _make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_ch, stem_k, stride=2, padding=stem_k // 2, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.SiLU(inplace=True),
        )

        # MBConv blocks
        blocks = []
        in_ch = stem_ch
        total_blocks = sum(c[2] for c in configs)
        block_idx = 0
        for t, c, n, s, k in configs:
            out_ch = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                drop_rate = 0.2 * block_idx / total_blocks
                blocks.append(MBConv1D(in_ch, out_ch, k, stride, t, drop_path_rate=drop_rate))
                in_ch = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        # Head
        if head_ch is None:
            head_ch = _make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm1d(head_ch),
            nn.SiLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(head_ch, 256)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = EfficientNet1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EfficientNet1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
