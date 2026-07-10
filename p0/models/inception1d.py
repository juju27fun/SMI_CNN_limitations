"""InceptionTime 1D for particle signal classification.

Based on Fawaz et al. (2020) InceptionTime architecture:
- Inception modules with multi-scale parallel convolutions
- Bottleneck 1x1 convs to reduce dimensionality before large kernels
- Residual connections every 3 modules
Scaled to ~5.3M params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule1D(nn.Module):
    """Single Inception module: bottleneck -> parallel convs + maxpool branch."""

    def __init__(self, in_channels, num_filters=32, bottleneck_size=32):
        super().__init__()
        # Bottleneck 1x1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, 1, bias=False)

        # Parallel convolutions with different kernel sizes (odd for symmetric padding)
        self.conv_11 = nn.Conv1d(bottleneck_size, num_filters, kernel_size=11,
                                 padding=5, bias=False)
        self.conv_21 = nn.Conv1d(bottleneck_size, num_filters, kernel_size=21,
                                 padding=10, bias=False)
        self.conv_41 = nn.Conv1d(bottleneck_size, num_filters, kernel_size=41,
                                 padding=20, bias=False)

        # MaxPool branch
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, num_filters, 1, bias=False)

        # Output channels = num_filters * 4 (3 conv branches + 1 pool branch)
        self.bn = nn.BatchNorm1d(num_filters * 4)

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        c10 = self.conv_11(bottleneck)
        c20 = self.conv_21(bottleneck)
        c40 = self.conv_41(bottleneck)
        pool = self.conv_pool(self.maxpool(x))
        out = torch.cat([c10, c20, c40, pool], dim=1)
        return F.relu(self.bn(out))


class InceptionTime1D(nn.Module):
    """InceptionTime classifier for 1D signals."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2,
                 num_filters: int = 148, depth: int = 6, bottleneck_size: int = 64):
        super().__init__()
        self.depth = depth
        module_out_ch = num_filters * 4  # 4 branches per module

        # Build inception modules
        self.modules_list = nn.ModuleList()
        for i in range(depth):
            in_ch = 1 if i == 0 else module_out_ch
            self.modules_list.append(
                InceptionModule1D(in_ch, num_filters, bottleneck_size)
            )

        # Residual shortcuts (every 3 modules)
        self.shortcuts = nn.ModuleDict()
        for i in range(depth):
            if i % 3 == 2:
                shortcut_in = 1 if i < 3 else module_out_ch
                self.shortcuts[str(i)] = nn.Sequential(
                    nn.Conv1d(shortcut_in, module_out_ch, 1, bias=False),
                    nn.BatchNorm1d(module_out_ch),
                )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(module_out_ch, 256)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        residual = x
        for i, module in enumerate(self.modules_list):
            x = module(x)
            # Residual connection every 3 modules
            if i % 3 == 2:
                shortcut = self.shortcuts[str(i)](residual)
                # Align lengths if needed (conv_20/40 with even kernels can shift length)
                if shortcut.shape[-1] != x.shape[-1]:
                    shortcut = F.adaptive_avg_pool1d(shortcut, x.shape[-1])
                x = F.relu(x + shortcut)
                residual = x
        x = self.gap(x)
        x = self.flatten(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = InceptionTime1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"InceptionTime1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
