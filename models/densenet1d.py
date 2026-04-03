"""1D DenseNet for particle signal classification.

DenseNet hallmarks: dense connectivity (each layer receives all preceding feature maps),
growth rate controls width expansion, transition layers for downsampling.
Scaled to ~5.3M params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer1D(nn.Module):
    """Single dense layer: BN -> ReLU -> Conv1d(1) -> BN -> ReLU -> Conv1d(3)."""

    def __init__(self, in_channels, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        mid = bn_size * growth_rate
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, mid, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, growth_rate, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock1D(nn.Module):
    """Dense block: stack of DenseLayers with concatenation."""

    def __init__(self, in_channels, num_layers, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer1D(in_channels + i * growth_rate, growth_rate, bn_size, dropout))
        self.layers = nn.Sequential(*layers)
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x):
        return self.layers(x)


class Transition1D(nn.Module):
    """Transition layer: BN -> ReLU -> Conv1d(1) -> AvgPool for compression."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet1D(nn.Module):
    """DenseNet 1D classifier."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2,
                 growth_rate: int = 40, block_config=(6, 12, 24), init_channels: int = 64,
                 compression: float = 0.5, bn_size: int = 4):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, init_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        ch = init_channels
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock1D(ch, num_layers, growth_rate, bn_size, dropout=0.0)
            self.dense_blocks.append(block)
            ch = block.out_channels
            if i < len(block_config) - 1:
                out_ch = int(ch * compression)
                self.transitions.append(Transition1D(ch, out_ch))
                ch = out_ch

        self.final_bn = nn.BatchNorm1d(ch)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(ch, 256)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = F.relu(self.final_bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = DenseNet1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DenseNet1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
