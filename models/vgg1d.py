"""1D VGG-style network for particle signal classification.

VGG hallmarks: stacked 3x3 convolutions, BN, doubling channels, large FC head.
Scaled to ~5.3M params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_vgg_block(in_ch, out_ch, num_convs):
    """Stack of Conv1d(3)+BN+ReLU layers followed by MaxPool."""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv1d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class VGG1D(nn.Module):
    """VGG-style 1D classifier."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.3,
                 width_mult: float = 1.0):
        super().__init__()

        b1 = max(1, int(64 * width_mult))
        b2 = max(1, int(128 * width_mult))
        b3 = max(1, int(256 * width_mult))
        b4 = max(1, int(256 * width_mult))
        fc1_out = max(1, int(400 * width_mult))
        fc2_out = max(1, int(256 * width_mult))

        # Conv blocks: [in_ch, out_ch, num_convs]
        self.block1 = _make_vgg_block(1, b1, 2)      # -> /2
        self.block2 = _make_vgg_block(b1, b2, 2)     # -> /4
        self.block3 = _make_vgg_block(b2, b3, 3)     # -> /8
        self.block4 = _make_vgg_block(b3, b4, 3)     # -> /16

        flat_len = input_length // 2 // 2 // 2 // 2
        flatten_size = b4 * flat_len

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, fc1_out)
        self.drop1 = nn.Dropout(dropout)
        self.feature_layer = nn.Linear(fc1_out, fc2_out)
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc2_out, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop2(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = VGG1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VGG1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
