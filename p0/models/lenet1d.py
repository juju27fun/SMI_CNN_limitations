"""1D LeNet-5 adapted for particle signal classification.

Classic LeNet architecture: 2 conv layers + 3 FC layers.
Scaled to ~5.3M params via wider channels and large FC layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet1D(nn.Module):
    """LeNet-5 adapted for 1D signals."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2,
                 width_mult: float = 1.0):
        super().__init__()

        c1 = max(1, int(32 * width_mult))
        c2 = max(1, int(64 * width_mult))
        fc1_out = max(1, int(512 * width_mult))
        fc2_out = max(1, int(256 * width_mult))

        self.conv1 = nn.Conv1d(1, c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool2 = nn.MaxPool1d(2)

        flatten_size = c2 * (input_length // 2 // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, fc1_out)
        self.drop1 = nn.Dropout(dropout)
        self.feature_layer = nn.Linear(fc1_out, fc2_out)
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(fc2_out, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop2(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = LeNet1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LeNet1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
