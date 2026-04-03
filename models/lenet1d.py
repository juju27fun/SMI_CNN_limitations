"""1D LeNet-5 adapted for particle signal classification.

Classic LeNet architecture: 2 conv layers + 3 FC layers.
Scaled to ~5.3M params via wider channels and large FC layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet1D(nn.Module):
    """LeNet-5 adapted for 1D signals."""

    def __init__(self, input_length: int = 625, num_classes: int = 4, dropout: float = 0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        flatten_size = 64 * (input_length // 2 // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, 512)
        self.drop1 = nn.Dropout(dropout)
        self.feature_layer = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

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
