"""Conv1D classifier for particle signals."""

import torch.nn as nn
import torch.nn.functional as F


class Conv1DClassifier(nn.Module):
    """1D Convolutional classifier for particle signals."""

    def __init__(self, input_length: int = 250, num_classes: int = 3, dropout: float = 0.2):
        super(Conv1DClassifier, self).__init__()

        # Conv1D layers with increasing channels and max pooling to reduce sequence length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(dropout)

        # Flatten size: 256 channels * (seq_len / 8) width
        flatten_size = 256 * (input_length // 2 // 2 // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, 256)
        self.feature_layer = self.fc1  # alias for model zoo compatibility
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input shape: (batch, seq_len) - 1D signal format
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(self.pool3(x))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x
