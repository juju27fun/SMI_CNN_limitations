"""Conv1D classifier for particle signals."""

import torch.nn as nn
import torch.nn.functional as F


class Conv1DClassifier(nn.Module):
    """1D Convolutional classifier for particle signals."""

    def __init__(self, input_length: int = 250, num_classes: int = 3, dropout: float = 0.2,
                 width_mult: float = 1.0):
        super(Conv1DClassifier, self).__init__()

        c1 = max(1, int(64 * width_mult))
        c2 = max(1, int(128 * width_mult))
        c3 = max(1, int(256 * width_mult))
        fc_hidden = max(1, int(256 * width_mult))

        # Conv1D layers with increasing channels and max pooling to reduce sequence length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(in_channels=c2, out_channels=c3, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(c3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(dropout)

        # Flatten size: c3 channels * (seq_len / 8) width
        flatten_size = c3 * (input_length // 2 // 2 // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, fc_hidden)
        self.feature_layer = self.fc1  # alias for model zoo compatibility
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

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
