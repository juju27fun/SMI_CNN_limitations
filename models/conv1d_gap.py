"""Conv1D classifier with Global Average Pooling head.

Identical backbone to :class:`Conv1DClassifier` (three Conv1D blocks with
BatchNorm, MaxPool and Dropout), but the flatten + fully-connected head is
replaced by a Global Average Pooling over the temporal axis before the
classifier. This produces a much smaller latent space (``c3`` features
instead of ``c3 * input_length / 8``), which dramatically reduces the FC
parameter budget and the total model size while preserving the same
scaling schedule via ``width_mult``.
"""

import torch.nn as nn
import torch.nn.functional as F


class Conv1DGAPClassifier(nn.Module):
    """1D Convolutional classifier with Global Average Pooling head."""

    def __init__(self, input_length: int = 250, num_classes: int = 3, dropout: float = 0.2,
                 width_mult: float = 1.0):
        super(Conv1DGAPClassifier, self).__init__()

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

        # Global Average Pooling collapses the temporal axis to 1, so the
        # latent feature vector has just `c3` features regardless of
        # `input_length`.
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(c3, fc_hidden)
        self.feature_layer = self.fc1  # alias for model zoo compatibility
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        # Input shape: (batch, 1, seq_len) - 1D signal format
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(self.pool3(x))

        x = self.gap(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x
