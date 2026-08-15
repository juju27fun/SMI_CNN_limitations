"""Proposal-aware ROI classifier with event and conditional class heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from p0.models.conv1d_gap import Conv1DGAPClassifier


class ProposalAwareROIClassifier(nn.Module):
    """Conv1DGAP-S encoder shared by eventness and particle-class heads."""

    def __init__(
        self,
        *,
        input_length: int = 6144,
        num_classes: int = 3,
        width_mult: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = Conv1DGAPClassifier(
            input_length=input_length,
            num_classes=num_classes,
            width_mult=width_mult,
        )
        feature_dim = self.encoder.fc1.out_features
        self.encoder.fc2 = nn.Identity()
        self.event_head = nn.Linear(feature_dim, 1)
        self.class_head = nn.Linear(feature_dim, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(inputs)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        return self.event_head(features).squeeze(-1), self.class_head(features)
