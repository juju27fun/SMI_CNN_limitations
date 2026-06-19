"""Classifier wrappers for P1 transformer backbones.

The P1 Swin/PatchTST implementations expose a detection-backbone contract:
``forward(x) -> list[(B, C, T_i)]``.  P0 expects a classifier contract:
``forward(x) -> logits`` plus a ``feature_layer`` attribute for hooks.

These wrappers keep the transformer implementation single-sourced in P1 and
adapt only the final pooling/classification head for P0 experiments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_P1_ROOT = Path(__file__).resolve().parents[2] / "P1"
if str(_P1_ROOT) not in sys.path:
    sys.path.insert(0, str(_P1_ROOT))

from detseg.models.backbones.patchtst1d import PatchTST1DBackbone  # noqa: E402
from detseg.models.backbones.swin1d import Swin1DBackbone  # noqa: E402


class _PyramidBackboneClassifier(nn.Module):
    """Global-pooled classifier on top of a P1 feature pyramid."""

    def __init__(self, backbone: nn.Module, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_layer = nn.Linear(backbone.out_channels, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        deepest = feats[-1]
        pooled = F.adaptive_avg_pool1d(deepest, 1).squeeze(-1)
        features = F.gelu(self.feature_layer(pooled))
        return self.classifier(self.dropout(features))


class Swin1DClassifier(_PyramidBackboneClassifier):
    """Swin-1D classifier using the P1 hierarchical transformer backbone."""

    def __init__(
        self,
        input_length: int = 625,
        num_classes: int = 4,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        **_: object,
    ) -> None:
        del input_length
        backbone = Swin1DBackbone(
            embed_dim=embed_dim,
            proj_channels=256,
            drop_path_rate=0.1,
        )
        super().__init__(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)


class PatchTSTClassifier(_PyramidBackboneClassifier):
    """PatchTST classifier using the P1 transformer backbone."""

    def __init__(
        self,
        input_length: int = 625,
        num_classes: int = 4,
        embed_dim: int = 128,
        depth: int = 6,
        hidden_dim: int = 128,
        **_: object,
    ) -> None:
        del input_length
        backbone = PatchTST1DBackbone(
            embed_dim=embed_dim,
            depth=depth,
            proj_channels=256,
        )
        super().__init__(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)
