"""Classifier wrappers for the installed ``detseg`` transformer backbones.

The P1 Swin/PatchTST implementations expose a detection-backbone contract:
``forward(x) -> list[(B, C, T_i)]``.  P0 expects a classifier contract:
``forward(x) -> logits`` plus a ``feature_layer`` attribute for hooks.

These wrappers keep the transformer implementation single-sourced in P1 and
adapt only the final pooling/classification head for P0 experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from detseg.models.backbones.patchtst1d import PatchTST1DBackbone
from detseg.models.backbones.patchtst_pretrained import PatchTSTPretrained1DBackbone
from detseg.models.backbones.swin1d import Swin1DBackbone


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
        depths: tuple[int, int, int] = (2, 2, 2),
        num_heads: tuple[int, int, int] = (2, 4, 8),
        proj_channels: int = 256,
        hidden_dim: int = 128,
        drop_path_rate: float = 0.1,
        **_: object,
    ) -> None:
        del input_length
        backbone = Swin1DBackbone(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            proj_channels=proj_channels,
            drop_path_rate=drop_path_rate,
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
        num_heads: int = 4,
        proj_channels: int = 256,
        hidden_dim: int = 128,
        **_: object,
    ) -> None:
        del input_length
        backbone = PatchTST1DBackbone(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            proj_channels=proj_channels,
        )
        super().__init__(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)


class PatchTSTPretrainedClassifier(_PyramidBackboneClassifier):
    """PatchTST HF-pretrained classifier for P0 comparisons.

    ``finetune_mode="linear_probe"`` freezes the HF encoder and trains the
    projection/head. ``finetune_mode="full"`` fine-tunes the encoder too.
    """

    def __init__(
        self,
        input_length: int = 625,
        num_classes: int = 4,
        hidden_dim: int = 128,
        finetune_mode: str = "full",
        cache_dir: str | Path | None = None,
        **_: object,
    ) -> None:
        backbone = PatchTSTPretrained1DBackbone(
            input_length=input_length,
            finetune_mode=finetune_mode,
            cache_dir=cache_dir,
        )
        super().__init__(backbone=backbone, num_classes=num_classes, hidden_dim=hidden_dim)
        self.pretrained_metadata = backbone.pretrained_metadata
