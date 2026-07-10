"""1D ConvNeXt for particle signal classification.

ConvNeXt-style architecture adapted for 1D signals: patchify stem, depthwise
convolutions with large kernels, inverted bottleneck MLP blocks, LayerNorm,
GELU activation, and layer scale. Scaled to ~5.3M params via ``base_dim``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm1d(nn.Module):
    """Channel-first LayerNorm for (B, C, L) tensors."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None] * x + self.bias[:, None]


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt block: DWConv -> LayerNorm -> 1x1 expand -> GELU -> 1x1 project.

    Includes layer scale (learnable per-channel scaling of the residual branch,
    initialised to a small value) for training stability.
    """

    def __init__(self, dim, drop_path=0.0, kernel_size=7, mlp_ratio=4,
                 layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim * mlp_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * mlp_ratio, dim, 1)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path_rate = drop_path

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None] * x
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1).item() < self.drop_path_rate:
                return shortcut
        return shortcut + x


class ConvNeXt1D(nn.Module):
    """ConvNeXt 1D classifier.

    4-stage architecture with [3, 3, 9, 3] blocks (ConvNeXt-Tiny layout).
    Channel widths double at each stage: [d, 2d, 4d, 8d] where d = base_dim.
    """

    def __init__(self, input_length: int = 625, num_classes: int = 4,
                 dropout: float = 0.2, base_dim: int = 42,
                 depths=(3, 3, 9, 3), kernel_size=7):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(len(depths))]

        # Patchify stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm1d(dims[0]),
        )

        # Build stages with inter-stage downsampling
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        total_blocks = sum(depths)
        block_idx = 0

        for i in range(len(depths)):
            if i > 0:
                self.downsamples.append(nn.Sequential(
                    LayerNorm1d(dims[i - 1]),
                    nn.Conv1d(dims[i - 1], dims[i], 2, stride=2),
                ))
            else:
                self.downsamples.append(nn.Identity())

            blocks = []
            for j in range(depths[i]):
                dp = 0.1 * block_idx / max(total_blocks - 1, 1)
                blocks.append(ConvNeXtBlock1D(dims[i], drop_path=dp,
                                              kernel_size=kernel_size))
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

        # Head
        self.norm = LayerNorm1d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Linear(dims[-1], 256)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for ds, stage in zip(self.downsamples, self.stages):
            x = ds(x)
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.feature_layer(x))
        x = self.drop(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ConvNeXt1D(input_length=625, num_classes=4)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvNeXt1D: {total:,} params")
    x = torch.randn(2, 1, 625)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
