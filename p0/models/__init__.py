"""Model zoo for 1D particle signal classification.

All models follow the same interface:
    model = create_model(name, input_length=625, num_classes=4)
    output = model(x)  # x: (batch, 1, input_length) -> (batch, num_classes)

Every model exposes a `feature_layer` attribute (penultimate Linear) for
hook-based feature extraction in the training pipeline.
"""

from p0.models.conv1d_gap import Conv1DGAPClassifier
from p0.models.convnext1d import ConvNeXt1D
from p0.models.resnet1d import ResNet1D
from p0.models.inception1d import InceptionTime1D
from p0.models.mobilenet1d import MobileNet1D
from p0.models.efficientnet1d import EfficientNet1D
from p0.models.densenet1d import DenseNet1D
from p0.models.transformer1d_classifiers import (
    PatchTSTClassifier,
    PatchTSTPretrainedClassifier,
    Swin1DClassifier,
)

MODEL_REGISTRY = {
    "Conv1DGAP": Conv1DGAPClassifier,
    "ConvNeXt1D": ConvNeXt1D,
    "ResNet1D": ResNet1D,
    "InceptionTime1D": InceptionTime1D,
    "MobileNet1D": MobileNet1D,
    "EfficientNet1D": EfficientNet1D,
    "DenseNet1D": DenseNet1D,
    "Swin1D": Swin1DClassifier,
    "PatchTST": PatchTSTClassifier,
    "PatchTSTPretrained": PatchTSTPretrainedClassifier,
    "PatchTSTPretrained-Frozen": PatchTSTPretrainedClassifier,
}

# Scaling variants for accuracy-vs-capacity curves.
# Each entry: (model_class, variant-specific kwargs).
# "M" (base) entries use {} so default constructor args are preserved.
# Suffixes (smallest -> largest): -Pico, -Nano, -XXS, -XS, -S, (M = no suffix), -L
MODEL_VARIANTS = {
    # Conv1DGAP family (width_mult: default=1.0) — Conv1D backbone with
    # Global Average Pooling instead of flatten, yielding a much smaller
    # FC head and input-length agnostic inference.
    "Conv1DGAP-Pico":    (Conv1DGAPClassifier, {"width_mult": 0.0125}),
    "Conv1DGAP-Nano":    (Conv1DGAPClassifier, {"width_mult": 0.025}),
    "Conv1DGAP-XXS":     (Conv1DGAPClassifier, {"width_mult": 0.1}),
    "Conv1DGAP-XS":      (Conv1DGAPClassifier, {"width_mult": 0.25}),
    "Conv1DGAP-S":       (Conv1DGAPClassifier, {"width_mult": 0.5}),
    "Conv1DGAP":         (Conv1DGAPClassifier, {}),
    "Conv1DGAP-L":       (Conv1DGAPClassifier, {"width_mult": 2.0}),
    # ConvNeXt1D family (base_dim: default=42)
    "ConvNeXt1D-Nano":   (ConvNeXt1D, {"base_dim": 2}),
    "ConvNeXt1D-XXS":    (ConvNeXt1D, {"base_dim": 5}),
    "ConvNeXt1D-XS":     (ConvNeXt1D, {"base_dim": 10}),
    "ConvNeXt1D-S":      (ConvNeXt1D, {"base_dim": 21}),
    "ConvNeXt1D":        (ConvNeXt1D, {}),
    "ConvNeXt1D-L":      (ConvNeXt1D, {"base_dim": 84}),
    # ResNet1D family (base_width: default=74)
    "ResNet1D-Nano":     (ResNet1D, {"base_width": 2}),
    "ResNet1D-XXS":      (ResNet1D, {"base_width": 7}),
    "ResNet1D-XS":       (ResNet1D, {"base_width": 18}),
    "ResNet1D-S":        (ResNet1D, {"base_width": 37}),
    "ResNet1D":          (ResNet1D, {}),
    "ResNet1D-L":        (ResNet1D, {"base_width": 148}),
    # MobileNet1D family (width_mult: default=1.5)
    "MobileNet1D-Nano":  (MobileNet1D, {"width_mult": 0.0375, "last_ch": 32}),
    "MobileNet1D-XXS":   (MobileNet1D, {"width_mult": 0.15, "last_ch": 96}),
    "MobileNet1D-XS":    (MobileNet1D, {"width_mult": 0.375, "last_ch": 240}),
    "MobileNet1D-S":     (MobileNet1D, {"width_mult": 0.75}),
    "MobileNet1D":       (MobileNet1D, {}),
    "MobileNet1D-L":     (MobileNet1D, {"width_mult": 3.0}),
    # EfficientNet1D family (width_mult: default=0.85)
    "EfficientNet1D-Nano": (EfficientNet1D, {"width_mult": 0.02, "head_ch": 32}),
    "EfficientNet1D-XXS":  (EfficientNet1D, {"width_mult": 0.085, "head_ch": 96}),
    "EfficientNet1D-XS":   (EfficientNet1D, {"width_mult": 0.21, "head_ch": 256}),
    "EfficientNet1D-S":  (EfficientNet1D, {"width_mult": 0.425}),
    "EfficientNet1D":    (EfficientNet1D, {}),
    "EfficientNet1D-L":  (EfficientNet1D, {"width_mult": 1.7}),
    # InceptionTime1D family (num_filters: default=148, bottleneck_size: default=64)
    "InceptionTime1D-Nano": (InceptionTime1D, {"num_filters": 4, "bottleneck_size": 2}),
    "InceptionTime1D-XXS":  (InceptionTime1D, {"num_filters": 12, "bottleneck_size": 4}),
    "InceptionTime1D-XS":   (InceptionTime1D, {"num_filters": 32, "bottleneck_size": 8}),
    "InceptionTime1D-S": (InceptionTime1D, {"num_filters": 74, "bottleneck_size": 32}),
    "InceptionTime1D":   (InceptionTime1D, {}),
    "InceptionTime1D-L": (InceptionTime1D, {"num_filters": 296, "bottleneck_size": 128}),
    # DenseNet1D family (growth_rate: default=40, init_channels: default=64)
    "DenseNet1D-Nano":   (DenseNet1D, {"growth_rate": 2, "init_channels": 4}),
    "DenseNet1D-XXS":    (DenseNet1D, {"growth_rate": 4, "init_channels": 8}),
    "DenseNet1D-XS":     (DenseNet1D, {"growth_rate": 10, "init_channels": 16}),
    "DenseNet1D-S":      (DenseNet1D, {"growth_rate": 20, "init_channels": 32}),
    "DenseNet1D":        (DenseNet1D, {}),
    "DenseNet1D-L":      (DenseNet1D, {"growth_rate": 80, "init_channels": 128}),
    # Transformer families imported from P1 backbones.  Variants scale both
    # token width and classification head width while keeping the same patch
    # geometry as the base models, so Tier-1 scaling remains comparable.
    "Swin1D-Nano":       (Swin1DClassifier, {"embed_dim": 16, "depths": (1, 1, 1), "num_heads": (1, 2, 4), "proj_channels": 64, "hidden_dim": 64, "drop_path_rate": 0.02}),
    "Swin1D-XXS":        (Swin1DClassifier, {"embed_dim": 24, "depths": (1, 1, 2), "num_heads": (1, 2, 4), "proj_channels": 96, "hidden_dim": 64, "drop_path_rate": 0.04}),
    "Swin1D-XS":         (Swin1DClassifier, {"embed_dim": 32, "depths": (1, 2, 2), "num_heads": (1, 2, 4), "proj_channels": 128, "hidden_dim": 96, "drop_path_rate": 0.06}),
    "Swin1D-S":          (Swin1DClassifier, {"embed_dim": 48, "depths": (2, 2, 2), "num_heads": (2, 4, 8), "proj_channels": 192, "hidden_dim": 128, "drop_path_rate": 0.08}),
    "Swin1D":            (Swin1DClassifier, {}),
    "Swin1D-L":          (Swin1DClassifier, {"embed_dim": 96, "depths": (2, 3, 3), "num_heads": (3, 6, 12), "proj_channels": 384, "hidden_dim": 192, "drop_path_rate": 0.15}),
    "PatchTST-Nano":     (PatchTSTClassifier, {"embed_dim": 32, "depth": 2, "num_heads": 2, "proj_channels": 64, "hidden_dim": 64}),
    "PatchTST-XXS":      (PatchTSTClassifier, {"embed_dim": 48, "depth": 3, "num_heads": 3, "proj_channels": 96, "hidden_dim": 64}),
    "PatchTST-XS":       (PatchTSTClassifier, {"embed_dim": 64, "depth": 4, "num_heads": 4, "proj_channels": 128, "hidden_dim": 96}),
    "PatchTST-S":        (PatchTSTClassifier, {"embed_dim": 96, "depth": 5, "num_heads": 4, "proj_channels": 192, "hidden_dim": 128}),
    "PatchTST-Compact":  (PatchTSTClassifier, {"embed_dim": 96}),
    "PatchTST":          (PatchTSTClassifier, {}),
    "PatchTST-L":        (PatchTSTClassifier, {"embed_dim": 192, "depth": 8, "num_heads": 6, "proj_channels": 384, "hidden_dim": 192}),
    "PatchTSTPretrained-Frozen": (PatchTSTPretrainedClassifier, {"finetune_mode": "linear_probe"}),
    "PatchTSTPretrained": (PatchTSTPretrainedClassifier, {"finetune_mode": "full"}),
}

# Auto-build family map by stripping size suffix
_SIZE_SUFFIXES = ("-Pico", "-Nano", "-XXS", "-XS", "-S", "-L")
FAMILY_MAP = {}
for _variant_name in MODEL_VARIANTS:
    _family = _variant_name
    for _suffix in _SIZE_SUFFIXES:
        if _variant_name.endswith(_suffix):
            _family = _variant_name[: -len(_suffix)]
            break
    FAMILY_MAP[_variant_name] = _family


def get_family(name: str) -> str:
    """Return the family name for a model/variant name."""
    return FAMILY_MAP.get(name, name)


def list_families() -> list[str]:
    """Return sorted list of unique family names."""
    return sorted(set(FAMILY_MAP.values()))


def create_model(name: str, input_length: int = 625, num_classes: int = 4, **kwargs):
    """Instantiate a model by name from the registry or variant table.

    Args:
        name: Model name (key in MODEL_REGISTRY or MODEL_VARIANTS).
        input_length: Length of the 1D input signal.
        num_classes: Number of output classes.
        **kwargs: Additional model-specific arguments (override variant defaults).

    Returns:
        nn.Module instance.
    """
    # Check variants first (includes base names)
    if name in MODEL_VARIANTS:
        cls, variant_kwargs = MODEL_VARIANTS[name]
        merged = {**variant_kwargs, **kwargs}
        return cls(input_length=input_length, num_classes=num_classes, **merged)

    # Fall back to base registry
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](input_length=input_length, num_classes=num_classes, **kwargs)

    available = ", ".join(sorted(set(list(MODEL_REGISTRY.keys()) + list(MODEL_VARIANTS.keys()))))
    raise ValueError(f"Unknown model '{name}'. Available: {available}")


def list_models(include_variants: bool = False) -> list[str]:
    """Return sorted list of available model names.

    Args:
        include_variants: If True, include S/L variants alongside base models.
    """
    if include_variants:
        return sorted(set(list(MODEL_REGISTRY.keys()) + list(MODEL_VARIANTS.keys())))
    return sorted(MODEL_REGISTRY.keys())
