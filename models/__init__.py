"""Model zoo for 1D particle signal classification.

All models follow the same interface:
    model = create_model(name, input_length=625, num_classes=4)
    output = model(x)  # x: (batch, 1, input_length) -> (batch, num_classes)

Every model exposes a `feature_layer` attribute (penultimate Linear) for
hook-based feature extraction in the training pipeline.
"""

from models.conv1d import Conv1DClassifier
from models.lenet1d import LeNet1D
from models.vgg1d import VGG1D
from models.resnet1d import ResNet1D
from models.inception1d import InceptionTime1D
from models.mobilenet1d import MobileNet1D
from models.efficientnet1d import EfficientNet1D
from models.densenet1d import DenseNet1D

MODEL_REGISTRY = {
    "Conv1D": Conv1DClassifier,
    "LeNet1D": LeNet1D,
    "VGG1D": VGG1D,
    "ResNet1D": ResNet1D,
    "InceptionTime1D": InceptionTime1D,
    "MobileNet1D": MobileNet1D,
    "EfficientNet1D": EfficientNet1D,
    "DenseNet1D": DenseNet1D,
}

# Scaling variants for accuracy-vs-capacity curves.
# Each entry: (model_class, variant-specific kwargs).
# "M" (base) entries use {} so default constructor args are preserved.
# Suffixes (smallest -> largest): -Pico, -Nano, -XXS, -XS, -S, (M = no suffix), -L
MODEL_VARIANTS = {
    # Conv1D family (width_mult: default=1.0)
    "Conv1D-Pico":       (Conv1DClassifier, {"width_mult": 0.0125}),
    "Conv1D-Nano":       (Conv1DClassifier, {"width_mult": 0.025}),
    "Conv1D-XXS":        (Conv1DClassifier, {"width_mult": 0.1}),
    "Conv1D-XS":         (Conv1DClassifier, {"width_mult": 0.25}),
    "Conv1D-S":          (Conv1DClassifier, {"width_mult": 0.5}),
    "Conv1D":            (Conv1DClassifier, {}),
    "Conv1D-L":          (Conv1DClassifier, {"width_mult": 2.0}),
    # LeNet1D family (width_mult: default=1.0)
    "LeNet1D-Pico":      (LeNet1D, {"width_mult": 0.01}),
    "LeNet1D-Nano":      (LeNet1D, {"width_mult": 0.025}),
    "LeNet1D-XXS":       (LeNet1D, {"width_mult": 0.1}),
    "LeNet1D-XS":        (LeNet1D, {"width_mult": 0.25}),
    "LeNet1D-S":         (LeNet1D, {"width_mult": 0.5}),
    "LeNet1D":           (LeNet1D, {}),
    "LeNet1D-L":         (LeNet1D, {"width_mult": 2.0}),
    # VGG1D family (width_mult: default=1.0)
    "VGG1D-Pico":        (VGG1D, {"width_mult": 0.0125}),
    "VGG1D-Nano":        (VGG1D, {"width_mult": 0.025}),
    "VGG1D-XXS":         (VGG1D, {"width_mult": 0.1}),
    "VGG1D-XS":          (VGG1D, {"width_mult": 0.25}),
    "VGG1D-S":           (VGG1D, {"width_mult": 0.5}),
    "VGG1D":             (VGG1D, {}),
    "VGG1D-L":           (VGG1D, {"width_mult": 2.0}),
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
