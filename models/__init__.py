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


def create_model(name: str, input_length: int = 625, num_classes: int = 4, **kwargs):
    """Instantiate a model by name from the registry.

    Args:
        name: Model name (key in MODEL_REGISTRY).
        input_length: Length of the 1D input signal.
        num_classes: Number of output classes.
        **kwargs: Additional model-specific arguments.

    Returns:
        nn.Module instance.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](input_length=input_length, num_classes=num_classes, **kwargs)


def list_models():
    """Return sorted list of available model names."""
    return sorted(MODEL_REGISTRY.keys())
