"""Benchmark family visual encodings.

Kept separate from the benchmark runner so plotting scripts can reuse the
benchmark2 symbolism without importing training dependencies such as torch.
"""

FAMILY_COLORS = {
    "Conv1D": "#0072B2",
    "Conv1DGAP": "#E6AB02",
    "ConvNeXt1D": "#882255",
    "DenseNet1D": "#999999",
    "EfficientNet1D": "#009E73",
    "InceptionTime1D": "#CC79A7",
    "LeNet1D": "#E69F00",
    "MobileNet1D": "#56B4E9",
    "ResNet1D": "#D55E00",
    "Swin1D": "#CC79A7",
    "PatchTST": "#332288",
    "PatchTSTPretrained": "#117733",
    "PatchTSTPretrained-Frozen": "#88CCEE",
    "VGG1D": "#000000",
}

FAMILY_MARKERS = {
    "Conv1D": "o",
    "Conv1DGAP": "h",
    "ConvNeXt1D": "p",
    "DenseNet1D": "s",
    "EfficientNet1D": "^",
    "InceptionTime1D": "D",
    "LeNet1D": "v",
    "MobileNet1D": "P",
    "ResNet1D": "X",
    "Swin1D": "h",
    "PatchTST": ">",
    "PatchTSTPretrained": "<",
    "PatchTSTPretrained-Frozen": "8",
    "VGG1D": "*",
}

FAMILY_LINESTYLES = {
    "Conv1D": "-",
    "Conv1DGAP": (0, (5, 1, 1, 1, 1, 1, 1, 1)),
    "ConvNeXt1D": (0, (3, 2, 1, 2)),
    "DenseNet1D": "--",
    "EfficientNet1D": "-.",
    "InceptionTime1D": ":",
    "LeNet1D": (0, (3, 1, 1, 1)),
    "MobileNet1D": (0, (5, 2)),
    "ResNet1D": (0, (1, 1, 1, 1, 5, 1)),
    "Swin1D": (0, (5, 1, 1, 1, 1, 1)),
    "PatchTST": (0, (2, 1, 1, 1, 1, 1)),
    "PatchTSTPretrained": "-",
    "PatchTSTPretrained-Frozen": "--",
    "VGG1D": (0, (3, 1, 1, 1, 1, 1)),
}
