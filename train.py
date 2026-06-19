"""Compatibility wrapper for the base training CLI and legacy train imports."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from p0.data import (  # noqa: E402,F401
    RAW_SIGNAL_LENGTH,
    AdaptiveBandpassDecimate,
    AmplitudeScale,
    BandpassFilter,
    CenterCrop,
    Decimate,
    GaussianNoise,
    ParticleDataset,
    RealNoise,
    TimeMasking,
    TimeShift,
    Truncate,
)
from p0.training import evaluate, train_one_epoch  # noqa: E402,F401


def main():
    from scripts.training.train import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
