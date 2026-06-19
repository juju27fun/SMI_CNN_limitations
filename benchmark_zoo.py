"""Compatibility wrapper for ``scripts/benchmarks/benchmark_zoo.py``."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    from scripts.benchmarks.benchmark_zoo import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
