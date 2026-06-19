"""Compatibility wrapper for ``scripts/datasets/generate_ui.py``."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    from scripts.datasets import generate_ui as _module

    return getattr(_module, "main", lambda: None)()


if __name__ == "__main__":
    raise SystemExit(main())
