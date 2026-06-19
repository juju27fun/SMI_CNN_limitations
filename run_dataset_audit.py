"""Compatibility wrapper for ``scripts/audit/run_dataset_audit.py``."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    from scripts.audit.run_dataset_audit import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
