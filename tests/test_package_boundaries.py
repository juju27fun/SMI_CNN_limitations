"""Package boundary checks for active source files."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_scripts_do_not_import_reusable_symbols_from_train_shim():
    offenders = []
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from train import" in text:
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
