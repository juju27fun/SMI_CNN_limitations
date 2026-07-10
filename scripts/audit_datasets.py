#!/usr/bin/env python3
"""Lightweight dataset inventory for P0 data folders.

The script avoids modifying datasets. For .npy files it uses mmap where
possible, records shape/dtype, and samples values for basic sanity checks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def dataset_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 2 and rel.parts[0] in {"raw", "processed"}:
        return "/".join(rel.parts[:2])
    return rel.parts[0] if rel.parts else "."


def sample_array(arr: np.ndarray, max_values: int) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    if flat.size <= max_values:
        return flat
    idx = np.linspace(0, flat.size - 1, num=max_values, dtype=np.int64)
    return flat[idx]


def npy_metadata(path: Path, max_values: int) -> dict[str, Any]:
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        meta: dict[str, Any] = {
            "shape": "x".join(str(dim) for dim in arr.shape),
            "dtype": str(arr.dtype),
            "npy_error": "",
        }
        if np.issubdtype(arr.dtype, np.number):
            sample = sample_array(arr, max_values)
            meta.update(
                {
                    "sample_count": int(sample.size),
                    "sample_nan": int(np.isnan(sample).sum())
                    if np.issubdtype(arr.dtype, np.floating)
                    else 0,
                    "sample_inf": int(np.isinf(sample).sum())
                    if np.issubdtype(arr.dtype, np.floating)
                    else 0,
                    "sample_min": float(np.nanmin(sample)) if sample.size else None,
                    "sample_max": float(np.nanmax(sample)) if sample.size else None,
                    "sample_mean": float(np.nanmean(sample)) if sample.size else None,
                }
            )
        return meta
    except Exception as exc:  # noqa: BLE001 - audit should continue per file.
        return {"shape": "", "dtype": "", "npy_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", type=Path)
    parser.add_argument("--out-dir", default=Path("audit_artifacts/SMI_CNN_limitations/dataset_inventory"), type=Path)
    parser.add_argument("--max-values", default=2048, type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "files": 0,
            "bytes": 0,
            "extensions": Counter(),
            "shapes": Counter(),
            "dtypes": Counter(),
            "npy_errors": 0,
            "sample_nan": 0,
            "sample_inf": 0,
        }
    )

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root)
        ext = path.suffix.lower() or "<none>"
        name = dataset_name(path, root)
        stat = path.stat()
        row: dict[str, Any] = {
            "dataset": name,
            "path": str(rel_path),
            "bytes": stat.st_size,
            "extension": ext,
            "shape": "",
            "dtype": "",
            "sample_count": "",
            "sample_nan": "",
            "sample_inf": "",
            "sample_min": "",
            "sample_max": "",
            "sample_mean": "",
            "npy_error": "",
        }
        if ext == ".npy":
            row.update(npy_metadata(path, args.max_values))
        rows.append(row)

        summary = summaries[name]
        summary["files"] += 1
        summary["bytes"] += stat.st_size
        summary["extensions"][ext] += 1
        if row["shape"]:
            summary["shapes"][row["shape"]] += 1
        if row["dtype"]:
            summary["dtypes"][row["dtype"]] += 1
        if row["npy_error"]:
            summary["npy_errors"] += 1
        summary["sample_nan"] += int(row["sample_nan"] or 0)
        summary["sample_inf"] += int(row["sample_inf"] or 0)

    file_csv = out_dir / "files.csv"
    with file_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary_rows = []
    for name, summary in sorted(summaries.items()):
        summary_rows.append(
            {
                "dataset": name,
                "files": summary["files"],
                "bytes": summary["bytes"],
                "extensions": json.dumps(dict(summary["extensions"]), sort_keys=True),
                "shapes": json.dumps(dict(summary["shapes"].most_common()), sort_keys=True),
                "dtypes": json.dumps(dict(summary["dtypes"]), sort_keys=True),
                "npy_errors": summary["npy_errors"],
                "sample_nan": summary["sample_nan"],
                "sample_inf": summary["sample_inf"],
            }
        )

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as handle:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Wrote {file_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Datasets: {len(summary_rows)}")
    print(f"Files: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
