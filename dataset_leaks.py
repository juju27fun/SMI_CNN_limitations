#!/usr/bin/env python3
"""Dataset leak detection script for deep learning datasets.

Analyzes a dataset folder with train/test splits for common data leaks:
  1. Source-level leaks (same source recording in both splits)
  2. Exact content duplicates across splits
  3. Near-duplicate samples across splits (cosine similarity)
  4. Intra-split duplicates
  5. Class distribution & signal statistics comparison
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ── Filename parsing ─────────────────────────────────────────────────────────

# Pattern: HFocusing_5_10_{class}_0_{sourceID}.npy{cropID}.npy
SOURCE_RE = re.compile(r"^(.+?)(\d+)\.npy(\d+)\.npy$")


def parse_source_id(filename: str) -> str | None:
    """Extract the source ID from a filename like HFocusing_5_10_4um_0_660.npy2035.npy."""
    m = SOURCE_RE.match(filename)
    if m:
        return f"{m.group(1)}{m.group(2)}.npy"  # e.g. HFocusing_5_10_4um_0_660.npy
    return None


# ── File discovery ───────────────────────────────────────────────────────────

def discover_dataset(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Return {split: {class: [file_paths]}} for all .npy files found."""
    dataset = {}
    if not root.is_dir():
        sys.exit(f"ERROR: dataset root not found: {root}")

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        dataset[split_name] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            files = sorted(class_dir.glob("*.npy"))
            if files:
                dataset[split_name][class_dir.name] = files
    return dataset


# ── Check 1: source-level leaks ─────────────────────────────────────────────

def check_source_leaks(dataset: dict, splits: tuple[str, str]) -> list[dict]:
    """Find source recordings that appear in both splits."""
    split_a, split_b = splits
    leaks = []

    classes_a = set(dataset.get(split_a, {}).keys())
    classes_b = set(dataset.get(split_b, {}).keys())

    for cls in sorted(classes_a & classes_b):
        sources_a = defaultdict(list)
        sources_b = defaultdict(list)

        for f in dataset[split_a][cls]:
            src = parse_source_id(f.name)
            if src:
                sources_a[src].append(f)

        for f in dataset[split_b][cls]:
            src = parse_source_id(f.name)
            if src:
                sources_b[src].append(f)

        shared = set(sources_a.keys()) & set(sources_b.keys())
        if shared:
            for src in sorted(shared):
                leaks.append({
                    "class": cls,
                    "source": src,
                    f"{split_a}_files": [str(f) for f in sources_a[src]],
                    f"{split_b}_files": [str(f) for f in sources_b[src]],
                })
    return leaks


# ── Check 2: exact content duplicates across splits ─────────────────────────

def hash_file(path: Path) -> str:
    data = np.load(path)
    return hashlib.md5(data.tobytes()).hexdigest()


def check_exact_duplicates(dataset: dict, splits: tuple[str, str]) -> list[dict]:
    """Find files with identical content across two splits."""
    split_a, split_b = splits
    dupes = []

    classes_a = set(dataset.get(split_a, {}).keys())
    classes_b = set(dataset.get(split_b, {}).keys())

    for cls in sorted(classes_a & classes_b):
        hashes_a = defaultdict(list)
        for f in dataset[split_a][cls]:
            hashes_a[hash_file(f)].append(f)

        for f in dataset[split_b][cls]:
            h = hash_file(f)
            if h in hashes_a:
                for fa in hashes_a[h]:
                    dupes.append({
                        "class": cls,
                        f"{split_a}_file": str(fa),
                        f"{split_b}_file": str(f),
                        "md5": h,
                    })
    return dupes


# ── Check 3: near-duplicate detection ───────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def check_near_duplicates(
    dataset: dict, splits: tuple[str, str], threshold: float = 0.99
) -> list[dict]:
    """Find sample pairs across splits with cosine similarity above threshold."""
    split_a, split_b = splits
    near_dupes = []

    classes_a = set(dataset.get(split_a, {}).keys())
    classes_b = set(dataset.get(split_b, {}).keys())

    for cls in sorted(classes_a & classes_b):
        files_a = dataset[split_a][cls]
        files_b = dataset[split_b][cls]

        # Load all signals into matrices for vectorized comparison
        sigs_a = np.array([np.load(f).flatten().astype(np.float64) for f in files_a])
        sigs_b = np.array([np.load(f).flatten().astype(np.float64) for f in files_b])

        # Normalize rows
        norms_a = np.linalg.norm(sigs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(sigs_b, axis=1, keepdims=True)
        norms_a[norms_a == 0] = 1.0
        norms_b[norms_b == 0] = 1.0
        sigs_a_normed = sigs_a / norms_a
        sigs_b_normed = sigs_b / norms_b

        # Cosine similarity matrix (a x b)
        sim_matrix = sigs_a_normed @ sigs_b_normed.T

        # Find pairs above threshold
        idxs = np.argwhere(sim_matrix >= threshold)
        for i, j in idxs:
            near_dupes.append({
                "class": cls,
                f"{split_a}_file": str(files_a[i]),
                f"{split_b}_file": str(files_b[j]),
                "cosine_similarity": round(float(sim_matrix[i, j]), 6),
            })
    return near_dupes


# ── Check 4: intra-split duplicates ─────────────────────────────────────────

def check_intra_duplicates(dataset: dict) -> list[dict]:
    """Find exact duplicates within each split."""
    dupes = []
    for split, classes in dataset.items():
        for cls, files in classes.items():
            seen = {}
            for f in files:
                h = hash_file(f)
                if h in seen:
                    dupes.append({
                        "split": split,
                        "class": cls,
                        "file_a": str(seen[h]),
                        "file_b": str(f),
                        "md5": h,
                    })
                else:
                    seen[h] = f
    return dupes


# ── Check 5: distribution stats ─────────────────────────────────────────────

def compute_class_stats(dataset: dict) -> dict:
    """Compute per-split per-class signal statistics."""
    stats = {}
    for split, classes in dataset.items():
        stats[split] = {}
        for cls, files in classes.items():
            signals = [np.load(f).astype(np.float64) for f in files]
            all_vals = np.concatenate([s.flatten() for s in signals])
            stats[split][cls] = {
                "count": len(files),
                "signal_length": int(signals[0].shape[-1]) if signals else 0,
                "mean": round(float(np.mean(all_vals)), 4),
                "std": round(float(np.std(all_vals)), 4),
                "min": round(float(np.min(all_vals)), 4),
                "max": round(float(np.max(all_vals)), 4),
            }
    return stats


# ── Report writing ───────────────────────────────────────────────────────────

def write_report(report_dir: Path, name: str, items: list[dict], header: str,
                 severity: str = "WARNING", description: str = ""):
    """Write a detailed leak report as JSON + Markdown."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # Always write JSON (even empty) so stale reports don't persist
    json_path = report_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(items, f, indent=2)

    if not items:
        # Remove stale markdown report if it exists
        md_path = report_dir / f"{name}.md"
        if md_path.exists():
            md_path.unlink()
        return

    severity_icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(severity, "⚪")
    md_path = report_dir / f"{name}.md"
    with open(md_path, "w") as f:
        f.write(f"# {severity_icon} {header}\n\n")
        f.write(f"> **Severity:** {severity} &nbsp;|&nbsp; "
                f"**Issues found:** {len(items)} &nbsp;|&nbsp; "
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        if description:
            f.write(f"{description}\n\n")
        f.write("---\n\n")

        # Group items by class (or split for intra-duplicates)
        group_key = "class" if "class" in items[0] else "split"
        groups = defaultdict(list)
        for item in items:
            groups[item.get(group_key, "unknown")].append(item)

        for group_name in sorted(groups.keys()):
            group_items = groups[group_name]
            f.write(f"## Class `{group_name}` ({len(group_items)} issue{'s' if len(group_items) != 1 else ''})\n\n")

            # Build a table from the item keys (excluding the group key)
            keys = [k for k in group_items[0].keys() if k != group_key]

            # Check if any value is a list — use expandable details instead of table
            has_lists = any(isinstance(v, list) for item in group_items for v in item.values())

            if has_lists:
                for i, item in enumerate(group_items, 1):
                    label_parts = [f"{k}: `{v}`" for k, v in item.items()
                                   if k != group_key and not isinstance(v, list)]
                    summary_label = " &nbsp;|&nbsp; ".join(label_parts) if label_parts else f"Issue #{i}"
                    f.write(f"<details>\n<summary><b>#{i}</b> &mdash; {summary_label}</summary>\n\n")
                    for k, v in item.items():
                        if k == group_key:
                            continue
                        if isinstance(v, list):
                            f.write(f"**{k}:**\n\n")
                            for entry in v:
                                f.write(f"- `{entry}`\n")
                            f.write("\n")
                    f.write("</details>\n\n")
            else:
                # Render as a markdown table
                col_headers = [_format_col_header(k) for k in keys]
                f.write("| # | " + " | ".join(col_headers) + " |\n")
                f.write("|--:" + "".join(f" | ---" for _ in keys) + " |\n")
                for i, item in enumerate(group_items, 1):
                    cols = []
                    for k in keys:
                        v = item.get(k, "")
                        cols.append(f"`{v}`" if isinstance(v, str) and "/" in str(v) else str(v))
                    f.write(f"| {i} | " + " | ".join(cols) + " |\n")
                f.write("\n")

        f.write("---\n\n")
        f.write(f"*Full data available in [`{name}.json`]({name}.json)*\n")


def _format_col_header(key: str) -> str:
    """Turn a snake_case key into a readable column header."""
    return key.replace("_", " ").title()


def write_summary_report(report_dir: Path, results: dict, stats: dict,
                         splits: tuple[str, str], threshold: float, root: Path):
    """Write a top-level summary.md linking to all individual reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / "summary.md"

    total = sum(r["count"] for r in results.values())
    has_critical = any(r["severity"] == "CRITICAL" and r["count"] > 0 for r in results.values())
    status = "🔴 CRITICAL leaks detected" if has_critical else (
        f"🟡 {total} warnings" if total > 0 else "🟢 No leaks detected"
    )

    with open(md_path, "w") as f:
        f.write("# Dataset Leak Detection Report\n\n")
        f.write(f"> **Status:** {status} &nbsp;|&nbsp; "
                f"**Total issues:** {total} &nbsp;|&nbsp; "
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|---|---|\n")
        f.write(f"| Dataset root | `{root.resolve()}` |\n")
        f.write(f"| Splits compared | `{splits[0]}` vs `{splits[1]}` |\n")
        f.write(f"| Similarity threshold | {threshold} |\n\n")

        # Results overview table
        f.write("## Results Overview\n\n")
        f.write("| Check | Severity | Issues | Report |\n")
        f.write("|---|---|--:|---|\n")
        for name, r in results.items():
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(r["severity"], "⚪")
            count = r["count"]
            link = f"[{name}.md]({name}.md)" if count > 0 else "—"
            f.write(f"| {r['label']} | {icon} {r['severity']} | {count} | {link} |\n")
        f.write("\n")

        # Distribution stats table
        f.write("## Distribution Statistics\n\n")
        f.write("| Split | Class | Count | Length | Mean | Std | Min | Max |\n")
        f.write("|---|---|--:|--:|--:|--:|--:|--:|\n")
        for split in splits:
            for cls in sorted(stats.get(split, {}).keys()):
                s = stats[split][cls]
                f.write(f"| {split} | {cls} | {s['count']} | {s['signal_length']} "
                        f"| {s['mean']:.4f} | {s['std']:.4f} "
                        f"| {s['min']:.4f} | {s['max']:.4f} |\n")
        f.write("\n---\n\n")
        f.write("*Reports generated by `dataset_leaks.py`*\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect data leaks in a deep learning dataset."
    )
    parser.add_argument(
        "dataset", type=str, nargs="?", default="dataset",
        help="Path to dataset root (default: dataset)",
    )
    parser.add_argument(
        "--report-dir", type=str, default="leak_reports",
        help="Directory to save detailed leak reports (default: leak_reports)",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.99,
        help="Cosine similarity threshold for near-duplicates (default: 0.99)",
    )
    parser.add_argument(
        "--splits", type=str, nargs=2, default=["train", "test"],
        help="Names of the two splits to compare (default: train test)",
    )
    args = parser.parse_args()

    root = Path(args.dataset)
    report_dir = Path(args.report_dir)
    splits = tuple(args.splits)
    has_critical = False

    print(f"Dataset Leak Detection")
    print(f"======================")
    print(f"Root:       {root.resolve()}")
    print(f"Splits:     {splits[0]} vs {splits[1]}")
    print(f"Reports:    {report_dir.resolve()}")
    print(f"Threshold:  {args.similarity_threshold}")
    print()

    # Discover dataset
    dataset = discover_dataset(root)
    if len(dataset) < 2:
        sys.exit(f"ERROR: expected at least 2 splits, found: {list(dataset.keys())}")

    for split_name in splits:
        if split_name not in dataset:
            sys.exit(f"ERROR: split '{split_name}' not found in {root}")

    # Track results for summary
    results = {}

    # ── Check 1: source-level leaks ──────────────────────────────────────
    print("[1/5] Checking source-level leaks...")
    source_leaks = check_source_leaks(dataset, splits)
    results["source_leaks"] = {
        "label": "Source-level leaks", "severity": "CRITICAL", "count": len(source_leaks),
    }
    if source_leaks:
        has_critical = True
        n_sources = len(set(l["source"] for l in source_leaks))
        print(f"  CRITICAL: {n_sources} shared source(s) across splits "
              f"({len(source_leaks)} entries)")
        by_class = defaultdict(int)
        for l in source_leaks:
            by_class[l["class"]] += 1
        for cls, count in sorted(by_class.items()):
            print(f"    {cls}: {count} shared source(s)")
    else:
        print("  OK: no shared sources between splits")
    write_report(
        report_dir, "source_leaks", source_leaks,
        "Source-Level Leaks",
        severity="CRITICAL",
        description=(
            "Crops from the **same source recording** appear in both splits. "
            "These samples share noise characteristics, baseline drift, and "
            "signal artifacts, allowing the model to memorize source-specific "
            "patterns instead of learning class features.\n\n"
            "**How to fix:** Split the dataset by source recording, not by "
            "individual crop. All crops from one source must stay in the same split."
        ),
    )

    # ── Check 2: exact content duplicates ────────────────────────────────
    print("[2/5] Checking exact content duplicates across splits...")
    exact_dupes = check_exact_duplicates(dataset, splits)
    results["exact_duplicates"] = {
        "label": "Exact content duplicates", "severity": "CRITICAL", "count": len(exact_dupes),
    }
    if exact_dupes:
        has_critical = True
        print(f"  CRITICAL: {len(exact_dupes)} identical file(s) across splits")
    else:
        print("  OK: no identical files across splits")
    write_report(
        report_dir, "exact_duplicates", exact_dupes,
        "Exact Content Duplicates Across Splits",
        severity="CRITICAL",
        description=(
            "Files with **byte-identical content** exist in both splits. "
            "The model has seen these exact samples during training and will "
            "trivially predict them at test time, inflating accuracy.\n\n"
            "**How to fix:** Remove duplicates from one split (preferably test)."
        ),
    )

    # ── Check 3: near-duplicates ─────────────────────────────────────────
    print(f"[3/5] Checking near-duplicates (threshold={args.similarity_threshold})...")
    near_dupes = check_near_duplicates(dataset, splits, args.similarity_threshold)
    exact_pairs = set()
    for d in exact_dupes:
        exact_pairs.add((d[f"{splits[0]}_file"], d[f"{splits[1]}_file"]))
    near_dupes_only = [
        nd for nd in near_dupes
        if (nd[f"{splits[0]}_file"], nd[f"{splits[1]}_file"]) not in exact_pairs
    ]
    results["near_duplicates"] = {
        "label": "Near-duplicate pairs", "severity": "WARNING", "count": len(near_dupes_only),
    }
    if near_dupes_only:
        print(f"  WARNING: {len(near_dupes_only)} near-duplicate pair(s) across splits")
    else:
        print("  OK: no near-duplicates above threshold")
    write_report(
        report_dir, "near_duplicates", near_dupes_only,
        f"Near-Duplicate Pairs (cosine >= {args.similarity_threshold})",
        severity="WARNING",
        description=(
            "Sample pairs across splits have **very high cosine similarity**, "
            "suggesting they may be near-copies (e.g. overlapping windows from "
            "the same recording, or augmented versions of the same sample).\n\n"
            "**How to fix:** Investigate the flagged pairs. If they originate "
            "from the same measurement, ensure source-level splitting."
        ),
    )

    # ── Check 4: intra-split duplicates ──────────────────────────────────
    print("[4/5] Checking intra-split duplicates...")
    intra_dupes = check_intra_duplicates(dataset)
    results["intra_duplicates"] = {
        "label": "Intra-split duplicates", "severity": "WARNING", "count": len(intra_dupes),
    }
    if intra_dupes:
        print(f"  WARNING: {len(intra_dupes)} duplicate(s) within splits")
        by_split = defaultdict(int)
        for d in intra_dupes:
            by_split[d["split"]] += 1
        for split, count in sorted(by_split.items()):
            print(f"    {split}: {count} duplicate(s)")
    else:
        print("  OK: no duplicates within splits")
    write_report(
        report_dir, "intra_duplicates", intra_dupes,
        "Intra-Split Exact Duplicates",
        severity="WARNING",
        description=(
            "Identical files exist **within the same split**. In train, this "
            "over-represents certain samples; in test, it inflates metrics.\n\n"
            "**How to fix:** Deduplicate by removing the extra copy."
        ),
    )

    # ── Check 5: distribution stats ──────────────────────────────────────
    print("[5/5] Computing distribution statistics...")
    stats = compute_class_stats(dataset)

    all_classes = sorted(
        set().union(*(classes.keys() for classes in dataset.values()))
    )
    print()
    print(f"  {'Split':<8} {'Class':<8} {'Count':>6} {'Length':>7} "
          f"{'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for split in splits:
        for cls in all_classes:
            s = stats.get(split, {}).get(cls, {})
            if not s:
                continue
            print(f"  {split:<8} {cls:<8} {s['count']:>6} {s['signal_length']:>7} "
                  f"{s['mean']:>10.4f} {s['std']:>10.4f} "
                  f"{s['min']:>10.4f} {s['max']:>10.4f}")

    # Save stats JSON
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── Write summary report ─────────────────────────────────────────────
    write_summary_report(report_dir, results, stats, splits,
                         args.similarity_threshold, root)

    # ── Console summary ──────────────────────────────────────────────────
    print()
    print("=" * 50)
    total_issues = sum(r["count"] for r in results.values())
    if has_critical:
        print(f"RESULT: CRITICAL leaks detected ({total_issues} total issues)")
        print(f"Detailed reports saved to: {report_dir.resolve()}/")
        print(f"  -> Start with: {report_dir.resolve()}/summary.md")
        sys.exit(1)
    elif total_issues > 0:
        print(f"RESULT: {total_issues} warnings (no critical leaks)")
        print(f"Detailed reports saved to: {report_dir.resolve()}/")
        print(f"  -> Start with: {report_dir.resolve()}/summary.md")
        sys.exit(0)
    else:
        print("RESULT: No leaks detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
