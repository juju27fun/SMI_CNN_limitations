"""
run_filter_combinations.py
--------------------------
Runs training for every combination of the 3 bandpass filters (8 total)
and writes a summary report.

Filters:
  F1 – Generation filter  (Butterworth 7–80 kHz)  applied during data generation
  F2 – Training filter     (FFT 5–100 kHz)         applied at training time
  F3 – Notebook filter     (FFT 8–40 kHz)          applied at training time

Combinations (subsets of {F1, F2, F3}):
  0 filters: {}
  1 filter : {F1}, {F2}, {F3}
  2 filters: {F1,F2}, {F1,F3}, {F2,F3}
  3 filters: {F1,F2,F3}

Usage:
    python run_filter_combinations.py [--epochs 150] [--seed 42]
"""

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path


FILTERS = ["F1", "F2", "F3"]
DATASET_WITH_F1 = "./dataset_f1"
DATASET_NO_F1 = "./dataset_nof1"
OUTPUT_ROOT = Path("./filter_study_outputs")
RESULTS_FILE = OUTPUT_ROOT / "results.json"


def powerset(items):
    """All subsets of `items` (from empty set to full set)."""
    result = []
    for r in range(len(items) + 1):
        for combo in combinations(items, r):
            result.append(set(combo))
    return result


def generate_dataset(output_dir: str, with_f1: bool, force: bool = True):
    """Generate a dataset, optionally skipping the generation bandpass (F1)."""
    cmd = [
        sys.executable, "generate_dataset.py", "auto",
        "--output", output_dir,
    ]
    if force:
        cmd.append("--force")
    if not with_f1:
        cmd.append("--no-filter")

    label = "WITH F1" if with_f1 else "WITHOUT F1"
    print(f"\n{'='*60}")
    print(f"  Generating dataset {label} → {output_dir}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Dataset generation failed for {output_dir}")
    print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)


def run_training(combo_name: str, data_dir: str, filter_training: bool,
                 filter_notebook: bool, epochs: int, seed: int):
    """Run train.py with selected filters. Returns (test_acc, test_loss)."""
    output_dir = OUTPUT_ROOT / combo_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        "--data-dir", data_dir,
        "--output-dir", str(output_dir),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if filter_training:
        cmd.append("--filter-training")
    if filter_notebook:
        cmd.append("--filter-notebook")

    print(f"\n{'─'*60}")
    print(f"  Training: {combo_name}")
    print(f"  Filters : F2={'ON' if filter_training else 'OFF'}  F3={'ON' if filter_notebook else 'OFF'}")
    print(f"  Dataset : {data_dir}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        print(f"STDOUT:\n{stdout}")
        raise RuntimeError(f"Training failed for {combo_name}")

    # Parse test accuracy and loss from stdout
    test_acc = None
    test_loss = None
    best_val_acc = None
    for line in stdout.splitlines():
        if "Test Loss:" in line and "Test Accuracy:" in line:
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Test Loss:"):
                    test_loss = float(part.split(":")[1].strip())
                if part.startswith("Test Accuracy:"):
                    test_acc = float(part.split(":")[1].strip())
        if "Best validation accuracy:" in line:
            best_val_acc = float(line.split(":")[1].strip())

    # Print last lines of output
    lines = stdout.strip().splitlines()
    for line in lines[-15:]:
        print(f"  {line}")

    return {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": best_val_acc,
    }


def write_report(results: list, report_path: Path):
    """Write the markdown report."""
    # Sort by test accuracy descending
    ranked = sorted(results, key=lambda r: r["test_accuracy"] or 0, reverse=True)

    lines = [
        "# Filter Combination Study — Report",
        "",
        "## Objective",
        "",
        "Determine which combination of the 3 bandpass filters yields the best",
        "classification accuracy for the Conv1D particle classifier.",
        "",
        "## Filters",
        "",
        "| ID | Name | Type | Band | Stage |",
        "|----|------|------|------|-------|",
        "| F1 | Generation filter | Butterworth (scipy) | 7–80 kHz | Data generation |",
        "| F2 | Training filter | FFT bandpass (torch) | 5–100 kHz | Training transform |",
        "| F3 | Notebook filter | FFT bandpass (torch) | 8–40 kHz | Training transform |",
        "",
        "## Results",
        "",
        "| Rank | Combination | Filters active | Test Accuracy | Test Loss | Best Val Accuracy |",
        "|------|------------|----------------|---------------|-----------|-------------------|",
    ]

    for i, r in enumerate(ranked, 1):
        active = ", ".join(sorted(r["filters"])) if r["filters"] else "(none)"
        acc = f"{r['test_accuracy']:.4f}" if r["test_accuracy"] is not None else "N/A"
        loss = f"{r['test_loss']:.4f}" if r["test_loss"] is not None else "N/A"
        val = f"{r['best_val_accuracy']:.4f}" if r["best_val_accuracy"] is not None else "N/A"
        marker = " **best**" if i == 1 else ""
        lines.append(f"| {i} | {r['name']} | {active} | {acc}{marker} | {loss} | {val} |")

    best = ranked[0]
    lines += [
        "",
        "## Conclusion",
        "",
        f"The best filter combination is **{best['name']}** "
        f"(filters: {', '.join(sorted(best['filters'])) if best['filters'] else 'none'}) "
        f"with a test accuracy of **{best['test_accuracy']:.4f}**.",
        "",
        "### Observations",
        "",
    ]

    # Group by number of filters for analysis
    by_count = {}
    for r in ranked:
        n = len(r["filters"])
        by_count.setdefault(n, []).append(r)

    for n in sorted(by_count):
        accs = [r["test_accuracy"] for r in by_count[n] if r["test_accuracy"] is not None]
        if accs:
            avg = sum(accs) / len(accs)
            lines.append(f"- **{n} filter(s)**: average test accuracy = {avg:.4f}")

    lines += [
        "",
        "---",
        f"*Generated automatically by `run_filter_combinations.py`*",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all filter combinations")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs per run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_combos = powerset(FILTERS)
    print(f"Will run {len(all_combos)} filter combinations, {args.epochs} epochs each.\n")

    # Step 1 — generate the two datasets (with/without F1)
    needs_f1 = any("F1" in c for c in all_combos)
    needs_nof1 = any("F1" not in c for c in all_combos)

    if needs_f1:
        generate_dataset(DATASET_WITH_F1, with_f1=True)
    if needs_nof1:
        generate_dataset(DATASET_NO_F1, with_f1=False)

    # Step 2 — run training for each combination
    results = []
    total_start = time.time()

    for combo in all_combos:
        combo_name = "_".join(sorted(combo)) if combo else "no_filter"
        data_dir = DATASET_WITH_F1 if "F1" in combo else DATASET_NO_F1

        metrics = run_training(
            combo_name=combo_name,
            data_dir=data_dir,
            filter_training="F2" in combo,
            filter_notebook="F3" in combo,
            epochs=args.epochs,
            seed=args.seed,
        )

        results.append({
            "name": combo_name,
            "filters": sorted(combo),
            **metrics,
        })

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All {len(results)} runs completed in {total_time:.0f}s")
    print(f"{'='*60}")

    # Save raw JSON results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Raw results saved to {RESULTS_FILE}")

    # Step 3 — write the report
    write_report(results, Path("filter_study_report.md"))


if __name__ == "__main__":
    main()
