# Dataset Leak Detection Script

`dataset_leaks.py` analyzes a deep learning dataset for data leaks that can silently inflate model performance and produce unreliable results.

## Why data leaks matter

A data leak occurs when information from the test set is available to the model during training. This makes evaluation metrics **artificially high** — the model appears to perform well, but it has effectively "seen the answers" and will fail on truly unseen data.

## What the script checks

The script runs **5 checks** in order, from most critical to informational.

### 1. Source-level leaks (CRITICAL)

**What it detects:** Crops extracted from the same source recording appearing in both train and test.

**How it works:** Filenames follow the pattern `HFocusing_5_10_{class}_0_{sourceID}.npy{cropID}.npy`. The script parses out the `sourceID` and flags any source that has crops in both splits.

**Why it matters:** Crops from the same recording share noise characteristics, baseline drift, and signal artifacts. A model can memorize these source-specific patterns instead of learning the actual class differences, leading to over-optimistic test accuracy.

### 2. Exact content duplicates across splits (CRITICAL)

**What it detects:** Files with byte-identical content in both train and test.

**How it works:** Computes an MD5 hash of each `.npy` file's array data and checks for hash collisions between splits, per class.

**Why it matters:** If the model trains on the exact same sample it will be tested on, it can simply recall it from memory. This is the most direct form of data leak.

### 3. Near-duplicate detection (WARNING)

**What it detects:** Sample pairs across splits with very high cosine similarity.

**How it works:** For each class, loads all signals into matrices, L2-normalizes them, and computes the full cosine similarity matrix between train and test. Flags pairs above the threshold (default: 0.99). Pairs already caught as exact duplicates are excluded.

**Why it matters:** Near-identical samples (e.g. overlapping windows, minor noise variations) give the model an unfair advantage even if they are not byte-identical.

### 4. Intra-split duplicates (WARNING)

**What it detects:** Exact duplicate files within the same split.

**How it works:** Same MD5 hashing approach as check 2, but comparing files within each split individually.

**Why it matters:** Duplicates in train over-represent certain samples (biasing the model). Duplicates in test inflate the weight of those samples in accuracy metrics.

### 5. Distribution statistics (INFO)

**What it computes:** Per-split, per-class summary statistics — sample count, signal length, mean, standard deviation, min, and max.

**Why it matters:** Large distribution mismatches between splits can indicate a non-random or biased split. Also useful as a quick sanity check on the dataset.

## Usage

```bash
# Basic usage (assumes dataset/ in current directory)
python dataset_leaks.py

# Specify a custom dataset path
python dataset_leaks.py /path/to/dataset

# Change the cosine similarity threshold
python dataset_leaks.py --similarity-threshold 0.95

# Change where reports are saved
python dataset_leaks.py --report-dir my_reports

# Compare different split names
python dataset_leaks.py --splits train val
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `dataset` | `dataset` | Path to the dataset root directory |
| `--report-dir` | `leak_reports` | Directory where reports are saved |
| `--similarity-threshold` | `0.99` | Cosine similarity threshold for near-duplicate detection |
| `--splits` | `train test` | Names of the two splits to compare |

## Expected dataset structure

```
dataset/
  train/
    class_a/
      sample_001.npy
      sample_002.npy
    class_b/
      ...
  test/
    class_a/
      ...
    class_b/
      ...
```

The script auto-discovers all subdirectories as splits and all sub-subdirectories as classes. Files must be `.npy` (NumPy arrays).

## Output

### Console

A progress log with pass/fail status for each check and a final summary:

```
[1/5] Checking source-level leaks...
  CRITICAL: 105 shared source(s) across splits
[2/5] Checking exact content duplicates across splits...
  OK: no identical files across splits
...
==================================================
RESULT: CRITICAL leaks detected (105 total issues)
  -> Start with: leak_reports/summary.md
```

### Reports directory

Each check always writes a JSON file (even when no issues are found, to prevent stale data from previous runs). Markdown reports are only generated when issues exist; stale ones are cleaned up automatically.

| File | Format | Purpose |
|---|---|---|
| `summary.md` | Markdown | Top-level dashboard linking to all reports |
| `{check_name}.json` | JSON | Machine-readable data for scripting automated fixes (empty `[]` if no issues) |
| `{check_name}.md` | Markdown | Human-readable report grouped by class (only when issues > 0) |
| `stats.json` | JSON | Distribution statistics for all splits and classes |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | No critical leaks (warnings may exist) |
| `1` | Critical leaks detected |

## Part of the Full Analysis Pipeline

`dataset_leaks.py` is the first step of the `run_full_analysis.py` pipeline. When run through the pipeline, reports are saved to `<dataset_name>_full_analysis/leak_reports/` and a non-zero exit code (critical leaks) aborts the rest of the pipeline (noise analysis and training).

```bash
# Standalone
python dataset_leaks.py dataset

# As part of the full pipeline
python run_full_analysis.py dataset
```

## Dependencies

- Python 3.10+
- NumPy
