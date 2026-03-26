# Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

# Install requirements
```bash
python -m pip install -r requirements.txt
```

# Notebook
Open the notebook in vscode, run cell by cell, and analyse and understand the code.
Specially, look at the data and the model

# Training
Run the training script :
```bash
python train.py
```
Check the training_plots.png figure
Play with the different hyperparameters (batch_size, lr, decimation, epochs)

# Benchmarking
Run the benchmark pipeline which wraps training with structured W&B metric logging, convergence tracking, early stopping, and full post-training evaluation:
```bash
# Basic benchmark (synthetic test set only)
python benchmark.py --data-dir dataset

# With real test set (measures generalization gap)
python benchmark.py --data-dir dataset --real-test-dir dataset_real/test

# With noise samples for OOD evaluation
python benchmark.py --data-dir dataset --noise-dir Noise

# Full benchmark: synthetic + real test + OOD noise
python benchmark.py --data-dir dataset --real-test-dir dataset_real/test --noise-dir Noise

# Custom hyperparameters and early stopping
python benchmark.py --data-dir dataset --epochs 200 --lr 1e-3 --patience 20

# Disable LR scheduler (constant LR) or use plateau scheduler
python benchmark.py --data-dir dataset --scheduler none
python benchmark.py --data-dir dataset --scheduler plateau

# Offline W&B mode (no internet required)
python benchmark.py --data-dir dataset --wandb-offline
```
The benchmark runs in 5 phases: (1) pre-training config logging, (2) training loop with per-epoch metrics, (3) post-training evaluation (confusion matrix, F1 per class, PR/ROC curves), (4) dimensionality reduction (PCA / t-SNE latent space visualizations), and (5) OOD noise evaluation (MSP, Energy, ODIN, Mahalanobis — AUROC, FPR@95, score histograms, ROC comparison). Phases 4–5 require `--noise-dir`. All metrics are logged to the W&B project `particle-benchmark`. See `Benchmarking.md` for the full W&B reference guide.

# Dataset Generation Interface
Launch the Streamlit UI to configure and generate OFI particle signal datasets:
```bash
streamlit run generate_ui.py
```
This opens a web interface (default http://localhost:8501) where you can configure signal presets, noise types, output directory, and generate datasets in **auto** mode (full train/test split) or **test** mode (3 samples per class for quick inspection).

# Noise Analysis
Generate a PDF report characterizing the noise in a folder of `.npy` signal files:
```bash
python analyze_noise.py ./Noise
python analyze_noise.py dataset/test/10um
python analyze_noise.py v_colored/2um --output custom_report.pdf
python analyze_noise.py ./Noise --fs 2000000 --segment 2500
```
The report includes amplitude statistics, PSD, spectral slope, frequency-band energy, noise-type classification, inter-file variability, and stationarity analysis.

# Dataset Leaks Analysis
Detect data leaks (source-level, exact duplicates, near-duplicates, intra-split duplicates) between train/test splits:
```bash
python dataset_leaks.py dataset
python dataset_leaks.py dataset --report-dir leak_reports --similarity-threshold 0.99
```
Detailed JSON reports are saved to the `leak_reports/` directory.

# Fix Leaks
If leaks are detected, run the fix script to remove leaked files and replace them with fresh data from a replacement database:
```bash
# Default paths (dataset in ./dataset, reports in ./leak_reports, DB in ./replacement_db)
python fix_leaks.py

# Custom paths
python fix_leaks.py --dataset-root /path/to/dataset --reports-dir /path/to/leak_reports --db-root /path/to/replacement_db
```
This script reads the reports from the reports directory, removes the offending files, and copies replacement files that don't introduce new leaks. After running it, re-run `dataset_leaks.py` to verify the dataset is clean.

# Dataset Audit (batch leak + noise analysis)
Run leak detection and noise analysis on **all** datasets at once. Already-processed datasets are skipped automatically (tracked via `audit_manifest.json`):
```bash
# Process all datasets found in the project directory
python run_dataset_audit.py

# Preview what would be processed without running anything
python run_dataset_audit.py --dry-run

# Force re-processing of all datasets
python run_dataset_audit.py --force

# Process only specific datasets
python run_dataset_audit.py --datasets dataset_f1 dataset_nof1
```
The script auto-discovers two kinds of directories:
- **Split datasets** (containing `train/` + `test/`): runs leak detection + noise analysis per split/class
- **Standalone noise folders** (containing `.npy` files directly, e.g. `Noise/`): runs noise analysis only

Results are saved to `audit_results/<dataset_name>/` with leak reports, noise PDFs, and a summary. If a dataset is modified (files added/removed), it will be re-processed on the next run.
