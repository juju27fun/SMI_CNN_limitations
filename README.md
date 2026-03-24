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

# Custom hyperparameters and early stopping
python benchmark.py --data-dir dataset --epochs 200 --lr 1e-3 --patience 20

# Offline W&B mode (no internet required)
python benchmark.py --data-dir dataset --wandb-offline
```
The benchmark runs in 3 phases: pre-training config logging, training loop with per-epoch metrics, and post-training evaluation (confusion matrix, F1 per class, PR/ROC curves). All metrics are logged to the W&B project `particle-benchmark`. See `Benchmarking.md` for the full W&B reference guide.

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
If leaks are detected, run the fix script to remove leaked files and replace them with fresh data from the download databases (`~/Downloads/<size>_DB`):
```bash
python fix_leaks.py
```
This script reads the reports from `leak_reports/`, removes the offending files, and copies replacement files that don't introduce new leaks. After running it, re-run `dataset_leaks.py` to verify the dataset is clean.

# Full Analysis Pipeline
Run the complete analysis pipeline (leak detection, noise analysis, training & benchmarking) in one command:
```bash
python run_full_analysis.py dataset
python run_full_analysis.py dataset --dataset-name synthetic_v2 --epochs 100
python run_full_analysis.py dataset --real-test-dir dataset_real/test --wandb-offline
```
This orchestrates `dataset_leaks.py`, `analyze_noise.py` (once per class per split), and `benchmark.py` sequentially. If critical leaks are detected, the pipeline aborts before training. All local results are saved to a `<dataset_name>_full_analysis/` folder containing leak reports, noise analysis PDFs, model checkpoints, and a summary report. Training metrics are logged to W&B as usual.
