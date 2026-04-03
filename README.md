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

# 4-Class Training with Model Zoo
Train using any of the 8 model architectures on a 4-class dataset (2um, 4um, 10um, Noise) with optional OOD evaluation and cluster distance analysis:
```bash
# Basic 4-class training (Conv1D, default)
python train4classes.py --data-dir S1_white_4c --epochs 150 --wandb-offline

# Use a different model from the zoo
python train4classes.py --model ResNet1D --data-dir S1_white_4c --epochs 150
python train4classes.py --model InceptionTime1D --data-dir S2_colored_4c --epochs 150

# With OOD evaluation (MSP, Energy, ODIN, Mahalanobis, Energy_tuned)
python train4classes.py --model Conv1D --data-dir S1_white_4c --noise-dir Noise

# With real test set (measures generalization gap)
python train4classes.py --model ResNet1D --data-dir S1_white_4c --real-test-dir dataset_real/test

# Full pipeline: model zoo + OOD + cluster distances + generalization gap
python train4classes.py --model EfficientNet1D --data-dir S1_white_4c --noise-dir Noise --real-test-dir dataset_real/test

# Custom hyperparameters, early stopping, offline W&B
python train4classes.py --model VGG1D --data-dir S1_white_4c --epochs 200 --lr 1e-3 --patience 20 --wandb-offline

# With data augmentation (training set only)
python train4classes.py --data-dir S1_white_4c --augment
python train4classes.py --data-dir S1_white_4c --augment --aug-snr 15 --aug-scale-min 0.7 --aug-scale-max 1.3

# With different optimizers
python train4classes.py --data-dir S1_white_4c --optimizer adamw --weight-decay 0.01
python train4classes.py --data-dir S1_white_4c --optimizer sgd --lr 0.01 --momentum 0.9
```
Available models: Conv1D, LeNet1D, VGG1D, ResNet1D, InceptionTime1D, MobileNet1D, EfficientNet1D, DenseNet1D (all ~5.3M params, see `architecture_et_entrainement.md` for details).

When `--noise-dir` is provided, the pipeline runs OOD evaluation (5 methods with AUROC/FPR@95/AUPR, score histograms, ROC comparison, temperature sweep, per-class analysis, silhouette score) and cluster distance analysis (cosine distance heatmap between class centroids). All metrics are logged to the W&B project `particle-benchmark`.

> **Note:** The legacy 3-class benchmark script (`benchmark.py`) and the standalone cluster distance script (`compute_cluster_distances.py`) have been moved to `archive/`. All their functionality is available in `train4classes.py`.

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
