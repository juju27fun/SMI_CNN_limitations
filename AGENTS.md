# P0 Codex Instructions

## Project Snapshot

- P0 is the particle-classification benchmark project.
- Core workflow: train 1D neural classifiers, track runs with W&B, compare model families, and produce publication-quality figures.
- Main W&B project: `particle-benchmark`.
- Important outputs live under `output/`, `results/`, `audit_results/`, `wandb/`, and `artifacts/`; treat them as generated or experimental unless the task explicitly targets them.
- P1 intentionally imports profiling helpers from `P0/training_utils.py`; preserve this compatibility when editing profiling utilities.

## Source Priority

- Prefer current code, tests, and scripts over old `.claude` documents.
- Use this file for project-wide behavior and `.agents/skills/` for focused workflows.
- Load only the smallest relevant skill for the task.
- Treat `.claude/` as historical migration context, not active operating instructions.
- Do not bulk-read W&B runs, artifacts, generated outputs, datasets, or result folders unless the task needs them.

## Collaboration And Efficiency

- Use focused skills instead of loading broad historical context.
- For broad investigations, split independent questions by domain: training, W&B, metrics, benchmarking, or publication figures.
- Use parallel exploration only for independent read-only questions.
- Keep implementation local when the next step depends directly on the result.
- Do not duplicate context across skills; link work back to the smallest relevant skill.

## Project Areas

- Training entry points include `train.py`, `train3classes_proof.py`, `train4classes.py`, and benchmark scripts such as `benchmark_zoo.py` / `benchmark_base.py`.
- Model code lives under `models/`.
- Shared training and profiling utilities live in `training_utils.py`.
- Plotting, reporting, and remeasurement helpers live mostly under `scripts/` and `archive/`.
- Tests live under `tests/` when present.
- Dataset folders under `data/` and `C1_HF_*` may be large; inspect metadata or small samples first.

## Commands

- Install dependencies: `pip install -r requirements.txt`.
- Run tests: `pytest tests`.
- Focused test: `pytest tests/<file>.py -q`.
- Train or benchmark commands vary by script; inspect `--help` before launching a long run.
- W&B online runs require authentication; use offline or dry-run modes when available for validation.

## W&B And Metrics

- Required metric names use `/` namespaces such as `train/loss`, `val/accuracy`, and `test_synthetic/accuracy`.
- Use snake_case metric components. Avoid hyphens, spaces, and commas.
- Required per-epoch keys: `epoch`, `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `epoch_time_sec`, and `learning_rate` when a scheduler exists.
- Required summary keys include `best_val_accuracy`, `best_epoch`, `total_training_time_sec`, `convergence_time_sec`, `final_val_accuracy`, `final_val_loss`, `model_size_params`, and `dataset_size`.
- W&B run config must include model name, parameter count, dataset, dataset size, epochs, batch size, learning rate, optimizer, seed, and patience when available.

## Publication Figures

- Save paper figures as vector PDF only. Do not create PNG/JPG siblings for paper outputs.
- Use fixed canvas sizes and explicit margins; do not use `tight_layout()` or `constrained_layout()`.
- Do not use `ax.set_title()` for paper figures; titles belong in LaTeX captions.
- Use Okabe-Ito or project constants for categorical color, marker, and linestyle encodings.
- Confusion matrices and heatmaps must be annotated and include a labelled colorbar.
- Retrace plots locally from raw exported data. Do not rely on W&B native image exports for publication PDFs.

## Verification

- For code changes, run the narrowest relevant tests first.
- For training-loop changes, run a smoke or very small epoch count if available.
- For W&B changes, validate config keys, metric names, summaries, and offline behavior.
- For plotting changes, regenerate the smallest target figure and inspect output file type/path.
- If a full benchmark is too expensive, report the skipped command and the residual risk.

## Skill Map

- `training`: PyTorch training loops, validation, checkpoints, convergence, parameter counts, and profiling helper compatibility.
- `wandb`: W&B initialization, metric logging, summaries, artifacts, sweeps, and reports.
- `metrics`: canonical classification, OOD, config, and naming conventions.
- `benchmarking`: multi-model benchmark orchestration, comparison, leaderboards, and result exports.
- `publication-figures`: paper-quality plotting rules, canvas sizes, palettes, matrices, Pareto plots, and review checklist.

## Done Criteria

- Summarize changed files and behavior.
- Report commands run and results.
- Mention skipped checks, expensive validations not run, and remaining risks.
