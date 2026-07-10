# Project structure

- `p0/`: importable training, model, plotting, and benchmark implementation.
- `scripts/`: grouped command wrappers and one-off analysis/dataset utilities.
- `tests/`: source and CLI contract tests.
- `docs/`: project methods and tracked documentation assets.
- `archive/`: historical source retained for provenance, not active imports.

Datasets, environments, caches, and experiment outputs are owned by the parent
workspace. Use `../datasets`, `../.venv`, `../.cache`, and
`../artifacts/SMI_CNN_limitations`; do not add compatibility symlinks or local
payload directories here.
