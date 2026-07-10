# SMI CNN Limitations

Supervised 1D particle-classification baselines and model-family benchmarks.
Reusable code is in `p0/`; command-line programs are grouped by purpose under
`scripts/`.

## Workspace contract

This repository is one component of the parent research workspace. It does not
own a virtual environment, dataset copy, cache, or output tree.

- Environment: `../.venv`
- Registered datasets: `../datasets/`
- Run artifacts: `../artifacts/SMI_CNN_limitations/`
- Dataset generation and provenance owner: `../particles2SNR-pipeline/`

Use `workspace datasets list` and `workspace datasets resolve <id>` from the
workspace root. Common classification datasets include
`p0-baseline-3class@v1`, `p0-baseline-4class@v1`, and
`particles2snr-f-dual-clean-c1-class-folders@v1`.

## Development

From the workspace root:

```bash
.venv/bin/python -m pip install -e SMI_CNN_limitations
.venv/bin/python -m pytest -q SMI_CNN_limitations/tests
.venv/bin/python SMI_CNN_limitations/scripts/training/train4classes.py --help
```

Generated checkpoints, W&B files, figures, and reports must be directed to
`artifacts/SMI_CNN_limitations/<run-id>/`. Root-level compatibility entry points
were intentionally removed; use the grouped script path or import `p0.*`.
