# SMI CNN Limitations agent context

- Treat this directory as an independent Git repository inside the parent
  workspace; also obey the workspace-root `AGENTS.md`.
- Reusable classification code belongs in `p0/`; CLIs belong in the existing
  `scripts/{training,benchmarks,analysis,audit,datasets,plotting}/` groups.
- Import models from `p0.models` and shared helpers from installed packages. Do
  not restore root wrappers or add `sys.path` manipulation.
- Resolve input through the workspace dataset registry. Do not create `data/`,
  `outputs/`, a local venv, or a project-local model cache.
- Write runs to `artifacts/SMI_CNN_limitations/<run-id>/` with a `run.json`.
- Run `.venv/bin/python -m pytest -q SMI_CNN_limitations/tests` from the
  workspace root. Use only small CPU smoke checks for organizational changes.
- Before pfcalcul work, read `docs/operations/pfcalcul/current-state.md` and
  `docs/operations/pfcalcul/runbook.md` at the workspace root.
