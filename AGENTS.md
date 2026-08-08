# P0 classification contract

- If `../workspace-repos.lock` exists, read `../AGENTS.md` first; Git roots do
  not inherit parent instructions.
- Put reusable code in `p0/` and CLIs in
  `scripts/{training,benchmarks,analysis,audit,datasets,plotting}`. Import
  `p0.models` and installed packages; never restore wrappers or inject paths.
- Resolve registered dataset IDs. Do not create local data/output trees, venvs,
  or caches. From the workspace root, write manifested runs only under
  `artifacts/SMI_CNN_limitations/<run-id>/`.
- For specialized work, read the matching `.agents/skills/*/SKILL.md`.
- Verify from the workspace root with
  `.venv/bin/python -m pytest -q SMI_CNN_limitations/tests` and CPU-only smoke
  checks. Before pfcalcul work, read the root current-state and runbook.
