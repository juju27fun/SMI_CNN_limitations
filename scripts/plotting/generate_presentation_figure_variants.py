"""Generate CNN-only and transformer-inclusive presentation figure variants.

The variants are meant for slide animations: first show the CNN-only view, then
replace it with the matching view that includes PatchTST and Swin1D.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from p0 import benchmarking as bz  # noqa: E402
from p0 import plot_doublet_comparison as doublet_cmp  # noqa: E402
from p0 import plot_realtime_factor_doublet as doublet_rt  # noqa: E402


TRANSFORMER_FAMILIES = {"PatchTST", "Swin1D"}
BENCHMARK2_DIR = PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/benchmark2"
DOUBLET_DIR = PROJECT_ROOT.parent / "artifacts/SMI_CNN_limitations/benchmarks/doublet_3fam_retrained"
CNN_DOUBLET_FAMILIES = ["EfficientNet1D", "ResNet1D", "Conv1DGAP"]
WITH_TRANSFORMER_DOUBLET_FAMILIES = [
    "Swin1D",
    "PatchTST",
    "EfficientNet1D",
    "ResNet1D",
    "Conv1DGAP",
]
BENCHMARK2_VARIANT_PDFS = [
    "pareto.pdf",
    "pareto_latency.pdf",
    "pareto_size.pdf",
    "scaling_grid.pdf",
    "scaling_grid_size.pdf",
    "tier_grid.pdf",
]


def _family_from_run(run: dict) -> str:
    if "model_family" in run:
        return run["model_family"]
    return bz.get_family(run["model_name"])


def _current_pdf_names(figures_dir: Path) -> list[str]:
    names = []
    for path in sorted(figures_dir.glob("*.pdf")):
        stem = path.stem
        if stem.endswith("_cnn_only") or stem.endswith("_with_transformers"):
            continue
        names.append(path.name)
    return names


def _copy_filtered_runs(src_runs: Path, dst_runs: Path, *, include_transformers: bool) -> int:
    dst_runs.mkdir(parents=True, exist_ok=True)
    kept = 0
    for src in sorted(src_runs.glob("*.json")):
        with open(src) as f:
            run = json.load(f)
        is_transformer = _family_from_run(run) in TRANSFORMER_FAMILIES
        if is_transformer and not include_transformers:
            continue
        shutil.copy2(src, dst_runs / src.name)
        kept += 1
    return kept


def _render_benchmark2_variant(*, include_transformers: bool, suffix: str) -> None:
    figures_dir = BENCHMARK2_DIR / "figures"
    with tempfile.TemporaryDirectory(prefix=f"p0_benchmark2_{suffix}_") as tmp_name:
        tmp_output = Path(tmp_name) / "benchmark2"
        kept = _copy_filtered_runs(
            BENCHMARK2_DIR / "runs",
            tmp_output / "runs",
            include_transformers=include_transformers,
        )
        print(f"[benchmark2:{suffix}] rendering from {kept} run JSONs")

        bz.generate_plots(tmp_output, wandb_run=None)
        bz.generate_tier_robustness(tmp_output, wandb_run=None)
        bz.generate_tier_grid(tmp_output, wandb_run=None)
        bz.generate_tier6_domain_gap(tmp_output, wandb_run=None)
        bz.generate_scaling_curves(tmp_output, wandb_run=None)
        bz.generate_scaling_grid(tmp_output, wandb_run=None)
        bz.generate_scaling_grid(
            tmp_output,
            x_col="params",
            x_label="Parameters",
            fname="scaling_grid_size.pdf",
            wandb_run=None,
        )
        bz.generate_pareto_publication(tmp_output, wandb_run=None)
        bz.generate_pareto_latency_focus(tmp_output, wandb_run=None)
        bz.generate_pareto_publication(
            tmp_output,
            x_col="size_mb",
            x_label="Model size (MB)",
            x_log=True,
            fname="pareto_size.pdf",
            wandb_run=None,
        )

        tmp_figures = tmp_output / "figures"
        for name in BENCHMARK2_VARIANT_PDFS:
            src = tmp_figures / name
            if not src.exists():
                print(f"[benchmark2:{suffix}] skipped missing {name}")
                continue
            dst = figures_dir / f"{Path(name).stem}_{suffix}.pdf"
            shutil.copy2(src, dst)
            print(f"[benchmark2:{suffix}] wrote {dst}")


def _render_doublet_realtime(entries: list[dict], suffix: str) -> None:
    fig = doublet_rt._make_figure(entries)
    out = doublet_rt._save_pdf(
        fig,
        DOUBLET_DIR,
        f"{Path(doublet_rt.OUT_NAME).stem}_{suffix}.pdf",
    )
    print(f"[doublet:{suffix}] wrote {out}")


def _render_doublet_comparison(families: list[str], suffix: str) -> None:
    acc_before, acc_after, lat_before, lat_after = doublet_cmp._load_comparison_data()
    fig = doublet_cmp._make_figure(
        acc_before,
        acc_after,
        lat_before,
        lat_after,
        families=families,
    )
    out = doublet_cmp._save_pdf(
        fig,
        DOUBLET_DIR,
        f"doublet_comparison_acc_latency_{suffix}.pdf",
    )
    print(f"[doublet:{suffix}] wrote {out}")


def _render_doublet_variants() -> None:
    all_entries = doublet_rt._load_entries()
    cnn_entries = [
        entry for entry in all_entries
        if entry["family"] not in TRANSFORMER_FAMILIES
    ]
    _render_doublet_realtime(cnn_entries, "cnn_only")
    _render_doublet_realtime(all_entries, "with_transformers")
    _render_doublet_comparison(CNN_DOUBLET_FAMILIES, "cnn_only")
    _render_doublet_comparison(WITH_TRANSFORMER_DOUBLET_FAMILIES, "with_transformers")


def main() -> int:
    _render_benchmark2_variant(include_transformers=False, suffix="cnn_only")
    _render_benchmark2_variant(include_transformers=True, suffix="with_transformers")
    _render_doublet_variants()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
