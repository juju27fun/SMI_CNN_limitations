"""Diagnose what the retrained long-input Conv1D-GAP-S is sensitive to.

The run compares detector-derived event intervals, fixed record-edge bands,
and remaining background using Grad-CAM density and regional masking.  These
are sensitivity diagnostics, not causal attributions or localization ground
truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from internship_workspace.visual_style import evidence_style, save_evidence_figure
from p0.data import BandpassFilter, Decimate
from p0.gradcam import event_concentration, regional_top_mask, temporal_gradcam, temporal_regions
from p0.models import create_model


CLASSES = ("2um", "4um", "10um")
RAW_LENGTH = 16_384
INPUT_LENGTH = 4_096
EDGE_FRACTION = 0.05
MASK_BUDGET = 0.02
SEED = 42
REGION_COLORS = {"event": "#009E73", "edge": "#D55E00", "background": "#7A7A7A"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(path: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True).stdout.strip()


def reference_intervals(labels_dir: Path, filename: str) -> list[tuple[float, float]]:
    path = labels_dir / f"{Path(filename).stem}.txt"
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            centre, width = float(fields[1]), float(fields[2])
            result.append((max(0.0, centre - width / 2), min(1.0, centre + width / 2)))
    return result


def load_population(test_dir: Path, labels_dir: Path) -> tuple[list[dict], torch.Tensor]:
    bandpass, decimate = BandpassFilter(), Decimate(4)
    records, signals = [], []
    for label, class_name in enumerate(CLASSES):
        for path in sorted((test_dir / class_name).glob("*.npy")):
            raw = np.load(path).astype(np.float32)
            if raw.shape != (RAW_LENGTH,):
                raise ValueError(f"unexpected shape {raw.shape}: {path}")
            processed = decimate(bandpass(torch.from_numpy(raw[None, :])))
            edge_width = int(INPUT_LENGTH * EDGE_FRACTION)
            edge_peak = float(torch.quantile(processed[0, torch.cat((torch.arange(edge_width), torch.arange(INPUT_LENGTH-edge_width, INPUT_LENGTH)))].abs(), .99))
            interior_peak = float(torch.quantile(processed[0, edge_width:-edge_width].abs(), .99)) or 1.0
            records.append({"filename": path.name, "label": label, "class_name": class_name, "intervals": reference_intervals(labels_dir, path.name), "edge_artifact_score": edge_peak / interior_peak})
            signals.append(processed)
    return records, torch.stack(signals)


def load_model(checkpoint: Path) -> torch.nn.Module:
    model = create_model("Conv1DGAP-S", input_length=INPUT_LENGTH, num_classes=3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def region_diagnostics(model: torch.nn.Module, signal: torch.Tensor, target: int, cam: np.ndarray, intervals: list[tuple[float, float]]) -> dict[str, dict[str, float | None]]:
    regions = temporal_regions(INPUT_LENGTH, intervals, EDGE_FRACTION)
    with torch.inference_mode():
        base_logit = float(model(signal)[0, target])
    output = {}
    total_cam = float(cam.sum())
    for name, mask in regions.items():
        coverage = float(mask.mean())
        if not mask.any():
            output[name] = {"coverage": 0.0, "cam_mass": None, "cam_density_enrichment": None, "target_logit_drop_top2pct": None}
            continue
        mass = float(cam[mask].sum() / total_cam) if total_cam > 0 else 0.0
        intervention = regional_top_mask(cam, mask, MASK_BUDGET)
        if not intervention.any():
            output[name] = {"coverage": coverage, "cam_mass": mass, "cam_density_enrichment": mass / coverage, "target_logit_drop_top2pct": None}
            continue
        masked = signal.clone()
        masked[..., torch.from_numpy(intervention)] = 0
        with torch.inference_mode():
            drop = base_logit - float(model(masked)[0, target])
        output[name] = {"coverage": coverage, "cam_mass": mass, "cam_density_enrichment": mass / coverage, "target_logit_drop_top2pct": drop}
    return output


def bootstrap_median(values: list[float], rng: np.random.Generator, draws: int = 2000) -> dict[str, object]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    samples = np.median(rng.choice(array, size=(draws, len(array)), replace=True), axis=1)
    return {"median": float(np.median(array)), "bootstrap_95_ci": np.quantile(samples, [.025, .975]).tolist(), "n": int(len(array))}


def select_cases(rows: list[dict]) -> list[tuple[str, dict]]:
    strata = [
        ("Single reference event · correct", [r for r in rows if len(r["intervals"]) == 1 and r["correct"]], lambda r: (r["confidence"], r["filename"]), max),
        ("Correct classification with two detected events", [r for r in rows if len(r["intervals"]) >= 2 and r["correct"]], lambda r: (r["margin"], r["filename"]), min),
        ("No reference interval · strongest edge transient", [r for r in rows if not r["intervals"] and r["correct"]], lambda r: (r["edge_artifact_score"], r["filename"]), max),
        ("Reference event · confident error", [r for r in rows if r["intervals"] and not r["correct"]], lambda r: (r["confidence"], r["filename"]), max),
        ("No reference interval · uncertain error", [r for r in rows if not r["intervals"] and not r["correct"]], lambda r: (r["margin"], r["filename"]), min),
    ]
    selected = []
    for title, candidates, key, chooser in strata:
        if not candidates:
            raise ValueError(f"empty deterministic case stratum: {title}")
        selected.append((title, chooser(candidates, key=key)))
    return selected


def render(metrics: dict, cases: list[tuple[str, dict]], destination: Path) -> None:
    with evidence_style():
        title, row = cases[1]
        concentration = metrics["selected_two_event_case"]["event_concentration"]
        fig, axis = plt.subplots(figsize=(11.2, 5.4))
        fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.68)
        signal = np.asarray(row["signal"])
        cam = np.asarray(row["cam"])
        x = np.arange(INPUT_LENGTH)
        scale = np.quantile(np.abs(signal), .995) or 1.0
        axis.plot(x, signal / scale, color="#222222", linewidth=.75)
        for start, end in row["intervals"]:
            axis.axvspan(start * INPUT_LENGTH, end * INPUT_LENGTH, color=REGION_COLORS["event"], alpha=.18, linewidth=0)
        axis.imshow(cam[None, :], extent=(0, INPUT_LENGTH, -1.58, -1.30), aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axis.set(xlim=(0, INPUT_LENGTH), ylim=(-1.62, 1.18), xlabel="Decimated sample", ylabel="Scaled OFI")
        axis.set_yticks([-1, 0, 1])
        axis.tick_params(labelsize=12)
        axis.xaxis.label.set_fontsize(12)
        axis.yaxis.label.set_fontsize(12)
        axis.grid(axis="x", alpha=.15, linewidth=.5)

        fig.suptitle(title, x=.09, y=.96, ha="left", fontsize=18, fontweight="bold")
        subtitle = fig.text(.09, .86, f"True class: {row['class_name']}   ·   Prediction: {CLASSES[row['pred']]}   ·   Confidence: {row['confidence']:.0%}", fontsize=12, color="#5B7088")
        subtitle.set_gid("secondary")
        fig.text(.98, .77, f"{concentration['temporal_coverage']:.1%} of signal  →  {concentration['cam_mass']:.1%} of Grad-CAM activation", ha="right", fontsize=18, fontweight="bold", color="#0B3558")
        save_evidence_figure(fig, destination, visual_form="narrative", dominant_object="complete 4096-sample signal with two detected-event windows and one Grad-CAM ribbon", regions=2, color_semantics={"#009E73": "detector-derived event window", "#0072B2": "retrained Conv1D-GAP-S Grad-CAM"}, dpi=220)
        fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source_workspace.resolve(), args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        metrics = json.loads((output / "metrics.json").read_text())
        metadata = json.loads((output / "cases.json").read_text()); arrays = np.load(output / "case_arrays.npz")
        cases = []
        for index, item in enumerate(metadata):
            row = dict(item["record"]); row["signal"] = arrays[f"signal_{index}"]; row["cam"] = arrays[f"cam_{index}"]; cases.append((item["title"], row))
        render(metrics, cases, output / "chapter7_two_event_gradcam.png"); return

    test_dir = source / "datasets/processed/doublet/v1/test"
    labels_dir = source / "datasets/processed/particles2snr-f-dual-clean-c1-yolo-3class/v1/test/labels"
    checkpoint = source / "artifacts/SMI_CNN_limitations/training/output/Conv1DGAP-S-dataset_doublet-train/best_model.pth"
    records, inputs = load_population(test_dir, labels_dir); model = load_model(checkpoint)
    rows = []
    for record, value in zip(records, inputs):
        signal = value.unsqueeze(0); probs, cam = temporal_gradcam(model, signal, model.pool3); pred = int(np.argmax(probs)); ordered = np.sort(probs)
        row = dict(record); row.update({"pred": pred, "confidence": float(probs[pred]), "margin": float(ordered[-1]-ordered[-2]), "correct": pred == record["label"], "cam": cam, "signal": value[0].numpy(), "regions": region_diagnostics(model, signal, pred, cam, record["intervals"])})
        rows.append(row)
    rng = np.random.default_rng(SEED); aggregate = {}
    for region in ("event", "background", "edge"):
        aggregate[region] = {}
        for metric in ("cam_density_enrichment", "target_logit_drop_top2pct"):
            aggregate[region][metric] = bootstrap_median([row["regions"][region][metric] for row in rows], rng)
    selected = select_cases(rows)
    selected_two_event = selected[1][1]
    metrics = {"schema_version": 1, "dataset": "doublet@v1", "n_test": len(rows), "model": "Conv1DGAP-S", "checkpoint": str(checkpoint.relative_to(source)), "input_contract": {"raw_samples": RAW_LENGTH, "decimation": 4, "model_samples": INPUT_LENGTH, "edge_band_fraction_each_side": EDGE_FRACTION, "regional_mask_budget": MASK_BUDGET}, "aggregate": aggregate, "selected_two_event_case": {"filename": selected_two_event["filename"], "true_class": selected_two_event["class_name"], "predicted_class": CLASSES[selected_two_event["pred"]], "confidence": selected_two_event["confidence"], "event_concentration": event_concentration(selected_two_event["cam"], selected_two_event["intervals"])}, "interpretation_boundary": "Detector-derived intervals are not ground truth. Grad-CAM diagnoses sensitivity concentration, not causal feature use; fixed-budget masking remains supporting evidence only."}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    cases_json = []
    arrays = {}
    for index, (title, row) in enumerate(selected):
        record = {key: value for key, value in row.items() if key not in {"signal", "cam", "regions"}}
        cases_json.append({"title": title, "record": record}); arrays[f"signal_{index}"] = row["signal"]; arrays[f"cam_{index}"] = row["cam"]
    (output / "cases.json").write_text(json.dumps(cases_json, indent=2, sort_keys=True) + "\n"); np.savez_compressed(output / "case_arrays.npz", **arrays)
    if not args.analysis_only: render(metrics, selected, output / "chapter7_two_event_gradcam.png")
    provenance = {"datasets": ["doublet@v1", "particles2snr-f-dual-clean-c1-yolo-3class@v1"], "inputs": {"checkpoint_sha256": sha256(checkpoint)}, "parameters": {"seed": SEED, "edge_fraction_each_side": EDGE_FRACTION, "regional_mask_budget": MASK_BUDGET, "input_length": INPUT_LENGTH}, "metric_definitions": {"cam_density_enrichment": "regional CAM mass fraction divided by regional time coverage", "target_logit_drop_top2pct": "predicted-class logit drop after zero masking the highest-CAM 2% of input positions available within each region", "selected_event_temporal_coverage": "union of detector-derived event-window samples divided by 4096", "selected_event_cam_mass": "Grad-CAM activation inside the union of detector-derived event windows divided by total Grad-CAM activation"}, "code": "scripts/analysis/analyze_long_input_regions.py", "git_revision": git_sha(Path(__file__).resolve().parents[2])}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metrics_manifest = {"schema_version": 1, "analysis_run_id": output.name, "computation_fingerprint": fingerprint, "computation_provenance": provenance, "metrics": [{"path": "metrics.json", "sha256": sha256(output / "metrics.json"), "computation_fingerprint": fingerprint}]}
    (output / "metrics_manifest.json").write_text(json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n")
    outputs = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "run.json")
    run = {"schema_version": 1, "project": "SMI_CNN_limitations", "run_id": output.name, "kind": "long_input_regional_gradcam", "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(), "training_performed": False, "dataset": "doublet@v1", "computation_fingerprint": fingerprint, "metric_provenance": provenance, "outputs": outputs, "output_sha256": {name: sha256(output/name) for name in outputs}}
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "aggregate": aggregate, "cases": [row["filename"] for _, row in selected]}, indent=2))


if __name__ == "__main__":
    main()
