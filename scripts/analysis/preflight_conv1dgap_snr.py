"""Preflight a Conv1DGAP checkpoint before accuracy-vs-SNR evaluation.

The goal is to catch scientific mismatches before producing an SNR threshold:
class order, checkpoint head size, training metadata, raw signal length, and
preprocessing/input-length alignment.
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

from p0.models import create_model  # noqa: E402
from p0.data import RAW_SIGNAL_LENGTH  # noqa: E402

from p0.snr_utils import load_checkpoint_state  # noqa: E402


DEFAULT_CLASS_NAMES = ["2um", "4um", "10um"]
DEFAULT_BANDPASS = {"low_khz": 5.0, "high_khz": 100.0, "sample_rate_mhz": 2.0}


def parse_run_tag(tag):
    out = {"run_tag": tag}
    dataset_pos = tag.find("-dataset")
    if dataset_pos > 0:
        out["model_name"] = tag[:dataset_pos]
    model_match = re.match(r"(?P<model>.+?)(?:-k(?P<kernel>\d+))?(?:-L(?P<length>\d+))?(?:-decim)?-dataset", tag)
    if model_match:
        out["model_name"] = model_match.group("model")
        if model_match.group("kernel"):
            out["kernel_size"] = int(model_match.group("kernel"))
        if model_match.group("length"):
            out["input_length_from_tag"] = int(model_match.group("length"))
    tier_match = re.search(r"-tier(?P<tier>\d+)-seed(?P<seed>\d+)$", tag)
    if tier_match:
        out["tier"] = int(tier_match.group("tier"))
        out["seed"] = int(tier_match.group("seed"))
    out["decim_sweep"] = "-decim-" in tag or tag.endswith("-decim")
    return out


def find_run_metadata(checkpoint_path):
    tag = checkpoint_path.parent.name
    candidates = [
        WORKSPACE_ROOT / "artifacts" / "SMI_CNN_limitations" / "benchmark2" / "runs" / f"{tag}.json",
        checkpoint_path.parents[2] / "runs" / f"{tag}.json",
        checkpoint_path.parent.with_suffix(".json"),
    ]
    for path in candidates:
        if path.exists():
            return path, json.loads(path.read_text())
    return None, None


def infer_checkpoint_head_classes(state):
    for key in ("fc2.weight", "classifier.weight", "head.weight"):
        value = state.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
    return None


def checkpoint_compatibility(checkpoint_path, model_name, input_length, num_classes, kernel_size=None):
    state = load_checkpoint_state(str(checkpoint_path))
    kwargs = {}
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size
    model = create_model(model_name, input_length=input_length, num_classes=num_classes, **kwargs)
    model_state = model.state_dict()
    compatible = []
    skipped = []
    for key, value in state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible.append(key)
        else:
            skipped.append(key)
    return {
        "total_keys": len(state),
        "compatible_keys": len(compatible),
        "skipped_keys": len(skipped),
        "compatible_ratio": (len(compatible) / len(state)) if state else 0.0,
        "checkpoint_out_classes": infer_checkpoint_head_classes(state),
        "skipped_key_names": skipped[:20],
    }


def audit_dataset(data_dir, class_names, decimate):
    data_dir = Path(data_dir)
    classes = {}
    all_lengths = []
    for class_name in class_names:
        class_dir = data_dir / class_name
        files = sorted(class_dir.glob("*.npy")) if class_dir.exists() else []
        lengths = []
        dtypes = Counter()
        for path in files:
            arr = np.load(path, mmap_mode="r")
            lengths.append(int(arr.shape[-1]))
            dtypes[str(arr.dtype)] += 1
        all_lengths.extend(lengths)
        classes[class_name] = {
            "exists": class_dir.exists(),
            "n_files": len(files),
            "raw_length_min": min(lengths) if lengths else None,
            "raw_length_median": float(np.median(lengths)) if lengths else None,
            "raw_length_max": max(lengths) if lengths else None,
            "dtype_counts": dict(dtypes),
        }
    transformed_lengths = sorted({int((length + decimate - 1) // decimate) for length in all_lengths})
    return {
        "data_dir": str(data_dir.resolve()),
        "classes": classes,
        "total_files": sum(info["n_files"] for info in classes.values()),
        "raw_lengths": sorted(set(all_lengths)),
        "after_decimate_lengths": transformed_lengths,
    }


def build_recommendation(run_info, args):
    input_length = args.input_length
    preprocess = args.preprocess
    if input_length is None:
        input_length = (
            run_info.get("input_length_from_metadata")
            or run_info.get("input_length_from_tag")
            or (RAW_SIGNAL_LENGTH // args.decimate)
        )
    if preprocess == "auto":
        preprocess = "adaptive-bandpass" if run_info.get("decim_sweep") else "bandpass-decimate"
    return {
        "model_name": args.model_name or run_info.get("model_name") or "Conv1DGAP",
        "input_length": int(input_length),
        "preprocess": preprocess,
        "decimate": args.decimate,
        "bandpass": DEFAULT_BANDPASS,
        "class_names": args.class_names.split(","),
        "kernel_size": args.kernel_size or run_info.get("kernel_size"),
    }


def evaluate_status(report):
    failures = []
    warnings = []
    dataset = report["dataset"]
    rec = report["recommended_accuracy_vs_snr_args"]
    compat = report["checkpoint_compatibility"]

    missing = [name for name, info in dataset["classes"].items() if not info["exists"] or info["n_files"] == 0]
    if missing:
        failures.append(f"Missing or empty class folders: {', '.join(missing)}")
    if compat["checkpoint_out_classes"] is not None and compat["checkpoint_out_classes"] != len(rec["class_names"]):
        failures.append(
            f"Checkpoint head has {compat['checkpoint_out_classes']} classes, expected {len(rec['class_names'])}"
        )
    if compat["compatible_ratio"] < 1.0:
        warnings.append(
            f"Checkpoint is not strict-compatible ({compat['compatible_keys']}/{compat['total_keys']} keys)"
        )
    raw_lengths = dataset["raw_lengths"]
    if raw_lengths and raw_lengths != [RAW_SIGNAL_LENGTH]:
        native_length = (
            report.get("run_metadata", {}).get("native_length")
            if report.get("run_metadata")
            else None
        )
        if native_length is None or raw_lengths != [native_length]:
            warnings.append(
                f"Dataset raw lengths {raw_lengths} differ from P0 training RAW_SIGNAL_LENGTH={RAW_SIGNAL_LENGTH}"
            )
    after_decimate = dataset["after_decimate_lengths"]
    if rec["preprocess"] == "bandpass-decimate" and after_decimate and rec["input_length"] not in after_decimate:
        warnings.append(
            f"Decimated lengths {after_decimate} will be center-cropped/padded to input_length={rec['input_length']}"
        )
    if report.get("run_metadata") is None:
        warnings.append("No run metadata JSON found next to benchmark2 checkpoint tag")

    status = "fail" if failures else ("warn" if warnings else "ok")
    return {"status": status, "failures": failures, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(
        description="Preflight Conv1DGAP checkpoint/data compatibility before accuracy-vs-SNR."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--class-names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--preprocess", choices=("auto", "none", "bandpass-decimate", "adaptive-bandpass"),
                        default="auto")
    parser.add_argument("--decimate", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--snr-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero when the report status is fail.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    run_info = parse_run_tag(checkpoint_path.parent.name)
    run_metadata_path, run_metadata = find_run_metadata(checkpoint_path)
    if run_metadata:
        run_info.update({
            "model_name": run_metadata.get("model_name", run_info.get("model_name")),
            "kernel_size": run_metadata.get("kernel_size", run_info.get("kernel_size")),
            "input_length_from_metadata": run_metadata.get("input_length"),
            "native_length": run_metadata.get("native_length"),
            "decim_sweep": run_metadata.get("decim_mode", run_info.get("decim_sweep")),
            "tier": run_metadata.get("tier", run_info.get("tier")),
            "seed": run_metadata.get("seed", run_info.get("seed")),
            "tier_name": run_metadata.get("tier_name"),
            "benchmark_accuracy": run_metadata.get("accuracy"),
            "benchmark_best_val_accuracy": run_metadata.get("best_val_accuracy"),
        })

    rec = build_recommendation(run_info, args)
    class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    rec["class_names"] = class_names
    dataset = audit_dataset(args.data_dir, class_names, args.decimate)
    compat = checkpoint_compatibility(
        checkpoint_path,
        rec["model_name"],
        rec["input_length"],
        len(class_names),
        rec.get("kernel_size"),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_tag": checkpoint_path.parent.name,
        "run_metadata_path": str(run_metadata_path) if run_metadata_path else None,
        "run_metadata": run_info if run_metadata else None,
        "dataset": dataset,
        "p0_training_defaults": {
            "class_names": DEFAULT_CLASS_NAMES,
            "raw_signal_length": RAW_SIGNAL_LENGTH,
            "preprocess": "BandpassFilter(5-100 kHz, 2 MHz) -> Decimate(4)",
            "default_input_length": RAW_SIGNAL_LENGTH // args.decimate,
        },
        "recommended_accuracy_vs_snr_args": rec,
        "checkpoint_compatibility": compat,
        "snr_csv": str(Path(args.snr_csv).resolve()) if args.snr_csv else None,
    }
    report["assessment"] = evaluate_status(report)

    output_json = Path(args.output_json) if args.output_json else Path(args.data_dir) / "conv1dgap_snr_preflight.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Status: {report['assessment']['status']}")
    for item in report["assessment"]["failures"]:
        print(f"FAIL: {item}")
    for item in report["assessment"]["warnings"]:
        print(f"WARN: {item}")
    print(f"Recommended model: {rec['model_name']}")
    print(f"Recommended preprocessing: {rec['preprocess']} input_length={rec['input_length']}")
    print(f"Compatible keys: {compat['compatible_keys']}/{compat['total_keys']}")
    print(f"JSON: {output_json}")

    if args.strict and report["assessment"]["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
