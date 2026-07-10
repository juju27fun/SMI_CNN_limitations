"""Reusable classifier/SNR helpers shared with the data pipeline."""

from __future__ import annotations

import numpy as np
import torch


def macro_f1(rows: list[dict]) -> float:
    labels = sorted({row["y_true"] for row in rows} | {row["y_pred"] for row in rows})
    scores = []
    for label in labels:
        tp = sum(row["y_true"] == label and row["y_pred"] == label for row in rows)
        fp = sum(row["y_true"] != label and row["y_pred"] == label for row in rows)
        fn = sum(row["y_true"] == label and row["y_pred"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores)) if scores else 0.0


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_model_weights(model, checkpoint_path, strict: bool = True) -> dict:
    state = load_checkpoint_state(checkpoint_path)
    if strict:
        model.load_state_dict(state, strict=True)
        return {"loaded_keys": len(state), "skipped_keys": []}
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped = [key for key in state if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    return {"loaded_keys": len(compatible), "skipped_keys": skipped}

