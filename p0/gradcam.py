"""One-dimensional Grad-CAM utilities for temporal classifiers.

The functions in this module deliberately separate attribution computation
from scientific interpretation.  Grad-CAM is a diagnostic of local gradient
sensitivity; it is not evidence of causal feature use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def temporal_gradcam(
    model: torch.nn.Module,
    signal: torch.Tensor,
    target_layer: torch.nn.Module,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted-class probabilities and a normalized temporal CAM.

    ``signal`` must have shape ``(1, channels, time)``.  The CAM is linearly
    interpolated to the input length and normalized to ``[0, 1]``.
    """

    if signal.ndim != 3 or signal.shape[0] != 1:
        raise ValueError("signal must have shape (1, channels, time)")

    captured: dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = target_layer.register_forward_hook(_forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(signal)
        probabilities = torch.softmax(logits, dim=1)
        class_index = int(probabilities.argmax(dim=1).item()) if target_class is None else target_class
        logits[0, class_index].backward()
    finally:
        handle.remove()

    activation = captured["activation"]
    gradient = captured["gradient"]
    weights = gradient.mean(dim=-1, keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=signal.shape[-1], mode="linear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().float()
    span = cam.max() - cam.min()
    if float(span) > 0:
        cam = (cam - cam.min()) / span
    else:
        cam = torch.zeros_like(cam)
    return probabilities[0].detach().cpu().numpy(), cam.numpy()


def interval_mask(length: int, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    """Rasterize normalized ``[start, end]`` intervals into a Boolean mask."""

    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        lo = max(0, min(length, int(np.floor(start * length))))
        hi = max(lo + 1, min(length, int(np.ceil(end * length))))
        mask[lo:hi] = True
    return mask


def attention_enrichment(cam: np.ndarray, mask: np.ndarray) -> float:
    """Return CAM mass enrichment inside intervals relative to their coverage."""

    cam = np.asarray(cam, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if cam.ndim != 1 or mask.shape != cam.shape:
        raise ValueError("cam and mask must be one-dimensional and equally sized")
    coverage = float(mask.mean())
    total = float(cam.sum())
    if coverage <= 0 or total <= 0:
        return float("nan")
    return float(cam[mask].sum() / total / coverage)


def top_fraction_mask(cam: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Return a deterministic mask for the highest-CAM temporal positions."""

    if not 0 < fraction < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    cam = np.asarray(cam, dtype=float)
    count = max(1, int(np.ceil(cam.size * fraction)))
    order = np.argsort(cam, kind="stable")
    mask = np.zeros(cam.size, dtype=bool)
    mask[order[-count:]] = True
    return mask


def temporal_regions(
    length: int,
    event_intervals: Sequence[tuple[float, float]],
    edge_fraction: float = 0.05,
) -> dict[str, np.ndarray]:
    """Build mutually exclusive event, edge, and background masks.

    Detector-derived event intervals take precedence over the fixed edge band;
    background is everything assigned to neither region.
    """

    if not 0 < edge_fraction < 0.5:
        raise ValueError("edge_fraction must lie between zero and one half")
    event = interval_mask(length, event_intervals)
    edge = np.zeros(length, dtype=bool)
    width = max(1, int(np.ceil(length * edge_fraction)))
    edge[:width] = True
    edge[-width:] = True
    edge &= ~event
    return {"event": event, "edge": edge, "background": ~(event | edge)}


def regional_top_mask(cam: np.ndarray, region: np.ndarray, fraction: float) -> np.ndarray:
    """Select a fixed input-length budget of highest-CAM points in a region."""

    cam = np.asarray(cam, dtype=float)
    region = np.asarray(region, dtype=bool)
    if cam.shape != region.shape or cam.ndim != 1:
        raise ValueError("cam and region must be equally sized one-dimensional arrays")
    count = max(1, int(np.ceil(cam.size * fraction)))
    candidates = np.flatnonzero(region)
    if len(candidates) < count:
        return np.zeros_like(region)
    chosen = candidates[np.argsort(cam[candidates], kind="stable")[-count:]]
    mask = np.zeros_like(region)
    mask[chosen] = True
    return mask


def event_concentration(
    cam: np.ndarray,
    intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Summarize temporal coverage and Grad-CAM mass inside event windows."""

    cam = np.asarray(cam, dtype=float)
    if cam.ndim != 1:
        raise ValueError("cam must be one-dimensional")
    mask = interval_mask(cam.size, intervals)
    coverage = float(mask.mean())
    total = float(cam.sum())
    mass = float(cam[mask].sum() / total) if total > 0 and mask.any() else 0.0
    enrichment = mass / coverage if coverage > 0 else float("nan")
    return {
        "temporal_coverage": coverage,
        "cam_mass": mass,
        "uniform_attention_enrichment": enrichment,
    }
