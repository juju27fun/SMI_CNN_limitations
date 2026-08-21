import numpy as np
import pytest
import torch

from p0.gradcam import attention_enrichment, event_concentration, interval_mask, regional_top_mask, temporal_gradcam, temporal_regions, top_fraction_mask
from p0.models.conv1d_gap import Conv1DGAPClassifier


def test_temporal_gradcam_matches_input_length_and_is_normalized():
    torch.manual_seed(7)
    model = Conv1DGAPClassifier(input_length=64, num_classes=3, width_mult=0.1).eval()
    probabilities, cam = temporal_gradcam(model, torch.randn(1, 1, 64), model.pool3)
    assert probabilities.shape == (3,)
    assert cam.shape == (64,)
    assert np.isfinite(cam).all()
    assert 0 <= cam.min() <= cam.max() <= 1


def test_interval_enrichment_is_relative_to_temporal_coverage():
    mask = interval_mask(100, [(0.2, 0.4)])
    uniform = np.ones(100)
    concentrated = mask.astype(float)
    assert attention_enrichment(uniform, mask) == pytest.approx(1.0)
    assert attention_enrichment(concentrated, mask) == pytest.approx(5.0)


def test_top_fraction_mask_has_deterministic_size():
    mask = top_fraction_mask(np.arange(10), 0.2)
    assert mask.sum() == 2
    assert np.flatnonzero(mask).tolist() == [8, 9]


def test_temporal_regions_are_exclusive_and_exhaustive():
    regions = temporal_regions(100, [(0.0, 0.03), (0.45, 0.55)], edge_fraction=0.05)
    stacked = np.stack(list(regions.values()))
    assert np.all(stacked.sum(axis=0) == 1)
    assert regions["event"][:3].all()
    assert not regions["edge"][:3].any()


def test_regional_top_mask_uses_fixed_global_budget():
    cam = np.arange(100)
    region = np.zeros(100, dtype=bool); region[20:80] = True
    mask = regional_top_mask(cam, region, 0.05)
    assert mask.sum() == 5
    assert np.flatnonzero(mask).tolist() == [75, 76, 77, 78, 79]


def test_event_concentration_reports_coverage_mass_and_enrichment():
    cam = np.ones(100)
    cam[20:40] = 4
    result = event_concentration(cam, [(0.2, 0.4)])
    assert result["temporal_coverage"] == pytest.approx(0.2)
    assert result["cam_mass"] == pytest.approx(80 / 160)
    assert result["uniform_attention_enrichment"] == pytest.approx(2.5)
