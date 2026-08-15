from __future__ import annotations

import torch

from p0.models.conv1d_gap import Conv1DGAPClassifier
from p0.models.proposal_aware_roi import ProposalAwareROIClassifier


def test_forward_features_is_backward_compatible_with_classifier_head() -> None:
    torch.manual_seed(42)
    model = Conv1DGAPClassifier(input_length=128, num_classes=3, width_mult=0.5)
    model.eval()
    inputs = torch.randn(4, 1, 128)

    with torch.no_grad():
        features = model.forward_features(inputs)
        logits = model(inputs)

    assert features.shape == (4, model.fc1.out_features)
    assert torch.equal(logits, model.fc2(features))


def test_proposal_aware_roi_uses_shared_encoder_for_both_heads() -> None:
    model = ProposalAwareROIClassifier(input_length=128, num_classes=3)
    event_logits, class_logits = model(torch.randn(2, 1, 128))

    assert event_logits.shape == (2,)
    assert class_logits.shape == (2, 3)
    assert isinstance(model.encoder.fc2, torch.nn.Identity)
