"""Model smoke tests (forward pass, output shape)."""
from __future__ import annotations

import torch

from chest_xray_classifier.models import build_model


def test_resnet50_forward():
    model = build_model("resnet50", num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3)
