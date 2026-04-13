"""Model factory — returns a torch.nn.Module by name."""
from __future__ import annotations

from torch import nn

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "convnextv2_tiny":
        from transformers import ConvNextV2ForImageClassification

        return ConvNextV2ForImageClassification.from_pretrained(
            "facebook/convnextv2-tiny-22k-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")
