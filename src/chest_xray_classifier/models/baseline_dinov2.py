"""DINOv2 ViT-S feature extractor + linear classification head."""

from __future__ import annotations

import torch
from torch import nn


class DinoV2LinearProbe(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained("facebook/dinov2-small")
        if pretrained:
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        else:
            self.backbone = AutoModel.from_config(config)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(pixel_values=x)
            feats = out.last_hidden_state[:, 0]
        return self.head(feats)
