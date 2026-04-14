"""DINOv2 ViT-S feature extractor + linear classification head."""

from __future__ import annotations

import torch
from torch import nn


class DinoV2LinearProbe(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            feat_dim = self.backbone.config.hidden_size
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            feat_dim = 384
            self.backbone = None
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            with torch.no_grad():
                out = self.backbone(pixel_values=x)
                feats = out.last_hidden_state[:, 0]
        else:
            feats = torch.zeros(x.shape[0], self.head.in_features, device=x.device)
        return self.head(feats)
