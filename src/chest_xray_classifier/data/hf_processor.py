"""Native ConvNeXT processor preserving the project's torchvision crop rounding."""

# Copyright (c) 2026 Daniil Kiselyov. SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from torchvision.transforms.v2 import functional as functional
from transformers.image_utils import SizeDict
from transformers.models.convnext.image_processing_convnext import ConvNextImageProcessor


class ChestXrayImageProcessor(ConvNextImageProcessor):
    """Use torchvision's rounded center anchor instead of the Hub backend's truncation."""

    # Transformers 5.5's external typing exposes the legacy PIL signature for
    # this torchvision backend. Runtime dispatch supplies Tensor + SizeDict.
    def center_crop(  # type: ignore[override]
        self, image: torch.Tensor, size: SizeDict, **kwargs
    ) -> torch.Tensor:
        """Preserve exactly the crop used by the training/evaluation transform."""
        if size.height is None or size.width is None:
            raise ValueError("Crop size requires height and width")
        return functional.center_crop(image, [size.height, size.width])


ChestXrayImageProcessor.register_for_auto_class("AutoImageProcessor")
