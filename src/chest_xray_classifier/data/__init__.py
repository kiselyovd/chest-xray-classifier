"""Data layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodule import ImageDataModule
    from .dataset import ImageDataset

__all__ = ["ImageDataModule", "ImageDataset"]


def __getattr__(name: str):
    """Avoid importing the training runtime for dataset preparation and inference."""
    if name == "ImageDataModule":
        from .datamodule import ImageDataModule

        return ImageDataModule
    if name == "ImageDataset":
        from .dataset import ImageDataset

        return ImageDataset
    raise AttributeError(name)
