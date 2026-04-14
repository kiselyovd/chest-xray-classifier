"""Training smoke — one-epoch fit on data/sample/."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from chest_xray_classifier.data import ImageDataModule
from chest_xray_classifier.models import ClassificationModule, build_model


def test_fit_one_epoch_on_sample(sample_data_dir: Path) -> None:
    torch.manual_seed(0)
    dm = ImageDataModule(
        data_dir=str(sample_data_dir),
        batch_size=2,
        num_workers=0,
        val_split=0.2,
    )

    def _setup(stage=None):
        from chest_xray_classifier.data import ImageDataset
        from chest_xray_classifier.data.transforms import (
            build_eval_transforms,
            build_train_transforms,
        )

        train_tf = build_train_transforms(224)
        eval_tf = build_eval_transforms(224)
        dm.train_ds = ImageDataset(sample_data_dir, transform=train_tf)
        dm.val_ds = ImageDataset(sample_data_dir, transform=eval_tf)
        dm.test_ds = ImageDataset(sample_data_dir, transform=eval_tf)

    dm.setup = _setup  # type: ignore[method-assign]

    model = build_model("resnet50", num_classes=3, pretrained=False)
    lit = ClassificationModule(model, num_classes=3, lr=1e-3)

    trainer = L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    trainer.fit(lit, dm)
    assert "train/loss_epoch" in trainer.callback_metrics
