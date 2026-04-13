"""Training smoke — one-batch overfit."""
from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from chest_xray_classifier.models import ClassificationModule, build_model


def test_overfit_on_batch():
    torch.manual_seed(0)
    model = build_model("resnet50", num_classes=3, pretrained=False)
    lit = ClassificationModule(model, num_classes=3, lr=1e-3)
    x = torch.randn(4, 3, 64, 64)
    y = torch.tensor([0, 1, 2, 0])
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    trainer = L.Trainer(
        max_epochs=3,
        overfit_batches=1,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    trainer.fit(lit, loader, loader)
    assert trainer.callback_metrics["train/loss_epoch"].item() < 2.0
