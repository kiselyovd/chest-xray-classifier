"""Lightning module wrappers."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class ClassificationModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.save_hyperparameters(ignore=["model"])

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self._forward_logits(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self._forward_logits(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)
