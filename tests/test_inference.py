"""End-to-end inference smoke on data/sample/."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from chest_xray_classifier.data import ImageDataset
from chest_xray_classifier.data.transforms import build_train_transforms
from chest_xray_classifier.inference.predict import load_model, predict
from chest_xray_classifier.models import ClassificationModule, build_model


def test_predict_on_sample(sample_data_dir: Path, tmp_path: Path) -> None:
    torch.manual_seed(0)
    ds = ImageDataset(sample_data_dir, transform=build_train_transforms(64))
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    model = build_model("resnet50", num_classes=3, pretrained=False)
    lit = ClassificationModule(model, num_classes=3, lr=1e-3, model_name="resnet50")
    trainer = L.Trainer(
        max_epochs=1,
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    trainer.fit(lit, loader)
    ckpt = tmp_path / "smoke.ckpt"
    trainer.save_checkpoint(str(ckpt))

    reloaded = load_model(ckpt)
    sample_img = next((sample_data_dir / "normal").glob("*.jpeg"))
    result = predict(reloaded, sample_img)
    assert "pred" in result and "probs" in result
    assert 0 <= result["pred"] <= 2
    assert len(result["probs"]) == 3
    assert abs(sum(result["probs"]) - 1.0) < 1e-5
