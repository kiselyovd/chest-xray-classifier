"""Data layer smoke tests."""

from __future__ import annotations

from pathlib import Path

from chest_xray_classifier.data import ImageDataModule
from chest_xray_classifier.data.prepare import prepare_data


def test_datamodule_instantiates(sample_data_dir):
    dm = ImageDataModule(data_dir=str(sample_data_dir), batch_size=2, num_workers=0)
    assert dm.hparams.batch_size == 2


def test_prepare_splits_bacterial_viral(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    (raw / "train" / "NORMAL").mkdir(parents=True)
    (raw / "train" / "PNEUMONIA").mkdir(parents=True)
    (raw / "train" / "NORMAL" / "NORMAL-1.jpeg").write_bytes(b"\x89PNG\r\n\x1a\n")
    (raw / "train" / "PNEUMONIA" / "person1_bacteria_1.jpeg").write_bytes(b"\x89PNG\r\n\x1a\n")
    (raw / "train" / "PNEUMONIA" / "person1_virus_6.jpeg").write_bytes(b"\x89PNG\r\n\x1a\n")

    prepare_data(raw, out)

    assert (out / "train" / "normal" / "NORMAL-1.jpeg").exists()
    assert (out / "train" / "bacterial_pneumonia" / "person1_bacteria_1.jpeg").exists()
    assert (out / "train" / "viral_pneumonia" / "person1_virus_6.jpeg").exists()
