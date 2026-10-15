"""Evaluation must reject altered files, stale membership and inconsistent labels."""

import hashlib
import json
from pathlib import Path

import pytest

from chest_xray_classifier.evaluation.evaluate import verify_test_data


def cohort(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "test"
    image = data / "normal/example.jpeg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    records = [
        {
            "image": "normal/example.jpeg",
            "class": "normal",
            "sha256": hashlib.sha256(b"image").hexdigest(),
        }
    ]
    serialized = json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n"
    manifest = {
        "count": 1,
        "records": records,
        "records_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return data, path


def test_cohort_verifies_exact_image_bytes(tmp_path: Path) -> None:
    data, manifest = cohort(tmp_path)
    assert verify_test_data(data, manifest)["count"] == 1
    (data / "normal/example.jpeg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        verify_test_data(data, manifest)


def test_stale_file_fails_before_model_loading(tmp_path: Path) -> None:
    data, manifest = cohort(tmp_path)
    (data / "normal/stale.jpeg").write_bytes(b"stale")
    with pytest.raises(ValueError, match="membership"):
        verify_test_data(data, manifest)


def test_changed_manifest_is_rejected(tmp_path: Path) -> None:
    data, path = cohort(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["records"][0]["class"] = "viral_pneumonia"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest content hash"):
        verify_test_data(data, path)
