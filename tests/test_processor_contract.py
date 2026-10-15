"""Native processor tensors must match production evaluation preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor

from chest_xray_classifier.data.transforms import build_eval_transforms, build_hf_processor


@pytest.mark.parametrize("shape", [(224, 224), (320, 280), (91, 173), (401, 289)])
def test_native_processor_matches_eval_tensor(tmp_path: Path, shape) -> None:
    pixels = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    build_hf_processor().save_pretrained(tmp_path)
    processor = AutoImageProcessor.from_pretrained(
        tmp_path, backend="torchvision", trust_remote_code=True
    )
    reference = build_eval_transforms()(image)
    native = processor(image, return_tensors="pt").pixel_values[0]
    torch.testing.assert_close(native, reference, rtol=0, atol=1e-6)
    assert int(224 / processor.crop_pct) == 255
    assert native.shape == (3, 224, 224)


def test_committed_samples_match_production_tensors() -> None:
    root = Path(__file__).resolve().parents[1] / "data/sample"
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    assert paths
    processor, transform = build_hf_processor(), build_eval_transforms()
    for path in paths:
        image = Image.open(path).convert("RGB")
        native = processor(image, return_tensors="pt").pixel_values[0]
        torch.testing.assert_close(native, transform(image), rtol=0, atol=1e-6)


def test_unsupported_native_crop_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="validated for 224"):
        build_hf_processor(384)
