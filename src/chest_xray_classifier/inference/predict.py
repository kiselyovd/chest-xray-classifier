"""Inference CLI — load a checkpoint and predict on input(s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def load_model(checkpoint_path: str | Path):
    """Load a Lightning module from checkpoint, rebuilding the backbone from hparams."""
    import torch

    from ..models import ClassificationModule, build_model

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    model_name = hp.get("model_name")
    num_classes = hp.get("num_classes")
    if model_name is None or num_classes is None:
        raise ValueError(
            "Checkpoint missing model_name/num_classes hparams — "
            "re-train after upgrading ClassificationModule."
        )
    backbone = build_model(model_name, num_classes=num_classes, pretrained=False)
    return ClassificationModule.load_from_checkpoint(
        str(checkpoint_path), model=backbone
    )


def predict(model, input_path: str | Path):
    """Run a single prediction. Returns a task-specific result dict."""
    import torch
    from PIL import Image

    from ..data.transforms import build_eval_transforms

    model.eval()
    tf = build_eval_transforms()
    img = Image.open(input_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model._forward_logits(x) if hasattr(model, "_forward_logits") else model(x)
        probs = logits.softmax(-1).squeeze(0).tolist()
    return {"probs": probs, "pred": int(max(range(len(probs)), key=probs.__getitem__))}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
