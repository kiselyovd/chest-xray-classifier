"""Export the trained ConvNeXt-V2 classifier into HF-native format (safetensors + config.json)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from chest_xray_classifier.data.transforms import build_hf_processor
from chest_xray_classifier.inference.predict import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    parser.add_argument("--out", default="artifacts/hf_export")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Evaluation crop size; the resize recipe comes from this project.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lit = load_model(args.checkpoint)
    backbone = lit.model
    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit("Backbone is not transformers-compatible; cannot export natively.")

    # Inject real class labels so the HF inference widget shows meaningful names.
    # Authoritative order is alphabetical: the dataset loader sorts class dir names
    # (see src/chest_xray_classifier/data/dataset.py).
    class_names = sorted(["bacterial_pneumonia", "normal", "viral_pneumonia"])
    if hasattr(backbone, "config"):
        backbone.config.id2label = {i: name for i, name in enumerate(class_names)}
        backbone.config.label2id = {name: i for i, name in enumerate(class_names)}

    backbone.save_pretrained(out)

    build_hf_processor(args.image_size).save_pretrained(out)
    shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", out / "LICENSE.processor.txt")
    for name in ("preprocessor_config.json", "hf_processor.py", "LICENSE.processor.txt"):
        path = out / name
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    print(f"Exported HF-native model to {out}")


if __name__ == "__main__":
    main()
