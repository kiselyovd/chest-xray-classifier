"""Pick one image per class for HF widget examples (one per class)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/processed/test")
    parser.add_argument("--dst", default="data/sample/widget")
    parser.add_argument(
        "--classes", nargs="+", default=["normal", "bacterial_pneumonia", "viral_pneumonia"]
    )
    args = parser.parse_args()

    src = Path(args.src)
    # Fallback: if processed/test doesn't exist, use data/sample directly
    if not src.is_dir():
        fallback = Path("data/sample")
        if fallback.is_dir():
            print(f"Source {src} not found; falling back to {fallback}")
            src = fallback
        else:
            raise SystemExit(f"Neither {src} nor {fallback} found.")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    picked = []
    for cls in args.classes:
        cls_dir = src / cls
        if not cls_dir.is_dir():
            raise SystemExit(f"Class dir not found: {cls_dir}")
        candidates = sorted(
            p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if not candidates:
            raise SystemExit(f"No images in {cls_dir}")
        chosen = candidates[0]
        out_png = dst / f"{cls}.png"
        Image.open(chosen).convert("RGB").save(out_png, format="PNG")
        picked.append(out_png.name)

    print(f"Wrote {len(picked)} widget samples to {dst}: {picked}")


if __name__ == "__main__":
    main()
