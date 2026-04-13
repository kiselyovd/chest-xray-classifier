"""Populate data/sample/ with a tiny subset for smoke tests and CI."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image


CLASS_COUNTS = {"normal": 5, "bacterial_pneumonia": 3, "viral_pneumonia": 2}


def build_sample(src: Path, dst: Path, seed: int = 42, size: int = 224) -> None:
    rng = random.Random(seed)
    dst.mkdir(parents=True, exist_ok=True)
    for cls, n in CLASS_COUNTS.items():
        cls_dst = dst / cls
        cls_dst.mkdir(parents=True, exist_ok=True)
        candidates = sorted((src / cls).glob("*.jpeg"))
        if len(candidates) < n:
            raise SystemExit(f"Not enough in {src / cls}: need {n}, have {len(candidates)}")
        chosen = rng.sample(candidates, n)
        for path in chosen:
            img = Image.open(path).convert("RGB")
            img.thumbnail((size, size))
            img.save(cls_dst / path.name, "JPEG", quality=85)
            print(f"wrote {cls_dst / path.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/processed/train")
    p.add_argument("--dst", default="data/sample")
    args = p.parse_args()
    build_sample(Path(args.src), Path(args.dst))
