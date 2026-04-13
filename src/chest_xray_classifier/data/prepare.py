"""Split Kaggle chest X-ray dataset into 3-class layout."""
from __future__ import annotations

import shutil
from pathlib import Path

from ..utils import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")
TARGET_CLASSES = ("normal", "bacterial_pneumonia", "viral_pneumonia")


def _classify(filename: str, src_class: str) -> str | None:
    if src_class.lower() == "normal":
        return "normal"
    low = filename.lower()
    if "_bacteria_" in low or "bacteria" in low:
        return "bacterial_pneumonia"
    if "_virus_" in low or "virus" in low:
        return "viral_pneumonia"
    return None


def prepare_data(raw_dir: Path | str, processed_dir: Path | str) -> None:
    raw = Path(raw_dir)
    out = Path(processed_dir)
    log.info("prepare_data.start", raw=str(raw), out=str(out))
    copied = {cls: 0 for cls in TARGET_CLASSES}
    skipped = 0
    for split in SPLITS:
        split_src = raw / split
        if not split_src.is_dir():
            log.warning("prepare_data.missing_split", split=split)
            continue
        for src_class_dir in split_src.iterdir():
            if not src_class_dir.is_dir():
                continue
            for img in src_class_dir.iterdir():
                if not img.is_file():
                    continue
                target = _classify(img.name, src_class_dir.name)
                if target is None:
                    skipped += 1
                    continue
                dst = out / split / target / img.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst)
                copied[target] += 1
    log.info("prepare_data.done", copied=copied, skipped=skipped)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    prepare_data(args.raw, args.out)
