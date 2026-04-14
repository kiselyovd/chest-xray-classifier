"""Run model on test set, write reports/metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from ..data import ImageDataset
from ..data.transforms import build_eval_transforms
from ..inference.predict import load_model
from ..utils import configure_logging, get_logger

log = get_logger(__name__)

CLASSES = ("bacterial_pneumonia", "normal", "viral_pneumonia")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="reports/metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    configure_logging()
    model = load_model(args.checkpoint)
    model.eval()

    test_ds = ImageDataset(Path(args.data) / "test", transform=build_eval_transforms())
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for x, y in loader:
            logits = model._forward_logits(x) if hasattr(model, "_forward_logits") else model(x)
            probs = logits.softmax(-1)
            all_preds.extend(probs.argmax(-1).tolist())
            all_labels.extend(y.tolist())
            all_probs.extend(probs.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=list(CLASSES), output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(CLASSES),
        "auroc_macro_ovr": float(auroc),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("done", out=str(out_path), macro_f1=metrics["macro_f1"])


if __name__ == "__main__":
    main()
