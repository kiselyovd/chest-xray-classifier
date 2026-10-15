"""Run model on test set, write reports/metrics.json."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
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


def file_sha256(path: Path) -> str:
    """Hash exact artifact bytes, independently of checkout line endings."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_test_data(root: Path, manifest_path: Path) -> dict:
    """Require exact original test membership and file contents before inference."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest["records"], sort_keys=True, separators=(",", ":")) + "\n"
    if hashlib.sha256(serialized.encode()).hexdigest() != manifest["records_sha256"]:
        raise ValueError("Test manifest content hash mismatch")
    expected = {r["image"] for r in manifest["records"]}
    if len(expected) != len(manifest["records"]) or len(expected) != manifest["count"]:
        raise ValueError("Test manifest contains duplicate or inconsistent records")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Test image membership differs from manifest")
    for record in manifest["records"]:
        if record["class"] not in CLASSES or Path(record["image"]).parts[0] != record["class"]:
            raise ValueError("Test record label differs from its class directory")
        path = (root / record["image"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Test manifest path escapes data directory")
        if file_sha256(path) != record["sha256"]:
            raise ValueError("Test image content hash mismatch")
    log.info("evaluation.cohort_verified", images=len(expected), hash=manifest["records_sha256"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--hf-model")
    parser.add_argument("--revision")
    parser.add_argument("--weights-sha256")
    parser.add_argument("--source-revision")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument(
        "--test-manifest", type=Path, default=Path("reports/provenance/chest-test-manifest.json")
    )
    parser.add_argument("--out", default="reports/replay-native.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.source_revision and not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("Source revision must be an immutable commit")

    configure_logging()
    torch.set_num_threads(4)
    test_dir = args.test_dir or Path(args.data) / "test"
    manifest = verify_test_data(test_dir, args.test_manifest)
    if args.hf_model:
        if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            parser.error("Native HF replay requires an immutable revision")
        if not args.weights_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.weights_sha256):
            parser.error("Native HF replay requires the expected weights SHA256")
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForImageClassification

        weights = Path(
            hf_hub_download(args.hf_model, "model.safetensors", revision=args.revision, token=False)
        )
        if file_sha256(weights) != args.weights_sha256:
            raise ValueError("Native model weights hash mismatch")
        model = AutoModelForImageClassification.from_pretrained(
            args.hf_model,
            revision=args.revision,
            use_safetensors=True,
            trust_remote_code=False,
            token=False,
        )
        if [model.config.id2label[i] for i in range(3)] != list(CLASSES):
            raise ValueError("Native output labels differ from dataset classes")
        artifact = {
            "repo_id": args.hf_model,
            "revision": args.revision,
            "sha256": args.weights_sha256,
        }
    else:
        model = load_model(args.checkpoint)
        artifact = {"kind": "lightning", "sha256": file_sha256(Path(args.checkpoint))}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_ds = ImageDataset(test_dir, transform=build_eval_transforms())
    test_ds.samples.sort(key=lambda item: item[0].as_posix())
    if test_ds.class_to_idx != {name: i for i, name in enumerate(CLASSES)}:
        raise ValueError("Dataset class indices differ from evaluation labels")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            result = model._forward_logits(x) if hasattr(model, "_forward_logits") else model(x)
            logits = result.logits if hasattr(result, "logits") else result
            probs = logits.softmax(-1)
            all_preds.extend(probs.argmax(-1).tolist())
            all_labels.extend(y.tolist())
            all_probs.extend(probs.tolist())
            log.info("evaluation.progress", completed=len(all_labels), total=len(test_ds))

    report = classification_report(
        all_labels, all_preds, target_names=list(CLASSES), output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")

    metrics = {
        "status": "replayed_on_reconstructed_original_split",
        "training_membership_verified": False,
        "artifact": artifact,
        "source_revision": args.source_revision,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "test_size": len(test_ds),
        "test_manifest_sha256": manifest["records_sha256"],
        "images": [p.relative_to(test_dir).as_posix() for p, _ in test_ds.samples],
        "preprocessing": {
            "resize_shortest": 255,
            "interpolation": "bilinear antialias",
            "crop": "torchvision CenterCrop224 with rounded anchor",
            "rescale": "1/255",
            "normalize": "ImageNet mean/std",
        },
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "transformers", "scikit-learn", "numpy")
        },
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(CLASSES),
        "auroc_macro_ovr": float(auroc),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        arrays_path, probabilities=np.asarray(all_probs), target=np.asarray(all_labels)
    )
    metrics["prediction_arrays"] = {"file": arrays_path.name, "sha256": file_sha256(arrays_path)}
    package_root = Path(__file__).resolve().parents[1]
    metrics["source_files_sha256"] = {
        p.relative_to(package_root).as_posix(): file_sha256(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("done", out=str(out_path), macro_f1=metrics["macro_f1"])


if __name__ == "__main__":
    main()
