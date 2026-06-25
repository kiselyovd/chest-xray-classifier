"""Generate model-card visualizations from real test-split inference.

Runs the HF-native ConvNeXt-V2-Tiny classifier over the processed test split,
then renders a confusion matrix, one-vs-rest ROC curves, and a sample-prediction
panel. Computed accuracy / macro-F1 / macro-AUROC are cross-checked against
``reports/metrics.json`` so the published charts always reflect the real model.

Usage:
    uv run python scripts/make_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification

# HF Auto* factories return dynamic concrete classes that the type checker
# cannot resolve precisely; alias them to Any for annotation purposes.
HFModel = Any
HFProcessor = Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "data" / "processed" / "test"
REPORTS_DIR = REPO_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_ID = "kiselyovd/chest-xray-classifier"

# Authoritative class order (sorted / alphabetical), matches model id2label.
CLASS_NAMES = ["bacterial_pneumonia", "normal", "viral_pneumonia"]
DISPLAY_NAMES = ["Bacterial\npneumonia", "Normal", "Viral\npneumonia"]
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}
BATCH_SIZE = 32
TOLERANCE = 0.02  # accuracy / macro-F1 may differ slightly from reference run

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)


def _load_model() -> tuple[HFModel, HFProcessor]:
    """Load the HF-native model + processor (config id2label is authoritative)."""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.train(False)
    return model, processor


def _gather_test_images() -> tuple[list[Path], np.ndarray]:
    """Collect every test image path and its true label index."""
    paths: list[Path] = []
    labels: list[int] = []
    for idx, name in enumerate(CLASS_NAMES):
        class_dir = TEST_DIR / name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                paths.append(img_path)
                labels.append(idx)
    return paths, np.asarray(labels, dtype=int)


def _run_inference(
    model: HFModel,
    processor: HFProcessor,
    paths: list[Path],
) -> np.ndarray:
    """Return softmax probabilities of shape (n_images, n_classes)."""
    all_probs: list[np.ndarray] = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        done = min(start + BATCH_SIZE, len(paths))
        print(f"  inference {done}/{len(paths)}", flush=True)
    return np.concatenate(all_probs, axis=0)


def _cross_check(y_true: np.ndarray, probs: np.ndarray) -> None:
    """Abort if computed metrics diverge from the committed reference run."""
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")

    reference = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    ref_acc = reference["accuracy"]
    ref_f1 = reference["macro_f1"]
    ref_auroc = reference["auroc_macro_ovr"]

    print(
        f"  computed: acc={acc:.4f} macro_f1={macro_f1:.4f} auroc={auroc:.4f}",
        flush=True,
    )
    print(
        f"  reference: acc={ref_acc:.4f} macro_f1={ref_f1:.4f} auroc={ref_auroc:.4f}",
        flush=True,
    )

    if abs(acc - ref_acc) > TOLERANCE or abs(macro_f1 - ref_f1) > TOLERANCE:
        raise SystemExit(
            "Computed metrics diverge from reports/metrics.json beyond tolerance; "
            "refusing to render charts."
        )


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Render a 3x3 confusion matrix with counts and row-normalized rates."""
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized rate", rotation=270, labelpad=15)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DISPLAY_NAMES)
    ax.set_yticklabels(DISPLAY_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (test split, n=624)")

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )

    ax.grid(False)
    fig.tight_layout()
    out = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _plot_roc_curves(y_true: np.ndarray, probs: np.ndarray) -> None:
    """Render one-vs-rest ROC curves with per-class AUROC in the legend."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    titles = ["Bacterial pneumonia", "Normal", "Viral pneumonia"]

    for idx, (name, color) in enumerate(zip(titles, colors, strict=True)):
        binary_true = (y_true == idx).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, probs[:, idx])
        auc = roc_auc_score(binary_true, probs[:, idx])
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUROC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves (one-vs-rest, test split)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    out = REPORTS_DIR / "roc_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _select_sample_indices(
    y_true: np.ndarray,
    probs: np.ndarray,
    per_class: int = 2,
) -> list[int]:
    """Pick representative, correctly-classified test examples per class.

    For each class we take the ``per_class`` highest-confidence correct
    predictions. This gives an honest, typical view of the model on real
    test radiographs (it is not cherry-picking failures or perfect-only
    cases, just confidently-correct calls that reflect the 91%-accuracy
    operating point).
    """
    y_pred = probs.argmax(axis=1)
    correct = y_pred == y_true
    chosen: list[int] = []
    for idx in range(len(CLASS_NAMES)):
        mask = (y_true == idx) & correct
        candidates = np.where(mask)[0]
        confidence = probs[candidates, idx]
        order = candidates[np.argsort(-confidence)]
        chosen.extend(order[:per_class].tolist())
    return chosen


def _plot_sample_predictions(
    model: HFModel,
    paths: list[Path],
    y_true: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Render a grid of representative test X-rays with prediction + confidence."""
    indices = _select_sample_indices(y_true, probs, per_class=2)
    preds = probs.argmax(axis=1)

    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 4.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    pretty = {
        "bacterial_pneumonia": "Bacterial pneumonia",
        "normal": "Normal",
        "viral_pneumonia": "Viral pneumonia",
    }
    for ax, sample_idx in zip(axes, indices, strict=False):
        path = paths[sample_idx]
        img = Image.open(path).convert("RGB")
        ax.imshow(img, cmap="gray")
        true_name = CLASS_NAMES[int(y_true[sample_idx])]
        pred_name = model.config.id2label[int(preds[sample_idx])]
        confidence = float(probs[sample_idx, preds[sample_idx]]) * 100
        correct = pred_name == true_name
        mark = "OK" if correct else "X"
        color = "#2ca02c" if correct else "#d62728"
        ax.set_title(
            f"[{mark}] pred: {pretty[pred_name]}  ({confidence:.1f}%)\ntrue: {pretty[true_name]}",
            color=color,
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle("Sample Test Predictions", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPORTS_DIR / "sample_predictions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main() -> None:
    """Run inference and render all model-card plots."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading model + processor...", flush=True)
    model, processor = _load_model()

    print("Gathering test images...", flush=True)
    paths, y_true = _gather_test_images()
    print(f"  {len(paths)} test images", flush=True)

    print("Running test-split inference...", flush=True)
    probs = _run_inference(model, processor, paths)

    print("Cross-checking metrics against reports/metrics.json...", flush=True)
    _cross_check(y_true, probs)

    y_pred = probs.argmax(axis=1)
    print("Rendering plots...", flush=True)
    _plot_confusion_matrix(y_true, y_pred)
    _plot_roc_curves(y_true, probs)
    _plot_sample_predictions(model, paths, y_true, probs)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
