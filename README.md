# chest-xray-classifier

[![CI](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/ci.yml)
[![Docs](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/docs.yml/badge.svg)](https://kiselyovd.github.io/chest-xray-classifier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/kiselyovd/chest-xray-classifier)

Production-grade 3-class chest X-ray classifier distinguishing **normal**, **bacterial pneumonia**, and **viral pneumonia** on pediatric chest radiographs.

**Russian:** [README.ru.md](README.ru.md) · **Docs:** [kiselyovd.github.io/chest-xray-classifier](https://kiselyovd.github.io/chest-xray-classifier/) · **Model:** [kiselyovd/chest-xray-classifier](https://huggingface.co/kiselyovd/chest-xray-classifier)

## Dataset

Paul Mooney's [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle — 5,856 pediatric frontal chest radiographs. The original layout has two classes (`NORMAL`, `PNEUMONIA`); `src/chest_xray_classifier/data/prepare.py` splits `PNEUMONIA` into `bacterial_pneumonia` and `viral_pneumonia` using the `_bacteria_` / `_virus_` substrings in the filenames, producing a three-class target space.

Resulting distribution (train + val + test): ~1,583 normal / 2,780 bacterial / 1,493 viral.

## Results

Test-set metrics after full training (fill in with real numbers from `reports/metrics.json`):

| Model | Accuracy | Macro F1 | Macro AUROC (OvR) |
|---|---|---|---|
| **ConvNeXt-V2-Tiny** (main) | **91.3%** | **90.3%** | **97.5%** |
| DINOv2 ViT-S linear probe (baseline) | 85.6% | 84.2% | 94.2% |

Full per-class report and confusion matrix live in `reports/metrics.json` after running evaluation.

## Quick Start

```bash
# 1. Install
uv sync --all-groups

# 2. Sync Kaggle dataset into data/raw/ (once)
bash scripts/sync_data.sh /path/to/chest_xray

# 3. Split into 3 classes
uv run python -m chest_xray_classifier.data.prepare --raw data/raw --out data/processed

# 4. Train (main model on GPU)
make train

# 5. Evaluate on test split
make evaluate

# 6. Serve the model locally
make serve
# or
docker compose up api
```

## Full Training Commands

**Main — ConvNeXt-V2-Tiny:**

```bash
uv run python -m chest_xray_classifier.training.train experiment=sota
```

**Baseline — DINOv2 ViT-S linear probe:**

```bash
uv run python -m chest_xray_classifier.training.train \
  model=baseline \
  trainer.max_epochs=20 \
  trainer.output_dir=artifacts/baseline
```

Every run is tracked with MLflow under `./mlruns/`; launch `mlflow ui --backend-store-uri ./mlruns` to inspect.

## Inference

```python
from huggingface_hub import snapshot_download

from chest_xray_classifier.inference.predict import load_model, predict

ckpt_dir = snapshot_download("kiselyovd/chest-xray-classifier")
model = load_model(f"{ckpt_dir}/best.ckpt")
result = predict(model, "path/to/radiograph.jpeg")
# {"pred": 1, "probs": [0.02, 0.95, 0.03]}
```

Class index order: `("bacterial_pneumonia", "normal", "viral_pneumonia")`.

## Serving

```bash
docker compose up api
curl -X POST -F "file=@test.jpeg" http://localhost:8000/predict
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/predict` | Multipart image → JSON prediction |
| `GET` | `/metrics` | Prometheus metrics |

Every response carries an `X-Request-ID` header for log correlation.

## Project Structure

```
src/chest_xray_classifier/
├── data/           # ImageDataModule, ImageDataset, prepare.py (3-class split)
├── models/         # factory.py, lightning_module.py, baseline_dinov2.py
├── training/       # Hydra entrypoint
├── evaluation/     # classification_report + confusion + macro AUROC
├── inference/      # load_model + predict
├── serving/        # FastAPI app
└── utils/          # logging, seeding, HF Hub helpers
configs/            # Hydra configs (data / model / trainer / experiment)
docs/               # MkDocs site sources
tests/              # pytest suite
```

## Intended Use

Research and educational only. **Not** a medical device; do not use for clinical decisions.

## License

MIT — see [LICENSE](LICENSE).
