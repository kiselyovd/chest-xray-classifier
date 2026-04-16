# chest-xray-classifier

[![CI](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/ci.yml)
[![Docs](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/docs.yml/badge.svg)](https://kiselyovd.github.io/chest-xray-classifier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/kiselyovd/chest-xray-classifier)

Production-ready классификатор рентгенограмм грудной клетки на 3 класса: **норма**, **бактериальная пневмония**, **вирусная пневмония**.

**English:** [README.md](README.md) · **Docs:** [kiselyovd.github.io/chest-xray-classifier](https://kiselyovd.github.io/chest-xray-classifier/) · **Модель:** [kiselyovd/chest-xray-classifier](https://huggingface.co/kiselyovd/chest-xray-classifier)

## Датасет

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) от Paul Mooney на Kaggle — 5 856 педиатрических рентгенограмм. В оригинале два класса (`NORMAL`, `PNEUMONIA`); `src/chest_xray_classifier/data/prepare.py` разделяет `PNEUMONIA` на `bacterial_pneumonia` и `viral_pneumonia` по подстрокам `_bacteria_` / `_virus_` в именах файлов — получается 3-классовая задача.

Итоговое распределение (train + val + test): ~1 583 normal / 2 780 bacterial / 1 493 viral.

## Результаты

Метрики на тестовом сплите после полного обучения (проставить из `reports/metrics.json`):

| Модель | Accuracy | Macro F1 | Macro AUROC (OvR) |
|---|---|---|---|
| **ConvNeXt-V2-Tiny** (основная) | **91.3%** | **90.3%** | **97.5%** |
| DINOv2 ViT-S linear probe (baseline) | 85.6% | 84.2% | 94.2% |

Полный classification report и матрица ошибок лежат в `reports/metrics.json` после запуска evaluation.

## Быстрый старт

```bash
# 1. Установка
uv sync --all-groups

# 2. Скопировать датасет из Kaggle в data/raw/ (один раз)
bash scripts/sync_data.sh /path/to/chest_xray

# 3. Разделить на 3 класса
uv run python -m chest_xray_classifier.data.prepare --raw data/raw --out data/processed

# 4. Обучение (основная модель, на GPU)
make train

# 5. Оценка на test-сплите
make evaluate

# 6. Локальный сервинг
make serve
# или
docker compose up api
```

## Полные команды обучения

**Основная — ConvNeXt-V2-Tiny:**

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

Каждый запуск логируется в MLflow в `./mlruns/`; просмотр: `mlflow ui --backend-store-uri ./mlruns`.

## Инференс

```python
from huggingface_hub import snapshot_download

from chest_xray_classifier.inference.predict import load_model, predict

ckpt_dir = snapshot_download("kiselyovd/chest-xray-classifier")
model = load_model(f"{ckpt_dir}/best.ckpt")
result = predict(model, "path/to/radiograph.jpeg")
# {"pred": 1, "probs": [0.02, 0.95, 0.03]}
```

Порядок индексов классов: `("bacterial_pneumonia", "normal", "viral_pneumonia")`.

## Сервинг

```bash
docker compose up api
curl -X POST -F "file=@test.jpeg" http://localhost:8000/predict
```

Эндпоинты:

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | Liveness-проба |
| `POST` | `/predict` | multipart-изображение → JSON-ответ |
| `GET` | `/metrics` | Prometheus-метрики |

В каждом ответе есть заголовок `X-Request-ID` для корреляции логов.

## Структура проекта

```
src/chest_xray_classifier/
├── data/           # ImageDataModule, ImageDataset, prepare.py (3-class split)
├── models/         # factory.py, lightning_module.py, baseline_dinov2.py
├── training/       # Hydra-входная точка
├── evaluation/     # classification_report + confusion + macro AUROC
├── inference/      # load_model + predict
├── serving/        # FastAPI-приложение
└── utils/          # логирование, сиды, HF Hub
configs/            # Hydra-конфиги (data / model / trainer / experiment)
docs/               # исходники MkDocs
tests/              # pytest
```

## Назначение

Исследовательский и образовательный проект. **Не является медицинским изделием**; запрещается использовать для клинических решений.

Известные ограничения и режимы отказа — в [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Цитирование

```bibtex
@software{kiselyov2026chestxray,
  author  = {Kiselyov, Daniil},
  title   = {chest-xray-classifier: ConvNeXt-V2-Tiny pneumonia classifier},
  year    = {2026},
  url     = {https://github.com/kiselyovd/chest-xray-classifier},
  version = {v0.1.0}
}
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
