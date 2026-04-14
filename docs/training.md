# Training

## Prerequisites

```bash
uv sync --all-groups
bash scripts/sync_data.sh /path/to/kaggle/chest_xray
uv run python -m chest_xray_classifier.data.prepare --raw data/raw --out data/processed
```

## Main — ConvNeXt-V2-Tiny

```bash
uv run python -m chest_xray_classifier.training.train experiment=sota
```

Expected wall time: ~90 min on an A10/A100. Checkpoint written to `artifacts/checkpoints/best.ckpt`.

## Baseline — DINOv2 linear probe

```bash
uv run python -m chest_xray_classifier.training.train \
  model=baseline \
  trainer.max_epochs=20 \
  trainer.output_dir=artifacts/baseline
```

## MLflow tracking

```bash
mlflow ui --backend-store-uri ./mlruns
```

Browse at http://localhost:5000 — every Hydra run is one MLflow run with the full resolved config logged as params and `train/loss`, `val/loss`, `val/acc`, `val/f1_macro` as metrics.

## Hydra overrides (common)

| Override | Effect |
|---|---|
| `trainer.max_epochs=50` | Longer training |
| `trainer.accelerator=gpu` | Force GPU |
| `data.batch_size=64` | Larger batches |
| `model.lr=1e-4` | Different learning rate |
| `seed=7` | Reproducibility |

Multi-run sweep example:

```bash
uv run python -m chest_xray_classifier.training.train -m \
  model.lr=1e-5,3e-5,1e-4 \
  data.batch_size=32,64
```
