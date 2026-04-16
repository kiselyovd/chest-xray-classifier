# Reproducibility

This project is designed to produce identical results across re-runs on the same hardware. Here's how each moving part is pinned.

## Random seeds

- `seed: 42` in `configs/config.yaml` is threaded through `seed_everything()` at the start of `src/chest_xray_classifier/training/train.py`.
- Lightning's `deterministic="warn"` mode is set on the `Trainer` — non-deterministic CUDA ops (if any) emit a warning rather than silently using a non-deterministic impl.
- Data split is deterministic via `configs/data/default.yaml:val_split=0.1` + `seed: ${seed}`.

## Dependencies

- **`pyproject.toml`** declares direct deps; **`uv.lock`** is committed to git and pins every transitive package version + hash.
- **`.python-version`** = `3.13` — interpreters in CI and local dev must match.
- CUDA torch pinned via `[[tool.uv.index]] pytorch-cu124` on Win/Linux; re-pinning to a different CUDA requires a lockfile regen.
- CI runs the suite against Python 3.12 + 3.13 via a matrix strategy so regressions are caught in both.

## Data

- **`data/sample/`** (8 images + labels) is in git — CI + smoke tests work without any external download.
- **`data/raw/`** and **`data/processed/`** are DVC-tracked (see `dvc.yaml`). DVC remote is local-only by default; configure one for team use.
- Kaggle source: the Chest X-Ray Pneumonia dataset is pre-split (train / val / test) — no custom splitting, so splits are deterministic across machines.

## Docker

- `Dockerfile` is multi-stage (`base`, `training`, `serving`); `.dockerignore` keeps build context minimal.
- Published images live at `ghcr.io/kiselyovd/chest-xray-classifier:<tag>`; SHA256 digests are visible on the GHCR web UI.
- Base image: `python:3.13-slim-bookworm` — Debian package versions roll with Debian security updates but the Python version itself is frozen.

## Model weights

- Published to HuggingFace Hub under `kiselyovd/chest-xray-classifier`. Each release tag (e.g. `v0.1.0`) corresponds to a specific HF commit SHA visible in the repo's git log on huggingface.co.
- Weights ship as `model.safetensors` — a safe, reproducible binary format.

## One-command reproduction

```bash
git clone https://github.com/kiselyovd/chest-xray-classifier
cd chest-xray-classifier
uv sync --all-groups
bash scripts/sync_data.sh "/path/to/Chest X-Ray Images (Pneumonia)/chest_xray"
uv run python -m chest_xray_classifier.data.prepare --raw data/raw --out data/processed
uv run python -m chest_xray_classifier.training.train +experiment=sota
uv run python -m chest_xray_classifier.evaluation.evaluate --checkpoint artifacts/checkpoints/best.ckpt --out reports/metrics.json
```

Expected: the numbers in [BENCHMARKS.md](BENCHMARKS.md) ± 0.5% (floating-point noise across CUDA driver versions).
