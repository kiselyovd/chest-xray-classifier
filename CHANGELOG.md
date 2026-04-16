# Changelog

All notable changes to **chest-xray-classifier** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
spec.

## [Unreleased]

### Added

- `CHANGELOG.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md` at the repository
  root.
- `docs/REPRODUCIBILITY.md`, `docs/LIMITATIONS.md`, and `docs/BENCHMARKS.md`
  describing the reproducibility contract, known limitations, and an honest
  comparison against literature.
- Mermaid data-flow diagram in `docs/architecture.md` (replaces the ASCII
  version).
- `.github/dependabot.yml` covering `pip`, `github-actions`, and `docker`
  ecosystems.
- Codecov badge, portfolio cross-link, and `## Citation` section in
  `README.md` / `README.ru.md`.
- Custom MkDocs Material palette (primary `light blue`, accent `blue`) for
  this M1 classification project.

### Changed

- `ci.yml` now runs the `lint-test` job across a Python `3.12` / `3.13`
  matrix with `fail-fast: false`.

## [0.1.0] - 2026-04-14

### Added

- Initial production-grade release of the 3-class chest X-ray classifier
  (normal / bacterial pneumonia / viral pneumonia).
- ConvNeXt-V2-Tiny main model (91.3% accuracy, 90.3% macro F1,
  97.5% macro AUROC OvR on the 624-image Kaggle test split).
- DINOv2 ViT-S linear-probe baseline (85.6% / 84.2% / 94.2%) as a
  methodology sanity check.
- Hydra-driven training (`chest_xray_classifier.training.train`) with
  MLflow logging under `./mlruns/`.
- `ImageDataModule` / `ImageDataset` with a 3-class `prepare.py` splitter
  keyed off `_bacteria_` / `_virus_` filename substrings.
- Evaluation pipeline producing `reports/metrics.json` and
  `reports/metrics_summary.json` (per-class report, confusion matrix,
  macro AUROC).
- FastAPI serving app with `GET /health`, `POST /predict`, and
  `GET /metrics` endpoints plus an `X-Request-ID` correlation header.
- Multi-stage `Dockerfile` (training + serving targets) and
  `docker-compose.yml` for local deployment.
- Hugging Face publisher (`scripts/publish_to_hf.py`) producing a native
  `safetensors` export with widget samples, rich frontmatter, and a
  templated model card.
- Bilingual documentation (`README.md` / `README.ru.md` and MkDocs site)
  and MIT license.
- Quality-gate stack in CI: `ruff`, `mypy`, `deptry`, `bandit`,
  `interrogate`, `actionlint`, `codespell`, `pytest` with Codecov upload.
- Docker build job in CI validating both `training` and `serving` stages.

## Links

- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
- [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
- Releases: <https://github.com/kiselyovd/chest-xray-classifier/releases>

[Unreleased]: https://github.com/kiselyovd/chest-xray-classifier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kiselyovd/chest-xray-classifier/releases/tag/v0.1.0
