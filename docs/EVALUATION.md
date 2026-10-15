# Original-source replay and preprocessing

The original [Kaggle chest X-ray archive](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) was retrieved on 2026-10-15: dataset 17810/version23812, 2,463,365,435 bytes, SHA256 `f569fe885b0f921e836f3d6bcc8d7b3442f5e0ca4db4533d06b8cf25d2114ea1`.

It contains two identical test trees and a separate `__MACOSX` metadata tree. Corresponding image bytes in the two normal test trees were compared; one 624-image cohort was used. The test manifest records every filename, derived class and SHA256. Its canonical records SHA256 is `c763fd1a0cee678815db830f59c54164d57e82b16df6c630c4d1a8a4bdc14280`.

[Source receipt](https://github.com/kiselyovd/chest-xray-classifier/blob/main/reports/provenance/chest-source.json) · [Test manifest](https://github.com/kiselyovd/chest-xray-classifier/blob/main/reports/provenance/chest-test-manifest.json) · [Full native replay](https://github.com/kiselyovd/chest-xray-classifier/blob/main/reports/replay-native-2026-10-15.json)

## What was verified

The published native weights reproduce every cell of the committed main confusion matrix and the stored accuracy, macro-F1 and macro-AUROC. Model class ordering remains bacterial / normal / viral. An explicit alternative-order audit rejected the sample-driven swapped-label hypothesis; no labels were relabeled to improve results.

The baseline report is retained separately. Its checkpoint is not in this Hub snapshot, so a baseline replay is unavailable. The original test fold is reconstructed and hashed; historical training membership and patient-level independence were not recorded and are not inferred from an image-level replay.

## Exact preprocessing

Production evaluation applies RGB conversion, uint8 tensor bilinear resize to shortest edge 255 with antialiasing, torchvision `CenterCrop(224)`, float32 rescaling 1/255, and ImageNet mean/std normalization.

The old native processor specified resize 256 and bicubic interpolation. A config-only 255/bilinear correction still differed: Transformers truncates the center anchor, while torchvision rounds it. `data/hf_processor.py` overrides only that crop method with torchvision's implementation. The exported native processor carries this small MIT-licensed module and its license notice.

Tensor tests cover square/nonsquare inputs and every committed sample. Real prediction comparisons cover all 624 original test images: zero differing top-1 predictions, identical confusion matrix and AUROC, maximum softmax-score difference 1.67e-6. [Complete parity receipt](https://github.com/kiselyovd/chest-xray-classifier/blob/main/reports/provenance/processor-full-test-parity.json).

The explicit torchvision recipe in the README requires no downloaded processor code. Loading the custom native processor through `AutoImageProcessor` requires reviewing its pinned module and passing `trust_remote_code=True`; native model weights continue to load with safetensors and `trust_remote_code=False`.

## Repeat the run

Use one unpacked `chest_xray/` directory, preserving its original train/val/test folders. Preparation derives bacterial/viral classes from the filename suffixes.

```bash
uv sync --all-groups
uv run python -m chest_xray_classifier.data.prepare --raw /path/to/chest_xray --out data/processed
uv run python -m chest_xray_classifier.evaluation.evaluate \
  --hf-model kiselyovd/chest-xray-classifier \
  --revision d0de246d324b8b1a45523cd4b0a0a514a4aa7d89 \
  --weights-sha256 d3c0b8fba6caa340cfbabf06119c6a9988b018c277a72d9565e0f62acda2a7d0 \
  --out reports/replay-native.json
```

The evaluator rejects changed files or test membership before loading the model, then records class ordering, preprocessing, package/source identities, probabilities, labels and an array-file SHA256. Plot generation checks exact confusion counts and bounded AUROC drift and includes both correct and incorrect cases.
