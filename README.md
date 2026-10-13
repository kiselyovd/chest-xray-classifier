# Chest X-Ray Classification

A research classifier for normal, bacterial-pneumonia, and viral-pneumonia chest radiographs. ConvNeXt-V2-Tiny is the main model; a DINOv2 linear probe provides the comparison.

[![CI](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/chest-xray-classifier/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](https://kiselyovd.github.io/chest-xray-classifier/) [![Model](https://img.shields.io/badge/Hugging_Face-model-yellow)](https://huggingface.co/kiselyovd/chest-xray-classifier) [![MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)

[![Interactive comparison of stored test reports](docs/images/showcase.png)](https://kiselyovd.github.io/chest-xray-classifier/showcase/)

**[Explore the classification errors](https://kiselyovd.github.io/chest-xray-classifier/showcase/)** — change the model, read counts or row percentages, and inspect a true/predicted-class cell. The UI uses committed JSON reports; the labelled image gallery is separate from the test results.

## Reported results

Stored evaluation on 624 pediatric radiographs, ordered as bacterial / normal / viral:

| Model | Accuracy | Macro F1 | Macro AUROC (OvR) | Classification errors |
|---|---|---|---|---|
| ConvNeXt-V2-Tiny | 91.3% | 90.3% | 97.5% | 54 |
| DINOv2 linear probe | 85.6% | 84.2% | 94.2% | 90 |

[Main report](reports/metrics.json) · [Baseline report](reports/metrics_baseline.json) · [Protocol and limitations](docs/project-guide.md)

## Try the published model

```python
from urllib.request import urlopen
from PIL import Image
from transformers import pipeline

classifier = pipeline("image-classification", model="kiselyovd/chest-xray-classifier")
image = Image.open(urlopen("https://huggingface.co/kiselyovd/chest-xray-classifier/resolve/main/samples/normal.png")).convert("RGB")
print(classifier(image))
```

The model card contains the direct processor/model API and its class-index order. [Training and local serving commands](docs/project-guide.md#quick-start) · [Русский README](README.ru.md)

## Documentation

- [Interactive report explorer](https://kiselyovd.github.io/chest-xray-classifier/showcase/)
- [Project guide](docs/project-guide.md): dataset, full recipes, inference, and Docker.
- [Architecture](docs/architecture.md) · [Training](docs/training.md) · [Serving](docs/serving.md)

## Intended use and terms

Research and education only. The model has not been clinically validated and is not a medical device. The benchmark uses a limited pediatric dataset.

Repository code is [MIT](LICENSE); dataset, pretrained models, and model artifacts retain their respective terms. See the [model card](https://huggingface.co/kiselyovd/chest-xray-classifier).
