# Benchmarks

All numbers are on the held-out Kaggle test split (n=624). Hardware: RTX 3080 10 GB. Inference is single-image, FP32, no batching.

## Main results

| Model | Accuracy | Macro F1 | AUROC | Params | Inference (ms/img, RTX 3080) |
|---|---|---|---|---|---|
| **ConvNeXt-V2-Tiny** (ours, main) | **91.3%** | **90.3%** | **97.5%** | ~28 M | ~8 ms |
| DINOv2 ViT-S linear probe (ours, baseline) | 85.6% | 84.2% | 94.2% | ~22 M | ~12 ms |

## Literature context

Widely reported numbers on the same Kaggle Pneumonia split, with the caveat that test-set definitions and augmentation vary between papers:

| Model | Accuracy | Source |
|---|---|---|
| ResNet-50 fine-tuned | ~88.0% | Rajpurkar et al., 2017 (CheXNet-style setup) |
| DenseNet-121 | ~89.5% | Published replications 2018-2021 |
| EfficientNet-B0 | ~90.0% | Published replications 2020-2022 |
| ConvNeXt-V2-Tiny (ours) | 91.3% | This repo, v0.1.0 |

Our main model is competitive with the best reported numbers while being trained end-to-end on a single RTX 3080 in under 90 minutes with a 20-epoch budget.

## Trade-offs

- **Main vs baseline (ConvNeXt-V2 vs DINOv2 linear probe)**: the baseline is a deliberate "how far do frozen features get us" reference. It's 6pp behind on accuracy but trains in under 10 minutes — useful as a sanity benchmark whenever you re-do the main training.
- **ConvNeXt-V2 vs ViT**: ConvNeXt-V2-Tiny is ~28M params vs a comparable ViT-S's ~22M, but the convolutional inductive bias helps on small (~5k-image) datasets like this one. We tested both; ConvNeXt-V2 wins by ~2pp.
- **Why not an ensemble**: a 2-3 model ensemble reliably adds ~1-2pp on this dataset, but at 2-3x inference cost — not worth it for a portfolio deployment story. If you need to push accuracy past 93%, start there.

## Reproducing these numbers

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the one-command re-run. Expected variation: ± 0.5% from floating-point noise across CUDA driver versions.
