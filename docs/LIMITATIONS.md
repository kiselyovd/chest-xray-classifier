# Limitations & Failure Modes

This page is a deliberate call-out of where the published model falls short. Read it before using the weights for anything beyond demo / research.

## Dataset scope

- **Pediatric population bias**: the Kaggle Chest X-Ray Pneumonia set (Kermany et al., 2018) is sourced from a pediatric cohort (Guangzhou Women and Children's Medical Center). Adult X-rays — different cardiothoracic ratio, bone density, calcifications — are out of distribution; expect accuracy to drop on them.
- **Single-institution sourcing**: All 5856 images come from one hospital's imaging pipeline. Scanner model, acquisition protocol, and post-processing differ across institutions and will shift the input distribution.
- **Small held-out test set**: 624 test images. Reported 91.3% accuracy has a ~±1.5% confidence interval — don't over-interpret fractional gains.

## Class boundaries

- **Bacterial vs viral pneumonia**: these classes overlap in radiographic appearance; even experienced radiologists disagree in a meaningful fraction of cases. The model's per-class recall reflects this — `viral` recall is ~3pp lower than `normal`.
- **Only 3 classes**: Normal / Bacterial / Viral. Real clinical chest X-rays contain many other findings (TB, lung cancer, COVID-19, pneumothorax, pleural effusion) that the model has never seen. Any prediction on such an input is spurious.

## Known failure modes

- **Heavy infiltrate patterns** (whiteout consolidation) get predicted as bacterial pneumonia regardless of etiology.
- **Grid-like artifacts** from bedside portable radiographs sometimes trigger false-positive classifications.
- **Low-contrast or underexposed** films degrade performance sharply — no automatic contrast normalisation is applied beyond standard ImageNet mean/std.
- **Lateral views** were not in the training set; the model will confidently mis-classify them.

## Adversarial & reliability

- No adversarial robustness testing has been performed. The model is trained on clean images only.
- No uncertainty estimation — the model outputs a single softmax probability. For clinical use you'd want calibrated confidence or an ensemble.

## Not a medical device

- This model is **not FDA-cleared, CE-marked, or clinically validated**. It is a computer-vision portfolio project.
- Any use beyond research / educational demos requires independent clinical validation by qualified professionals.

## What this project *is* good for

- Demonstrating a production ML pipeline: Hydra configs, Lightning training loop, FastAPI serving, Docker/CI/CD, HuggingFace Hub distribution.
- A reproducible baseline others can fork and extend with better data, calibration, and explainability.
- Comparing ConvNeXt-V2 vs DINOv2 on a small chest-X-ray dataset with a deliberately modest training budget.
