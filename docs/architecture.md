# Architecture

## Data flow

```mermaid
flowchart TD
    A["Kaggle: Chest X-Ray Pneumonia<br/>(pre-split train/val/test)"]:::external
    A -->|sync_data.sh| B["data/raw/<br/>NORMAL / PNEUMONIA"]:::data
    B -->|prepare.py<br/>bacterial/viral filename split| C["data/processed/<br/>3-class PNG dataset"]:::data
    C -->|ImageDataModule| D["Lightning + Hydra<br/>training loop"]:::code
    D --> E["ConvNeXt-V2-Tiny (main)<br/>or DINOv2 linear probe (baseline)"]:::model
    E -->|MLflow logging| F["artifacts/checkpoints/best.ckpt"]:::artifact
    F -->|evaluate.py| R["reports/metrics.json<br/>(acc + F1 + AUROC + CM)"]:::artifact
    F -->|publish_to_hf.py| G["HuggingFace Hub<br/>kiselyovd/chest-xray-classifier"]:::external
    F -->|FastAPI| H["POST /predict<br/>Docker + GHCR"]:::serve

    classDef external fill:#FFE4B5,stroke:#FF8C00,color:#000
    classDef data fill:#E6F3FF,stroke:#4A90E2,color:#000
    classDef code fill:#F0F0F0,stroke:#666,color:#000
    classDef model fill:#E8F5E9,stroke:#4CAF50,color:#000
    classDef artifact fill:#FFF9C4,stroke:#F9A825,color:#000
    classDef serve fill:#F3E5F5,stroke:#9C27B0,color:#000
```

## Model choices

- **Main — ConvNeXt-V2-Tiny.** Modern CNN with strong ImageNet-22k pretraining; good accuracy/compute balance for medical imaging. Pre-trained via `transformers.ConvNextV2ForImageClassification` with a 3-label classification head.
- **Baseline — DINOv2 ViT-S linear probe.** Frozen self-supervised backbone (`facebook/dinov2-small`) with a single trainable linear layer on top of the `[CLS]` embedding. Acts as a methodology sanity check — the main model must clearly beat a frozen generic feature extractor to justify the compute.

## Metrics

| Metric | Why |
|---|---|
| Per-class precision/recall/F1 | Detect class-specific weakness (viral is the hardest + smallest class) |
| Macro F1 | Headline number tolerant of class imbalance |
| Confusion matrix | Expose bacterial↔viral confusions (clinically meaningful) |
| Macro AUROC (OvR) | Threshold-independent separability, averaged over classes |

Accuracy alone is avoided — bacterial_pneumonia dominates the train split.

## Key conventions

- Class index order is alphabetical, matching `torchvision.ImageFolder`: `("bacterial_pneumonia", "normal", "viral_pneumonia")`.
- Checkpoint contains `model_name` in its hyperparameters so `inference.load_model` can rebuild the backbone without metadata supplied by the caller.
- Lightning trainer seed-controlled via Hydra `seed`; deterministic mode on.
