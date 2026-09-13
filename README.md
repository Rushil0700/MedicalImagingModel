# Medical Imaging AI — ChestMedicalNet

A ConvNeXt-based, multi-pathway chest X-ray classifier for NIH ChestX-ray14, with a Feature Pyramid Network, CBAM attention, Bayesian uncertainty quantification, and an auxiliary CORAL ordinal severity head.

## Project status (read this first)

This project's scope was expanded mid-stream to a larger "chest + lungs + RAG router" system. Here is what's real versus what's a placeholder, stated plainly:

| Component | Status |
|---|---|
| **ChestMedicalNet** (`models/custom_architectures.py`) | Real, implemented, unit-tested (`tests/test_models.py`), smoke-tested end-to-end against real data (forward/backward pass, no divergence over multiple optimizer steps). |
| Pneumonia pathway | Real and trainable — NIH ChestX-ray14 has a genuine Pneumonia label. |
| General pathway (13 other NIH pathologies) | Real and trainable. |
| **TB pathway** | Architecturally present, **not trainable on this project's data**. NIH ChestX-ray14 has no TB label at all. Targets are always "unknown" and masked out of the loss (see `config/config.py::TB_LABEL_AVAILABLE`). Do not treat its output as a real TB classifier without adding a TB-labeled dataset (e.g. Shenzhen, Montgomery, TBX11K). |
| Severity head (CORAL ordinal) | Architecturally present, **not trainable** — NIH ChestX-ray14 has no severity grading either. Trains as a no-op unless `DataConfig.severity_labels_path` is supplied. |
| **LungsMedicalNet** (`models/custom_architectures.py`) | **Skeleton only.** Raises `NotImplementedError` on instantiation. This project has no lung CT data (LUNA16/LIDC-IDRI) and no loader for it. |
| CheXpert / MIMIC-CXR loaders (`data/dataset.py`) | Stubbed, raise `NotImplementedError`. Both need separate, credentialed access — MIMIC-CXR specifically requires a PhysioNet data-use agreement and a completed human-subjects research training course, which cannot be automated. |
| RAG router (`inference/rag_router.py`) | Real, working chest-only routing + embedding-based similar-case retrieval (in-memory, not a persistent vector DB) + reasoning/recommendation text. "Image type detection" always returns `chest` since there's no lungs model to route to. |
| Grad-CAM localization (`inference/visualization.py`) | Real, implemented as the actual post-hoc technique (gradient-weighted activation maps), not a trained "head" — the original spec's phrasing was imprecise about what Grad-CAM is. |
| 6 modified pretrained-model variants (ViT/ConvNeXt/EfficientNet/ResNet/DenseNet/hybrid) | **Not built.** Sequenced as future work once ChestMedicalNet itself has real training results. |

Realistic performance targets stated elsewhere (99% TB/cancer sensitivity, 95-96% F1) are **not validated claims** — they require data and validation this project does not currently have. Treat them as aspirational, not as guaranteed outcomes.

## Architecture

`models/custom_architectures.py::ChestMedicalNet`:
1. **ConvNeXt-Base backbone** (`models/backbone.py`) — ~89M params, chosen over ConvNeXt-Large (~198M) to stay trainable on a free-tier Colab T4 (16GB); configurable via `ChestModelConfig.backbone`.
2. **Feature Pyramid Network** (`models/fpn.py`) — fuses strides 4/8/16/32.
3. **CBAM attention** (`models/attention.py`) — channel + spatial, per pyramid level.
4. **Disease-specific pathways**: TB (binary, untrained — see above), Pneumonia (binary, real), General (13-class, Bayesian mean+log-variance for uncertainty).
5. **CORAL ordinal severity head** (untrained — see above).
6. **MC Dropout** (`models/uncertainty.py`) for epistemic uncertainty, combined with the general pathway's learned aleatoric variance.

Total measured parameters: ~91M.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires the NIH ChestX-ray14 archive at `archive/` (`Data_Entry_2017.csv` + `images_001/` … `images_012/`).

## Training

```bash
python main.py train --model chest --architecture custom
```

`--model lungs` and any `--architecture` other than `custom` raise `NotImplementedError` — see the status table above for why.

Patient-level 70/15/15 split (by `Patient ID`, never by raw image — see `data/dataset.py`), AdamW + cosine schedule + warmup + AMP + gradient clipping, early stopping on validation mean per-class F1 (patience 10), full per-class metrics logged every epoch (F1, sensitivity, specificity, AUC-ROC, calibration). Checkpoints in `checkpoints/`, with resume support (`--resume path/to/last_checkpoint.pt`) for continuing after an interrupted session.

For quick pipeline validation before committing real compute: `python sanity_check.py`.

## Training on Google Colab

See `colab_training.ipynb`. Downloads the dataset via `kagglehub`, mounts Drive for checkpoint persistence, and runs `main.py train`. Push this repo to GitHub and set `REPO_URL` in the notebook's clone cell.

## Background documents

`docs/` contains the original Phase 1 design analysis for an earlier, MaxViT-based single-architecture version of this project (`architecture_analysis.md`, `enhanced_architecture_spec.md`, `training_strategy.md`). The training-strategy reasoning (patient-level splitting, class-balanced focal loss, medical-safe augmentation, sensitivity-prioritized metrics) carried forward into ChestMedicalNet unchanged; the architecture-specific sections describe the superseded MaxViT design, not the current ConvNeXt-based one.
