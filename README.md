# ChestXRayMaxViT-v2

Enhanced NIH ChestX-ray14 classifier, targeting 92%+ per-class F1 / 0.95+ AUC-ROC
from an 87% F1 / 0.935 AUC baseline (`ChestXRayMaxViT`, 38M params).

Design rationale for every architectural and training decision lives in `docs/`:

- [`docs/architecture_analysis.md`](docs/architecture_analysis.md) — baseline breakdown, measured bottlenecks, SOTA comparison
- [`docs/enhanced_architecture_spec.md`](docs/enhanced_architecture_spec.md) — full v2 architecture spec, with a correction note recording where the original param-count estimate was wrong and how it was fixed
- [`docs/training_strategy.md`](docs/training_strategy.md) — optimizer/schedule/loss/augmentation/metrics strategy, grounded in the real NIH ChestX-ray14 label distribution

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires the NIH ChestX-ray14 archive at `archive/` (already present: `Data_Entry_2017.csv` + `images_001/` … `images_012/`).

## Architecture (implemented, measured)

`src/models/chest_xray_maxvit_v2.py` assembles:

1. **`MaxViTBackbone`** (`src/models/backbone.py`) — `torchvision.models.maxvit.MaxVit` configured with `[3,3,9,3]` blocks at the baseline's `[64,128,256,512]` channel widths, exposing all 4 stage outputs instead of only the last.
2. **`FeaturePyramidNetwork`** (`src/models/fpn.py`) — fuses strides 4/8/16/32 top-down into a common 256-channel width.
3. **`CBAM`** (`src/models/attention.py`) — channel + spatial attention at each fused pyramid level.
4. **`SharedTrunk` + `DiseaseHead` + `SeverityHead`** (`src/models/heads.py`) — multi-task head: a Bayesian (mean, log-variance) disease head plus an auxiliary CORAL ordinal severity head.
5. **MC Dropout + Bayesian uncertainty** (`src/models/uncertainty.py`) — `mc_dropout_predict()` runs T=20 stochastic passes at inference and combines epistemic + aleatoric uncertainty into a single `total_std` per class, for flagging low-confidence predictions.

Measured parameter count: **52.0M** (backbone 48.8M + FPN 2.6M + CBAM 0.03M + heads 0.55M) — verified via a smoke test, not estimated. See the correction note at the top of `docs/enhanced_architecture_spec.md`: the original design draft proposed widening channels *and* deepening blocks together, which measured at ~113M, more than double target; keeping channels at baseline width and only deepening hits ~52M.

## Known limitation: no real severity labels

NIH ChestX-ray14 has no ordinal severity annotations — only binary presence/absence per pathology. `SeverityHead` and `CoralOrdinalLoss` are fully implemented and wired in, but `ChestXray14Dataset` returns severity targets of `-1` ("unknown") for every image unless a `severity_labels_path` CSV (Image Index → 14 ordinal grades, sourced from e.g. a radiologist re-annotation effort) is supplied in `DataConfig`. With no such file, the severity loss masks out every sample and trains as a no-op — this is by design, not a bug, and is logged as a warning at startup (`train.py`).

## Training

```bash
python train.py
```

`train.py` builds a patient-level 70/15/15 split (by `Patient ID`, never by raw image — see `src/data/dataset.py` for why), trains with AdamW + cosine schedule + 5-epoch warmup + AMP + gradient clipping (all per `docs/training_strategy.md`), early-stops on validation mean per-class F1 (patience 10), and logs a full per-class metrics table (F1, sensitivity, specificity, AUC-ROC, ECE) every epoch. Checkpoints land in `checkpoints/`, keyed on best validation F1, with the full metrics table saved alongside each one.

All hyperparameters are in `src/config.py` (`TrainConfig`), grouped into `DataConfig`, `AugmentationConfig`, `ModelConfig`, `LossConfig`, `OptimConfig` — edit there rather than passing CLI flags.

## Status

Phase 1 (architecture design) and the Phase 2 implementation scaffold (model, losses, data pipeline, training loop, metrics) are complete and have been smoke-tested end-to-end against the real archive data on this machine (forward/backward pass, real `DataLoader` batches, a short real training run). Not yet done: a full training run to convergence, per-component ablation to validate the expected-gain table in `docs/enhanced_architecture_spec.md` section 7, and FLOPs profiling (currently estimates, not measured).
