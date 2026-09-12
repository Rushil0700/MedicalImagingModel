# Training Strategy: ChestXRayMaxVit-v2

**Phase 1, Document 3 — Days 1-3**
**Target:** 92%+ per-class F1, 0.95+ AUC-ROC, sensitivity-prioritized for medical safety

## 1. Optimizer & Schedule

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay avoids the L2-in-Adam interaction that distorts effective regularization strength |
| Learning rate | 1e-4 | Standard for fine-tuning a ~50M-param hybrid conv-transformer at batch size 32 |
| Weight decay | 1e-5 | Light regularization; the model already has dropout + focal loss + augmentation as primary regularizers |
| Scheduler | CosineAnnealingLR, T_max=100, eta_min=1e-6 | Smooth decay avoids the sharp drops of step schedules, which tend to destabilize the Bayesian output layer's variance term late in training |
| Warmup | Linear, 5 epochs, 0 → 1e-4 | Prevents early large updates from destabilizing the newly-added FPN/attention/uncertainty layers, which start from random init while the backbone may be pretrained |
| Gradient clipping | max norm 1.0 | Bounds gradient spikes from the focal loss's hard-example weighting and the Gaussian NLL term's log σ² sensitivity |
| Mixed precision | AMP (fp16/bf16) | ~1.5-2× throughput and memory headroom for the larger 256² / 50M-param model |
| Batch size | 32 | Fits comfortably in memory at 256² with AMP on a single 40GB GPU; gradient accumulation (×2-4) available if a larger effective batch is needed for BatchNorm stability |
| Epochs | 100 (max) | See §5 for epoch-by-epoch expected trajectory |
| Early stopping | patience=10, monitored on validation mean per-class F1 (not loss) | Loss can keep improving via the auxiliary severity/uncertainty terms after the primary classification metric plateaus; F1 is the metric that matters |

## 2. Data Split — Patient-Level, Not Image-Level

**Critical methodological requirement:** NIH ChestX-ray14 contains multiple follow-up images per patient (e.g., `00000001_000.png`, `00000001_001.png`, ... for the same patient at different visits — confirmed in `Data_Entry_2017.csv`, column `Patient ID`). A naive random 70/15/15 split **by image** would place different scans of the same patient across train/val/test, leaking patient-specific anatomy and creating inflated validation/test metrics that would not hold on truly unseen patients. The split must therefore be performed **by unique Patient ID**, with all images for a given patient assigned entirely to one split.

| Split | Fraction | Approx. images (of 112,120) | Purpose |
|---|---|---|---|
| Train | 70% | ~78,484 | Model fitting |
| Validation | 15% | ~16,818 | Threshold tuning, early stopping, checkpoint selection |
| Test | 15% | ~16,818 | Final, untouched evaluation only |

This differs intentionally from the dataset's official `train_val_list.txt` / `test_list.txt` (≈77%/23%, already patient-disjoint) — the custom 70/15/15 patient-level re-split adds a dedicated validation set for early stopping and calibration, which the official two-way split does not provide.

## 3. Loss Function — Focal Loss for Class Imbalance

Baseline focal loss: `FL(p_t) = -α(1-p_t)^γ log(p_t)`, with **α=0.25, γ=2.0**, applied per-class per-sample and summed over the 14 disease logits (multi-label, one binary focal loss per class, not a single softmax).

Measured class prevalence (from `Data_Entry_2017.csv`, 112,120 images) confirms this is necessary and shows the imbalance is more severe than typically assumed:

| Class | Prevalence | Notes |
|---|---|---|
| No Finding | 53.84% | Majority "negative" class across all 14 heads |
| Infiltration | 17.74% | Most common finding |
| Effusion | 11.88% | |
| Atelectasis | 10.31% | |
| Nodule | 5.65% | Small-lesion class — benefits most from Doc 2's FPN |
| Mass | 5.16% | Small-lesion class |
| Pneumothorax | 4.73% | High clinical urgency — sensitivity floor applies (see §6) |
| Consolidation | 4.16% | |
| Pleural_Thickening | 3.02% | |
| Cardiomegaly | 2.48% | |
| Emphysema | 2.24% | |
| Edema | 2.05% | |
| Fibrosis | 1.50% | |
| Pneumonia | 1.28% | High clinical urgency — sensitivity floor applies |
| **Hernia** | **0.20%** | Extreme minority class (227/112,120 images) |

Because Hernia's true prevalence (0.20%) is more extreme than a flat α=0.25 assumes, the fixed focal-loss α/γ is supplemented with **per-class α weighting via the effective number of samples** (Cui et al.): `α_c = (1-β) / (1-β^n_c)`, β≈0.999, normalized across classes. This gives Hernia meaningfully more gradient weight than a flat scheme would, without manual per-class tuning. The ordinal severity auxiliary head (Doc 2 §5) uses its own CORAL/cumulative-link loss, weighted 0.3× relative to the primary focal loss.

## 4. Augmentation — Medical-Safe Only

All augmentations must preserve anatomical plausibility; standard natural-image augmentations that would corrupt clinical meaning are explicitly excluded.

| Augmentation | Setting | Included? | Rationale |
|---|---|---|---|
| Rotation | ±15° | Yes | Small rotations mimic real positioning variance without inverting anatomy |
| Elastic deformation | Low magnitude (α=20-30, σ=4-6) | Yes | Simulates natural soft-tissue/breathing variation |
| Brightness/contrast jitter | ±10-15% | Yes | Mimics exposure/detector variation across machines |
| Gaussian noise | Low σ, ~1-3% of intensity range | Yes | Simulates detector noise; improves robustness |
| **Vertical flip** | — | **No** | Inverts superior/inferior anatomy (e.g., aortic arch, diaphragm position) — produces anatomically impossible images and would teach the model incorrect priors |
| **Horizontal flip** | — | **No (excluded by default)** | Chest X-rays carry laterality (L/R) markers, and pathologies like dextrocardia are rare but real; flipping risks corrupting laterality-dependent findings (e.g., which lung has an effusion) and is disabled by default, gated behind a config flag if a future ablation wants to test it |
| Random crop / zoom | Not used | No | Risks cropping out the pathology-bearing region entirely in a small-lesion class (Nodule, Mass) |

Augmentations are applied only to the training split; validation/test images receive resize + normalize only.

## 5. Training Trajectory & Expected Milestones

| Epoch | Expected mean per-class F1 | Notes |
|---|---|---|
| 1 | ~70% | End of warmup; backbone pretrained weights dominate, new FPN/attention/head layers still adapting |
| 20 | ~83% | Cosine decay well underway; FPN and attention fusion converging |
| 50 | ~89% | Fine-grained convergence; severity auxiliary head fully contributing to shared-feature regularization |
| 100 (or early-stopped) | ~92%+ | Target reached; early stopping (patience=10 on val F1) expected to trigger before epoch 100 if plateau occurs earlier |

These are **design-stage expected milestones** for monitoring training health, not guarantees — if actual F1 significantly lags these checkpoints (e.g., <75% at epoch 20), it signals a bug (label leakage, LR mismatch, loss weighting) rather than expected variance, and training should be paused for diagnosis rather than run to completion.

## 6. Metrics — Per-Class Only, Sensitivity-Prioritized

**No single averaged F1 is treated as the primary metric.** Given the prevalence range (0.20%-53.84%), a macro-average is dominated by common classes and a micro-average is dominated by "No Finding," both masking failure on rare-but-critical classes. All of the following are reported **per class**:

- **Per-class F1** at a per-class-tuned decision threshold (threshold selected on validation set to maximize F1 or to hit a sensitivity floor — see below — not a fixed 0.5 cutoff).
- **Per-class sensitivity (recall) and specificity**, with sensitivity explicitly prioritized over specificity: for high-urgency findings (Pneumothorax, Pneumonia, Mass), the decision threshold is chosen to guarantee **sensitivity ≥ 90%** on the validation set even at some specificity cost, since a missed pathology (false negative) is categorically more dangerous than a false alarm in a screening/triage context.
- **Per-class AUC-ROC**, threshold-independent, used for model comparison and early-stopping-adjacent diagnostics separate from the operating-point-dependent F1/sensitivity numbers.
- **Calibration**: per-class Expected Calibration Error (ECE), Brier score, and reliability diagrams, evaluated against the Bayesian output layer's predicted uncertainty (Doc 2 §4) — a well-calibrated model's stated confidence should match its actual accuracy, which is what makes the uncertainty-flagging pathway for radiologist review trustworthy rather than decorative.

## 7. Logging & Code Quality

Training code for Phase 2+ should follow: full type hints on all public functions, docstrings (Google or NumPy style) on every module/class/function, `logging` (not `print`) for all training/eval output with per-epoch summaries at INFO level and per-batch diagnostics at DEBUG level, and checkpointing keyed on validation mean per-class F1 with the full metrics table (per-class F1/sensitivity/specificity/AUC/calibration) persisted alongside each checkpoint for later audit.
