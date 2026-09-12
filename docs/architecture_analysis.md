# Architecture Analysis: ChestXRayMaxVit Baseline

**Phase 1, Document 1 — Days 1-3**
**Baseline performance:** 87% F1 (per-class average), 0.935 AUC-ROC
**Dataset:** NIH ChestX-ray14, 112,120 frontal chest radiographs, 14 pathology labels (multi-label)

## 1. Architecture Breakdown

ChestXRayMaxVit adapts the MaxViT (Multi-Axis Vision Transformer) hybrid CNN-Transformer backbone to 14-way multi-label disease classification.

| Component | Detail |
|---|---|
| Input | 224×224×3, ImageNet-normalized single-channel X-ray replicated to 3 channels |
| Stem | 2× Conv3×3 (stride 2, stride 1), 224→112, channels 3→64 |
| Stage 1 | 2 MaxViT blocks, C=64, resolution 56×56 (stride 2 downsample at stage entry) |
| Stage 2 | 2 MaxViT blocks, C=128, resolution 28×28 |
| Stage 3 | 6 MaxViT blocks, C=256, resolution 14×14 |
| Stage 4 | 2 MaxViT blocks, C=512, resolution 7×7 |
| Head | Global Average Pool (7×7→1×1) → Dropout(0.2) → Linear(512→14) → Sigmoid |
| Depth config | [2, 2, 6, 2] — 12 MaxViT blocks total |
| Parameters | ~38M |
| Loss | Binary Cross-Entropy (per-class, summed/averaged) |

Each MaxViT block follows the canonical structure: **MBConv** (depthwise-separable inverted bottleneck with squeeze-excitation) → **Block Attention** (windowed local self-attention, 7×7 windows) → **Grid Attention** (dilated/global self-attention over a fixed grid). This Conv-Block-Grid sequence gives each block a local-to-global receptive field in a single pass at linear (not quadratic) complexity in image size, which is the core reason MaxViT was chosen over a pure ViT for a domain with limited labeled data relative to natural-image corpora.

## 2. What Works Well

- **Hybrid inductive bias.** MBConv's convolutional locality prior helps on a dataset of this size (112K images is small by ViT standards) where pure attention models tend to overfit or underperform without heavy pretraining.
- **Linear-complexity global context.** Grid attention lets every stage see the full image without O(n²) full self-attention cost, useful since pathology findings (e.g., cardiomegaly, effusion) are defined relationally (heart size vs. thoracic width, fluid level vs. lung field) rather than by a single local texture.
- **Squeeze-excitation in MBConv** already gives a coarse, implicit form of per-channel recalibration, which is likely a meaningful part of why the baseline reaches 0.935 AUC without any explicit attention head beyond the backbone's own blocks.
- **Single unified backbone + linear head** keeps the model simple to train and reason about — a reasonable MVP architecture and a fair baseline to improve on.

## 3. Bottlenecks

1. **Single-scale features at the head.** Only the Stage 4 output (7×7, stride 32) reaches the classifier. Small or subtle findings — early nodules, focal infiltrates, hairline pneumothorax lines — are exactly the signals most attenuated by 32× downsampling. There is no path for Stage 1/2 high-resolution detail to influence the final prediction.
2. **No explicit spatial or channel re-attention beyond backbone SE blocks.** There is nothing that re-weights *which regions* or *which fused channels* matter most for a *specific* pathology after the backbone has produced its feature map — every disease head reads the same pooled vector.
3. **Limited capacity for a 14-way, long-tailed multi-label problem.** 38M parameters is on the small side (comparable to MaxViT-Tiny) for jointly modeling 14 correlated-but-distinct pathologies whose prevalence spans nearly two orders of magnitude (Hernia at 0.20% of images vs. Infiltration at 17.74%, measured from `Data_Entry_2017.csv`).
4. **Plain BCE loss with no class-imbalance correction.** With Hernia present in only 227 of 112,120 images, unweighted BCE is dominated by the easy majority-negative gradient for rare classes, capping achievable sensitivity on exactly the pathologies where misses are costliest.
5. **No uncertainty quantification.** A single deterministic sigmoid score gives no signal about *when the model doesn't know* — a material gap for a clinical-decision-support context where flagging low-confidence cases for radiologist review matters as much as raw accuracy.
6. **No multi-task or severity signal.** The model outputs binary presence/absence only; it cannot express "mild vs. severe" gradation, which limits clinical usefulness and discards ordinal information that could otherwise regularize training.
7. **Fixed low input resolution (224²).** Chest radiographs are natively high-resolution (the archive's originals are ~2500-3000px); downsampling to 224 before the model ever sees the image discards detail that especially hurts small-lesion classes (Nodule, Mass, early Pneumothorax).

## 4. SOTA Comparison

Illustrative reference points from the chest-radiograph literature (architectures and rough parameter/AUC scale — exact numbers vary by paper's label taxonomy, split, and image resolution, and are shown here for architectural positioning, not as a benchmark claim):

| Model | Params | Relative Strength | Relative Weakness (for this task) |
|---|---|---|---|
| ResNet-50/152 (original Wang et al. 2017 baseline family) | 25M / 60M | Strong, well-understood CNN baseline; fast | Single-scale global pooling; no attention; residual-only inductive bias plateaus on subtle findings |
| DenseNet-121 (CheXNet-style) | 8M | Feature reuse via dense connections aids gradient flow at small param count | Limited capacity; no long-range attention; historically strong on this exact dataset despite small size |
| EfficientNet-B4/B7 | 19M / 66M | Compound-scaled depth/width/resolution is efficient per-FLOP; strong ImageNet transfer | Still convolution-only — same single global-context limitation as ResNet without an attention mechanism |
| ViT-B/L | 86M / 307M | Full global self-attention from layer 1; strong with enough data/pretraining | Quadratic attention cost; weak locality prior makes it data-hungry and prone to overfitting on ~100K images without heavy pretraining |
| Swin-B / ConvNeXt-B | 88M | Hierarchical multi-scale features (exactly what the baseline lacks) + strong pretraining recipes | Larger and more expensive than the current baseline; ConvNeXt has no explicit attention |
| **ChestXRayMaxVit (current)** | **38M** | Hybrid conv+attention, linear-complexity global context | Single-scale head, no re-attention at fusion, no uncertainty, no imbalance-aware loss |

The clear gap versus Swin/ConvNeXt-class models is **hierarchical multi-scale feature use** — those architectures feed every stage's output forward, not just the last. That is the single highest-leverage architectural change available and motivates Improvement #1 below.

## 5. Proposed Improvements

1. **Multi-scale feature pyramid (FPN-style).** Fuse Stage 1-4 outputs (strides 4/8/16/32) instead of using Stage 4 alone, so small/subtle findings retain high-resolution signal at classification time.
2. **Explicit spatial + channel attention at fusion.** Add CBAM-style spatial and channel attention when combining pyramid levels, letting the model learn *where* and *which channels* matter per pathology rather than relying solely on backbone SE blocks.
3. **Deeper backbone ([2,2,6,2] → [3,3,9,3], ~50M params).** Added capacity specifically in Stage 3 (where most semantic disease-relevant features form) to better separate 14 correlated, long-tailed classes.
4. **Imbalance-aware loss (Focal Loss, α=0.25, γ=2.0, optionally per-class α).** Directly targets the Hernia-at-0.20% / Infiltration-at-17.74% imbalance measured in the dataset, down-weighting easy negatives and up-weighting hard/rare positives.
5. **Uncertainty quantification (MC Dropout + Bayesian output layer).** Gives per-prediction epistemic + aleatoric uncertainty, enabling a "flag for radiologist review" pathway for low-confidence cases — a medical-safety requirement, not a nice-to-have.
6. **Multi-task head with ordinal severity regression.** Adds a second head predicting severity grade (none/mild/moderate/severe) per pathology via ordinal regression, which both adds clinical value and acts as an auxiliary regularizer on the shared backbone.
7. **Higher input resolution (224²→256²).** Preserves more of the original high-resolution radiograph detail, directly benefiting small-lesion classes identified as a bottleneck above.

These six architectural changes (1-3, 5-7) plus the loss/training changes (4, covered fully in `training_strategy.md`) form the basis of `enhanced_architecture_spec.md`.
