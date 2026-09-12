# Enhanced Architecture Specification: ChestXRayMaxVit-v2

**Phase 1, Document 2 — Days 1-3**
**Target:** 92%+ per-class F1, 0.95+ AUC-ROC (from 87% F1 / 0.935 AUC baseline)

This spec implements the six architectural improvements identified in `architecture_analysis.md`. Parameter counts below are **measured** from the reference implementation (`src/models/chest_xray_maxvit_v2.py`, verified via `scratch_smoketest.py`), not estimates; FLOPs remain engineering estimates pending profiling with `fvcore`.

> **Correction from the original draft of this spec:** the first version of this document proposed widening backbone channels (64/128/256/512 → 96/192/384/768) *and* deepening blocks ([2,2,6,2] → [3,3,9,3]) simultaneously. Building and measuring that configuration showed it produces a ~110M-parameter backbone (~113M total) — more than double the ~50M target — because MaxViT's parameter count scales roughly with the product of width² and depth, not their sum. The corrected design below keeps the depth increase only and leaves channel widths unchanged from the baseline, which measures at ~49M backbone / **~52M total** params, matching the target. See §6 for the full measured breakdown.

## 1. Input Resolution: 224×224 → 256×256

Increasing resolution preserves more native detail from the source radiographs (originals ~2500-3000px) before the backbone's first downsample. 256 is chosen over larger options (e.g., 320/384) because it divides evenly through 5 stride-2 stages (256→128→64→32→16→8) while keeping the FLOPs increase to a manageable ~1.3× over 224² rather than the ~2×+ that 320² would cost.

## 2. Backbone: [2,2,6,2] → [3,3,9,3], Multi-Scale Feature Pyramid

### 2.1 Layer breakdown

| Stage | Input res | Output res | Stride (cumulative) | Channels (unchanged from v1) | Blocks (v1 → v2) | Block type |
|---|---|---|---|---|---|---|
| Stem | 256×256 | 128×128 | 2 | 3→64 | Conv3×3(s2)→Conv3×3(s2) | — |
| Stage 1 | 128×128 | 64×64 | 4 | 64 | 2 → **3** | MBConv+Block Attn+Grid Attn |
| Stage 2 | 64×64 | 32×32 | 8 | 128 | 2 → **3** | MBConv+Block Attn+Grid Attn |
| Stage 3 | 32×32 | 16×16 | 16 | 256 | 6 → **9** | MBConv+Block Attn+Grid Attn |
| Stage 4 | 16×16 | 8×8 | 32 | 512 | 2 → **3** | MBConv+Block Attn+Grid Attn |

Total blocks: 12 → **18**. Channel widths are kept **unchanged** from the baseline (64/128/256/512) — only depth increases. Measured backbone params: baseline [2,2,6,2] ≈ 32M; enhanced [3,3,9,3] at the same width ≈ 49M. Widening channels on top of this (as originally drafted) would push the backbone alone past 100M — see the correction note above.

### 2.2 Multi-scale feature pyramid (4 levels)

Each stage output is tapped **before** the next stage's downsample, giving exactly the four pyramid levels requested:

| Pyramid level | Source | Stride | Resolution (256 input) | Raw channels | FPN lateral→ | FPN channels |
|---|---|---|---|---|---|---|
| P1 | Stage 1 out | 1/4 | 64×64 | 64 | 1×1 conv | 256 |
| P2 | Stage 2 out | 1/8 | 32×32 | 128 | 1×1 conv | 256 |
| P3 | Stage 3 out | 1/16 | 16×16 | 256 | 1×1 conv | 256 |
| P4 | Stage 4 out | 1/32 | 8×8 | 512 | 1×1 conv | 256 |

Top-down pathway: P4 → upsample 2× → add to lateral(P3) → upsample 2× → add to lateral(P2) → upsample 2× → add to lateral(P1), each followed by a 3×3 conv to smooth aliasing (standard FPN). This lets fine-grained P1/P2 detail (needed for Nodule, Mass, early Pneumothorax) combine with the semantically rich P4 context (needed for Cardiomegaly, Effusion) at every level, directly resolving the single-scale bottleneck identified in Document 1.

## 3. Attention: Spatial + Channel (CBAM-style)

Applied at each FPN level **after** the top-down fusion, before the smoothing conv:

- **Channel attention:** GAP + GMP (global avg/max pool) over the feature map → shared 2-layer MLP (reduction ratio 16) → sigmoid → rescale channels. Answers "which of the 256 fused channels matter for this level."
- **Spatial attention:** channel-wise avg + max pool → concat → 7×7 conv → sigmoid → rescale spatial locations. Answers "which pixels matter" — critical for localizing small/focal findings.

Cost: ~0.15M params per level × 4 levels ≈ 0.6M params total — negligible relative to backbone, disproportionately valuable for localization-sensitive classes (Nodule, Mass, Pneumothorax).

## 4. Uncertainty Quantification

Two complementary mechanisms, combined at inference:

| Mechanism | Captures | Implementation |
|---|---|---|
| **MC Dropout** | Epistemic (model) uncertainty | Dropout(p=0.3) kept **active at inference**; run T=20 stochastic forward passes; report mean ± std of sigmoid outputs per class |
| **Bayesian output layer** | Aleatoric (data/label-noise) uncertainty | Final linear layer outputs (μ, log σ²) per class instead of a point logit; trained with a Gaussian NLL / heteroscedastic loss term, sampled via reparameterization during training |

Total predictive uncertainty ≈ epistemic (MC Dropout variance) + aleatoric (learned σ²). Cases where 95% confidence interval spans the decision threshold are flagged for radiologist review — this is the concrete clinical-safety payoff of Document 1's Improvement #5.

## 5. Enhanced Multi-Task Head

Replaces the baseline's single GAP→Linear(512→14) head. Feeds from the fused, attention-weighted pyramid:

```
P1..P4 (attention-weighted) → GAP each → concat [256×4 = 1024-d]
        → shared trunk: Linear(1024→512) → GELU → Dropout(0.3, MC-active)
        ├─ Disease head:   Linear(512→14) → (μ,logσ²) per class → sigmoid(μ) at inference
        └─ Severity head:  Linear(512→14×4) → ordinal/CORAL logits (none/mild/moderate/severe per class)
```

- **Disease head** is the primary multi-label output (unchanged task, richer features + uncertainty).
- **Severity head** uses ordinal regression (CORAL: rank-consistent binary classifiers, or cumulative-link/proportional-odds formulation) rather than plain 4-way softmax, so a "moderate" misclassified as "severe" is penalized less than "moderate" misclassified as "none" — matching how ordinal clinical grades actually behave. This is trained as an auxiliary loss (weight 0.3× relative to the primary disease loss) and additionally regularizes the shared trunk.

## 6. Parameter & FLOPs Summary

Measured from `src/models/chest_xray_maxvit_v2.py` ([3,3,9,3] depths, [64,128,256,512] channels — see the correction note in the introduction):

| Component | v1 (baseline, measured) | v2 (enhanced, measured) |
|---|---|---|
| Backbone (MaxViT blocks) | ~32.4M | **48.8M** |
| FPN (lateral + smoothing convs) | — | 2.6M |
| CBAM attention (×4 levels) | — | 0.03M |
| Shared trunk + disease head + severity head | ~2M (single linear head) | 0.55M |
| **Total parameters** | **~34M** | **~52.0M** |
| Input resolution | 224² | 256² |
| **Estimated FLOPs (single forward pass)** | ~5.6 GFLOPs (estimate) | ~11-13 GFLOPs (estimate, not yet profiled) |
| Estimated inference latency (1×A100, fp16, batch=1) | ~8 ms (estimate) | ~14-16 ms (estimate; ×20 for MC Dropout passes if uncertainty requested) |

The backbone accounts for essentially all of the parameter growth (32.4M → 48.8M from the depth increase alone); FPN, CBAM, and the multi-task head together add only ~3.2M — the "enhancement" machinery is cheap, the backbone depth is what buys capacity. FLOPs and latency figures remain engineering estimates pending profiling with `fvcore`/`torch.profiler`, unlike the parameter counts above which are measured, not estimated.

## 7. Expected Gains by Component (design targets)

| Change | Primary benefit | Est. incremental F1 gain* |
|---|---|---|
| 256² input | More detail for small lesions | +0.5-1.0 pt |
| Multi-scale FPN | Fixes single-scale bottleneck; biggest single lever | +2.0-3.0 pt |
| CBAM attention | Better localization on focal findings | +0.5-1.0 pt |
| Deeper backbone [3,3,9,3] | More capacity for 14-way long-tail | +0.5-1.5 pt |
| Focal loss + imbalance handling (see `training_strategy.md`) | Directly targets Hernia/rare-class recall | +1.0-2.0 pt |
| Multi-task ordinal severity (auxiliary regularization) | Better-structured shared features | +0.3-0.8 pt |
| **Combined (with interaction effects, not purely additive)** | | **+5-8 pt → 92-95% F1** |

*Estimates are directional design targets based on architectural reasoning and comparable published ablations (e.g., FPN and attention ablations in general object-detection/classification literature); they must be validated empirically per-component during Phase 3-4 ablation studies, not treated as guaranteed.

## 8. Training Time Estimate

- Dataset: 112,120 images, 70/15/15 patient-level split → ~78,484 train images (see `training_strategy.md` §2 for why the split must be patient-level, not image-level).
- Batch size 32 → ~2,453 steps/epoch.
- At ~50M params, 256² input, mixed precision on a single A100 (40GB): estimated **~45-55 min/epoch** including validation pass.
- 100 epochs (with early stopping typically firing well before, per the epoch milestones in `training_strategy.md`) → **~50-70 GPU-hours** total, or ~2.5-3 days wall-clock on a single GPU, ~12-18 hours on a 4-GPU DDP setup.
- MC Dropout uncertainty inference (T=20 passes) is inference-time-only overhead and does not affect training time.
