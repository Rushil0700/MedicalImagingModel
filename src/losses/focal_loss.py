"""Focal loss with class-balanced per-class alpha, and heteroscedastic NLL.

Implements docs/training_strategy.md section 3. Two loss terms combine here:

1. Binary focal loss per class, down-weighting easy negatives (the
   "No Finding" majority) and up-weighting hard/rare positives. The flat
   alpha=0.25 from the design brief is refined into a per-class alpha via
   the "effective number of samples" re-weighting (Cui et al., 2019),
   because measured NIH ChestX-ray14 prevalence is far more skewed than a
   flat alpha assumes (Hernia at 0.20% vs. Infiltration at 17.74%).
2. A Gaussian heteroscedastic NLL term that uses the Bayesian head's
   learned log-variance, which is what teaches the aleatoric-uncertainty
   estimate in src/models/uncertainty.py to be meaningful rather than a
   frozen constant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def effective_number_alpha(class_counts: Tensor, beta: float = 0.999) -> Tensor:
    """Per-class alpha weights via the effective-number-of-samples formula.

    alpha_c = (1 - beta) / (1 - beta^n_c), normalized to mean 1 across classes
    so the overall loss scale stays comparable to a flat-alpha focal loss.

    Args:
        class_counts: (C,) positive-sample count per class from the training split.
        beta: Effective-number hyperparameter, beta -> 1 approaches inverse-frequency
            weighting; beta=0.999 (per docs/training_strategy.md) is a moderate choice.

    Returns:
        (C,) alpha weights, mean-normalized to 1.0.
    """
    effective_num = 1.0 - torch.pow(beta, class_counts.clamp(min=1))
    alpha = (1.0 - beta) / effective_num
    return alpha * (len(alpha) / alpha.sum())


class BinaryFocalLoss(nn.Module):
    """Multi-label binary focal loss with optional per-class alpha weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Tensor = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "alpha", alpha if isinstance(alpha, Tensor) else torch.tensor(alpha), persistent=False
        )
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Args:
            logits: (B, C) raw disease logits (pre-sigmoid).
            targets: (B, C) binary labels in {0, 1}.
        """
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma

        # NOTE: this intentionally does NOT use the textbook
        # `alpha*targets + (1-alpha)*(1-targets)` split. That formula only
        # stays non-negative when alpha is a single scalar in [0, 1]
        # (RetinaNet's original use case: one alpha shared across all
        # classes). Here `alpha` can be a per-class tensor from
        # `effective_number_alpha`, which deliberately produces values above
        # 1 for rare classes (e.g. Hernia measured at ~3.65, see
        # docs/training_strategy.md class-prevalence table) so that rare
        # positives get amplified. Plugging alpha>1 into `(1-alpha)` for the
        # negative term goes negative, which flips the loss's sign on every
        # negative sample of that class and rewards the model for being
        # *more* wrong -- this caused the model's logits to diverge without
        # bound during a real training run. Weighting only the positive term
        # by alpha and leaving negatives at a constant weight of 1 keeps the
        # loss non-negative for any alpha >= 0, matching the actual intent
        # (correct for positive-class rarity, not negative-class balance).
        alpha_t = self.alpha * targets + (1 - targets)

        loss = alpha_t * modulating * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none": (B, C), per-class losses preserved for logging


def heteroscedastic_nll(mean: Tensor, log_var: Tensor, targets: Tensor) -> Tensor:
    """Gaussian negative log-likelihood over sigmoid-space residuals.

    Applied on top of (not instead of) the focal loss: focal loss drives
    classification accuracy, this term additionally teaches `log_var` to
    reflect per-sample aleatoric uncertainty (larger predicted variance where
    the model's mean prediction is less reliable).
    """
    residual = torch.sigmoid(mean) - targets
    precision = torch.exp(-log_var)
    return (0.5 * precision * residual**2 + 0.5 * log_var).mean()


class DiseaseLoss(nn.Module):
    """Combines class-balanced focal loss with the heteroscedastic NLL term."""

    def __init__(
        self,
        class_counts: Tensor | None = None,
        gamma: float = 2.0,
        flat_alpha: float = 0.25,
        use_class_balanced_alpha: bool = True,
        class_balanced_beta: float = 0.999,
        use_heteroscedastic_nll: bool = True,
        heteroscedastic_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if use_class_balanced_alpha and class_counts is not None:
            alpha = effective_number_alpha(class_counts, beta=class_balanced_beta)
        else:
            alpha = torch.tensor(flat_alpha)
        self.focal = BinaryFocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
        self.use_heteroscedastic_nll = use_heteroscedastic_nll
        self.heteroscedastic_weight = heteroscedastic_weight

    def forward(self, disease_mean: Tensor, disease_log_var: Tensor, targets: Tensor) -> dict[str, Tensor]:
        focal = self.focal(disease_mean, targets)
        total = focal
        nll = torch.tensor(0.0, device=disease_mean.device)
        if self.use_heteroscedastic_nll:
            nll = heteroscedastic_nll(disease_mean, disease_log_var, targets)
            total = total + self.heteroscedastic_weight * nll
        return {"total": total, "focal": focal, "heteroscedastic_nll": nll}
