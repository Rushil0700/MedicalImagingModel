"""Loss functions for ChestMedicalNet's pathway-split output.

Focal loss and the class-balanced alpha weighting are carried over unchanged
from validated prior work, INCLUDING a fix for a real bug found in that
work: the textbook `alpha_t = alpha*target + (1-alpha)*(1-target)` formula
only stays non-negative when alpha is a scalar in [0, 1]. The class-balanced
per-class alpha (effective-number-of-samples weighting) deliberately
produces values above 1 for rare classes, which makes `(1-alpha)` go
negative for negative samples -- flipping the loss's sign and rewarding the
model for being *more* wrong. This was caught via a step-by-step diagnostic
against real data (logits diverged from O(1) to 79+ within 15 steps before
the fix). The fix here weights only the positive term by alpha and leaves
negatives at a constant weight of 1, which stays non-negative for any
alpha >= 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.config import NUM_SEVERITY_LEVELS
from utils.helpers import (
    disease_target_to_general,
    disease_target_to_pneumonia,
    severity_target_to_pathway_order,
)


def effective_number_alpha(class_counts: Tensor, beta: float = 0.999) -> Tensor:
    """Per-class alpha weights via the effective-number-of-samples formula,
    mean-normalized to 1 across classes.
    """
    effective_num = 1.0 - torch.pow(beta, class_counts.clamp(min=1))
    alpha = (1.0 - beta) / effective_num
    return alpha * (len(alpha) / alpha.sum())


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | Tensor = 0.25, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "alpha", alpha if isinstance(alpha, Tensor) else torch.tensor(alpha), persistent=False
        )
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        # See module docstring: only the positive term is alpha-weighted.
        alpha_t = self.alpha * targets + (1 - targets)
        loss = alpha_t * modulating * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def heteroscedastic_nll(mean: Tensor, log_var: Tensor, targets: Tensor) -> Tensor:
    """Gaussian NLL over sigmoid-space residuals, teaching the Bayesian
    head's log_var to reflect per-sample aleatoric uncertainty.
    """
    residual = torch.sigmoid(mean) - targets
    precision = torch.exp(-log_var)
    return (0.5 * precision * residual**2 + 0.5 * log_var).mean()


def masked_bce(logits: Tensor, targets: Tensor) -> Tensor:
    """BCE with targets of -1 treated as 'unknown' and excluded (e.g. the
    TB pathway, which has no genuine label in NIH ChestX-ray14 -- see
    config.config.TB_LABEL_AVAILABLE). Returns 0.0 if nothing is known.
    """
    known = targets >= 0
    if not known.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[known], targets[known])


def levels_to_coral_targets(levels: Tensor, num_severity_levels: int) -> Tensor:
    thresholds = torch.arange(num_severity_levels - 1, device=levels.device)
    return (levels.unsqueeze(-1) > thresholds).float()


class CoralOrdinalLoss(nn.Module):
    """CORAL ordinal loss with masking for unknown severity labels (-1)."""

    def __init__(self, num_severity_levels: int = NUM_SEVERITY_LEVELS) -> None:
        super().__init__()
        self.num_severity_levels = num_severity_levels

    def forward(self, severity_logits: Tensor, severity_targets: Tensor) -> Tensor:
        known_mask = severity_targets >= 0
        if not known_mask.any():
            return severity_logits.sum() * 0.0

        safe_targets = severity_targets.clamp(min=0)
        coral_targets = levels_to_coral_targets(safe_targets, self.num_severity_levels)
        per_threshold = F.binary_cross_entropy_with_logits(
            severity_logits, coral_targets, reduction="none"
        ).mean(dim=-1)
        return per_threshold[known_mask].mean()


def decode_coral_levels(severity_logits: Tensor) -> Tensor:
    return (torch.sigmoid(severity_logits) > 0.5).sum(dim=-1)


class ChestMedicalNetLoss(nn.Module):
    """Combines the pneumonia, general, TB, and severity losses for
    ChestMedicalNet's pathway-split output.
    """

    def __init__(
        self,
        general_class_counts: Tensor | None = None,
        pneumonia_count: float | None = None,
        gamma: float = 2.0,
        flat_alpha: float = 0.25,
        use_class_balanced_alpha: bool = True,
        class_balanced_beta: float = 0.999,
        use_heteroscedastic_nll: bool = True,
        heteroscedastic_weight: float = 0.1,
        severity_weight: float = 0.3,
        num_severity_levels: int = NUM_SEVERITY_LEVELS,
    ) -> None:
        super().__init__()
        if use_class_balanced_alpha and general_class_counts is not None:
            general_alpha = effective_number_alpha(general_class_counts, beta=class_balanced_beta)
        else:
            general_alpha = torch.tensor(flat_alpha)
        self.general_focal = BinaryFocalLoss(gamma=gamma, alpha=general_alpha, reduction="mean")

        if use_class_balanced_alpha and pneumonia_count is not None:
            pneumonia_alpha = effective_number_alpha(torch.tensor([pneumonia_count]), beta=class_balanced_beta)[0]
        else:
            pneumonia_alpha = torch.tensor(flat_alpha)
        self.pneumonia_focal = BinaryFocalLoss(gamma=gamma, alpha=pneumonia_alpha, reduction="mean")

        self.use_heteroscedastic_nll = use_heteroscedastic_nll
        self.heteroscedastic_weight = heteroscedastic_weight
        self.severity_weight = severity_weight
        self.severity_loss = CoralOrdinalLoss(num_severity_levels)

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        general_target = disease_target_to_general(batch["disease_target"])
        pneumonia_target = disease_target_to_pneumonia(batch["disease_target"])

        general_focal = self.general_focal(outputs["general_mean"], general_target)
        pneumonia_focal = self.pneumonia_focal(outputs["pneumonia_logit"], pneumonia_target)

        nll = outputs["general_mean"].new_tensor(0.0)
        if self.use_heteroscedastic_nll:
            nll = heteroscedastic_nll(outputs["general_mean"], outputs["general_log_var"], general_target)

        tb_loss = outputs["general_mean"].new_tensor(0.0)
        if "tb_logit" in outputs:
            tb_loss = masked_bce(outputs["tb_logit"], batch["tb_target"])

        severity = outputs["general_mean"].new_tensor(0.0)
        if "severity_logits" in outputs:
            severity_target = severity_target_to_pathway_order(batch["severity_target"])
            severity = self.severity_loss(outputs["severity_logits"], severity_target)

        total = (
            general_focal
            + pneumonia_focal
            + self.heteroscedastic_weight * nll
            + tb_loss  # masked to 0 unless real TB labels are supplied
            + self.severity_weight * severity
        )

        return {
            "total": total,
            "general_focal": general_focal,
            "pneumonia_focal": pneumonia_focal,
            "heteroscedastic_nll": nll,
            "tb_loss": tb_loss,
            "severity": severity,
        }
