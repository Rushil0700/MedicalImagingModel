"""Combines the primary disease loss with the auxiliary severity loss.

Total = disease_loss + severity_loss_weight * severity_loss
(docs/training_strategy.md section 3: severity auxiliary weight 0.3x).
"""
from __future__ import annotations

from torch import Tensor, nn

from src.losses.focal_loss import DiseaseLoss
from src.losses.ordinal_loss import CoralOrdinalLoss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        disease_loss: DiseaseLoss,
        severity_loss: CoralOrdinalLoss | None,
        severity_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.disease_loss = disease_loss
        self.severity_loss = severity_loss
        self.severity_weight = severity_weight

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        disease_terms = self.disease_loss(
            outputs["disease_mean"], outputs["disease_log_var"], batch["disease_target"]
        )
        total = disease_terms["total"]

        severity = disease_terms["total"].new_tensor(0.0)
        if self.severity_loss is not None and "severity_logits" in outputs:
            severity = self.severity_loss(outputs["severity_logits"], batch["severity_target"])
            total = total + self.severity_weight * severity

        return {**disease_terms, "severity": severity, "total": total}
