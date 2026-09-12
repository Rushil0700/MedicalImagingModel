"""Multi-task head: disease presence (Bayesian, multi-label) + ordinal severity.

Implements docs/enhanced_architecture_spec.md section 5. The shared trunk
consumes GAP-pooled, attention-weighted features from all 4 FPN levels; the
disease head is the primary task, the severity head is an auxiliary,
rank-consistent ordinal regression (CORAL) trained at 0.3x weight.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.uncertainty import BayesianLinear


class SharedTrunk(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 512, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DiseaseHead(nn.Module):
    """Primary multi-label disease head with a Bayesian (mean, log-var) output."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.bayesian_linear = BayesianLinear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.bayesian_linear(x)


class SeverityHead(nn.Module):
    """CORAL-style ordinal regression head: `num_severity_levels - 1` rank-consistent
    binary classifiers per class, whose summed sigmoid outputs give the predicted
    ordinal level (see src/losses/ordinal_loss.py for the corresponding loss and
    decoding logic).
    """

    def __init__(self, hidden_dim: int, num_classes: int, num_severity_levels: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_severity_levels = num_severity_levels
        num_thresholds = num_severity_levels - 1
        # Shared per-class feature projection, then per-class rank-consistent biases.
        self.shared_logit = nn.Linear(hidden_dim, num_classes)
        self.thresholds = nn.Parameter(torch.zeros(num_classes, num_thresholds))

    def forward(self, x: Tensor) -> Tensor:
        """Returns CORAL logits, shape (B, num_classes, num_severity_levels - 1)."""
        base_logit = self.shared_logit(x)  # (B, num_classes)
        return base_logit.unsqueeze(-1) + self.thresholds.unsqueeze(0)
