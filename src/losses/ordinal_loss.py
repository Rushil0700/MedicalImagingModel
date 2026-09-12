"""CORAL ordinal regression loss for the auxiliary severity head.

Implements docs/enhanced_architecture_spec.md section 5: rank-consistent
binary classifiers (Cao et al., "Rank consistent ordinal regression for
neural networks", 2020) so that predicting "severe" for a true "moderate"
case is penalized less than predicting "none" for the same case, matching
how ordinal clinical grades behave.

Severity ground truth does not exist in the released NIH ChestX-ray14
labels (see src/data/dataset.py docstring) -- targets of -1 mean "unknown"
and are masked out of the loss entirely. With no `severity_labels_path`
configured, this loss always evaluates to 0 and the severity head trains
as a no-op auxiliary branch until real severity labels are supplied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def levels_to_coral_targets(levels: Tensor, num_severity_levels: int) -> Tensor:
    """Expand ordinal level indices into CORAL's cumulative binary targets.

    Args:
        levels: (B, C) integer severity levels in [0, num_severity_levels).
        num_severity_levels: Total number of ordinal grades (4: none/mild/moderate/severe).

    Returns:
        (B, C, num_severity_levels - 1) binary targets, where entry k is 1 iff
        `levels > k`.
    """
    thresholds = torch.arange(num_severity_levels - 1, device=levels.device)
    return (levels.unsqueeze(-1) > thresholds).float()


class CoralOrdinalLoss(nn.Module):
    """CORAL loss with per-(sample, class) masking for unknown severity labels."""

    def __init__(self, num_severity_levels: int = 4) -> None:
        super().__init__()
        self.num_severity_levels = num_severity_levels

    def forward(self, severity_logits: Tensor, severity_targets: Tensor) -> Tensor:
        """Args:
            severity_logits: (B, C, num_severity_levels - 1) from SeverityHead.
            severity_targets: (B, C) integer levels, or -1 where unknown/unlabeled.

        Returns:
            Scalar loss, masked-mean over known (sample, class) pairs. Returns
            0.0 if no severity labels are present in this batch.
        """
        known_mask = severity_targets >= 0  # (B, C)
        if not known_mask.any():
            return severity_logits.sum() * 0.0

        safe_targets = severity_targets.clamp(min=0)
        coral_targets = levels_to_coral_targets(safe_targets, self.num_severity_levels)  # (B, C, K-1)

        per_threshold = F.binary_cross_entropy_with_logits(
            severity_logits, coral_targets, reduction="none"
        ).mean(dim=-1)  # (B, C)

        return per_threshold[known_mask].mean()


def decode_coral_levels(severity_logits: Tensor) -> Tensor:
    """Decode CORAL logits back to a single predicted ordinal level per class.

    Sums the number of thresholds exceeded (sigmoid > 0.5), which is the
    standard CORAL decoding rule and stays rank-consistent by construction.
    """
    return (torch.sigmoid(severity_logits) > 0.5).sum(dim=-1)
