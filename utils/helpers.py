"""Small shared utilities for reconciling ChestMedicalNet's pathway-split
output layout (tb / pneumonia / general) with the NIH ChestX-ray14 dataset's
fixed 14-column layout (config.config.NIH_PATHOLOGY_NAMES).
"""
from __future__ import annotations

import torch
from torch import Tensor

from config.config import GENERAL_PATHWAY_NAMES, NIH_PATHOLOGY_NAMES, PNEUMONIA_INDEX

# Column order used by ChestMedicalNet's severity head (14 = 13 general + Pneumonia).
SEVERITY_PATHWAY_NAMES = GENERAL_PATHWAY_NAMES + ["Pneumonia"]
_NIH_TO_SEVERITY_PERM = [NIH_PATHOLOGY_NAMES.index(c) for c in SEVERITY_PATHWAY_NAMES]
_GENERAL_INDICES_IN_NIH = [i for i, c in enumerate(NIH_PATHOLOGY_NAMES) if c != "Pneumonia"]


def disease_target_to_general(disease_target: Tensor) -> Tensor:
    """Select the 13 general-pathway columns from a (B, 14) NIH-ordered target."""
    return disease_target[:, _GENERAL_INDICES_IN_NIH]


def disease_target_to_pneumonia(disease_target: Tensor) -> Tensor:
    """Select the Pneumonia column from a (B, 14) NIH-ordered target."""
    return disease_target[:, PNEUMONIA_INDEX]


def severity_target_to_pathway_order(severity_target: Tensor) -> Tensor:
    """Reorder a (B, 14) NIH-ordered severity target into
    (general-13, Pneumonia) order, matching ChestMedicalNet's severity head.
    """
    return severity_target[:, _NIH_TO_SEVERITY_PERM]


def assemble_disease_probs(outputs: dict[str, Tensor]) -> Tensor:
    """Recombine pneumonia_logit + general_mean into a single (B, 14)
    probability tensor in NIH_PATHOLOGY_NAMES order, for metrics/reporting.
    """
    general_prob = torch.sigmoid(outputs["general_mean"])  # (B, 13)
    pneumonia_prob = torch.sigmoid(outputs["pneumonia_logit"]).unsqueeze(-1)  # (B, 1)

    batch_size = general_prob.shape[0]
    combined = torch.zeros(batch_size, len(NIH_PATHOLOGY_NAMES), device=general_prob.device)
    combined[:, _GENERAL_INDICES_IN_NIH] = general_prob
    combined[:, PNEUMONIA_INDEX] = pneumonia_prob.squeeze(-1)
    return combined
