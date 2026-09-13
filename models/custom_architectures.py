"""Custom architectures: ChestMedicalNet (real, trainable) and LungsMedicalNet
(structural skeleton only -- see its docstring for why).
"""
from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from config.config import GENERAL_PATHWAY_NAMES, NUM_SEVERITY_LEVELS, TB_LABEL_AVAILABLE
from models.attention import CBAM
from models.backbone import ConvNeXtBackbone
from models.fpn import FeaturePyramidNetwork
from models.uncertainty import BayesianLinear

logger = logging.getLogger(__name__)


class SeverityHead(nn.Module):
    """CORAL-style ordinal regression head: `num_severity_levels - 1`
    rank-consistent binary classifiers per class. See training/losses.py's
    `CoralOrdinalLoss` for the matching loss and decoding logic. Applies to
    all 14 NIH pathologies (severity grading is a per-finding concept,
    independent of the TB/Pneumonia/General pathway split used for disease
    presence).
    """

    def __init__(self, hidden_dim: int, num_classes: int, num_severity_levels: int) -> None:
        super().__init__()
        num_thresholds = num_severity_levels - 1
        self.shared_logit = nn.Linear(hidden_dim, num_classes)
        self.thresholds = nn.Parameter(torch.zeros(num_classes, num_thresholds))

    def forward(self, x: Tensor) -> Tensor:
        base_logit = self.shared_logit(x)
        return base_logit.unsqueeze(-1) + self.thresholds.unsqueeze(0)


class ChestMedicalNet(nn.Module):
    """ConvNeXt backbone + FPN + CBAM + disease-specific pathways.

    Pathways (per the project spec's disease-specific pathway design):
      - `tb_pathway`: binary logit. Architecturally present, but NIH
        ChestX-ray14 (the only dataset available in this project) has no TB
        label at all -- see `config.config.TB_LABEL_AVAILABLE`. Training
        against this dataset alone will only ever see target=-1 ("unknown")
        for this pathway (matching the masking pattern used for severity
        labels), so it cannot become a real TB classifier until a
        TB-labeled dataset (Shenzhen/Montgomery/TBX11K) is added.
      - `pneumonia_pathway`: binary logit. This one IS genuinely trainable
        -- NIH ChestX-ray14 has a real "Pneumonia" label.
      - `general_pathway`: the other 13 NIH pathologies, as a Bayesian
        (mean, log-variance) multi-label head -- this is where the model's
        uncertainty quantification (MC Dropout epistemic + learned
        aleatoric variance) lives, satisfying the spec's "uncertainty head"
        requirement without duplicating that machinery three times over.
      - `severity_head`: auxiliary CORAL ordinal regression over all 14
        pathologies (see its own docstring for why severity labels are
        also unavailable in NIH ChestX-ray14 and how that's handled).

    Localization ("Grad-CAM output" in the spec) is deliberately NOT a
    trained head here -- Grad-CAM is a post-hoc technique computed from
    gradients flowing back through a chosen conv layer, not something a
    network outputs directly. See `inference/visualization.py::GradCAM`.
    """

    def __init__(
        self,
        input_size: int = 256,
        backbone_variant: str = "convnext_base",
        pretrained: bool = True,
        fpn_channels: int = 256,
        cbam_reduction: int = 16,
        head_hidden_dim: int = 512,
        mc_dropout_p: float = 0.3,
        num_general_classes: int | None = None,
        num_severity_levels: int = NUM_SEVERITY_LEVELS,
        enable_severity_head: bool = True,
        enable_tb_pathway: bool = True,
    ) -> None:
        super().__init__()
        num_general_classes = num_general_classes or len(GENERAL_PATHWAY_NAMES)

        self.backbone = ConvNeXtBackbone(variant=backbone_variant, pretrained=pretrained)
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=fpn_channels)
        self.cbam_levels = nn.ModuleList([CBAM(fpn_channels, reduction=cbam_reduction) for _ in range(4)])

        self.trunk = nn.Sequential(
            nn.Linear(fpn_channels * 4, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(mc_dropout_p),
        )

        self.enable_tb_pathway = enable_tb_pathway
        if enable_tb_pathway:
            if not TB_LABEL_AVAILABLE:
                logger.warning(
                    "ChestMedicalNet's TB pathway is enabled but NIH ChestX-ray14 has no "
                    "TB label -- this pathway will train as a no-op (masked loss) until a "
                    "TB-labeled dataset is added. See config.config.TB_LABEL_AVAILABLE."
                )
            self.tb_pathway = nn.Linear(head_hidden_dim, 1)

        self.pneumonia_pathway = nn.Linear(head_hidden_dim, 1)
        self.general_pathway = BayesianLinear(head_hidden_dim, num_general_classes)

        self.enable_severity_head = enable_severity_head
        if enable_severity_head:
            self.severity_head = SeverityHead(head_hidden_dim, num_general_classes + 1, num_severity_levels)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Returns a dict with `tb_logit` (if enabled), `pneumonia_logit`,
        `general_mean`, `general_log_var`, `severity_logits` (if enabled),
        and `fpn_features` (the attention-weighted pyramid, for Grad-CAM /
        embedding-based case retrieval in inference/rag_router.py).
        """
        stage_features = self.backbone(x)
        fused = self.fpn(stage_features)
        attended = [cbam(level) for cbam, level in zip(self.cbam_levels, fused)]

        pooled = [level.mean(dim=(2, 3)) for level in attended]
        trunk_input = torch.cat(pooled, dim=1)
        shared = self.trunk(trunk_input)

        out: dict[str, Tensor] = {"embedding": shared, "fpn_features": attended}
        if self.enable_tb_pathway:
            out["tb_logit"] = self.tb_pathway(shared).squeeze(-1)
        out["pneumonia_logit"] = self.pneumonia_pathway(shared).squeeze(-1)
        out["general_mean"], out["general_log_var"] = self.general_pathway(shared)
        if self.enable_severity_head:
            out["severity_logits"] = self.severity_head(shared)
        return out


class LungsMedicalNet(nn.Module):
    """Structural skeleton only -- NOT trained, NOT validated.

    This project has no lung CT data (LUNA16/LIDC-IDRI or similar) and no
    loader for it (see `data.dataset.load_lung_ct_volumes`, which raises
    NotImplementedError). Instantiating this class will not crash, but its
    outputs are meaningless: every weight is randomly initialized and there
    is no training pipeline behind it. It exists so the module/class shape
    described in the project spec is present and importable, not as a
    claim that lung-CT cancer detection works.

    Do not wire this into `inference/rag_router.py`'s routing logic as if it
    were a real, usable model.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "LungsMedicalNet requires 3D lung CT data (e.g. LUNA16, LIDC-IDRI) that is "
            "not present in this project, and a 3D backbone/data pipeline that has not "
            "been built or validated. Acquire the data and implement "
            "data.dataset.load_lung_ct_volumes first."
        )
