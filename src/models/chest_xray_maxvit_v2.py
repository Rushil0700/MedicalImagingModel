"""ChestXRayMaxViT-v2: the full enhanced model from docs/enhanced_architecture_spec.md.

Assembles, in order:
  1. MaxViTBackbone   -- [3,3,9,3] blocks, 4 hierarchical feature maps (src/models/backbone.py)
  2. FeaturePyramidNetwork -- fuses strides 4/8/16/32 into a common width (src/models/fpn.py)
  3. CBAM             -- channel + spatial re-attention at each fused level (src/models/attention.py)
  4. SharedTrunk + DiseaseHead + SeverityHead -- multi-task, uncertainty-aware head (src/models/heads.py)
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.attention import CBAM
from src.models.backbone import MaxViTBackbone
from src.models.fpn import FeaturePyramidNetwork
from src.models.heads import DiseaseHead, SeverityHead, SharedTrunk


class ChestXRayMaxViTv2(nn.Module):
    """Enhanced multi-scale, attention-augmented, uncertainty-aware chest X-ray classifier."""

    def __init__(
        self,
        input_size: int = 256,
        stem_channels: int = 64,
        block_channels: list[int] | None = None,
        block_layers: list[int] | None = None,
        fpn_channels: int = 256,
        cbam_reduction: int = 16,
        head_hidden_dim: int = 512,
        mc_dropout_p: float = 0.3,
        num_classes: int = 14,
        num_severity_levels: int = 4,
        enable_severity_head: bool = True,
    ) -> None:
        super().__init__()
        block_channels = block_channels or [64, 128, 256, 512]
        block_layers = block_layers or [3, 3, 9, 3]

        self.backbone = MaxViTBackbone(
            input_size=input_size,
            stem_channels=stem_channels,
            block_channels=block_channels,
            block_layers=block_layers,
        )
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=fpn_channels)
        self.cbam_levels = nn.ModuleList(
            [CBAM(fpn_channels, reduction=cbam_reduction) for _ in range(4)]
        )
        self.trunk = SharedTrunk(
            in_features=fpn_channels * 4, hidden_dim=head_hidden_dim, dropout_p=mc_dropout_p
        )
        self.disease_head = DiseaseHead(head_hidden_dim, num_classes)
        self.enable_severity_head = enable_severity_head
        if enable_severity_head:
            self.severity_head = SeverityHead(head_hidden_dim, num_classes, num_severity_levels)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input images, shape (B, 3, H, W), H=W=input_size.

        Returns:
            Dict with:
                "disease_mean": (B, num_classes) logit mean (apply sigmoid for probability).
                "disease_log_var": (B, num_classes) learned aleatoric log-variance.
                "severity_logits": (B, num_classes, num_severity_levels - 1) CORAL logits,
                    present only if `enable_severity_head` is True.
        """
        stage_features = self.backbone(x)
        fused = self.fpn(stage_features)
        attended = [cbam(level) for cbam, level in zip(self.cbam_levels, fused)]

        pooled = [level.mean(dim=(2, 3)) for level in attended]  # GAP each level
        trunk_input = torch.cat(pooled, dim=1)  # (B, fpn_channels * 4)

        shared = self.trunk(trunk_input)
        disease_mean, disease_log_var = self.disease_head(shared)

        out = {"disease_mean": disease_mean, "disease_log_var": disease_log_var}
        if self.enable_severity_head:
            out["severity_logits"] = self.severity_head(shared)
        return out
