"""ConvNeXt backbone wrapper exposing 4-level hierarchical features.

Wraps `torchvision.models.convnext_base` / `convnext_large`, returning the
per-stage feature maps (strides 4/8/16/32) instead of pooled logits, so a
Feature Pyramid Network can consume all four scales.

torchvision's ConvNeXt exposes `.features` as a flat `nn.Sequential` of 8
children: [stem(/4), stage1(/4), downsample(/8), stage2(/8), downsample(/16),
stage3(/16), downsample(/32), stage4(/32)] -- verified empirically (see
project history), so the 4 pyramid levels are the outputs after children
[1, 3, 5, 7].
"""
from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import convnext_base, convnext_large

_STAGE_OUTPUT_INDICES = (1, 3, 5, 7)

_BACKBONE_BUILDERS = {
    "convnext_base": (convnext_base, [128, 256, 512, 1024]),
    "convnext_large": (convnext_large, [192, 384, 768, 1536]),
}


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone returning a list of 4 hierarchical feature maps."""

    def __init__(self, variant: str = "convnext_base", pretrained: bool = True) -> None:
        super().__init__()
        if variant not in _BACKBONE_BUILDERS:
            raise ValueError(f"Unknown ConvNeXt variant '{variant}', expected one of {list(_BACKBONE_BUILDERS)}")
        builder, channels = _BACKBONE_BUILDERS[variant]
        weights = "DEFAULT" if pretrained else None
        backbone = builder(weights=weights)
        self.features = backbone.features
        self.out_channels = channels

    def forward(self, x: Tensor) -> list[Tensor]:
        """Returns 4 feature maps at strides [4, 8, 16, 32]."""
        features = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in _STAGE_OUTPUT_INDICES:
                features.append(h)
        return features
