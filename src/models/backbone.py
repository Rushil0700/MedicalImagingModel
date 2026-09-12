"""MaxViT backbone wrapper exposing 4-level hierarchical features.

Wraps `torchvision.models.maxvit.MaxVit`, configured per
docs/enhanced_architecture_spec.md section 2 ([3,3,9,3] blocks,
channels [96,192,384,768]), and returns the per-stage feature maps
(strides 4/8/16/32) instead of pooled logits, so a Feature Pyramid Network
can consume all four scales rather than only the final stage output.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.maxvit import MaxVit


class MaxViTBackbone(nn.Module):
    """MaxViT backbone returning a list of 4 hierarchical feature maps.

    Attributes:
        out_channels: Channel count of each returned feature map, in the
            same [stride-4, stride-8, stride-16, stride-32] order as the
            returned list.
    """

    def __init__(
        self,
        input_size: int = 256,
        stem_channels: int = 64,
        block_channels: list[int] | None = None,
        block_layers: list[int] | None = None,
        partition_size: int = 8,
        head_dim: int = 32,
        stochastic_depth_prob: float = 0.2,
    ) -> None:
        super().__init__()
        block_channels = block_channels or [64, 128, 256, 512]
        block_layers = block_layers or [3, 3, 9, 3]
        assert len(block_channels) == len(block_layers) == 4, "expected exactly 4 stages"
        assert input_size % 32 == 0, "input_size must be divisible by 32 for 4 stride-2 stages + stem"

        backbone = MaxVit(
            input_size=(input_size, input_size),
            stem_channels=stem_channels,
            partition_size=partition_size,
            block_channels=block_channels,
            block_layers=block_layers,
            head_dim=head_dim,
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.stem = backbone.stem
        self.stages = backbone.blocks  # nn.ModuleList of 4 stage modules
        self.out_channels = list(block_channels)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the backbone and return all 4 stage feature maps.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            List of 4 feature maps at strides [4, 8, 16, 32] relative to
            input resolution, channels matching `self.out_channels`.
        """
        h = self.stem(x)
        features = []
        for stage in self.stages:
            h = stage(h)
            features.append(h)
        return features
