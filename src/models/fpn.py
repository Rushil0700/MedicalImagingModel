"""Feature Pyramid Network fusing the 4 backbone stages.

Implements docs/enhanced_architecture_spec.md section 2.2: lateral 1x1
convs project each backbone stage to a common channel width, a top-down
pathway fuses coarse semantic context (stride 32) into finer levels
(strides 16/8/4), and a 3x3 smoothing conv removes upsampling aliasing at
each fused level.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class FeaturePyramidNetwork(nn.Module):
    """Standard top-down FPN with lateral connections, 4 input levels."""

    def __init__(self, in_channels_list: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list]
        )
        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list]
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Fuse backbone features top-down.

        Args:
            features: 4 feature maps ordered fine-to-coarse, i.e.
                [stride-4, stride-8, stride-16, stride-32].

        Returns:
            4 fused feature maps at the same resolutions and a common
            `out_channels` width, same fine-to-coarse order.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        fused = [laterals[-1]]
        for lateral in reversed(laterals[:-1]):
            top_down = F.interpolate(fused[-1], size=lateral.shape[-2:], mode="nearest")
            fused.append(lateral + top_down)
        fused.reverse()  # back to fine-to-coarse order

        return [smooth(f) for smooth, f in zip(self.smooth_convs, fused)]
