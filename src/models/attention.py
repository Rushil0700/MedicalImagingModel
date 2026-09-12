"""CBAM-style spatial and channel attention, applied at each FPN level.

Implements docs/enhanced_architecture_spec.md section 3: channel attention
answers "which fused channels matter", spatial attention answers "which
pixels matter" — the latter is what gives the model an explicit way to
localize small/focal findings (Nodule, Mass, Pneumothorax) that the
baseline's global-pool-only head could not express.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention, applied per FPN level."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
