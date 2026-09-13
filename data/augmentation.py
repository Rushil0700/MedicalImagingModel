"""Medical-safe augmentation pipeline.

Every transform here preserves anatomical plausibility. Vertical flip and
horizontal flip are deliberately absent: vertical flip inverts
superior/inferior anatomy (aortic arch, diaphragm position), and horizontal
flip risks corrupting laterality-dependent findings (chest X-rays carry L/R
markers; a flipped image would misrepresent which lung a finding is in).
"""
from __future__ import annotations

import torch
import torchvision.transforms.v2 as T
from torch import nn

from config.config import AugmentationConfig, DataConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(data_cfg: DataConfig, aug_cfg: AugmentationConfig) -> nn.Module:
    return T.Compose(
        [
            T.Resize((data_cfg.image_size, data_cfg.image_size)),
            T.RandomRotation(degrees=aug_cfg.rotation_degrees),
            T.RandomApply(
                [T.ElasticTransform(alpha=aug_cfg.elastic_alpha[1], sigma=aug_cfg.elastic_sigma[1])],
                p=aug_cfg.elastic_p,
            ),
            T.ColorJitter(
                brightness=aug_cfg.brightness_contrast_jitter,
                contrast=aug_cfg.brightness_contrast_jitter,
            ),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomApply(
                [T.GaussianNoise(mean=0.0, sigma=aug_cfg.gaussian_noise_std_frac)],
                p=aug_cfg.gaussian_noise_p,
            ),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(data_cfg: DataConfig) -> nn.Module:
    return T.Compose(
        [
            T.Resize((data_cfg.image_size, data_cfg.image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
