"""Builds train/val/test DataLoaders from the NIH ChestX-ray14 archive."""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from config.config import AugmentationConfig, DataConfig
from data.augmentation import build_eval_transform, build_train_transform
from data.dataset import NIHChestXrayDataset, build_image_index, load_data_entry, patient_level_split

logger = logging.getLogger(__name__)


def get_dataloaders(
    data_cfg: DataConfig, aug_cfg: AugmentationConfig
) -> tuple[DataLoader, DataLoader, DataLoader, "pd.DataFrame", dict]:
    df = load_data_entry(data_cfg.data_entry_csv)
    image_index = build_image_index(data_cfg.archive_dir)
    splits = patient_level_split(
        df, data_cfg.train_frac, data_cfg.val_frac, data_cfg.test_frac, data_cfg.split_seed
    )

    common_ds_kwargs = dict(
        severity_labels_path=data_cfg.severity_labels_path,
        tb_labels_path=data_cfg.tb_labels_path,
    )
    train_ds = NIHChestXrayDataset(
        df, splits["train"], image_index, transform=build_train_transform(data_cfg, aug_cfg), **common_ds_kwargs
    )
    val_ds = NIHChestXrayDataset(
        df, splits["val"], image_index, transform=build_eval_transform(data_cfg), **common_ds_kwargs
    )
    test_ds = NIHChestXrayDataset(
        df, splits["test"], image_index, transform=build_eval_transform(data_cfg), **common_ds_kwargs
    )

    common_kwargs = dict(
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data_cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True, drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, df, splits
