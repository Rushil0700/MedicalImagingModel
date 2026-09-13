"""Quick pipeline validation: a few epochs on a data subset.

Not a real training run -- this exists to answer one question before
committing real compute to a full run: does the model actually learn on
real data, or is something in the pipeline broken in a way a tiny smoke
test would be too small to reveal?

Usage:
    python sanity_check.py
    python sanity_check.py --archive-dir /content/archive --num-workers 2 --device cuda
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from config.config import NIH_PATHOLOGY_NAMES, PNEUMONIA_INDEX, TrainConfig
from data.dataset import compute_class_counts
from data.preprocessing import get_dataloaders
from models.chest_model import ChestMedicalNet
from training.losses import ChestMedicalNetLoss
from training.trainer import MedicalImageTrainer
from utils.logger import setup_logging

TRAIN_SUBSET_SIZE = 3000
VAL_SUBSET_SIZE = 600
NUM_EPOCHS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--backbone", choices=["convnext_base", "convnext_large"], default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()
    cfg = TrainConfig()
    if args.archive_dir is not None:
        cfg.data.archive_dir = args.archive_dir
        cfg.data.data_entry_csv = args.archive_dir / "Data_Entry_2017.csv"
    if args.device is not None:
        cfg.device = args.device
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    cfg.data.batch_size = 8
    cfg.data.num_workers = args.num_workers if args.num_workers is not None else 2
    cfg.optim.max_epochs = NUM_EPOCHS
    cfg.optim.warmup_epochs = 1
    cfg.optim.cosine_t_max = NUM_EPOCHS
    cfg.optim.early_stop_patience = NUM_EPOCHS
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, test_loader, df, splits = get_dataloaders(cfg.data, cfg.aug)

    train_subset = Subset(train_loader.dataset, range(min(TRAIN_SUBSET_SIZE, len(train_loader.dataset))))
    val_subset = Subset(val_loader.dataset, range(min(VAL_SUBSET_SIZE, len(val_loader.dataset))))
    train_loader_small = DataLoader(train_subset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader_small = DataLoader(val_subset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    logger.info("Sanity check on subset: train=%d val=%d images", len(train_subset), len(val_subset))

    class_counts = torch.tensor(
        compute_class_counts(df, splits["train"][:TRAIN_SUBSET_SIZE]), dtype=torch.float32
    )
    general_indices = [i for i, c in enumerate(NIH_PATHOLOGY_NAMES) if c != "Pneumonia"]
    general_class_counts = class_counts[general_indices]
    pneumonia_count = float(class_counts[PNEUMONIA_INDEX])

    model = ChestMedicalNet(
        input_size=cfg.model.input_size,
        backbone_variant=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        fpn_channels=cfg.model.fpn_channels,
        cbam_reduction=cfg.model.cbam_reduction,
        head_hidden_dim=cfg.model.head_hidden_dim,
        mc_dropout_p=cfg.model.mc_dropout_p,
        num_general_classes=cfg.model.num_general_classes,
        num_severity_levels=cfg.model.num_severity_levels,
        enable_severity_head=cfg.model.enable_severity_head,
        enable_tb_pathway=cfg.model.enable_tb_pathway,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("ChestMedicalNet (%s) parameter count: %.1fM", cfg.model.backbone, num_params / 1e6)

    loss_fn = ChestMedicalNetLoss(
        general_class_counts=general_class_counts,
        pneumonia_count=pneumonia_count,
        gamma=cfg.loss.focal_gamma,
        flat_alpha=cfg.loss.focal_alpha,
        use_class_balanced_alpha=cfg.loss.use_class_balanced_alpha,
        class_balanced_beta=cfg.loss.class_balanced_beta,
        use_heteroscedastic_nll=cfg.loss.use_heteroscedastic_nll,
        heteroscedastic_weight=cfg.loss.heteroscedastic_weight,
        severity_weight=cfg.loss.severity_loss_weight,
        num_severity_levels=cfg.model.num_severity_levels,
    )

    cfg.checkpoint_dir = cfg.checkpoint_dir / "sanity_check"
    trainer = MedicalImageTrainer(model, loss_fn, train_loader_small, val_loader_small, cfg, model_type="chest", model_name="custom")
    logger.info("Device: %s", trainer.device)
    trainer.train()

    logger.info("Sanity check complete. If val F1 improves and train loss stays small/bounded "
                "(not diverging), the pipeline is learning correctly on this subset.")


if __name__ == "__main__":
    main()
