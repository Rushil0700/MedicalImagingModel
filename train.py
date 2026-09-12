"""Entry point: train ChestXRayMaxViT-v2 end-to-end.

Usage:
    python train.py
    python train.py --archive-dir /content/archive --checkpoint-dir /content/drive/MyDrive/chestxray/checkpoints
    python train.py --resume /content/drive/MyDrive/chestxray/checkpoints/last_checkpoint.pt

CLI overrides exist so the same script runs unmodified locally and on a
Colab VM, where the dataset and checkpoint directories are not `archive/`
and `checkpoints/` relative to the repo -- and so a Colab session that gets
disconnected mid-run can resume from `last_checkpoint.pt` (see
src/training/train.py::Trainer.resume) instead of restarting from scratch.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.config import TrainConfig
from src.data.dataset import compute_class_counts
from src.data.datamodule import build_dataloaders
from src.losses.combined_loss import CombinedLoss
from src.losses.focal_loss import DiseaseLoss
from src.losses.ordinal_loss import CoralOrdinalLoss
from src.models.chest_xray_maxvit_v2 import ChestXRayMaxViTv2
from src.training.metrics import format_metrics_table
from src.training.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=None, help="Root dir containing images_*/images and Data_Entry_2017.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Where to save/read checkpoints")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume training from")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    cfg = TrainConfig()
    if args.archive_dir is not None:
        cfg.data.archive_dir = args.archive_dir
        cfg.data.data_entry_csv = args.archive_dir / "Data_Entry_2017.csv"
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.max_epochs is not None:
        cfg.optim.max_epochs = args.max_epochs
    if args.device is not None:
        cfg.device = args.device

    torch.manual_seed(cfg.seed)

    logger.info("Building dataloaders from %s", cfg.data.archive_dir)
    train_loader, val_loader, test_loader, df, splits = build_dataloaders(cfg.data, cfg.aug)
    class_counts = torch.tensor(compute_class_counts(df, splits["train"]), dtype=torch.float32)
    logger.info("Train class counts: %s", class_counts.tolist())

    model = ChestXRayMaxViTv2(
        input_size=cfg.model.input_size,
        stem_channels=cfg.model.stem_channels,
        block_channels=cfg.model.block_channels,
        block_layers=cfg.model.block_layers,
        fpn_channels=cfg.model.fpn_channels,
        cbam_reduction=cfg.model.cbam_reduction,
        head_hidden_dim=cfg.model.head_hidden_dim,
        mc_dropout_p=cfg.model.mc_dropout_p,
        num_classes=cfg.model.num_classes,
        num_severity_levels=cfg.model.num_severity_levels,
        enable_severity_head=cfg.model.enable_severity_head,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameter count: %.1fM", num_params / 1e6)

    disease_loss = DiseaseLoss(
        class_counts=class_counts,
        gamma=cfg.loss.focal_gamma,
        flat_alpha=cfg.loss.focal_alpha,
        use_class_balanced_alpha=cfg.loss.use_class_balanced_alpha,
        class_balanced_beta=cfg.loss.class_balanced_beta,
        use_heteroscedastic_nll=cfg.loss.use_heteroscedastic_nll,
        heteroscedastic_weight=cfg.loss.heteroscedastic_weight,
    )
    severity_loss = CoralOrdinalLoss(cfg.model.num_severity_levels) if cfg.model.enable_severity_head else None
    if cfg.data.severity_labels_path is None:
        logger.warning(
            "No severity_labels_path configured: severity head will train as a no-op "
            "(NIH ChestX-ray14 has no native severity ground truth; see docs/enhanced_architecture_spec.md)."
        )
    loss_fn = CombinedLoss(disease_loss, severity_loss, severity_weight=cfg.loss.severity_loss_weight)

    trainer = Trainer(model, loss_fn, train_loader, val_loader, cfg)
    logger.info("Training on device: %s", trainer.device)
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.fit()

    logger.info("Final test-set evaluation:")
    test_metrics = trainer.evaluate(test_loader)
    logger.info("\n%s", format_metrics_table(test_metrics))


if __name__ == "__main__":
    main()
