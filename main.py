#!/usr/bin/env python3
"""Main entry point: train and run inference with ChestMedicalNet.

Usage:
    python main.py train --model chest --architecture custom
    python main.py train --model chest --architecture custom --archive-dir /content/archive \
        --checkpoint-dir /content/drive/MyDrive/ckpt --resume /content/drive/MyDrive/ckpt/last_checkpoint.pt

Only `--model chest --architecture custom` is implemented. `--model lungs`
and the 6 modified pretrained-model variants (`--architecture vit|convnext|
efficientnet|resnet|densenet|hybrid`) are not -- see
models/custom_architectures.py::LungsMedicalNet and project status notes.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config.config import NIH_PATHOLOGY_NAMES, PNEUMONIA_INDEX, TrainConfig
from data.dataset import compute_class_counts
from data.preprocessing import get_dataloaders
from models.chest_model import ChestMedicalNet
from training.losses import ChestMedicalNetLoss
from training.metrics import format_metrics_table
from training.trainer import MedicalImageTrainer
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def build_train_parser(subparsers) -> None:
    p = subparsers.add_parser("train")
    p.add_argument("--model", choices=["chest", "lungs"], required=True)
    p.add_argument("--architecture", choices=["custom"], default="custom",
                    help="Only 'custom' (ChestMedicalNet) is implemented")
    p.add_argument("--archive-dir", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--backbone", choices=["convnext_base", "convnext_large"], default=None)
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"])


def train(args: argparse.Namespace) -> None:
    if args.model != "chest" or args.architecture != "custom":
        raise NotImplementedError(
            f"model={args.model!r} architecture={args.architecture!r} is not implemented. "
            "Only chest/custom (ChestMedicalNet) is wired up currently."
        )

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
    if args.backbone is not None:
        cfg.model.backbone = args.backbone
    if args.device is not None:
        cfg.device = args.device

    torch.manual_seed(cfg.seed)

    logger.info("Building dataloaders from %s", cfg.data.archive_dir)
    train_loader, val_loader, test_loader, df, splits = get_dataloaders(cfg.data, cfg.aug)

    class_counts = torch.tensor(compute_class_counts(df, splits["train"]), dtype=torch.float32)
    # class_counts is in NIH_PATHOLOGY_NAMES order; split into general (13) + pneumonia (1)
    general_indices = [i for i, c in enumerate(NIH_PATHOLOGY_NAMES) if c != "Pneumonia"]
    general_class_counts = class_counts[general_indices]
    pneumonia_count = float(class_counts[PNEUMONIA_INDEX])
    logger.info("General class counts: %s | Pneumonia count: %.0f", general_class_counts.tolist(), pneumonia_count)

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
    if cfg.data.severity_labels_path is None:
        logger.warning("No severity_labels_path configured: severity head trains as a no-op.")
    if cfg.data.tb_labels_path is None:
        logger.warning("No tb_labels_path configured: TB pathway trains as a no-op (see config.config.TB_LABEL_AVAILABLE).")

    trainer = MedicalImageTrainer(model, loss_fn, train_loader, val_loader, cfg, model_type="chest", model_name="custom")
    logger.info("Training on device: %s", trainer.device)
    if args.resume is not None:
        trainer.resume(args.resume)
    trainer.train()

    logger.info("Final test-set evaluation:")
    test_metrics = trainer.validate_epoch(test_loader)
    logger.info("\n%s", format_metrics_table(test_metrics))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_train_parser(subparsers)

    args = parser.parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
