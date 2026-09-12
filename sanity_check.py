"""Quick pipeline validation: a few epochs on a data subset.

Not a real training run -- this exists to answer one question before
committing real compute (many hours on this machine's MPS backend, or
money on a cloud GPU) to the full 100-epoch schedule in
docs/training_strategy.md: does the model actually learn on real data, or
is something in the pipeline broken in a way the earlier 16-image smoke
test was too small to reveal?

Usage:
    python sanity_check.py
    python sanity_check.py --archive-dir /content/archive --num-workers 2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import TrainConfig
from src.data.dataset import compute_class_counts
from src.data.datamodule import build_dataloaders
from src.losses.combined_loss import CombinedLoss
from src.losses.focal_loss import DiseaseLoss
from src.losses.ordinal_loss import CoralOrdinalLoss
from src.models.chest_xray_maxvit_v2 import ChestXRayMaxViTv2
from src.training.train import Trainer

TRAIN_SUBSET_SIZE = 3000
VAL_SUBSET_SIZE = 600
NUM_EPOCHS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    cfg = TrainConfig()
    if args.archive_dir is not None:
        cfg.data.archive_dir = args.archive_dir
        cfg.data.data_entry_csv = args.archive_dir / "Data_Entry_2017.csv"
    if args.device is not None:
        cfg.device = args.device
    cfg.data.batch_size = 8
    cfg.data.num_workers = args.num_workers if args.num_workers is not None else 2
    cfg.optim.max_epochs = NUM_EPOCHS
    cfg.optim.warmup_epochs = 1
    cfg.optim.cosine_t_max = NUM_EPOCHS
    cfg.optim.early_stop_patience = NUM_EPOCHS  # don't early-stop during a sanity check
    torch.manual_seed(cfg.seed)

    train_loader, val_loader, test_loader, df, splits = build_dataloaders(cfg.data, cfg.aug)

    train_subset = Subset(train_loader.dataset, range(min(TRAIN_SUBSET_SIZE, len(train_loader.dataset))))
    val_subset = Subset(val_loader.dataset, range(min(VAL_SUBSET_SIZE, len(val_loader.dataset))))
    train_loader_small = DataLoader(
        train_subset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader_small = DataLoader(
        val_subset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    logger.info("Sanity check on subset: train=%d val=%d images", len(train_subset), len(val_subset))

    class_counts = torch.tensor(
        compute_class_counts(df, splits["train"][:TRAIN_SUBSET_SIZE]), dtype=torch.float32
    )

    model = ChestXRayMaxViTv2(
        input_size=cfg.model.input_size,
        block_channels=cfg.model.block_channels,
        block_layers=cfg.model.block_layers,
        num_classes=cfg.model.num_classes,
        num_severity_levels=cfg.model.num_severity_levels,
        enable_severity_head=cfg.model.enable_severity_head,
    )
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
    loss_fn = CombinedLoss(disease_loss, severity_loss, severity_weight=cfg.loss.severity_loss_weight)

    cfg.checkpoint_dir = cfg.checkpoint_dir / "sanity_check"
    trainer = Trainer(model, loss_fn, train_loader_small, val_loader_small, cfg)
    logger.info("Device: %s", trainer.device)
    trainer.fit()

    logger.info("Sanity check complete. Compare epoch-over-epoch val_mean_f1 above:")
    logger.info(
        "  - If val F1 climbs meaningfully above the ~0.19 degenerate-baseline floor "
        "and train loss trends down, the pipeline is learning correctly."
    )
    logger.info(
        "  - If val F1 stays flat/near-random or loss diverges, something upstream "
        "(labels, loss weighting, LR) needs debugging before a full run is worth the compute."
    )


if __name__ == "__main__":
    main()
