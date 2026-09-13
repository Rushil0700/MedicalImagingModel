"""MedicalImageTrainer: AdamW + cosine schedule with warmup, mixed precision,
gradient clipping, early stopping on validation mean per-class F1, and
checkpoint resume support (for continuing after a Colab session disconnect).
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config.config import TrainConfig
from training.losses import ChestMedicalNetLoss
from training.metrics import compute_per_class_metrics, format_metrics_table
from utils.helpers import assemble_disease_probs

logger = logging.getLogger(__name__)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, t_max: int, eta_min: float, base_lr: float
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(t_max - warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        floor = eta_min / base_lr
        return floor + (1 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class MedicalImageTrainer:
    """Trainer for ChestMedicalNet (model_type='chest', model_name='custom').

    Other (model_type, model_name) combinations from the project spec (the
    lungs model; the 6 modified pretrained-model variants) are not wired up
    yet -- see models/custom_architectures.py::LungsMedicalNet and the
    project status notes for why.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: ChestMedicalNetLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        model_type: str = "chest",
        model_name: str = "custom",
    ) -> None:
        if model_type != "chest" or model_name != "custom":
            raise NotImplementedError(
                f"MedicalImageTrainer only supports model_type='chest', model_name='custom' "
                f"currently (got model_type={model_type!r}, model_name={model_name!r}). "
                "The lungs model and the 6 modified pretrained-model variants are not "
                "implemented -- see project status notes."
            )
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.device = resolve_device(cfg.device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            warmup_epochs=cfg.optim.warmup_epochs,
            t_max=cfg.optim.cosine_t_max,
            eta_min=cfg.optim.cosine_eta_min,
            base_lr=cfg.optim.lr,
        )
        self.scaler = torch.amp.GradScaler(enabled=cfg.optim.use_amp and self.device.type == "cuda")

        self.best_val_f1 = -1.0
        self.epochs_without_improvement = 0
        self.start_epoch = 0

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir.mkdir(parents=True, exist_ok=True)

    def resume(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_f1 = ckpt.get("best_val_f1", ckpt["val_metrics"]["__mean__"]["f1"])
        self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        logger.info("Resumed from %s: starting at epoch %d, best_val_f1=%.4f",
                    checkpoint_path, self.start_epoch, self.best_val_f1)

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.cfg.optim.use_amp):
                outputs = self.model(batch["image"])
                losses = self.loss_fn(outputs, batch)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += losses["total"].item()
            if step % 50 == 0:
                logger.debug(
                    "epoch %d step %d/%d loss=%.4f general=%.4f pneumonia=%.4f tb=%.4f severity=%.4f",
                    epoch, step, len(self.train_loader), losses["total"].item(),
                    losses["general_focal"].item(), losses["pneumonia_focal"].item(),
                    losses["tb_loss"].item(), losses["severity"].item(),
                )

        self.scheduler.step()
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_probs, all_targets = [], []
        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(batch["image"])
            all_probs.append(assemble_disease_probs(outputs).cpu().numpy())
            all_targets.append(batch["disease_target"].cpu().numpy())

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        return compute_per_class_metrics(y_true, y_prob)

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.optim.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(self.val_loader)
            val_f1 = val_metrics["__mean__"]["f1"]

            logger.info("Epoch %d train_loss=%.4f val_mean_f1=%.4f", epoch, train_loss, val_f1)
            logger.info("\n%s", format_metrics_table(val_metrics))

            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self._save_checkpoint(epoch, val_metrics, filename="last_checkpoint.pt")
            if is_best:
                self._save_checkpoint(epoch, val_metrics, filename="best_model.pt")

            if self.epochs_without_improvement >= self.cfg.optim.early_stop_patience:
                logger.info("Early stopping at epoch %d (no val F1 improvement for %d epochs)",
                            epoch, self.cfg.optim.early_stop_patience)
                break

    def _save_checkpoint(self, epoch: int, val_metrics: dict, filename: str) -> None:
        path = self.cfg.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_val_f1": self.best_val_f1,
                "epochs_without_improvement": self.epochs_without_improvement,
                "val_metrics": val_metrics,
                "config": asdict(self.cfg),
            },
            path,
        )
        logger.info("Saved checkpoint to %s (val_mean_f1=%.4f)", path, val_metrics["__mean__"]["f1"])
