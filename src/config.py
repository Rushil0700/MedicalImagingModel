"""Central configuration for ChestXRayMaxVit-v2.

All hyperparameters here trace directly back to the Phase 1 design docs:
docs/enhanced_architecture_spec.md (architecture) and
docs/training_strategy.md (optimization, loss, data, augmentation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PATHOLOGY_NAMES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# Pathologies where a missed positive (false negative) carries high clinical
# risk. These classes get a sensitivity floor enforced at threshold-selection
# time (docs/training_strategy.md, section 6).
HIGH_URGENCY_CLASSES: list[str] = ["Pneumothorax", "Pneumonia", "Mass"]

NUM_CLASSES = len(PATHOLOGY_NAMES)
NUM_SEVERITY_LEVELS = 4  # none / mild / moderate / severe


@dataclass
class DataConfig:
    archive_dir: Path = Path("archive")
    data_entry_csv: Path = Path("archive/Data_Entry_2017.csv")
    images_glob: str = "archive/images_*/images"
    splits_dir: Path = Path("data/splits")
    severity_labels_path: Path | None = None  # see docs note in dataset.py

    image_size: int = 256
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 42

    batch_size: int = 32
    num_workers: int = 8


@dataclass
class AugmentationConfig:
    rotation_degrees: float = 15.0
    elastic_alpha: tuple[float, float] = (20.0, 30.0)
    elastic_sigma: tuple[float, float] = (4.0, 6.0)
    elastic_p: float = 0.5
    brightness_contrast_jitter: float = 0.15
    gaussian_noise_std_frac: float = 0.02
    gaussian_noise_p: float = 0.3
    # Deliberately absent: vertical_flip, horizontal_flip.
    # See docs/training_strategy.md section 4 for the medical-safety rationale.


@dataclass
class ModelConfig:
    input_size: int = 256
    stem_channels: int = 64
    # Channel widths are unchanged from the baseline (64/128/256/512) -- only
    # depth increases ([2,2,6,2] -> [3,3,9,3]). Widening channels AND
    # deepening simultaneously was the original design-doc proposal but
    # measures out to ~110M backbone params (see docs/enhanced_architecture_spec.md
    # section 6 correction note), well past the ~50M target. Depth-only
    # scaling at the original width measures at ~49M backbone params, which
    # lands on target once FPN/CBAM/head params are added.
    block_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    block_layers: list[int] = field(default_factory=lambda: [3, 3, 9, 3])
    fpn_channels: int = 256
    cbam_reduction: int = 16
    head_hidden_dim: int = 512
    mc_dropout_p: float = 0.3
    mc_dropout_passes: int = 20
    enable_severity_head: bool = True
    num_classes: int = NUM_CLASSES
    num_severity_levels: int = NUM_SEVERITY_LEVELS


@dataclass
class LossConfig:
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_class_balanced_alpha: bool = True
    class_balanced_beta: float = 0.999
    severity_loss_weight: float = 0.3
    use_heteroscedastic_nll: bool = True
    heteroscedastic_weight: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    max_epochs: int = 100
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    use_amp: bool = True


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    seed: int = 42
    device: str = "auto"  # "auto" resolves to cuda > mps > cpu
