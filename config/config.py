"""Central configuration for the medical imaging AI system.

Ported from the earlier ChestXRayMaxViT-v2 config with the same
patient-level split, augmentation, and loss design (that engineering was
correct and stays); the model section is rebuilt around ChestMedicalNet's
ConvNeXt backbone + disease-specific pathways.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

NIH_PATHOLOGY_NAMES: list[str] = [
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

# NIH ChestX-ray14 has no Tuberculosis label at all. The TB pathway below
# exists structurally (per the project spec's disease-pathway design) but
# has no genuine training signal on this dataset -- it will only become
# trainable if a TB-labeled dataset (e.g. Shenzhen, Montgomery, TBX11K) is
# added separately. Do not interpret its outputs as a real TB classifier
# until that happens.
TB_LABEL_AVAILABLE = False

PNEUMONIA_INDEX = NIH_PATHOLOGY_NAMES.index("Pneumonia")
GENERAL_PATHWAY_NAMES = [c for c in NIH_PATHOLOGY_NAMES if c != "Pneumonia"]  # 13 classes

HIGH_URGENCY_CLASSES: list[str] = ["Pneumothorax", "Pneumonia", "Mass"]

NUM_SEVERITY_LEVELS = 4  # none / mild / moderate / severe


@dataclass
class DataConfig:
    archive_dir: Path = Path("archive")
    data_entry_csv: Path = Path("archive/Data_Entry_2017.csv")
    severity_labels_path: Path | None = None
    tb_labels_path: Path | None = None  # see TB_LABEL_AVAILABLE above

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
    # No vertical or horizontal flip -- see docs/training_strategy.md
    # section 4 for the medical-safety rationale (anatomical plausibility,
    # laterality markers).


@dataclass
class ChestModelConfig:
    input_size: int = 256
    # ConvNeXt-Large (~198M params) is what the spec calls for, but it is
    # a tight fit on a free-tier Colab T4 (16GB) once FPN/attention/heads,
    # gradients, and AdamW's two momentum buffers are added on top. Default
    # to ConvNeXt-Base (~89M) for practical trainability; set to "large" if
    # running on a bigger GPU (A100/L4 40GB+).
    backbone: str = "convnext_base"  # "convnext_base" | "convnext_large"
    pretrained: bool = True
    fpn_channels: int = 256
    cbam_reduction: int = 16
    head_hidden_dim: int = 512
    mc_dropout_p: float = 0.3
    mc_dropout_passes: int = 20
    num_general_classes: int = len(GENERAL_PATHWAY_NAMES)
    num_severity_levels: int = NUM_SEVERITY_LEVELS
    enable_severity_head: bool = True
    enable_tb_pathway: bool = True  # architecturally present; see TB_LABEL_AVAILABLE


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
    model: ChestModelConfig = field(default_factory=ChestModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    seed: int = 42
    device: str = "auto"
