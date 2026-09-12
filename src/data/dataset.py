"""NIH ChestX-ray14 dataset loading with patient-level splitting.

Patient-level splitting matters: the dataset contains multiple follow-up
images per patient (see docs/training_strategy.md, section 2). Splitting by
image instead of by Patient ID would leak a patient's anatomy across
train/val/test and inflate held-out metrics. Every function here that builds
a split operates on unique Patient ID, never on raw image rows.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import NUM_CLASSES, NUM_SEVERITY_LEVELS, PATHOLOGY_NAMES, DataConfig

logger = logging.getLogger(__name__)

_NO_FINDING = "No Finding"


def load_data_entry(csv_path: Path) -> pd.DataFrame:
    """Load and lightly clean the NIH `Data_Entry_2017.csv` metadata file.

    Args:
        csv_path: Path to `Data_Entry_2017.csv`.

    Returns:
        DataFrame with one row per image, including a multi-hot label matrix
        for the 14 pathology classes.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    labels = df["Finding Labels"].str.split("|")
    for cls in PATHOLOGY_NAMES:
        df[cls] = labels.apply(lambda finds, c=cls: int(c in finds))
    return df


def build_image_index(archive_dir: Path) -> dict[str, Path]:
    """Map `Image Index` filenames to their absolute path on disk.

    NIH ChestX-ray14 ships images split across `images_001/images/` ...
    `images_012/images/` subdirectories, so a filename-to-path index must be
    built once up front rather than assuming a single flat directory.
    """
    index: dict[str, Path] = {}
    for images_dir in sorted(archive_dir.glob("images_*/images")):
        for path in images_dir.glob("*.png"):
            index[path.name] = path
    if not index:
        raise FileNotFoundError(
            f"No images found under {archive_dir}/images_*/images — check DataConfig.archive_dir"
        )
    return index


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split image indices into train/val/test by unique Patient ID.

    Args:
        df: DataFrame from `load_data_entry`, must contain "Patient ID".
        train_frac, val_frac, test_frac: Split fractions, must sum to ~1.0.
        seed: RNG seed for reproducible patient shuffling.

    Returns:
        Dict with keys "train"/"val"/"test" mapping to arrays of row indices
        into `df`.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1.0"

    patients = df["Patient ID"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    splits = {
        "train": df.index[df["Patient ID"].isin(train_patients)].to_numpy(),
        "val": df.index[df["Patient ID"].isin(val_patients)].to_numpy(),
        "test": df.index[df["Patient ID"].isin(test_patients)].to_numpy(),
    }
    logger.info(
        "Patient-level split: %d patients (train=%d, val=%d, test=%d) -> "
        "%d images (train=%d, val=%d, test=%d)",
        n,
        len(train_patients),
        len(val_patients),
        len(test_patients),
        len(df),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


class ChestXray14Dataset(Dataset):
    """NIH ChestX-ray14 multi-label dataset with optional severity targets.

    Severity ground truth is NOT part of the released NIH ChestX-ray14
    dataset — only binary presence/absence per pathology is annotated. The
    severity head described in docs/enhanced_architecture_spec.md therefore
    has no genuine training signal unless a `severity_labels_path` CSV
    (Image Index -> 14 ordinal grades) is supplied separately (e.g. from a
    radiologist re-annotation effort). When no such file is provided,
    `severity` targets are returned as -1 ("unknown") for every class and
    must be masked out of the severity loss (see src/losses/ordinal_loss.py).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        image_index: dict[str, Path],
        transform=None,
        severity_labels_path: Path | None = None,
    ) -> None:
        self.df = df.loc[indices].reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform
        self.severity = self._load_severity(severity_labels_path)

    def _load_severity(self, path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        sev_df = pd.read_csv(path).set_index("Image Index")
        missing = [c for c in PATHOLOGY_NAMES if c not in sev_df.columns]
        if missing:
            raise ValueError(f"severity_labels_path is missing columns: {missing}")
        logger.info("Loaded severity labels for %d images from %s", len(sev_df), path)
        return sev_df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_name = row["Image Index"]
        image_path = self.image_index.get(image_name)
        if image_path is None:
            raise FileNotFoundError(f"Image {image_name} not found in image index")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        disease_target = row[PATHOLOGY_NAMES].to_numpy(dtype=np.float32)

        if self.severity is not None and image_name in self.severity.index:
            severity_target = self.severity.loc[image_name, PATHOLOGY_NAMES].to_numpy(dtype=np.int64)
        else:
            severity_target = np.full(NUM_CLASSES, -1, dtype=np.int64)

        return {
            "image": image,
            "disease_target": disease_target,
            "severity_target": severity_target,
            "image_name": image_name,
        }


def compute_class_counts(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Positive-sample counts per class, used for class-balanced loss weighting."""
    return df.loc[indices, PATHOLOGY_NAMES].sum(axis=0).to_numpy()
