"""NIH ChestX-ray14 dataset loading with patient-level splitting.

Patient-level splitting matters: the dataset contains multiple follow-up
images per patient. Splitting by image instead of by Patient ID would leak
a patient's anatomy across train/val/test and inflate held-out metrics.
Every split-building function here operates on unique Patient ID, never on
raw image rows.

CheXpert and MIMIC-CXR loaders are stubbed below (not implemented): both
require separate, credentialed access -- MIMIC-CXR specifically requires a
PhysioNet data-use agreement plus a completed human-subjects research
training course, which is not something that can be automated here. Lung
CT loading (LUNA16/LIDC-IDRI) is likewise stubbed pending that data being
acquired; see LungsMedicalNet's docstring in models/custom_architectures.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config.config import NIH_PATHOLOGY_NAMES, DataConfig

logger = logging.getLogger(__name__)


def load_data_entry(csv_path: Path) -> pd.DataFrame:
    """Load and lightly clean the NIH `Data_Entry_2017.csv` metadata file."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    labels = df["Finding Labels"].str.split("|")
    for cls in NIH_PATHOLOGY_NAMES:
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
            f"No images found under {archive_dir}/images_*/images -- check DataConfig.archive_dir"
        )
    return index


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split image indices into train/val/test by unique Patient ID."""
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
        n, len(train_patients), len(val_patients), len(test_patients),
        len(df), len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


class NIHChestXrayDataset(Dataset):
    """NIH ChestX-ray14 multi-label dataset with optional severity/TB targets.

    Severity and TB ground truth are NOT part of the released NIH
    ChestX-ray14 labels -- only binary presence/absence per (non-TB)
    pathology is annotated. When `severity_labels_path` / `tb_labels_path`
    are not supplied, those targets come back as -1 ("unknown") and must be
    masked out of their respective losses (see training/losses.py).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        image_index: dict[str, Path],
        transform=None,
        severity_labels_path: Path | None = None,
        tb_labels_path: Path | None = None,
    ) -> None:
        self.df = df.loc[indices].reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform
        self.severity = self._load_side_labels(severity_labels_path, "severity")
        self.tb = self._load_side_labels(tb_labels_path, "TB")

    def _load_side_labels(self, path: Path | None, name: str) -> pd.DataFrame | None:
        if path is None:
            return None
        side_df = pd.read_csv(path).set_index("Image Index")
        logger.info("Loaded %s labels for %d images from %s", name, len(side_df), path)
        return side_df

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

        disease_target = row[NIH_PATHOLOGY_NAMES].to_numpy(dtype=np.float32)

        if self.severity is not None and image_name in self.severity.index:
            severity_target = self.severity.loc[image_name, NIH_PATHOLOGY_NAMES].to_numpy(dtype=np.int64)
        else:
            severity_target = np.full(len(NIH_PATHOLOGY_NAMES), -1, dtype=np.int64)

        if self.tb is not None and image_name in self.tb.index:
            tb_target = np.float32(self.tb.loc[image_name, "TB"])
        else:
            tb_target = np.float32(-1)  # unknown -- masked out of the TB loss

        return {
            "image": image,
            "disease_target": disease_target,
            "severity_target": severity_target,
            "tb_target": tb_target,
            "image_name": image_name,
        }


def compute_class_counts(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Positive-sample counts per class, used for class-balanced loss weighting."""
    return df.loc[indices, NIH_PATHOLOGY_NAMES].sum(axis=0).to_numpy()


def load_chexpert(*_args, **_kwargs):
    raise NotImplementedError(
        "CheXpert (223,648 images) requires a separate download from "
        "stanfordmlgroup.github.io/competitions/chexpert/ and its own "
        "uncertainty-label handling (the dataset marks some labels 'uncertain' "
        "rather than binary). Not wired up -- no CheXpert data is present in "
        "this project."
    )


def load_mimic_cxr(*_args, **_kwargs):
    raise NotImplementedError(
        "MIMIC-CXR (~371K images) requires a PhysioNet data-use agreement and "
        "a completed CITI human-subjects research training course before "
        "access is granted -- this is a manual credentialing process that "
        "cannot be automated. Not wired up -- no MIMIC-CXR data is present in "
        "this project."
    )


def load_lung_ct_volumes(*_args, **_kwargs):
    raise NotImplementedError(
        "Lung CT volumes (e.g. LUNA16, LIDC-IDRI) are not present in this "
        "project. LungsMedicalNet (models/custom_architectures.py) is a "
        "structural skeleton only until this data is acquired and a loader "
        "is implemented against its actual on-disk format."
    )
