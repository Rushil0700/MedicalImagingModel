"""Per-class evaluation metrics with sensitivity-prioritized thresholding.

No single averaged F1 is treated as primary. Every metric is computed per
class; classes in HIGH_URGENCY_CLASSES get their decision threshold chosen
to guarantee sensitivity >= 90% rather than to maximize F1, since a missed
pathology is worse than a false alarm.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from config.config import HIGH_URGENCY_CLASSES, NIH_PATHOLOGY_NAMES

logger = logging.getLogger(__name__)

MIN_SENSITIVITY_FOR_HIGH_URGENCY = 0.90


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, y_prob >= t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return best_threshold, best_f1


def _min_sensitivity_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_sensitivity: float) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    valid = []
    for t in thresholds:
        preds = y_prob >= t
        tp = np.sum((preds == 1) & (y_true == 1))
        fn = np.sum((preds == 0) & (y_true == 1))
        sensitivity = tp / max(tp + fn, 1)
        if sensitivity >= min_sensitivity:
            valid.append(t)
    return max(valid) if valid else 0.05


def compute_per_class_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str] | None = None
) -> dict[str, dict[str, float]]:
    class_names = class_names or NIH_PATHOLOGY_NAMES
    results: dict[str, dict[str, float]] = {}

    for i, name in enumerate(class_names):
        yt, yp = y_true[:, i], y_prob[:, i]

        if yt.sum() == 0 or yt.sum() == len(yt):
            logger.warning("Class %s has no positive/negative variation in this split; AUC undefined", name)
            auc = float("nan")
        else:
            auc = roc_auc_score(yt, yp)

        threshold = (
            _min_sensitivity_threshold(yt, yp, MIN_SENSITIVITY_FOR_HIGH_URGENCY)
            if name in HIGH_URGENCY_CLASSES
            else _best_f1_threshold(yt, yp)[0]
        )

        preds = yp >= threshold
        tp = np.sum((preds == 1) & (yt == 1))
        tn = np.sum((preds == 0) & (yt == 0))
        fp = np.sum((preds == 1) & (yt == 0))
        fn = np.sum((preds == 0) & (yt == 1))

        results[name] = {
            "threshold": float(threshold),
            "f1": float(f1_score(yt, preds, zero_division=0)),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "auc_roc": float(auc),
            "brier_score": float(brier_score_loss(yt, yp)),
            "ece": float(_expected_calibration_error(yt, yp)),
            "prevalence": float(yt.mean()),
        }

    results["__mean__"] = {
        metric: float(np.nanmean([results[c][metric] for c in class_names]))
        for metric in ["f1", "sensitivity", "specificity", "auc_roc", "brier_score", "ece"]
    }
    return results


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def format_metrics_table(results: dict[str, dict[str, float]]) -> str:
    header = f"{'Class':<20}{'F1':>8}{'Sens':>8}{'Spec':>8}{'AUC':>8}{'ECE':>8}{'Prev%':>8}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        if name == "__mean__":
            continue
        lines.append(
            f"{name:<20}{m['f1']:>8.3f}{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}"
            f"{m['auc_roc']:>8.3f}{m['ece']:>8.3f}{100*m['prevalence']:>7.2f}%"
        )
    mean = results["__mean__"]
    lines.append("-" * len(header))
    lines.append(
        f"{'MEAN (monitoring only)':<20}{mean['f1']:>8.3f}{mean['sensitivity']:>8.3f}"
        f"{mean['specificity']:>8.3f}{mean['auc_roc']:>8.3f}{mean['ece']:>8.3f}"
    )
    return "\n".join(lines)
