from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve


@dataclass
class EvaluationResult:
    detection_accuracy: float
    power: float              # True positive rate (TPR)
    type1_error: float        # False positive rate (FPR)
    localization_errors: np.ndarray  # |tau_hat - tau| for each true change
    mean_localization_error: float
    median_localization_error: float

    def __str__(self) -> str:
        return (
            f"Detection accuracy:       {self.detection_accuracy:.4f}\n"
            f"Power (TPR):              {self.power:.4f}\n"
            f"Type-I error (FPR):       {self.type1_error:.4f}\n"
            f"Mean localization error:  {self.mean_localization_error:.2f}\n"
            f"Median localization error:{self.median_localization_error:.2f}"
        )


def evaluate_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taus_true: np.ndarray,
    taus_pred: np.ndarray,
) -> EvaluationResult:
    """Compute detection and localization metrics.

    Args:
        y_true: (N,) binary ground truth — 1=change, 0=no change
        y_pred: (N,) binary predictions
        taus_true: (N,) true change locations (0 if y_true=0)
        taus_pred: (N,) predicted change locations (0 if y_pred=0)

    Returns:
        EvaluationResult
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    taus_true = np.asarray(taus_true)
    taus_pred = np.asarray(taus_pred)

    detection_accuracy = float((y_true == y_pred).mean())

    # Power: among true positives, fraction correctly detected
    pos_mask = y_true == 1
    power = float(y_pred[pos_mask].mean()) if pos_mask.any() else 0.0

    # Type-I error: among true negatives, fraction falsely flagged
    neg_mask = y_true == 0
    type1_error = float(y_pred[neg_mask].mean()) if neg_mask.any() else 0.0

    # Localization error on true-positive sequences
    tp_mask = (y_true == 1) & (y_pred == 1)
    if tp_mask.any():
        errors = np.abs(taus_pred[tp_mask] - taus_true[tp_mask]).astype(float)
    else:
        errors = np.array([], dtype=float)

    mean_loc = float(errors.mean()) if len(errors) > 0 else float("nan")
    median_loc = float(np.median(errors)) if len(errors) > 0 else float("nan")

    return EvaluationResult(
        detection_accuracy=detection_accuracy,
        power=power,
        type1_error=type1_error,
        localization_errors=errors,
        mean_localization_error=mean_loc,
        median_localization_error=median_loc,
    )


def localization_error(tau_true: int, tau_hat: int) -> int:
    return abs(tau_hat - tau_true)


def compute_roc(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve.

    Returns:
        (fpr, tpr, thresholds) arrays
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return fpr, tpr, thresholds
