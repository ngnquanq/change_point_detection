from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def minmax_scale(X: np.ndarray) -> np.ndarray:
    """Per-sequence min-max scaling to [0, 1].

    Args:
        X: shape (N, n)

    Returns:
        X_scaled: shape (N, n), values in [0, 1]
    """
    X = X.copy()
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    denom = maxs - mins
    # Avoid division by zero for constant sequences
    denom = np.where(denom == 0, 1.0, denom)
    return (X - mins) / denom


def trimmed_scale(X: np.ndarray, trim_fraction: float = 0.1) -> np.ndarray:
    """Robust per-sequence scaling using trimmed mean and trimmed std.

    Suitable for heavy-tailed (S3 Cauchy) sequences.

    Args:
        X: shape (N, n)
        trim_fraction: fraction to trim from each tail (e.g. 0.1 = 10% each side)

    Returns:
        X_scaled: shape (N, n)
    """
    X = X.copy().astype(np.float64)
    for i in range(len(X)):
        row = X[i]
        t_mean = stats.trim_mean(row, trim_fraction)
        # trimmed std via mstats
        t_std = stats.mstats.trimmed_std(row, limits=(trim_fraction, trim_fraction))
        if t_std == 0 or np.isnan(t_std):
            t_std = 1.0
        X[i] = (row - t_mean) / t_std
    return X.astype(np.float32)


def augment_reversed(
    X: np.ndarray,
    y: np.ndarray,
    taus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Double training samples by appending time-reversed sequences.

    Reversed sequences share the same label as the original.
    For change sequences: tau_reversed = n - tau (1-indexed convention).

    Args:
        X: shape (N, n)
        y: shape (N,) int
        taus: shape (N,) int — 0 for no-change sequences

    Returns:
        X_aug: shape (2N, n)
        y_aug: shape (2N,)
        taus_aug: shape (2N,)
    """
    n = X.shape[1]
    X_rev = X[:, ::-1].copy()

    # For a change at tau (1-indexed), reversed tau = n - tau
    taus_rev = np.where(y == 1, n - taus, 0)

    X_aug = np.concatenate([X, X_rev], axis=0)
    y_aug = np.concatenate([y, y], axis=0)
    taus_aug = np.concatenate([taus, taus_rev], axis=0)
    return X_aug, y_aug, taus_aug


def apply_pretransform(
    X: np.ndarray,
    use_squared: bool = False,
    use_cross_product: bool = False,
) -> np.ndarray:
    """Append optional pre-transform features for variance/slope detection.

    - use_squared: append x^2 (doubles feature width)
    - use_cross_product: append x_t * x_{t+1} padded with a zero (adds n cols)

    Args:
        X: shape (N, n)

    Returns:
        X_out: shape (N, n') where n' = n * (1 + use_squared + use_cross_product)
    """
    parts = [X]
    if use_squared:
        parts.append(X ** 2)
    if use_cross_product:
        # x_t * x_{t+1} for t=0..n-2, padded with 0 at the end
        cross = X[:, :-1] * X[:, 1:]
        pad = np.zeros((X.shape[0], 1), dtype=X.dtype)
        parts.append(np.concatenate([cross, pad], axis=1))
    return np.concatenate(parts, axis=1)


def build_preprocessing_pipeline(
    noise_type: str,
    use_squared: bool = False,
    use_cross_product: bool = False,
    trim_fraction: float = 0.1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Factory: returns a callable (X: ndarray) -> ndarray.

    For S3 (Cauchy) noise, uses trimmed scaling; otherwise min-max.
    Pre-transforms (x^2, cross-products) are applied before scaling.
    """
    def pipeline(X: np.ndarray) -> np.ndarray:
        X = apply_pretransform(X, use_squared=use_squared, use_cross_product=use_cross_product)
        if noise_type == "S3":
            X = trimmed_scale(X, trim_fraction=trim_fraction)
        else:
            X = minmax_scale(X)
        return X

    return pipeline
