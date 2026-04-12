from __future__ import annotations

import numpy as np


def cusum_detector(
    x: np.ndarray,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> tuple[int, float]:
    """Two-sided CUSUM change-point detector for mean shifts.

    Maintains upper (S+) and lower (S-) cumulative sums.
    Signals when either exceeds the threshold h.

    Args:
        x: 1-D float sequence of length n
        threshold: detection threshold h
        drift: allowance k (expected shift / 2 for optimal sensitivity)

    Returns:
        (tau_hat, max_cusum_value):
            tau_hat = first index where detection occurs (1-indexed),
                      or 0 if no change detected
            max_cusum_value = max of |S+|, |S-| at detection (or overall max)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s_pos = 0.0
    s_neg = 0.0
    tau_hat = 0
    max_val = 0.0

    for t in range(n):
        s_pos = max(0.0, s_pos + x[t] - drift)
        s_neg = max(0.0, s_neg - x[t] - drift)
        cur = max(s_pos, s_neg)
        if cur > max_val:
            max_val = cur
        if cur >= threshold and tau_hat == 0:
            tau_hat = t + 1  # 1-indexed

    return tau_hat, max_val


def run_cusum_on_dataset(
    X: np.ndarray,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CUSUM to all N sequences in dataset.

    Args:
        X: shape (N, n)
        threshold: CUSUM threshold h
        drift: allowance k

    Returns:
        predictions: (N,) int — 1 if change detected, 0 otherwise
        tau_hats: (N,) int — estimated change locations (0 if no detection)
    """
    N = len(X)
    predictions = np.zeros(N, dtype=np.int32)
    tau_hats = np.zeros(N, dtype=np.int64)

    for i, x in enumerate(X):
        tau, _ = cusum_detector(x, threshold=threshold, drift=drift)
        if tau > 0:
            predictions[i] = 1
            tau_hats[i] = tau

    return predictions, tau_hats
