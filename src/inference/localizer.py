from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn

from src.config import LocalizationConfig


@dataclass
class DetectedChangePoint:
    location: int           # Estimated tau in original series (0-indexed)
    segment_start: int      # Start of maximal segment
    segment_end: int        # End of maximal segment (inclusive)
    max_probability: float  # Peak probability in segment


class Localizer:
    """Algorithm 1: Sliding-window change-point localization.

    Given a trained model and a long time series of length M:
      1. Slide window of length n with step s, computing (label L_i, prob p_i)
         for each window starting at position i.
      2. Compute rolling average L̄_i over `rolling_window` width.
      3. Find maximal contiguous segments where L̄_i >= gamma.
      4. Within each segment: tau_hat = argmax(L̄_i).

    The window index i corresponds to window x[i : i+n]. The reported
    change-point location is i + tau_hat_within_window = i (since the window
    itself is scored; we report the start of the peak window as the location).

    Args:
        model: trained MLPDetector or ResidualCNN (will be set to eval mode)
        config: LocalizationConfig
        device: torch.device
        preprocess_fn: callable (np.ndarray shape (K, n)) -> (K, n')
                       Same per-sequence preprocessing used during training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: LocalizationConfig,
        device: torch.device,
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.preprocess_fn = preprocess_fn

        # Detect model type for correct input shaping
        from src.models.rescnn import ResidualCNN
        self._is_cnn = isinstance(model, ResidualCNN)

    def locate(
        self,
        series: np.ndarray,
        batch_size: int = 256,
    ) -> List[DetectedChangePoint]:
        """Run full localization on a 1-D time series.

        Args:
            series: 1-D float array of length M
            batch_size: number of windows to score per forward pass

        Returns:
            list of DetectedChangePoint, sorted by location
        """
        n = self.config.window_size
        step = self.config.step_size
        M = len(series)

        if M < n:
            raise ValueError(f"Series length {M} < window_size {n}")

        # Extract all windows
        starts = np.arange(0, M - n + 1, step)
        K = len(starts)
        windows = np.stack([series[s : s + n] for s in starts], axis=0).astype(np.float32)

        # Score all windows in batches
        labels, probs = self._score_windows(windows, batch_size=batch_size)

        # Rolling average of labels
        L_bar = self._rolling_average(labels, self.config.rolling_window)

        # Find maximal segments where L_bar >= gamma
        segments = self._find_maximal_segments(L_bar, self.config.gamma)

        results = []
        for seg_start, seg_end in segments:
            segment_L = L_bar[seg_start : seg_end + 1]
            peak_idx = seg_start + int(np.argmax(segment_L))
            location = starts[peak_idx] + n // 2  # map to series index (center of window)
            results.append(
                DetectedChangePoint(
                    location=int(location),
                    segment_start=int(starts[seg_start]),
                    segment_end=int(starts[seg_end] + n - 1),
                    max_probability=float(probs[peak_idx]),
                )
            )

        results.sort(key=lambda cp: cp.location)
        return results

    def _score_windows(
        self, windows: np.ndarray, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch-score windows with the model.

        Args:
            windows: shape (K, n)

        Returns:
            labels: (K,) int array
            probs:  (K,) float array
        """
        K = len(windows)
        all_probs = np.empty(K, dtype=np.float32)

        for start in range(0, K, batch_size):
            batch = windows[start : start + batch_size]
            # Apply same preprocessing as training (per-window independent)
            batch_proc = self.preprocess_fn(batch)
            x = torch.from_numpy(batch_proc.astype(np.float32))
            if self._is_cnn:
                x = x.unsqueeze(1)  # (B, 1, n')
            x = x.to(self.device)

            with torch.no_grad():
                logits = self.model(x).squeeze(1)
                p = torch.sigmoid(logits).cpu().numpy()
            all_probs[start : start + len(batch)] = p

        labels = (all_probs >= 0.5).astype(np.int32)
        return labels, all_probs

    def _rolling_average(self, labels: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean of binary labels using convolution."""
        kernel = np.ones(window) / window
        # 'same' mode: output same length as input, with edge padding
        return np.convolve(labels.astype(np.float32), kernel, mode="same")

    def _find_maximal_segments(
        self, L_bar: np.ndarray, gamma: float
    ) -> list[tuple[int, int]]:
        """Find maximal contiguous runs where L_bar >= gamma.

        Returns list of (start_idx, end_idx) pairs (inclusive).
        """
        above = L_bar >= gamma
        segments = []
        in_seg = False
        seg_start = 0
        for i, val in enumerate(above):
            if val and not in_seg:
                seg_start = i
                in_seg = True
            elif not val and in_seg:
                segments.append((seg_start, i - 1))
                in_seg = False
        if in_seg:
            segments.append((seg_start, len(above) - 1))
        return segments
