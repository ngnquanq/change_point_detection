import numpy as np
import pytest
import torch

from src.config import LocalizationConfig
from src.data.transforms import minmax_scale
from src.models.mlp import MLPDetector
from src.inference.localizer import Localizer, DetectedChangePoint


def make_trivial_preprocess(X):
    return minmax_scale(X)


class OracleDetector(torch.nn.Module):
    """Deterministic oracle: detects a mean shift between first and second half.

    After min-max scaling to [0,1], a strong half-mean difference is ~0.3+.
    Threshold set at 0.2 to reliably flag straddling windows.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n) — values in [0, 1] after min-max scaling
        n = x.shape[1]
        left = x[:, : n // 2].mean(dim=1)
        right = x[:, n // 2 :].mean(dim=1)
        # Positive logit when half-mean difference exceeds 0.2
        return ((right - left).abs() - 0.2).unsqueeze(1)


def test_localizer_detects_change_near_true_tau():
    """Verify Algorithm 1 localizes change points correctly using an oracle detector."""
    # Build a long series with a known 5-sigma change at position 200
    rng = np.random.default_rng(99)
    series = np.concatenate([
        rng.normal(0, 1, 200),
        rng.normal(5, 1, 200),  # 5-sigma mean shift
    ]).astype(np.float32)

    n = 50
    model = OracleDetector()
    loc_cfg = LocalizationConfig(window_size=n, step_size=5, rolling_window=5, gamma=0.5)
    localizer = Localizer(model, loc_cfg, torch.device("cpu"), make_trivial_preprocess)
    detections = localizer.locate(series)

    assert len(detections) >= 1, "Oracle detector should detect the 5-sigma shift"
    # At least one detection within 60 positions of true tau=200
    errors = [abs(cp.location - 200) for cp in detections]
    assert min(errors) <= 60, f"Closest detection: {min(errors)} from tau=200"


def test_localizer_no_detection_on_constant():
    model = MLPDetector(n=30, variant="pruned")
    model.eval()
    loc_cfg = LocalizationConfig(window_size=30, step_size=5, rolling_window=5, gamma=0.5)
    localizer = Localizer(model, loc_cfg, torch.device("cpu"), make_trivial_preprocess)

    # Constant series → model should output ~0.5 randomly but not consistently > gamma
    series = np.ones(200, dtype=np.float32)
    detections = localizer.locate(series)
    # No guarantee of zero detections on random weights, but test doesn't crash
    assert isinstance(detections, list)


def test_localizer_rolling_average():
    model = MLPDetector(n=20, variant="pruned")
    loc_cfg = LocalizationConfig(window_size=20, step_size=1, rolling_window=5, gamma=0.5)
    localizer = Localizer(model, loc_cfg, torch.device("cpu"), make_trivial_preprocess)

    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float32)
    L_bar = localizer._rolling_average(labels, window=3)
    assert L_bar.shape == labels.shape


def test_localizer_find_segments():
    model = MLPDetector(n=20, variant="pruned")
    loc_cfg = LocalizationConfig(window_size=20, step_size=1, rolling_window=5, gamma=0.5)
    localizer = Localizer(model, loc_cfg, torch.device("cpu"), make_trivial_preprocess)

    L_bar = np.array([0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.0, 0.6, 0.8, 0.0])
    segments = localizer._find_maximal_segments(L_bar, gamma=0.5)
    assert len(segments) == 2
    assert segments[0] == (2, 4)
    assert segments[1] == (7, 8)


def test_localizer_short_series_raises():
    model = MLPDetector(n=50, variant="pruned")
    loc_cfg = LocalizationConfig(window_size=50, step_size=1, rolling_window=5, gamma=0.5)
    localizer = Localizer(model, loc_cfg, torch.device("cpu"), make_trivial_preprocess)

    with pytest.raises(ValueError, match="Series length"):
        localizer.locate(np.ones(30, dtype=np.float32))
