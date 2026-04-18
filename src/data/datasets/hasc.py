"""HASC accelerometer dataset for change-point detection training."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.registry import DATASET_REGISTRY
from src.data.hasc_loader import (
    load_hasc_directory,
    extract_windows_from_recordings,
    balance_dataset,
)
from src.data.transforms import minmax_scale


@DATASET_REGISTRY.register("hasc")
class HASCDataset:
    """Load HASC accelerometer data, extract windows, balance, and preprocess.

    Registered as ``"hasc"`` in the dataset registry.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg.dataset

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load, window, balance, and scale HASC data.

        Returns:
            (X, y, taus) — windowed time series, labels, change-point positions
        """
        cfg = self.cfg
        hasc_dir = Path(cfg.hasc_dir)

        print(f"Loading HASC recordings from: {hasc_dir}")
        recordings = load_hasc_directory(hasc_dir, min_segments=2)
        print(f"  Found {len(recordings)} recordings with ≥2 activity segments")

        if len(recordings) == 0:
            raise FileNotFoundError(
                f"No valid recordings found in {hasc_dir}. "
                "Download HASC data from https://hub.hasc.jp"
            )

        # Summary
        total_changes = sum(len(r.change_points) for r in recordings)
        total_samples = sum(r.length for r in recordings)
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total change points: {total_changes}")

        all_labels = set()
        for rec in recordings:
            for seg in rec.segments:
                all_labels.add(seg.label)
        print(f"  Activities: {', '.join(sorted(all_labels))}")

        # Extract windows
        print(f"\nExtracting windows (size={cfg.window_size}, "
              f"stride={cfg.stride}, channel={cfg.channel})...")
        X, y, taus = extract_windows_from_recordings(
            recordings,
            window_size=cfg.window_size,
            channel=cfg.channel,
            stride=cfg.stride,
        )
        print(f"  Raw windows: {len(X)} (positive={y.sum()}, "
              f"negative={(y == 0).sum()})")

        # Balance
        X, y, taus = balance_dataset(X, y, taus, seed=cfg.seed)
        print(f"  After balancing: {len(X)} (50/50 split)")

        # Min-max scale
        X = minmax_scale(X)

        return X, y, taus
