"""HASC (Human Activity Sensing Consortium) dataset loader.

Loads accelerometer sequence data from the HASC corpus and prepares
it for change-point detection.

HASC data format:
  - Accelerometer CSV:  time(sec), X(G), Y(G), Z(G)
  - Label CSV:          start_time, end_time, activity_label

Download the HASC corpus from https://hub.hasc.jp (requires registration).
Place the data in data/hasc/ or specify the path via --hasc_dir.

Reference:
  Ichino et al., "HASC2010corpus: Large Scale Human Activity Corpus
  and Its Application", 2010.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivitySegment:
    """A labelled segment within a HASC recording."""
    start_time: float
    end_time: float
    label: str


@dataclass
class HASCRecording:
    """One HASC trial: accelerometer data + activity labels."""
    name: str
    timestamps: np.ndarray    # (T,) seconds
    acc_x: np.ndarray         # (T,) acceleration X in G
    acc_y: np.ndarray         # (T,) acceleration Y in G
    acc_z: np.ndarray         # (T,) acceleration Z in G
    segments: List[ActivitySegment] = field(default_factory=list)

    @property
    def acc_magnitude(self) -> np.ndarray:
        """Acceleration magnitude: sqrt(x² + y² + z²)."""
        return np.sqrt(self.acc_x ** 2 + self.acc_y ** 2 + self.acc_z ** 2)

    @property
    def length(self) -> int:
        return len(self.timestamps)

    @property
    def change_points(self) -> List[int]:
        """Get indices where activity transitions occur (change points).

        Returns list of sample indices corresponding to segment boundaries.
        """
        if len(self.segments) < 2:
            return []

        cps = []
        for i in range(1, len(self.segments)):
            boundary_time = self.segments[i].start_time
            # Find nearest sample index
            idx = int(np.searchsorted(self.timestamps, boundary_time))
            idx = min(idx, self.length - 1)
            cps.append(idx)
        return cps


# ──────────────────────────────────────────────────────────────────────────────
# File parsers
# ──────────────────────────────────────────────────────────────────────────────

def load_acc_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load HASC accelerometer CSV: time, x, y, z.

    Args:
        path: Path to *-acc.csv file

    Returns:
        (timestamps, acc_x, acc_y, acc_z) — each shape (T,)
    """
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                t, x, y, z = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                data.append((t, x, y, z))
            except ValueError:
                continue  # skip header or malformed rows

    if not data:
        raise ValueError(f"No valid data found in {path}")

    arr = np.array(data, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def load_label_file(path: Path) -> List[ActivitySegment]:
    """Load HASC label file: start_time, end_time, activity_label.

    Args:
        path: Path to *.label file

    Returns:
        List of ActivitySegment, sorted by start_time
    """
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2].strip()
                segments.append(ActivitySegment(start, end, label))
            except ValueError:
                continue

    segments.sort(key=lambda s: s.start_time)
    return segments


def load_recording(acc_path: Path, label_path: Optional[Path] = None) -> HASCRecording:
    """Load a single HASC recording (acc CSV + optional label file).

    Args:
        acc_path: Path to *-acc.csv
        label_path: Path to *.label (auto-detected if None)

    Returns:
        HASCRecording
    """
    timestamps, acc_x, acc_y, acc_z = load_acc_csv(acc_path)

    # Auto-detect label file
    if label_path is None:
        stem = acc_path.stem.replace("-acc", "")
        candidates = [
            acc_path.parent / f"{stem}.label",
            acc_path.parent / f"{acc_path.stem}.label",
        ]
        for c in candidates:
            if c.exists():
                label_path = c
                break

    segments = load_label_file(label_path) if label_path and label_path.exists() else []

    return HASCRecording(
        name=acc_path.stem,
        timestamps=timestamps,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        segments=segments,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_hasc_directory(
    hasc_dir: Path,
    min_segments: int = 2,
    require_labels: bool = True,
) -> List[HASCRecording]:
    """Load all HASC recordings from a directory tree.

    Recursively searches for *-acc.csv files and their associated label files.

    Args:
        hasc_dir: Root directory containing HASC data
        min_segments: Minimum number of activity segments required
        require_labels: If True, skip recordings without label files

    Returns:
        List of HASCRecording with at least min_segments segments
    """
    hasc_dir = Path(hasc_dir)
    if not hasc_dir.exists():
        raise FileNotFoundError(
            f"HASC directory not found: {hasc_dir}\n"
            "Download the HASC corpus from https://hub.hasc.jp (requires registration)\n"
            "and place data in data/hasc/"
        )

    acc_files = sorted(hasc_dir.rglob("*-acc.csv"))
    if not acc_files:
        # Fallback: try all CSV files
        acc_files = sorted(hasc_dir.rglob("*.csv"))

    recordings = []
    for acc_path in acc_files:
        try:
            rec = load_recording(acc_path)
        except (ValueError, OSError) as e:
            print(f"  ⚠ Skipping {acc_path.name}: {e}")
            continue

        if require_labels and len(rec.segments) == 0:
            continue
        if len(rec.segments) < min_segments:
            continue

        recordings.append(rec)

    return recordings


def extract_windows_from_recordings(
    recordings: List[HASCRecording],
    window_size: int = 100,
    channel: str = "magnitude",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract fixed-length windows from HASC recordings for binary classification.

    For each recording, slide a window and label it:
      y=1 if any change point falls within the window
      y=0 if no change point is in the window

    Args:
        recordings: List of HASCRecording with labels
        window_size: Length of each window (default 100)
        channel: "magnitude", "x", "y", or "z"
        stride: Step size between consecutive windows

    Returns:
        X: np.ndarray shape (N, window_size) — windowed time series
        y: np.ndarray shape (N,) — binary labels (1=change, 0=no change)
        taus: np.ndarray shape (N,) — relative change-point position within window
              (0 if y=0)
    """
    all_windows = []
    all_labels = []
    all_taus = []

    for rec in recordings:
        if channel == "magnitude":
            signal = rec.acc_magnitude
        elif channel == "x":
            signal = rec.acc_x
        elif channel == "y":
            signal = rec.acc_y
        elif channel == "z":
            signal = rec.acc_z
        else:
            raise ValueError(f"Unknown channel: {channel!r}")

        cps = set(rec.change_points)
        n = len(signal)

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window = signal[start:end].astype(np.float32)

            # Check if any change point falls within the window
            cp_in_window = [cp for cp in cps if start < cp < end]

            if cp_in_window:
                # Use the first change point within the window
                tau_relative = cp_in_window[0] - start
                all_windows.append(window)
                all_labels.append(1)
                all_taus.append(tau_relative)
            else:
                all_windows.append(window)
                all_labels.append(0)
                all_taus.append(0)

    if not all_windows:
        raise ValueError("No windows extracted. Check data and parameters.")

    X = np.stack(all_windows)
    y = np.array(all_labels, dtype=np.int8)
    taus = np.array(all_taus, dtype=np.int64)

    return X, y, taus


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    taus: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample the majority class to balance the dataset.

    Since windows without change points far outnumber those with,
    this function balances the two classes.

    Returns:
        Balanced (X, y, taus)
    """
    rng = np.random.default_rng(seed)

    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        return X, y, taus

    n_keep = min(n_pos, n_neg)

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    # Subsample the larger class
    if n_pos > n_keep:
        pos_indices = rng.choice(pos_indices, size=n_keep, replace=False)
    if n_neg > n_keep:
        neg_indices = rng.choice(neg_indices, size=n_keep, replace=False)

    indices = np.concatenate([pos_indices, neg_indices])
    rng.shuffle(indices)

    return X[indices], y[indices], taus[indices]
