#!/usr/bin/env python
"""Plot Algorithm 1 localization on one real HASC recording.

This script uses the binary HASC ResCNN trained by:
    python scripts/train.py --config configs/rescnn_hasc.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, PROJECT_ROOT
from src.data.hasc_loader import load_recording
from src.data.transforms import minmax_scale
from src.inference.localizer import Localizer
from src.registry import MODEL_REGISTRY

# Import to trigger model registration.
import src.models.rescnn  # noqa: F401


DEFAULT_RECORDING = PROJECT_ROOT / "data" / "hasc" / "person106" / "HASC1013-acc.csv"


def auto_detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_path(recording_path: Path) -> Path:
    return PROJECT_ROOT / "output" / f"hasc_{recording_path.stem}_result.png"


def extract_signal(recording, channel: str) -> np.ndarray:
    if channel == "magnitude":
        return recording.acc_magnitude.astype(np.float32)
    if channel == "x":
        return recording.acc_x.astype(np.float32)
    if channel == "y":
        return recording.acc_y.astype(np.float32)
    if channel == "z":
        return recording.acc_z.astype(np.float32)
    raise ValueError(f"Unsupported HASC channel: {channel}")


def score_track(localizer: Localizer, series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = localizer.config.window_size
    step = localizer.config.step_size
    starts = np.arange(0, len(series) - n + 1, step)
    windows = np.stack([series[start : start + n] for start in starts]).astype(np.float32)
    labels, probabilities = localizer._score_windows(windows)
    rolling = localizer._rolling_average(labels, localizer.config.rolling_window)
    centers = starts + n // 2
    return centers, probabilities, rolling


def load_model(checkpoint_dir: Path, device: torch.device) -> tuple[ExperimentConfig, torch.nn.Module]:
    config_path = checkpoint_dir / "config.yaml"
    checkpoint_path = checkpoint_dir / "best_model.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing HASC checkpoint in {checkpoint_dir}. "
            "Run: python scripts/train.py --config configs/rescnn_hasc.yaml --device cpu"
        )

    cfg = ExperimentConfig.from_yaml(config_path)
    model = MODEL_REGISTRY.build(cfg.model.architecture, cfg=cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return cfg, model


def plot_hasc(
    recording_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    cfg, model = load_model(checkpoint_dir, device)
    recording = load_recording(recording_path)
    series = extract_signal(recording, cfg.dataset.channel)

    localizer = Localizer(model, cfg.localization, device, minmax_scale)
    detections = localizer.locate(series)
    centers, probabilities, rolling = score_track(localizer, series)

    estimated_indices = [cp.location for cp in detections]
    true_indices = recording.change_points

    fig, (ax_signal, ax_score) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax_signal.plot(recording.timestamps, recording.acc_x, alpha=0.65, linewidth=0.9, label="x")
    ax_signal.plot(recording.timestamps, recording.acc_y, alpha=0.65, linewidth=0.9, label="y")
    ax_signal.plot(recording.timestamps, recording.acc_z, alpha=0.65, linewidth=0.9, label="z")
    ax_signal.plot(recording.timestamps, series, color="black", alpha=0.45, linewidth=1.0, label=cfg.dataset.channel)

    for idx, cp_idx in enumerate(true_indices):
        ax_signal.axvline(
            recording.timestamps[cp_idx],
            color="crimson",
            linewidth=1.6,
            label="True CP" if idx == 0 else None,
        )
    for idx, cp_idx in enumerate(estimated_indices):
        if 0 <= cp_idx < recording.length:
            ax_signal.axvline(
                recording.timestamps[cp_idx],
                color="royalblue",
                linestyle="--",
                linewidth=1.6,
                label="Estimated CP" if idx == 0 else None,
            )

    ax_signal.set_title(f"HASC Algorithm 1 localization: {recording_path.stem}")
    ax_signal.set_ylabel("Acceleration")
    ax_signal.legend(loc="upper right", ncol=3, fontsize=8)

    center_times = recording.timestamps[centers]
    ax_score.plot(center_times, probabilities, color="tab:gray", alpha=0.4, linewidth=0.8, label="P(change)")
    ax_score.plot(center_times, rolling, color="black", linewidth=1.4, label="L_bar")
    ax_score.axhline(cfg.localization.gamma, color="crimson", linestyle=":", linewidth=1.2, label=f"gamma={cfg.localization.gamma}")
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.set_xlabel("Time (s)")
    ax_score.set_ylabel("Score")
    ax_score.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"recording={recording_path.relative_to(PROJECT_ROOT)}")
    print(f"true_change_points={len(true_indices)}")
    print(f"estimated_change_points={len(estimated_indices)}")
    print(f"saved={output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HASC Algorithm 1 localization")
    parser.add_argument("--experiment", default="rescnn_hasc", help="Experiment name under the checkpoint root")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory; defaults to output/<experiment>")
    parser.add_argument("--recording", default=str(DEFAULT_RECORDING), help="Path to one HASC *-acc.csv recording")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    checkpoint_dir = resolve_path(args.checkpoint_dir) if args.checkpoint_dir else PROJECT_ROOT / "output" / args.experiment
    recording_path = resolve_path(args.recording)
    output_path = resolve_path(args.out) if args.out else default_output_path(recording_path)
    device = auto_detect_device(args.device)

    plot_hasc(recording_path, checkpoint_dir, output_path, device)


if __name__ == "__main__":
    main()
