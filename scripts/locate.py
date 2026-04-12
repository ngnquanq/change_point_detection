#!/usr/bin/env python
"""Run Algorithm 1 (sliding-window localization) on a synthetic long series.

Usage:
    python scripts/locate.py --experiment mlp_s1 --series_length 500 --n_changes 2
    python scripts/locate.py --experiment mlp_s1 --series_length 500 --n_changes 3 --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, MODELS_DIR
from src.data.transforms import build_preprocessing_pipeline
from src.models.mlp import MLPDetector
from src.models.rescnn import ResidualCNN
from src.inference.localizer import Localizer


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    n_input = cfg.input_length()
    if cfg.model.architecture == "mlp":
        return MLPDetector(n=n_input, variant=cfg.model.mlp_variant)
    elif cfg.model.architecture == "rescnn":
        return ResidualCNN(
            n=n_input,
            n_blocks=cfg.model.n_blocks,
            base_channels=cfg.model.base_channels,
            kernel_size=cfg.model.kernel_size,
            in_channels=1,
        )
    raise ValueError(f"Unknown architecture: {cfg.model.architecture!r}")


def generate_long_series(
    total_length: int,
    n_changes: int,
    noise_type: str,
    rho: float,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Generate a long series with n_changes evenly spaced change points."""
    rng = np.random.default_rng(seed)

    # Space change points evenly, avoid edges
    margin = total_length // (n_changes + 1)
    true_taus = [margin * (i + 1) for i in range(n_changes)]

    # Draw segment means
    means = rng.uniform(-2.0, 2.0, size=n_changes + 1)

    series = np.empty(total_length, dtype=np.float32)
    boundaries = [0] + true_taus + [total_length]

    for seg_idx in range(n_changes + 1):
        start = boundaries[seg_idx]
        end = boundaries[seg_idx + 1]
        length = end - start

        if noise_type == "S1":
            noise = rng.normal(0, sigma, size=length)
        elif noise_type in ("S1_prime", "S2"):
            noise = np.empty(length)
            sigma_eps = np.sqrt(max(1.0 - rho ** 2, 1e-8))
            noise[0] = rng.normal(0, sigma_eps)
            for t in range(1, length):
                noise[t] = rho * noise[t - 1] + rng.normal(0, sigma_eps)
        elif noise_type == "S3":
            u = rng.uniform(0, 1, size=length)
            noise = np.tan(np.pi * (u - 0.5))
        else:
            noise = rng.normal(0, sigma, size=length)

        series[start:end] = means[seg_idx] + noise

    return series, true_taus


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize change points in a long series")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--series_length", type=int, default=500)
    parser.add_argument("--n_changes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint_dir = MODELS_DIR / args.experiment
    cfg = ExperimentConfig.from_yaml(checkpoint_dir / "config.yaml")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Generate long series
    series, true_taus = generate_long_series(
        total_length=args.series_length,
        n_changes=args.n_changes,
        noise_type=cfg.simulation.noise_type,
        rho=cfg.simulation.rho,
        sigma=cfg.simulation.sigma,
        seed=args.seed,
    )
    print(f"Series length: {args.series_length}, True change points: {true_taus}")

    # Load model
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))

    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.simulation.noise_type,
        use_squared=cfg.model.use_squared,
        use_cross_product=cfg.model.use_cross_product,
    )

    localizer = Localizer(model, cfg.localization, device, preprocess)
    detections = localizer.locate(series)

    print(f"\nDetected {len(detections)} change point(s):")
    for i, cp in enumerate(detections):
        errors = [abs(cp.location - t) for t in true_taus]
        nearest_err = min(errors) if errors else "-"
        print(
            f"  [{i+1}] location={cp.location}, "
            f"max_prob={cp.max_probability:.3f}, "
            f"nearest_true_tau_error={nearest_err}"
        )


if __name__ == "__main__":
    main()
