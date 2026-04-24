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

from src.config import ExperimentConfig, PROJECT_ROOT
from src.registry import MODEL_REGISTRY
from src.data.transforms import build_preprocessing_pipeline
from src.inference.localizer import Localizer

# Import to trigger registration
import src.models.rescnn  # noqa: F401
import src.models.mlp     # noqa: F401


def auto_detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_long_series(
    total_length: int,
    n_changes: int,
    noise_type: str,
    rho: float,
    sigma: float,
    cauchy_scale: float,
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
        elif noise_type == "S1_prime":
            noise = np.empty(length)
            sigma_eps = np.sqrt(max(1.0 - rho ** 2, 1e-8))
            noise[0] = rng.normal(0, sigma_eps)
            for t in range(1, length):
                noise[t] = rho * noise[t - 1] + rng.normal(0, sigma_eps)
        elif noise_type == "S2":
            rho_t = rng.uniform(0.0, 1.0, size=length)
            noise = np.empty(length)
            noise[0] = rng.normal(0, np.sqrt(2.0))
            for t in range(1, length):
                noise[t] = rho_t[t] * noise[t - 1] + rng.normal(0, np.sqrt(2.0))
        elif noise_type == "S3":
            u = rng.uniform(0, 1, size=length)
            noise = cauchy_scale * np.tan(np.pi * (u - 0.5))
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

    config_dir = PROJECT_ROOT / "models" / args.experiment
    cfg = ExperimentConfig.from_yaml(config_dir / "config.yaml")
    checkpoint_dir = cfg.models_path / args.experiment
    device = auto_detect_device(args.device)

    # Generate long series
    series, true_taus = generate_long_series(
        total_length=args.series_length,
        n_changes=args.n_changes,
        noise_type=cfg.dataset.noise_type,
        rho=cfg.dataset.rho,
        sigma=cfg.dataset.sigma,
        cauchy_scale=cfg.dataset.cauchy_scale,
        seed=args.seed,
    )
    print(f"Series length: {args.series_length}, True change points: {true_taus}")

    # Load model via registry — no if/else
    model = MODEL_REGISTRY.build(cfg.model.architecture, cfg=cfg)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=True))

    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.dataset.noise_type,
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
