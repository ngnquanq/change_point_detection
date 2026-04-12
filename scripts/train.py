#!/usr/bin/env python
"""Train a change-point detection model from a YAML config.

Usage:
    python scripts/train.py --config configs/mlp_s1.yaml
    python scripts/train.py --config configs/rescnn_s1.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, MODELS_DIR
from src.data.simulator import simulate_dataset
from src.data.transforms import augment_reversed, build_preprocessing_pipeline
from src.data.dataset import make_dataloaders
from src.models.mlp import MLPDetector
from src.models.rescnn import ResidualCNN
from src.training.trainer import Trainer


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
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a change-point detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Experiment: {cfg.experiment_name}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 1. Simulate data
    print(f"Simulating {cfg.simulation.N} sequences of length {cfg.simulation.n} "
          f"({cfg.simulation.noise_type} noise)...")
    X, y, taus = simulate_dataset(
        N=cfg.simulation.N,
        n=cfg.simulation.n,
        noise_type=cfg.simulation.noise_type,
        rho=cfg.simulation.rho,
        mu_range=cfg.simulation.mu_range,
        sigma=cfg.simulation.sigma,
        seed=cfg.simulation.seed,
    )

    # 2. Augment with reversed sequences (before split)
    if cfg.training.augment_reversed:
        X, y, taus = augment_reversed(X, y, taus)
        print(f"After augmentation: {len(X)} sequences")

    # 3. Preprocess
    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.simulation.noise_type,
        use_squared=cfg.model.use_squared,
        use_cross_product=cfg.model.use_cross_product,
    )
    X_proc = preprocess(X)

    # 4. Build dataloaders
    flatten = cfg.model.architecture == "mlp"
    train_loader, val_loader = make_dataloaders(
        X_proc, y, taus,
        batch_size=cfg.training.batch_size,
        val_fraction=cfg.training.val_fraction,
        flatten=flatten,
        seed=cfg.simulation.seed,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 5. Build model
    model = build_model(cfg)
    print(f"Model: {cfg.model.architecture} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train
    checkpoint_dir = MODELS_DIR / cfg.experiment_name
    trainer = Trainer(model, cfg.training, device, checkpoint_dir)
    history = trainer.train(train_loader, val_loader)

    # 7. Save config and history
    cfg.save_yaml(checkpoint_dir / "config.yaml")
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_val_acc = history["val_acc"][-1]
    best_val_acc = max(history["val_acc"])
    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
