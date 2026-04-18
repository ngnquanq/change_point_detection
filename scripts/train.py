#!/usr/bin/env python
"""Train a change-point detection model from a YAML config.

Supports both simulated data and HASC accelerometer data via registry.

Usage:
    python scripts/train.py --config configs/rescnn_s1.yaml
    python scripts/train.py --config configs/rescnn_hasc.yaml --device auto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
import torch

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig
from src.registry import DATASET_REGISTRY, MODEL_REGISTRY
from src.data.dataset import make_dataloaders
from src.training.trainer import Trainer

# Import to trigger registration
import src.data.datasets.simulated  # noqa: F401  registers "simulated"
import src.data.datasets.hasc       # noqa: F401  registers "hasc"
import src.models.rescnn             # noqa: F401  registers "rescnn"
import src.models.mlp                # noqa: F401  registers "mlp"


def auto_detect_device(requested: str) -> torch.device:
    """Select device: MPS > CUDA > CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a change-point detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', or 'mps'")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    device = auto_detect_device(args.device)

    print(f"Experiment : {cfg.experiment_name}")
    print(f"Dataset    : {cfg.dataset.source}")
    print(f"Model      : {cfg.model.architecture}")
    print(f"Device     : {device}")

    # 1. Load data — registry looks up class by name, passes cfg
    dataset = DATASET_REGISTRY.build(cfg.dataset.source, cfg=cfg)
    X, y, taus = dataset.load()

    # 2. Build dataloaders
    flatten = cfg.model.architecture == "mlp"
    train_loader, val_loader = make_dataloaders(
        X, y, taus,
        batch_size=cfg.training.batch_size,
        val_fraction=cfg.training.val_fraction,
        flatten=flatten,
        seed=cfg.dataset.seed,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Build model — registry looks up class by name, passes cfg
    model = MODEL_REGISTRY.build(cfg.model.architecture, cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")

    # 4. Train
    checkpoint_dir = os.path.join(cfg.models_dir, cfg.experiment_name)
    trainer = Trainer(model, cfg.training, device, checkpoint_dir)
    history = trainer.train(train_loader, val_loader)

    # 5. Save config + history
    cfg.save_yaml(os.path.join(checkpoint_dir, "config.yaml"))
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_val_acc = max(history["val_acc"])
    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
