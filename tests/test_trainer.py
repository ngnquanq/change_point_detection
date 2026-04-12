import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import TrainingConfig
from src.data.simulator import simulate_dataset
from src.data.transforms import minmax_scale
from src.data.dataset import make_dataloaders
from src.models.mlp import MLPDetector
from src.training.trainer import Trainer


def make_tiny_loaders(n=20, N=100, batch_size=16):
    X, y, taus = simulate_dataset(N=N, n=n, seed=5)
    X = minmax_scale(X)
    train_loader, val_loader = make_dataloaders(
        X, y, taus, batch_size=batch_size, val_fraction=0.2, flatten=True, seed=0
    )
    return train_loader, val_loader


def test_trainer_one_epoch_completes():
    train_loader, val_loader = make_tiny_loaders()
    model = MLPDetector(n=20, variant="pruned")
    cfg = TrainingConfig(epochs=1, batch_size=16, lr=1e-3, patience=5)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model, cfg, device, Path(tmpdir))
        history = trainer.train(train_loader, val_loader)

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert np.isfinite(history["train_loss"][0])
    assert np.isfinite(history["val_loss"][0])


def test_trainer_saves_checkpoint():
    train_loader, val_loader = make_tiny_loaders()
    model = MLPDetector(n=20, variant="pruned")
    cfg = TrainingConfig(epochs=2, batch_size=16, lr=1e-3, patience=5)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model, cfg, device, Path(tmpdir))
        trainer.train(train_loader, val_loader)
        assert (Path(tmpdir) / "best_model.pt").exists()


def test_trainer_accuracy_is_valid():
    train_loader, val_loader = make_tiny_loaders()
    model = MLPDetector(n=20, variant="pruned")
    cfg = TrainingConfig(epochs=3, batch_size=16, lr=1e-3, patience=5)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model, cfg, device, Path(tmpdir))
        history = trainer.train(train_loader, val_loader)

    for acc in history["train_acc"] + history["val_acc"]:
        assert 0.0 <= acc <= 1.0


def test_trainer_early_stopping():
    train_loader, val_loader = make_tiny_loaders()
    model = MLPDetector(n=20, variant="pruned")
    # Very aggressive patience=1 should stop early
    cfg = TrainingConfig(epochs=50, batch_size=16, lr=1e-3, patience=1)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model, cfg, device, Path(tmpdir))
        history = trainer.train(train_loader, val_loader)

    # Should stop well before 50 epochs
    assert len(history["train_loss"]) < 50
