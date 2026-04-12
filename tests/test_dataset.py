import numpy as np
import pytest
import torch

from src.data.dataset import ChangePointDataset, make_dataloaders


def make_dummy_data(N=100, n=20):
    rng = np.random.default_rng(0)
    X = rng.random((N, n)).astype(np.float32)
    y = (rng.random(N) > 0.5).astype(np.int8)
    taus = np.where(y == 1, rng.integers(1, n - 1, size=N), 0).astype(np.int64)
    return X, y, taus


def test_dataset_len():
    X, y, taus = make_dummy_data()
    ds = ChangePointDataset(X, y, taus)
    assert len(ds) == 100


def test_dataset_getitem_flatten():
    X, y, taus = make_dummy_data(N=10, n=20)
    ds = ChangePointDataset(X, y, taus, flatten=True)
    x, label = ds[0]
    assert x.shape == (20,)
    assert label.dtype == torch.float32


def test_dataset_getitem_cnn():
    X, y, taus = make_dummy_data(N=10, n=20)
    ds = ChangePointDataset(X, y, taus, flatten=False)
    x, label = ds[0]
    assert x.shape == (1, 20)


def test_make_dataloaders_shapes():
    X, y, taus = make_dummy_data(N=200, n=20)
    train_loader, val_loader = make_dataloaders(
        X, y, taus, batch_size=16, val_fraction=0.1, flatten=True, seed=0
    )
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape[1] == 20
    assert y_batch.dtype == torch.float32


def test_make_dataloaders_split_sizes():
    X, y, taus = make_dummy_data(N=100, n=20)
    train_loader, val_loader = make_dataloaders(
        X, y, taus, batch_size=10, val_fraction=0.2, seed=0
    )
    train_total = sum(len(b[0]) for b in train_loader)
    val_total = sum(len(b[0]) for b in val_loader)
    assert train_total + val_total == 100


def test_make_dataloaders_cnn_shape():
    X, y, taus = make_dummy_data(N=50, n=20)
    train_loader, _ = make_dataloaders(
        X, y, taus, batch_size=8, flatten=False, seed=0
    )
    x_batch, _ = next(iter(train_loader))
    assert x_batch.ndim == 3  # (B, 1, n)
    assert x_batch.shape[1] == 1
    assert x_batch.shape[2] == 20
