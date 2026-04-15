from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split


class ChangePointDataset(Dataset):
    """PyTorch Dataset wrapping (X, y) arrays.

    Args:
        X: ndarray shape (N, n') — preprocessed sequences
        y: ndarray shape (N,) int — binary labels (1=change, 0=no change)
        taus: ndarray shape (N,) int — change-point locations (0 if y=0), optional
        flatten: if True, returns x as shape (n',); if False, shape (1, n') for CNN
        transform: optional additional transform applied per sample
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        taus: Optional[np.ndarray] = None,
        flatten: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.taus = torch.from_numpy(taus.astype(np.int64)) if taus is not None else None
        self.flatten = flatten
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.X[idx]
        label = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        if not self.flatten:
            x = x.unsqueeze(0)  # (1, n') for CNN

        return x, label


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    taus: np.ndarray,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    flatten: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from arrays.

    Augmentation (reversed sequences) should be applied BEFORE calling this
    function so that both original and reversed end up in the same split.

    Args:
        X: shape (N, n') — already preprocessed and augmented
        y: shape (N,)
        taus: shape (N,)
        batch_size: mini-batch size
        val_fraction: fraction of data for validation
        flatten: True for MLP, False for CNN (adds channel dim)
        seed: random seed for the split
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader
    """
    dataset = ChangePointDataset(X, y, taus, flatten=flatten)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
