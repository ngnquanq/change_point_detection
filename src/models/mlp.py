from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class MLPDetector(nn.Module):
    """Single-hidden-layer MLP for change-point binary classification.

    Architecture: Linear(n, h) -> ReLU -> Linear(h, 1)
    Outputs raw logits (use BCEWithLogitsLoss during training).

    Hidden size variants:
      full:   h = 2*n - 2  (theoretically replicates CUSUM for mean shifts)
      pruned: h = 4 * floor(log2(n))  (high accuracy with fewer parameters)

    Args:
        n: input length (after pre-transforms)
        variant: "full" or "pruned"
    """

    def __init__(self, n: int, variant: str = "pruned") -> None:
        super().__init__()
        if variant == "full":
            h = 2 * n - 2
        elif variant == "pruned":
            h = 4 * int(math.floor(math.log2(n)))
        else:
            raise ValueError(f"Unknown variant: {variant!r}. Choose 'full' or 'pruned'.")

        self._hidden_size = h
        self.net = nn.Sequential(
            nn.Linear(n, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, n) -> logits (B, 1)"""
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Hard binary prediction.

        Args:
            x: (B, n)
            threshold: decision threshold on sigmoid output

        Returns:
            preds: (B,) LongTensor
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze(1)
            return (probs >= threshold).long()

    def predict_proba(self, x: Tensor) -> Tensor:
        """Probability of change point (class 1).

        Returns:
            probs: (B,) FloatTensor in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)
