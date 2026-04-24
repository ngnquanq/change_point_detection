from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.registry import MODEL_REGISTRY


class ResidualBlock(nn.Module):
    """Single residual block for 1-D time series.

    Architecture:
        Conv1d -> BN -> ReLU -> Conv1d -> BN
        + skip connection (1x1 Conv if in_ch != out_ch)
        -> ReLU

    Same-padding (padding = kernel_size // 2) preserves sequence length.

    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: convolution kernel size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, L) -> (B, C_out, L)"""
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Trim or pad to match residual length (handles even kernel rounding)
        if out.shape[-1] != residual.shape[-1]:
            out = out[..., : residual.shape[-1]]
        out = self.relu(out + residual)
        return out


@MODEL_REGISTRY.register("rescnn")
class ResidualCNN(nn.Module):
    """Deep Residual CNN for change-point detection.

    Architecture (following paper — 21 residual blocks, Section 6 & Supp. C.3):
      Input projection: Conv1d(in_ch, base_ch, 1)
      Blocks 1–7:   base_ch   -> base_ch
      Block 8:      base_ch   -> base_ch*2  (channel expansion)
      Blocks 9–21:  base_ch*2 -> base_ch*2
      AdaptiveAvgPool1d(1) -> flatten
      Dense(50) -> ReLU -> Dropout(0.3)
      Dense(40) -> ReLU -> Dropout(0.3)
      Dense(30) -> ReLU -> Dropout(0.3)
      Dense(20) -> ReLU -> Dropout(0.3)
      Dense(10) -> ReLU -> Dropout(0.3)
      Dense(1)   [raw logits]

    Outputs raw logits (use BCEWithLogitsLoss during training).

    Args:
        n: input sequence length (used for shape validation; model is length-agnostic)
        n_blocks: total number of residual blocks (default 21)
        base_channels: base channel count (default 32)
        kernel_size: conv kernel size (default 8)
        in_channels: 1 for raw series; >1 when multi-channel pre-transforms applied
        dropout: dropout rate for dense layers (default 0.3, following paper)
    """

    def __init__(self, cfg=None, **kwargs) -> None:
        super().__init__()
        if cfg is not None:
            n = cfg.input_length()
            n_blocks = cfg.model.n_blocks
            base_channels = cfg.model.base_channels
            kernel_size = cfg.model.kernel_size
            in_channels = 1
            dropout = cfg.model.dropout
            num_classes = getattr(cfg.model, 'num_classes', 1)
        else:
            n = kwargs.get("n", 100)
            n_blocks = kwargs.get("n_blocks", 21)
            base_channels = kwargs.get("base_channels", 32)
            kernel_size = kwargs.get("kernel_size", 8)
            in_channels = kwargs.get("in_channels", 1)
            dropout = kwargs.get("dropout", 0.3)
            num_classes = kwargs.get("num_classes", 1)
        self.n = n
        self.num_classes = num_classes
        ch = base_channels
        transition_block = n_blocks // 3  # block index where channels expand

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, ch, kernel_size=1, bias=False)

        # Build residual blocks
        blocks = []
        current_ch = ch
        for i in range(n_blocks):
            if i == transition_block:
                out_ch = ch * 2
            else:
                out_ch = current_ch
            blocks.append(ResidualBlock(current_ch, out_ch, kernel_size=kernel_size))
            current_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        if num_classes == 1:
            # Original binary head
            self.head = nn.Sequential(
                nn.Linear(current_ch, 50),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(50, 40),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(40, 30),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(30, 20),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(20, 10),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(10, 1),
            )
        else:
            # Multi-class head (drop last two layers to agree with output dim, as in Section 6)
            self.head = nn.Sequential(
                nn.Linear(current_ch, 50),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(50, 40),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(40, 30),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(30, num_classes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_channels, n) -> logits (B, 1)"""
        out = self.input_proj(x)
        out = self.blocks(out)
        out = self.pool(out).squeeze(-1)  # (B, C)
        return self.head(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Hard binary prediction.

        Args:
            x: (B, in_channels, n)
            threshold: decision threshold

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
