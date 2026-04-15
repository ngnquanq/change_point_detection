from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


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


class ResidualCNN(nn.Module):
    """Deep Residual CNN for change-point detection.

    Architecture (following paper — 21 residual blocks):
      Input projection: Conv1d(in_ch, base_ch, 1)
      Blocks 1–7:   base_ch   -> base_ch
      Block 8:      base_ch   -> base_ch*2  (channel expansion)
      Blocks 9–21:  base_ch*2 -> base_ch*2
      AdaptiveAvgPool1d(1) -> flatten
      FC(base_ch*2, 64) -> ReLU -> FC(64, 1)   [raw logits]

    Outputs raw logits (use BCEWithLogitsLoss during training).

    Args:
        n: input sequence length (used for shape validation; model is length-agnostic)
        n_blocks: total number of residual blocks (default 21)
        base_channels: base channel count (default 32)
        kernel_size: conv kernel size (default 8)
        in_channels: 1 for raw series; >1 when multi-channel pre-transforms applied
    """

    def __init__(
        self,
        n: int,
        n_blocks: int = 21,
        base_channels: int = 32,
        kernel_size: int = 8,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.n = n
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
        self.head = nn.Sequential(
            nn.Linear(current_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
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
