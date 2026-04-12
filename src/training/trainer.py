from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainingConfig


class Trainer:
    """Generic binary classifier trainer for MLPDetector or ResidualCNN.

    Uses BCEWithLogitsLoss (models must output raw logits).
    Adam optimizer with configurable lr and weight_decay.
    Early stopping on validation loss; saves best checkpoint.

    Args:
        model: nn.Module
        config: TrainingConfig
        device: torch.device
        checkpoint_dir: directory where best_model.pt is saved
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """Run full training loop.

        Returns:
            history: {
                "train_loss": list[float],
                "val_loss": list[float],
                "train_acc": list[float],
                "val_acc": list[float],
            }
        """
        history: dict = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_ckpt = self.checkpoint_dir / "best_model.pt"

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(best_ckpt)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{self.config.epochs} | "
                    f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val loss={val_loss:.4f} acc={val_acc:.4f}"
                )

            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.config.patience})")
                break

        # Restore best weights
        if best_ckpt.exists():
            self.load_checkpoint(best_ckpt)

        return history

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, labels in loader:
            x = x.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(x).squeeze(1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += len(labels)

        return total_loss / total, correct / total

    def _val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, labels in loader:
                x = x.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(x).squeeze(1)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def save_checkpoint(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def load_best(self) -> None:
        best_ckpt = self.checkpoint_dir / "best_model.pt"
        self.load_checkpoint(best_ckpt)
