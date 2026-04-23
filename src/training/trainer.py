from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import TrainingConfig, OptimizerConfig, SchedulerConfig
from src.registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY


class Trainer:
    """Generic binary classifier trainer for MLPDetector or ResidualCNN.

    Uses BCEWithLogitsLoss (models must output raw logits).
    Optimizer, scheduler, and all hyper-parameters are driven by config
    and instantiated via registries — zero hardcoded types.
    Early stopping on validation loss; saves best checkpoint.
    TensorBoard logging for loss and metrics.

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
        self.optimizer = self._build_optimizer(config.optimizer)
        self.scheduler = self._build_scheduler(config.scheduler)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "runs"))

    def _build_optimizer(self, cfg: OptimizerConfig) -> torch.optim.Optimizer:
        """Build optimizer from registry — no if/else."""
        params = self.model.parameters()
        opt_class = OPTIMIZER_REGISTRY.get(cfg.name)

        # Build kwargs based on what each optimizer accepts
        kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.name in ("adam", "adamw"):
            kwargs["betas"] = cfg.betas
        elif cfg.name == "sgd":
            kwargs["momentum"] = cfg.momentum

        return opt_class(params, **kwargs)

    def _build_scheduler(self, cfg: SchedulerConfig):
        """Build LR scheduler from registry. Returns None if 'none'."""
        if cfg.name == "none":
            return None

        sched_class = SCHEDULER_REGISTRY.get(cfg.name)

        if cfg.name == "cosine":
            return sched_class(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.optimizer.lr * cfg.lrf,
            )
        elif cfg.name == "step":
            return sched_class(
                self.optimizer,
                step_size=cfg.step_size,
                gamma=cfg.gamma,
            )
        elif cfg.name == "onecycle":
            return None  # built lazily in train() when loader size is known

        return sched_class(self.optimizer)

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

        # Build OneCycleLR lazily (needs loader size)
        if self.config.scheduler.name == "onecycle":
            onecycle_cls = SCHEDULER_REGISTRY.get("onecycle")
            self.scheduler = onecycle_cls(
                self.optimizer,
                max_lr=self.config.optimizer.lr,
                epochs=self.config.epochs,
                steps_per_epoch=len(train_loader),
            )

        epoch_pbar = tqdm(
            range(1, self.config.epochs + 1),
            desc="Training",
            unit="epoch",
        )

        for epoch in epoch_pbar:
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            val_loss, val_acc = self._val_epoch(val_loader)

            # Step scheduler (except OneCycleLR which steps per batch)
            if self.scheduler is not None and self.config.scheduler.name != "onecycle":
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # --- TensorBoard logging ---
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            self.writer.flush()

            # --- tqdm progress bar ---
            is_best = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(best_ckpt)
                is_best = " ★"
            else:
                patience_counter += 1

            epoch_pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best": f"{best_val_loss:.4f}",
                "patience": f"{patience_counter}/{self.config.patience}",
            })

            # Periodic detailed log
            if epoch % 10 == 0 or epoch == 1:
                tqdm.write(
                    f"Epoch {epoch:3d}/{self.config.epochs} │ "
                    f"train loss={train_loss:.4f} acc={train_acc:.4f} │ "
                    f"val loss={val_loss:.4f} acc={val_acc:.4f}{is_best}"
                )

            if patience_counter >= self.config.patience:
                tqdm.write(
                    f"⏹ Early stopping at epoch {epoch} "
                    f"(patience={self.config.patience}, best_val_loss={best_val_loss:.4f})"
                )
                break

        # Restore best weights
        if best_ckpt.exists():
            self.load_checkpoint(best_ckpt)

        self.writer.close()
        tqdm.write(f"✓ Best val loss: {best_val_loss:.4f} | TensorBoard logs: {self.checkpoint_dir / 'runs'}")

        return history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        batch_pbar = tqdm(
            loader,
            desc=f"  Train",
            leave=False,
            unit="batch",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        for x, labels in batch_pbar:
            x = x.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(x).squeeze(1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None and self.config.scheduler.name == "onecycle":
                self.scheduler.step()


            batch_loss = loss.item()
            total_loss += batch_loss * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            batch_correct = (preds == labels.long()).sum().item()
            correct += batch_correct
            total += len(labels)

            batch_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "acc": f"{batch_correct / len(labels):.3f}",
            })

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
