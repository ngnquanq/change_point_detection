import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.rescnn import ResidualCNN


# ─── Utilities ───────────────────────────────────────────────────────────────

def increment_path(path):
    """Auto-increment path: output/hasc_runs → hasc_runs2 → hasc_runs3 etc.
    Similar to YOLO's increment_path behavior."""
    path = str(path)
    if not os.path.exists(path):
        return path
    # Try path2, path3, ...
    i = 2
    while True:
        new_path = f"{path}{i}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


# ─── Data loading ────────────────────────────────────────────────────────────

def load_split_data(split_dir="data/hasc/splits"):
    """Load pre-split train/val data and metadata from split_dir."""
    # Load metadata
    meta_path = os.path.join(split_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json found in {split_dir}. "
            f"Run 'python scripts/split_hasc.py' first to create the splits."
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load arrays
    train_data = np.load(os.path.join(split_dir, "train.npz"))
    val_data = np.load(os.path.join(split_dir, "val.npz"))

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    # Reconstruct label maps
    label2idx = meta["label2idx"]
    idx2label = {int(k): v for k, v in meta["idx2label"].items()}
    pure_indices = set(meta["pure_indices"])
    num_classes = meta["num_classes"]

    print(f"Loaded from {split_dir}:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Classes: {num_classes} (pure: {len(pure_indices)}, "
          f"transitions: {num_classes - len(pure_indices)})")

    return X_train, y_train, X_val, y_val, num_classes, idx2label, pure_indices


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model, X, y, device, batch_size=512):
    """Evaluate model on a dataset. Returns loss, accuracy, f1, per-class accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).long().to(device)

    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size]
            yb = y_t[i:i + batch_size]
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    preds = np.concatenate(all_preds)

    avg_loss = total_loss / len(y)
    accuracy = float((preds == y).mean())
    f1_macro = float(f1_score(y, preds, average='macro', zero_division=0))
    f1_weighted = float(f1_score(y, preds, average='weighted', zero_division=0))

    per_class_acc = {}
    for cls in np.unique(y):
        mask = y == cls
        if mask.sum() > 0:
            per_class_acc[int(cls)] = float((preds[mask] == y[mask]).mean())

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class_acc': per_class_acc,
        'preds': preds,
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, num_classes, device,
                idx2label, epochs=100, batch_size=128, lr=0.001,
                weight_decay=1e-4, patience=20, grad_clip=1.0,
                scheduler_type="cosine", log_dir="output/hasc_runs",
                checkpoint_dir="output/hasc_checkpoints"):
    """Train ResidualCNN with TensorBoard logging, validation, and early stopping.

    Args:
        X_train, y_train: training data and labels (numpy)
        X_val, y_val: validation data and labels (numpy)
        num_classes: number of output classes
        device: torch device
        idx2label: dict mapping class index to label name
        epochs: maximum number of training epochs
        batch_size: training batch size
        lr: initial learning rate
        weight_decay: L2 regularization
        patience: early stopping patience (0 = disable)
        grad_clip: max gradient norm (0 = disable)
        scheduler_type: 'cosine', 'step', or 'none'
        log_dir: TensorBoard log directory
        checkpoint_dir: directory to save model checkpoints
    """
    model = ResidualCNN(n=X_train.shape[-1], in_channels=3, num_classes=num_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # Checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).long().to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        scheduler = None

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False, ncols=100)
        for xb, yb in pbar:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss.item() * len(yb)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)

            # Update progress bar with running stats
            running_loss = train_loss / train_total
            running_acc = train_correct / train_total
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        train_avg_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # ── Validation ──
        val_metrics = evaluate_model(model, X_val, y_val, device, batch_size)

        # ── TensorBoard ──
        writer.add_scalar('Loss/train', train_avg_loss, epoch + 1)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch + 1)

        writer.add_scalar('Accuracy/train', train_acc, epoch + 1)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch + 1)

        writer.add_scalar('F1/val_macro', val_metrics['f1_macro'], epoch + 1)
        writer.add_scalar('F1/val_weighted', val_metrics['f1_weighted'], epoch + 1)

        writer.add_scalar('LearningRate', current_lr, epoch + 1)

        for cls_idx, cls_acc in val_metrics['per_class_acc'].items():
            label_name = idx2label.get(cls_idx, str(cls_idx))
            writer.add_scalar(f'Val_PerClass_Acc/{label_name}', cls_acc, epoch + 1)

        # ── Console ──
        print(f"Epoch {epoch+1}/{epochs} | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_avg_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"F1: {val_metrics['f1_macro']:.4f}")

        # ── Early stopping & best model ──
        improved = False
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            improved = True
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            improved = True

        if improved:
            best_epoch = epoch + 1
            epochs_without_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            epochs_without_improve += 1

        if patience > 0 and epochs_without_improve >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs, best was epoch {best_epoch})")
            break

    writer.close()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    print(f"\nBest Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Checkpoint saved: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train HASC change-point detection model on pre-split data"
    )
    # Data
    parser.add_argument("--split_dir", type=str, default="data/hasc/splits",
                        help="Directory containing train.npz, val.npz, meta.json")
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0=disable)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0=disable)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"], help="LR scheduler type")
    # Output
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_dir", type=str, default="output/hasc_runs")
    parser.add_argument("--checkpoint_dir", type=str, default="output/hasc_checkpoints")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load pre-split data ──
    X_train, y_train, X_val, y_val, num_classes, idx2label, pure_indices = \
        load_split_data(args.split_dir)

    # Print class distribution
    print("\nClass distribution:")
    print(f"  {'idx':>3s}  {'label':30s} {'train':>6s} {'val':>5s}")
    print("  " + "-" * 50)
    for cls_idx in sorted(idx2label.keys()):
        name = idx2label[cls_idx]
        n_train = int((y_train == cls_idx).sum())
        n_val = int((y_val == cls_idx).sum())
        marker = "" if cls_idx in pure_indices else " (trans)"
        print(f"  [{cls_idx:2d}] {name:30s} {n_train:6d} {n_val:5d}{marker}")

    # ── Auto-increment run directories ──
    log_dir = increment_path(args.log_dir)
    checkpoint_dir = increment_path(args.checkpoint_dir)
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # ── Train ──
    print("\nTraining ResCNN Model...")
    model = train_model(
        X_train, y_train, X_val, y_val,
        num_classes, device, idx2label,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, grad_clip=args.grad_clip,
        scheduler_type=args.scheduler,
        log_dir=log_dir, checkpoint_dir=checkpoint_dir,
    )

    # ── Final val evaluation (with best model) ──
    print("\n=== Final Validation Evaluation (best model) ===")
    val_metrics = evaluate_model(model, X_val, y_val, device)
    print(f"Val Loss: {val_metrics['loss']:.4f}")
    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
    print(f"Val F1 (weighted): {val_metrics['f1_weighted']:.4f}")

    print("\nPer-class val accuracy:")
    for cls_idx, cls_acc in sorted(val_metrics['per_class_acc'].items()):
        name = idx2label.get(cls_idx, str(cls_idx))
        print(f"  [{cls_idx:2d}] {name:30s} acc={cls_acc:.4f}")


if __name__ == "__main__":
    main()
