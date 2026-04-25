#!/usr/bin/env python
"""Evaluate a trained HASC model on the validation split.

Usage:
    python scripts/eval_hasc.py
    python scripts/eval_hasc.py --split_dir data/hasc/splits --checkpoint models/hasc/best_model.pt
    python scripts/eval_hasc.py --save_confusion output/hasc_confusion.png
"""

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.rescnn import ResidualCNN


# ─── Data ────────────────────────────────────────────────────────────────────

def load_val_data(split_dir: str):
    meta_path = os.path.join(split_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found in {split_dir}. "
            "Run 'python scripts/split_hasc.py' first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    val_data = np.load(os.path.join(split_dir, "val.npz"))
    X_val = val_data["X"]   # (N, 3, L)
    y_val = val_data["y"]   # (N,)

    idx2label = {int(k): v for k, v in meta["idx2label"].items()}
    pure_indices = set(meta["pure_indices"])
    num_classes = meta["num_classes"]

    return X_val, y_val, num_classes, idx2label, pure_indices


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             device: torch.device, batch_size: int = 512):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).long().to(device)

    all_preds, total_loss = [], 0.0

    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i : i + batch_size]
            yb = y_t[i : i + batch_size]
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    preds = np.concatenate(all_preds)
    avg_loss = total_loss / len(y)
    accuracy  = float((preds == y).mean())
    f1_macro  = float(f1_score(y, preds, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y, preds, average="weighted", zero_division=0))

    per_class_acc = {}
    for cls in np.unique(y):
        mask = y == cls
        per_class_acc[int(cls)] = float((preds[mask] == y[mask]).mean())

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_acc": per_class_acc,
        "preds": preds,
    }


# ─── Confusion matrix plot ────────────────────────────────────────────────────

def plot_confusion(y_true, preds, idx2label, save_path):
    labels_sorted = sorted(idx2label.keys())
    names = [idx2label[i] for i in labels_sorted]

    cm = confusion_matrix(y_true, preds, labels=labels_sorted)
    # Normalize per row (true class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    n = len(names)
    fig_size = max(12, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("HASC Val — Confusion Matrix (row-normalized)", fontsize=12, fontweight="bold")

    # Annotate cells with raw counts
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"Confusion matrix saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate HASC model on validation split")
    parser.add_argument("--split_dir",   default="data/hasc/splits",
                        help="Directory with val.npz and meta.json")
    parser.add_argument("--checkpoint",  default="models/hasc/best_model.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--save_confusion", default="",
                        help="(optional) path to save confusion matrix PNG")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    # ── Data ──
    print(f"\nLoading val data from {args.split_dir} ...")
    X_val, y_val, num_classes, idx2label, pure_indices = load_val_data(args.split_dir)
    print(f"  Val samples : {len(X_val)}")
    print(f"  Classes     : {num_classes}  "
          f"(pure: {len(pure_indices)}, transitions: {num_classes - len(pure_indices)})")
    print(f"  Window shape: {X_val.shape[1:]}")

    # ── Model ──
    print(f"\nLoading checkpoint from {args.checkpoint} ...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    in_channels = X_val.shape[1]   # 3
    seq_len     = X_val.shape[2]   # 700

    model = ResidualCNN(
        n=seq_len,
        in_channels=in_channels,
        num_classes=num_classes,
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {n_params:,}")

    # ── Evaluate ──
    print("\nRunning evaluation ...")
    metrics = evaluate(model, X_val, y_val, device, args.batch_size)

    print("\n" + "=" * 50)
    print("  HASC Validation Results (best model)")
    print("=" * 50)
    print(f"  Loss        : {metrics['loss']:.4f}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (Macro)  : {metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
    print("=" * 50)

    # ── Per-class breakdown ──
    print("\nPer-class accuracy:")
    header = f"  {'idx':>3}  {'label':<30} {'acc':>6}  {'type'}"
    print(header)
    print("  " + "-" * 55)
    for cls_idx in sorted(metrics["per_class_acc"].keys()):
        name    = idx2label.get(cls_idx, str(cls_idx))
        acc     = metrics["per_class_acc"][cls_idx]
        kind    = "pure" if cls_idx in pure_indices else "transition"
        flag    = "⚠" if acc < 0.5 else ""
        print(f"  [{cls_idx:2d}]  {name:<30} {acc:6.4f}  {kind} {flag}")

    # ── sklearn classification report ──
    target_names = [idx2label.get(i, str(i)) for i in sorted(idx2label.keys())]
    print("\nClassification Report:")
    print(classification_report(
        y_val, metrics["preds"],
        target_names=target_names,
        zero_division=0,
    ))

    # ── Confusion matrix ──
    if args.save_confusion:
        plot_confusion(y_val, metrics["preds"], idx2label, args.save_confusion)


if __name__ == "__main__":
    main()
