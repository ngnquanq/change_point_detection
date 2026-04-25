#!/usr/bin/env python
"""
Split HASC data into train/val sets and save to data/hasc/splits/.

Usage:
    python scripts/split_hasc.py
    python scripts/split_hasc.py --val_fraction 0.2 --window_size 700 --step 10
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ─── Data parsing ────────────────────────────────────────────────────────────

def parse_hasc_file(csv_path, label_path):
    """Parse a HASC accelerometer CSV and its label file."""
    df = pd.read_csv(csv_path, header=None, names=['time', 'x', 'y', 'z'])
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            start_t, end_t, label = float(parts[0]), float(parts[1]), parts[2]
            labels.append((start_t, end_t, label))
    return df, labels


def extract_windows(df, labels, window_size=700, step=10):
    """Extract sliding windows and assign activity labels."""
    windows = []
    window_labels = []

    times = df['time'].values
    xyz = df[['x', 'y', 'z']].values

    n_samples = len(times)
    for i in range(0, n_samples - window_size + 1, step):
        w_start_time = times[i]
        w_end_time = times[i + window_size - 1]

        active_labels = []
        for start_t, end_t, label in labels:
            if not (w_end_time < start_t or w_start_time > end_t):
                if label not in active_labels:
                    active_labels.append(label)

        if len(active_labels) == 0:
            continue
        elif len(active_labels) == 1:
            lbl = active_labels[0]
        else:
            lbl = "_to_".join(active_labels)

        windows.append(xyz[i:i + window_size].T)  # shape: (3, window_size)
        window_labels.append(lbl)

    if len(windows) == 0:
        return np.array([], dtype=np.float32).reshape(0, 3, window_size), []
    return np.array(windows, dtype=np.float32), window_labels


# ─── Split logic ─────────────────────────────────────────────────────────────

def build_label_map(all_labels):
    """Build label-to-index mapping. Pure labels first, then transitions."""
    unique_labels = sorted(set(all_labels))
    pure_labels = [lbl for lbl in unique_labels if "_to_" not in lbl]
    trans_labels = [lbl for lbl in unique_labels if "_to_" in lbl]

    label2idx = {}
    idx2label = {}
    for i, lbl in enumerate(pure_labels + trans_labels):
        label2idx[lbl] = i
        idx2label[i] = lbl

    pure_indices = list(range(len(pure_labels)))
    return label2idx, idx2label, pure_indices


def stratified_split(X, y, val_fraction=0.2, seed=42):
    """Stratified train/val split preserving class proportions."""
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y)

    train_idx, val_idx = [], []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Split HASC data into train/val")
    parser.add_argument("--data_dir", type=str, default="data/hasc",
                        help="Root directory containing person* subdirectories")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for splits (default: <data_dir>/splits)")
    parser.add_argument("--window_size", type=int, default=700)
    parser.add_argument("--step", type=int, default=10,
                        help="Sliding window step size")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, "splits")
    os.makedirs(output_dir, exist_ok=True)

    # ── Collect all windows from all persons ──
    print(f"Loading data from {args.data_dir}...")
    all_X, all_y_str = [], []

    persons = sorted(os.listdir(args.data_dir))
    for person in persons:
        person_dir = os.path.join(args.data_dir, person)
        if not os.path.isdir(person_dir):
            continue

        csv_files = sorted(glob.glob(os.path.join(person_dir, "*.csv")))
        for csv_path in csv_files:
            label_path = csv_path.replace("-acc.csv", ".label")
            if not os.path.exists(label_path):
                continue

            df, labels = parse_hasc_file(csv_path, label_path)
            X, y = extract_windows(df, labels, args.window_size, args.step)

            if len(X) > 0:
                all_X.append(X)
                all_y_str.extend(y)
                print(f"  {person}/{os.path.basename(csv_path)}: {len(X)} windows")

    if len(all_X) == 0:
        print("No data found. Check that data_dir has person* directories with CSVs and labels.")
        return

    X_all = np.concatenate(all_X, axis=0)
    print(f"\nTotal windows: {len(X_all)}")

    # ── Build label map ──
    label2idx, idx2label, pure_indices = build_label_map(all_y_str)
    num_classes = len(label2idx)
    n_pure = len(pure_indices)
    print(f"Classes: {num_classes} (pure: {n_pure}, transitions: {num_classes - n_pure})")

    y_all = np.array([label2idx[lbl] for lbl in all_y_str])

    # ── Split ──
    train_idx, val_idx = stratified_split(X_all, y_all, args.val_fraction, args.seed)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}")

    # ── Print class distribution ──
    print("\nClass distribution:")
    print(f"  {'idx':>3s}  {'label':30s} {'train':>6s} {'val':>5s} {'total':>6s}")
    print("  " + "-" * 55)
    for cls_idx in sorted(idx2label.keys()):
        name = idx2label[cls_idx]
        n_train = int((y_train == cls_idx).sum())
        n_val = int((y_val == cls_idx).sum())
        n_total = int((y_all == cls_idx).sum())
        marker = "" if cls_idx in pure_indices else " (transition)"
        print(f"  [{cls_idx:2d}] {name:30s} {n_train:6d} {n_val:5d} {n_total:6d}{marker}")

    # ── Save ──
    np.savez_compressed(os.path.join(output_dir, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, "val.npz"), X=X_val, y=y_val)

    meta = {
        "label2idx": label2idx,
        "idx2label": {str(k): v for k, v in idx2label.items()},
        "pure_indices": pure_indices,
        "num_classes": num_classes,
        "window_size": args.window_size,
        "step": args.step,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved to {output_dir}/")
    print(f"   train.npz  ({len(X_train)} samples)")
    print(f"   val.npz    ({len(X_val)} samples)")
    print(f"   meta.json")


if __name__ == "__main__":
    main()
