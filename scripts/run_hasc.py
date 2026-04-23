#!/usr/import/env python
"""
Run HASC change-point detection (Section 6 of the paper).
"""

import argparse
import glob
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.rescnn import ResidualCNN

def parse_hasc_file(csv_path, label_path):
    # read csv
    df = pd.read_csv(csv_path, header=None, names=['time', 'x', 'y', 'z'])
    # read label
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split(',')
        if len(parts) >= 3:
            start_t, end_t, label = float(parts[0]), float(parts[1]), parts[2]
            labels.append((start_t, end_t, label))
    return df, labels

def extract_windows(df, labels, window_size=700, step=10):
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
            # Overlap condition
            if not (w_end_time < start_t or w_start_time > end_t):
                if label not in active_labels:
                    active_labels.append(label)
                    
        if len(active_labels) == 0:
            continue # ignore background/unknown
        elif len(active_labels) == 1:
            lbl = active_labels[0]
        else:
            lbl = "_to_".join(active_labels)
            
        windows.append(xyz[i:i+window_size].T) # shape: (3, 700)
        window_labels.append(lbl)
        
    return np.array(windows, dtype=np.float32), window_labels

def load_all_data(base_dir="data/hasc", window_size=700, step=50):
    train_X, train_y = [], []
    test_X, test_y = [], []
    
    # We will also keep the raw df and labels for test set for Algorithm 1 plotting
    test_sequences = {}
    
    persons = sorted(os.listdir(base_dir))
    for person in persons:
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir): continue
        
        csv_files = glob.glob(os.path.join(person_dir, "*.csv"))
        for csv_path in csv_files:
            label_path = csv_path.replace("-acc.csv", ".label")
            if not os.path.exists(label_path):
                continue
                
            df, labels = parse_hasc_file(csv_path, label_path)
            
            is_test = (person in ["person106", "person107"])
            
            # Extract windows
            # Use smaller step for train, larger step or exactly 1 for test inference later
            w_step = 10 if not is_test else step
            X, y = extract_windows(df, labels, window_size, step=w_step)
            
            if is_test:
                test_X.extend(X)
                test_y.extend(y)
                seq_name = os.path.basename(csv_path)
                test_sequences[seq_name] = {"df": df, "labels": labels, "X": X, "y": y}
            else:
                train_X.extend(X)
                train_y.extend(y)
                
    return (np.array(train_X), np.array(train_y)), (np.array(test_X), np.array(test_y)), test_sequences

def build_label_map(y_train, y_test):
    unique_labels = sorted(list(set(y_train) | set(y_test)))
    
    # Pure labels don't contain "_to_"
    pure_labels = [lbl for lbl in unique_labels if "_to_" not in lbl]
    trans_labels = [lbl for lbl in unique_labels if "_to_" in lbl]
    
    # Map pure labels to first indices, then transitions
    label2idx = {}
    idx2label = {}
    for i, lbl in enumerate(pure_labels + trans_labels):
        label2idx[lbl] = i
        idx2label[i] = lbl
        
    pure_indices = set(range(len(pure_labels)))
    return label2idx, idx2label, pure_indices

def train_model(X_train, y_train, num_classes, device, epochs=30, batch_size=128):
    model = ResidualCNN(n=X_train.shape[-1], in_channels=3, num_classes=num_classes)
    model = model.to(device)
    model.train()
    
    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).long().to(device)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(y_train):.4f} | Acc: {correct/len(y_train):.4f}")
        
    return model

def algorithm_1(model, df, pure_indices, window_size=700, gamma=0.5, device='cpu'):
    """Implement Algorithm 1 from the paper for a single test sequence."""
    model.eval()
    times = df['time'].values
    xyz = df[['x', 'y', 'z']].values.astype(np.float32)
    
    n_star = len(times)
    
    # Step 1: Form windows and compute L_i
    # Doing step=1 is too slow. We will do step=1.
    # To speed up, we batch it.
    print(f"Algorithm 1: Evaluating {n_star - window_size + 1} windows...")
    L = np.zeros(n_star - window_size + 1)
    
    batch_size = 512
    windows = []
    indices = []
    
    for i in range(n_star - window_size + 1):
        windows.append(xyz[i:i+window_size].T)
        indices.append(i)
        
        if len(windows) == batch_size or i == n_star - window_size:
            w_batch = torch.from_numpy(np.array(windows)).to(device)
            with torch.no_grad():
                logits = model(w_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                
            # psi(X) = 0 if pred is pure class, else 1
            for j, p in enumerate(preds):
                L[indices[j]] = 0 if p in pure_indices else 1
                
            windows = []
            indices = []
            
    # Step 2: Compute L_bar
    L_bar = np.zeros(n_star - window_size + 1)
    # L_bar_i = 1/n sum_{j=i-n+1}^i L_j
    # We can use np.convolve
    kernel = np.ones(window_size) / window_size
    L_bar_full = np.convolve(L, kernel, mode='full')
    L_bar = L_bar_full[window_size-1:len(L)] # Shift to match i
    
    # Adjust padding to match time axis
    # L_bar_aligned[i] corresponds to time index i (the end of the window)
    
    # Step 3 & 4: Find maximal segments and argmax
    # To align with timestamps, we just work with indices
    estimated_tau_indices = []
    
    in_segment = False
    s_r = 0
    for i in range(len(L_bar)):
        if L_bar[i] >= gamma:
            if not in_segment:
                s_r = i
                in_segment = True
        else:
            if in_segment:
                e_r = i - 1
                # compute argmax
                tau_r = s_r + np.argmax(L_bar[s_r:e_r+1])
                # The paper says tau_r is the index. Since L_bar index i corresponds to window starting at i,
                # the actual change point in the sequence could be around i + window_size/2.
                # Algorithm 1: tau_r is computed directly on the L_bar index. We will add window_size/2 for the actual time.
                estimated_tau_indices.append(tau_r + window_size // 2)
                in_segment = False
                
    if in_segment:
        e_r = len(L_bar) - 1
        tau_r = s_r + np.argmax(L_bar[s_r:e_r+1])
        estimated_tau_indices.append(tau_r + window_size // 2)

    return times[estimated_tau_indices], L_bar

def plot_hasc_results(df, true_labels, estimated_times, L_bar, output_path, window_size=700):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    times = df['time'].values
    
    # Plot 1: Accelerometer data
    ax1.plot(times, df['x'], alpha=0.6, label='x')
    ax1.plot(times, df['y'], alpha=0.6, label='y')
    ax1.plot(times, df['z'], alpha=0.6, label='z')
    
    # Ground truth change points
    # A change point occurs at the boundary of a label
    gt_times = []
    for i in range(len(true_labels) - 1):
        gt_times.append(true_labels[i][1]) # end of current label
        
    for t in gt_times:
        ax1.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2, label='True CP' if t == gt_times[0] else "")
        
    for t in estimated_times:
        ax1.axvline(x=t, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Est CP' if t == estimated_times[0] else "")
        
    ax1.set_ylabel("Accelerometer")
    ax1.set_title("HASC Activity Data with Change Points")
    ax1.legend(loc="upper right")
    
    # Plot 2: L_bar
    # L_bar starts from index window_size-1 in terms of ending window, 
    # but the time assigned to index i of L_bar is roughly i + window_size/2
    l_bar_times = times[window_size//2 : window_size//2 + len(L_bar)]
    ax2.plot(l_bar_times, L_bar, color='black', label='L_bar')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Gamma=0.5')
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("L_bar (Change Probability)")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/hasc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Loading data from {args.data_dir}...")
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw), test_seqs = load_all_data(args.data_dir, window_size=700, step=10)
    
    print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")
    
    if len(X_train_raw) == 0:
        print("No training data found. Ensure data/hasc has person101-105 directories with CSVs.")
        return
        
    label2idx, idx2label, pure_indices = build_label_map(y_train_raw, y_test_raw)
    num_classes = len(label2idx)
    print(f"Detected {num_classes} classes (Pure classes: {len(pure_indices)}, Transitions: {num_classes - len(pure_indices)})")
    
    y_train = np.array([label2idx[lbl] for lbl in y_train_raw])
    y_test = np.array([label2idx[lbl] for lbl in y_test_raw])
    
    print("Training ResCNN Model...")
    model = train_model(X_train_raw, y_train, num_classes, device, epochs=args.epochs)
    
    # Test accuracy on test windows
    model.eval()
    X_t = torch.from_numpy(X_test_raw).to(device)
    y_t = torch.from_numpy(y_test).to(device)
    with torch.no_grad():
        logits = []
        for i in range(0, len(X_t), 512):
            logits.append(model(X_t[i:i+512]))
        logits = torch.cat(logits)
        preds = logits.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
        print(f"Test Accuracy on windows: {acc:.4f}")
        
    # Evaluate Algorithm 1 on one sequence
    if len(test_seqs) > 0:
        seq_name = list(test_seqs.keys())[0]
        print(f"Applying Algorithm 1 on test sequence: {seq_name}")
        df = test_seqs[seq_name]["df"]
        labels = test_seqs[seq_name]["labels"]
        
        estimated_times, L_bar = algorithm_1(model, df, pure_indices, window_size=700, gamma=0.5, device=device)
        print(f"Estimated Change Points: {estimated_times}")
        
        os.makedirs("output", exist_ok=True)
        plot_hasc_results(df, labels, estimated_times, L_bar, f"output/hasc_{seq_name}_algo1.png")

if __name__ == "__main__":
    main()
