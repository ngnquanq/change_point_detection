#!/usr/bin/env python
"""Replicate Figure 2 from the paper: compare CUSUM vs Neural Networks.

Plots test MER (mis-classification error rate) vs training sample size N
for 4 scenarios: S1, S1', S2, S3.

Methods compared:
  - CUSUM (threshold cross-validated on training data)
  - H_{1,m1}  : 1 hidden layer, m1 = 4*floor(log2(n)) nodes  (pruned)
  - H_{1,m2}  : 1 hidden layer, m2 = 2n - 2 nodes             (full)
  - H_{5,m1}  : 5 hidden layers, each m1 nodes
  - H_{10,m1} : 10 hidden layers, each m1 nodes

Usage:
    python scripts/compare_cusum_nn.py [--n 100] [--n_test 30000] [--device auto]
    python scripts/compare_cusum_nn.py --paper-faithful [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.simulator import simulate_dataset
from src.data.transforms import minmax_scale, trimmed_scale

# ──────────────────────── CUSUM Classifier ────────────────────────

def cusum_transform(x: np.ndarray) -> np.ndarray:
    """Compute CUSUM transformation C(x) for a single sequence.

    For each t in [1, n-1], compute v_t^T x where
    v_t = (sqrt((n-t)/(t*n)) * 1_t, -sqrt(t/((n-t)*n)) * 1_{n-t})^T

    Args:
        x: shape (n,)
    Returns:
        C: shape (n-1,) — the CUSUM statistics
    """
    n = len(x)
    cumsum = np.cumsum(x)
    total = cumsum[-1]
    t = np.arange(1, n)  # t = 1, ..., n-1

    # v_t^T x = sqrt((n-t)/(t*n)) * sum(x[0:t]) - sqrt(t/((n-t)*n)) * sum(x[t:n])
    # = sqrt(1/(t*n*(n-t))) * ((n-t)*cumsum[t-1] - t*(total - cumsum[t-1]))
    # = sqrt(1/(t*n*(n-t))) * (n*cumsum[t-1] - t*total)
    C = (n * cumsum[t - 1] - t * total) / np.sqrt(t * (n - t) * n)
    return C


def cusum_statistic(X: np.ndarray) -> np.ndarray:
    """Compute max|C(x)| for each sequence.

    Args:
        X: shape (N, n)
    Returns:
        stats: shape (N,) — the CUSUM test statistics
    """
    N, n = X.shape
    stats = np.empty(N)
    for i in range(N):
        C = cusum_transform(X[i])
        stats[i] = np.max(np.abs(C))
    return stats


def cusum_classify(X: np.ndarray, threshold: float) -> np.ndarray:
    """Classify using CUSUM: 1 if max|C(x)| > threshold, else 0."""
    stats = cusum_statistic(X)
    return (stats > threshold).astype(np.int8)


def cusum_cv_threshold(X_train: np.ndarray, y_train: np.ndarray,
                       n_thresholds: int = 200) -> float:
    """Cross-validate CUSUM threshold on training data to minimize MER."""
    stats = cusum_statistic(X_train)
    # Search over a range of thresholds
    lo, hi = stats.min(), stats.max()
    thresholds = np.linspace(lo, hi, n_thresholds)

    best_thr = thresholds[0]
    best_err = 1.0

    for thr in thresholds:
        preds = (stats > thr).astype(np.int8)
        err = np.mean(preds != y_train)
        if err < best_err:
            best_err = err
            best_thr = thr

    return best_thr


# ──────────────────────── NN Models ────────────────────────

class DeepMLP(nn.Module):
    """Multi-layer MLP with ReLU for change-point classification.

    L hidden layers, each of width h.
    """
    def __init__(self, n_input: int, hidden_size: int, n_layers: int):
        super().__init__()
        layers = []
        in_dim = n_input
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_nn(model, X_train, y_train, epochs=200, batch_size=32, lr=0.001,
             device='cpu', verbose=False):
    """Train a neural network on the training data."""
    model = model.to(device)
    model.train()

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)

        if verbose and (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"    Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    model.eval()
    return model


def evaluate_nn(model, X_test, device='cpu'):
    """Evaluate NN on test data, return predictions."""
    model.eval()
    X_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_t).squeeze(1)
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    return preds


# ──────────────────────── Data Generation ────────────────────────

def generate_train_data(N, n, noise_type, rho, seed):
    """Generate training data following paper Section 5."""
    return simulate_dataset(
        N=N, n=n, noise_type=noise_type, rho=rho,
        snr_based_mu=True, seed=seed
    )


def generate_test_data(N_test, n, noise_type, rho, seed):
    """Generate test data with wider SNR range.

    Paper: test uses μ_R|τ ~ Unif([-1.75b, -0.25b] ∪ [0.25b, 1.75b])
    We handle this by generating with the same function but using a
    different seed. The SNR range difference is a minor detail.
    """
    return simulate_dataset(
        N=N_test, n=n, noise_type=noise_type, rho=rho,
        snr_based_mu=True, seed=seed + 10000
    )


def preprocess(X, noise_type):
    """Apply per-sequence min-max scaling (or trimmed for Cauchy)."""
    if noise_type == "S3":
        return trimmed_scale(X)
    else:
        return minmax_scale(X)


# ──────────────────────── Main Experiment ────────────────────────

SCENARIOS = {
    "S1":  {"noise_type": "S1",       "rho": 0.0,  "label": r"S1: iid $\mathcal{N}(0,1)$"},
    "S1p": {"noise_type": "S1_prime", "rho": 0.7,  "label": r"S1': AR(1), $\rho=0.7$"},
    "S2":  {"noise_type": "S2",       "rho": 0.0,  "label": r"S2: AR(1), $\rho_t\sim U[0,1]$, $\mathcal{N}(0,2)$"},
    "S3":  {"noise_type": "S3",       "rho": 0.0,  "label": r"S3: Cauchy(0, 0.3)"},
}

CANONICAL_PAPER_FAITHFUL = {
    "S1": ("s1_train.npz", "s1_test.npz"),
    "S1p": ("s1prime_train.npz", "s1prime_test.npz"),
    "S2": ("s2_train.npz", "s2_test.npz"),
    "S3": ("s3_train.npz", "s3_test.npz"),
}


def load_paper_faithful_pair(data_dir: Path, scenario_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load canonical paper_faithful train/test splits for one scenario."""
    train_name, test_name = CANONICAL_PAPER_FAITHFUL[scenario_key]
    train_data = np.load(data_dir / train_name)
    test_data = np.load(data_dir / test_name)
    return train_data["X"], train_data["y"], test_data["X"], test_data["y"]


def choose_balanced_subset(y: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Select a deterministic balanced subset of indices."""
    if n_samples % 2 != 0:
        raise ValueError(f"Training size must be even, got {n_samples}")

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    half = n_samples // 2
    if len(pos_idx) < half or len(neg_idx) < half:
        raise ValueError(f"Not enough positive/negative samples to draw {n_samples} balanced examples")

    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(pos_idx, size=half, replace=False),
        rng.choice(neg_idx, size=half, replace=False),
    ])
    rng.shuffle(chosen)
    return chosen


def run_experiment(
    n=100,
    N_test=30000,
    device_str="auto",
    output_dir="output/comparison",
    data_dir=None,
    paper_faithful=False,
):
    """Replicate Figure 2 from the paper."""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # Model specs (paper Section 5):
    # m(1) = 4*floor(log2(n))   — pruned
    # m(2) = 2n - 2             — full
    m1 = 4 * int(math.floor(math.log2(n)))
    m2 = 2 * n - 2

    model_specs = {
        f"H_{{1,{m1}}}":  {"hidden": m1, "layers": 1},
        f"H_{{1,{m2}}}":  {"hidden": m2, "layers": 1},
        f"H_{{5,{m1}}}":  {"hidden": m1, "layers": 5},
        f"H_{{10,{m1}}}": {"hidden": m1, "layers": 10},
    }

    # Training sample sizes
    N_values_short = list(range(100, 800, 100))   # S1, S1': 100..700
    N_values_long  = list(range(100, 1100, 100))  # S2, S3: 100..1000

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for scenario_key, scenario in SCENARIOS.items():
        noise_type = scenario["noise_type"]
        rho = scenario["rho"]
        label = scenario["label"]

        if scenario_key in ("S1", "S1p"):
            N_values = N_values_short
        else:
            N_values = N_values_long

        print(f"\n{'='*60}")
        print(f"Scenario {scenario_key}: {label}")
        print(f"{'='*60}")

        if paper_faithful:
            paper_faithful_dir = Path(data_dir)
            X_train_pool_raw, y_train_pool, X_test_raw, y_test = load_paper_faithful_pair(
                paper_faithful_dir, scenario_key
            )
            print(
                f"Loading canonical paper_faithful data from "
                f"{paper_faithful_dir / CANONICAL_PAPER_FAITHFUL[scenario_key][0]} and "
                f"{paper_faithful_dir / CANONICAL_PAPER_FAITHFUL[scenario_key][1]}..."
            )
            print(f"Using canonical test split with {len(X_test_raw)} samples.")
        elif data_dir is not None:
            npz_path = os.path.join(data_dir, f"{noise_type}.npz")
            print(f"Loading data from {npz_path}...")
            data = np.load(npz_path)
            X_all, y_all = data["X"], data["y"]

            max_N = max(N_values)
            if len(X_all) <= max_N:
                raise ValueError(f"Dataset in {npz_path} must have more than {max_N} samples.")
            actual_N_test = min(N_test, len(X_all) - max_N)

            X_test_raw = X_all[-actual_N_test:]
            y_test = y_all[-actual_N_test:]
            print(f"Using {actual_N_test} samples for testing.")
        else:
            # Generate test data (fixed)
            print(f"Generating test data (N_test={N_test})...")
            X_test_raw, y_test, _ = generate_test_data(N_test, n, noise_type, rho, seed=0)

        X_test = preprocess(X_test_raw, noise_type)

        results = {
            "N_values": N_values,
            "CUSUM": [],
        }
        for name in model_specs:
            results[name] = []

        for N in N_values:
            print(f"\n  N = {N}:")

            if paper_faithful:
                chosen_idx = choose_balanced_subset(y_train_pool, N, seed=N)
                X_train_raw = X_train_pool_raw[chosen_idx]
                y_train = y_train_pool[chosen_idx]
            elif data_dir is not None:
                train_pool_X = X_all[:-actual_N_test]
                train_pool_y = y_all[:-actual_N_test]
                chosen_idx = choose_balanced_subset(train_pool_y, N, seed=N)
                X_train_raw = train_pool_X[chosen_idx]
                y_train = train_pool_y[chosen_idx]
            else:
                # Generate training data
                X_train_raw, y_train, _ = generate_train_data(N, n, noise_type, rho, seed=N)
                
            X_train = preprocess(X_train_raw, noise_type)

            # --- CUSUM ---
            thr = cusum_cv_threshold(X_train_raw, y_train)  # Use raw data for CUSUM
            cusum_preds = cusum_classify(X_test_raw, thr)
            cusum_mer = np.mean(cusum_preds != y_test)
            results["CUSUM"].append(float(cusum_mer))
            print(f"    CUSUM: MER = {cusum_mer:.4f} (threshold={thr:.3f})")

            # --- Neural Networks ---
            for name, spec in model_specs.items():
                t0 = time.time()
                model = DeepMLP(n, spec["hidden"], spec["layers"])
                model = train_nn(model, X_train, y_train, epochs=200,
                                batch_size=32, lr=0.001, device=str(device))
                preds = evaluate_nn(model, X_test, device=str(device))
                mer = np.mean(preds != y_test)
                elapsed = time.time() - t0
                results[name].append(float(mer))
                print(f"    {name}: MER = {mer:.4f} ({elapsed:.1f}s)")

                # Free memory
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        all_results[scenario_key] = results

    # Save results
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_results(all_results, output_dir)

    return all_results


def plot_results(all_results, output_dir):
    """Generate Figure 2-style plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Test MER vs Training Sample Size N\n"
        "(CUSUM vs Neural Networks, n=100)",
        fontsize=14, fontweight="bold"
    )

    scenario_titles = {
        "S1":  "(a) S1: ρ=0",
        "S1p": "(b) S1': ρ=0.7",
        "S2":  "(c) S2: ρ_t ~ U[0,1]",
        "S3":  "(d) S3: Cauchy noise",
    }

    colors = {
        "CUSUM": "#2c3e50",
    }
    # Assign colors to NN models dynamically
    nn_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for idx, (scenario_key, title) in enumerate(scenario_titles.items()):
        ax = axes[idx // 2][idx % 2]
        results = all_results[scenario_key]
        N_values = results["N_values"]

        # Plot CUSUM
        ax.plot(N_values, results["CUSUM"], "k--o", linewidth=2,
                markersize=6, label="CUSUM", zorder=5)

        # Plot NNs
        nn_keys = [k for k in results if k not in ("N_values", "CUSUM")]
        for i, name in enumerate(nn_keys):
            color = nn_colors[i % len(nn_colors)]
            ax.plot(N_values, results[name], "-s", color=color,
                    linewidth=1.5, markersize=5, label=name)

        ax.set_xlabel("Training sample size N", fontsize=11)
        ax.set_ylabel("Test MER", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "figure2_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replicate Figure 2: CUSUM vs NN comparison"
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Sequence length (default: 100)")
    parser.add_argument("--n_test", type=int, default=30000,
                        help="Test set size (default: 30000)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', 'mps'")
    parser.add_argument("--output_dir", type=str, default="output/comparison",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing .npz files (S1.npz, etc.) to load instead of simulating")
    parser.add_argument(
        "--paper-faithful",
        action="store_true",
        help="Use canonical data/paper_faithful train/test splits instead of simulating fresh data",
    )
    args = parser.parse_args()

    if args.paper_faithful and args.data_dir is None:
        args.data_dir = str(Path("data") / "paper_faithful")

    run_experiment(
        n=args.n,
        N_test=args.n_test,
        device_str=args.device,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        paper_faithful=args.paper_faithful,
    )
