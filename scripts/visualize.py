#!/usr/bin/env python
"""Generate visualizations: simulated data, training curves, localization demo,
and NN vs CUSUM performance comparison.

Usage:
    python scripts/visualize.py --experiment mlp_s1
    python scripts/visualize.py --experiment mlp_s1 --out_dir plots/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, PROJECT_ROOT
from src.data.simulator import simulate_dataset, simulate_sequence
from src.data.transforms import build_preprocessing_pipeline, minmax_scale
from src.evaluation.baselines import run_cusum_on_dataset
from src.evaluation.metrics import evaluate_detector, compute_roc
from src.models.mlp import MLPDetector
from src.models.rescnn import ResidualCNN
from src.inference.localizer import Localizer

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted")


# ──────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ──────────────────────────────────────────────��───────────────────────────────

def plot_simulated_sequences(axes: list, n: int = 100, seed: int = 0) -> None:
    """Row of 3 panels: S1, S1_prime (AR1), S3 — each with a change-point example."""
    configs = [
        ("S1\n(i.i.d. Gaussian)", "S1", 0.0),
        ("S1′ / S2\n(AR(1), ρ=0.7)", "S1_prime", 0.7),
        ("S3\n(Cauchy heavy-tail)", "S3", 0.0),
    ]
    rng = np.random.default_rng(seed)
    for ax, (title, noise_type, rho) in zip(axes, configs):
        x_change, tau = simulate_sequence(
            n=n, has_change=True, noise_type=noise_type, rho=rho,
            mu_range=(-2.0, 2.0), sigma=1.0, rng=rng,
        )
        x_flat, _ = simulate_sequence(
            n=n, has_change=False, noise_type=noise_type, rho=rho,
            mu_range=(-2.0, 2.0), sigma=1.0, rng=rng,
        )
        t = np.arange(n)
        ax.plot(t, x_flat, color=COLORS[0], alpha=0.6, linewidth=1.2, label="No change")
        ax.plot(t, x_change, color=COLORS[1], alpha=0.85, linewidth=1.2, label="With change")
        ax.axvline(tau, color="crimson", linewidth=1.8, linestyle="--", label=f"τ = {tau}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, loc="upper right")


def plot_training_curves(axes: list, history: dict) -> None:
    """Two panels: loss curve and accuracy curve across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss, ax_acc = axes

    ax_loss.plot(epochs, history["train_loss"], label="Train", color=COLORS[0])
    ax_loss.plot(epochs, history["val_loss"], label="Val", color=COLORS[1])
    ax_loss.set_title("Training Loss (BCE)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="Train", color=COLORS[0])
    ax_acc.plot(epochs, history["val_acc"], label="Val", color=COLORS[1])
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    ax_acc.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Best val acc {best_acc:.3f} @ ep {best_epoch}")
    ax_acc.set_title("Training Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=8)


def plot_performance_bars(ax, nn_result: dict, cusum_result: dict) -> None:
    """Grouped bar chart: NN vs CUSUM on three metrics."""
    metrics = ["Detection\nAccuracy", "Power\n(TPR)", "Type-I Error\n(FPR)"]
    nn_vals = [
        nn_result["detection_accuracy"],
        nn_result["power"],
        nn_result["type1_error"],
    ]
    cusum_vals = [
        cusum_result["detection_accuracy"],
        cusum_result["power"],
        cusum_result["type1_error"],
    ]
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width / 2, nn_vals, width, label="Neural Network", color=COLORS[0])
    bars2 = ax.bar(x + width / 2, cusum_vals, width, label="CUSUM", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate")
    ax.set_title("NN vs CUSUM Performance")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)


def plot_localization_errors(ax, nn_errors: list, cusum_errors: list) -> None:
    """Overlapping histograms of localization errors for NN and CUSUM."""
    bins = np.linspace(0, max(max(nn_errors, default=1), max(cusum_errors, default=1)) + 5, 30)
    ax.hist(nn_errors, bins=bins, alpha=0.6, color=COLORS[0], label="Neural Network", density=True)
    ax.hist(cusum_errors, bins=bins, alpha=0.6, color=COLORS[2], label="CUSUM", density=True)
    nn_med = np.median(nn_errors) if nn_errors else 0
    cs_med = np.median(cusum_errors) if cusum_errors else 0
    ax.axvline(nn_med, color=COLORS[0], linestyle="--", linewidth=1.5, label=f"NN median={nn_med:.1f}")
    ax.axvline(cs_med, color=COLORS[2], linestyle="--", linewidth=1.5, label=f"CUSUM median={cs_med:.1f}")
    ax.set_xlabel("|τ̂ - τ|  (positions)")
    ax.set_ylabel("Density")
    ax.set_title("Localization Error Distribution\n(true-positive sequences)")
    ax.legend(fontsize=8)


def plot_roc(ax, y_true: np.ndarray, nn_probs: np.ndarray, cusum_scores: np.ndarray) -> None:
    """ROC curves for NN and CUSUM."""
    fpr_nn, tpr_nn, _ = compute_roc(y_true, nn_probs)
    fpr_cs, tpr_cs, _ = compute_roc(y_true, cusum_scores)
    ax.plot(fpr_nn, tpr_nn, color=COLORS[0], linewidth=2, label="Neural Network")
    ax.plot(fpr_cs, tpr_cs, color=COLORS[2], linewidth=2, label="CUSUM")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)


def plot_localization_demo(
    ax_series: plt.Axes,
    ax_prob: plt.Axes,
    model,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int = 42,
) -> None:
    """Long series with sliding-window detection overlay."""
    from scripts.locate import generate_long_series

    total_length = 600
    n_changes = 3
    series, true_taus = generate_long_series(
        total_length=total_length,
        n_changes=n_changes,
        noise_type=cfg.simulation.noise_type,
        rho=cfg.simulation.rho,
        sigma=cfg.simulation.sigma,
        seed=seed,
    )

    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.simulation.noise_type,
        use_squared=cfg.model.use_squared,
        use_cross_product=cfg.model.use_cross_product,
    )
    localizer = Localizer(model, cfg.localization, device, preprocess)
    detections = localizer.locate(series)

    # Recompute window probabilities for the probability track
    n = cfg.localization.window_size
    step = cfg.localization.step_size
    starts = np.arange(0, total_length - n + 1, step)
    windows = np.stack([series[s: s + n] for s in starts])
    _, probs = localizer._score_windows(windows)
    L_bar = localizer._rolling_average((probs >= 0.5).astype(np.int32), cfg.localization.rolling_window)
    window_centers = starts + n // 2

    t = np.arange(total_length)
    ax_series.plot(t, series, color=COLORS[0], linewidth=0.9, alpha=0.85, label="Series")
    for tau in true_taus:
        ax_series.axvline(tau, color="crimson", linewidth=1.5, linestyle="--")
    for cp in detections:
        ax_series.axvline(cp.location, color="darkorange", linewidth=1.5, linestyle=":")
    legend_handles = [
        mpatches.Patch(color="crimson", label="True τ"),
        mpatches.Patch(color="darkorange", label="Detected τ̂"),
    ]
    ax_series.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax_series.set_title(f"Long Series ({total_length} pts, {n_changes} true changes)")
    ax_series.set_ylabel("Value")
    ax_series.set_xlabel("")

    ax_prob.plot(window_centers, probs, color=COLORS[1], linewidth=0.8, alpha=0.7, label="P(change)")
    ax_prob.plot(window_centers, L_bar, color=COLORS[3], linewidth=1.5, label="Rolling avg L̄")
    ax_prob.axhline(cfg.localization.gamma, color="gray", linestyle="--", linewidth=1.0,
                    label=f"γ = {cfg.localization.gamma}")
    for tau in true_taus:
        ax_prob.axvline(tau, color="crimson", linewidth=1.5, linestyle="--")
    for cp in detections:
        ax_prob.axvline(cp.location, color="darkorange", linewidth=1.5, linestyle=":")
    ax_prob.set_ylim(-0.05, 1.15)
    ax_prob.set_xlabel("Position in series")
    ax_prob.set_ylabel("Probability / Score")
    ax_prob.set_title("Algorithm 1 — Sliding Window Probabilities")
    ax_prob.legend(fontsize=8, loc="upper right")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory to save plots (default: models/<experiment>/plots/)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    checkpoint_dir = PROJECT_ROOT / "models" / args.experiment
    cfg = ExperimentConfig.from_yaml(checkpoint_dir / "config.yaml")
    device = torch.device(args.device)

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training history
    with open(checkpoint_dir / "history.json") as f:
        history = json.load(f)

    # Load model
    if cfg.model.architecture == "mlp":
        model = MLPDetector(n=cfg.input_length(), variant=cfg.model.mlp_variant)
    else:
        model = ResidualCNN(
            n=cfg.input_length(), n_blocks=cfg.model.n_blocks,
            base_channels=cfg.model.base_channels, kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        )
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.to(device).eval()

    # Generate test data
    print("Generating test data...")
    X_test, y_test, taus_test = simulate_dataset(
        N=args.n_test, n=cfg.simulation.n, noise_type=cfg.simulation.noise_type,
        rho=cfg.simulation.rho, mu_range=cfg.simulation.mu_range,
        sigma=cfg.simulation.sigma, seed=args.seed,
    )
    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.simulation.noise_type,
        use_squared=cfg.model.use_squared,
        use_cross_product=cfg.model.use_cross_product,
    )
    X_proc = preprocess(X_test)

    # NN inference
    flatten = cfg.model.architecture == "mlp"
    x_tensor = torch.from_numpy(X_proc.astype(np.float32))
    if not flatten:
        x_tensor = x_tensor.unsqueeze(1)
    all_probs, all_preds = [], []
    with torch.no_grad():
        for i in range(0, len(x_tensor), 256):
            batch = x_tensor[i: i + 256].to(device)
            logits = model(batch).squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(p)
            all_preds.append((p >= 0.5).astype(int))
    nn_probs = np.concatenate(all_probs)
    nn_preds = np.concatenate(all_preds)
    taus_pred_nn = np.where(nn_preds == 1, cfg.simulation.n // 2, 0)
    nn_result = evaluate_detector(y_test, nn_preds, taus_test, taus_pred_nn)

    # CUSUM inference
    cusum_preds, cusum_taus = run_cusum_on_dataset(X_test)
    # CUSUM scores: use raw max CUSUM stat as the probability proxy
    from src.evaluation.baselines import cusum_detector
    cusum_scores = np.array([cusum_detector(x)[1] for x in X_test])
    # Normalise to [0,1] for ROC
    cs_max = cusum_scores.max()
    if cs_max > 0:
        cusum_scores_norm = cusum_scores / cs_max
    else:
        cusum_scores_norm = cusum_scores
    cusum_result = evaluate_detector(y_test, cusum_preds, taus_test, cusum_taus)

    nn_res_dict = {
        "detection_accuracy": nn_result.detection_accuracy,
        "power": nn_result.power,
        "type1_error": nn_result.type1_error,
    }
    cusum_res_dict = {
        "detection_accuracy": cusum_result.detection_accuracy,
        "power": cusum_result.power,
        "type1_error": cusum_result.type1_error,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 1: Simulated sequences (3 noise types)
    # ─────────────────────────────────────────────���────────────────────────────
    print("Plotting simulated sequences...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    plot_simulated_sequences(axes1, n=cfg.simulation.n, seed=0)
    fig1.suptitle("Simulated Time Series — Three Noise Profiles", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig1_simulated_sequences.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ──────────────────────────────────────────────────────────────���───────────
    # Figure 2: Training curves
    # ─────────────────────────────────���────────────────────────────────────────
    print("Plotting training curves...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    plot_training_curves(axes2, history)
    fig2.suptitle(f"Training History — {args.experiment}", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig2_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 3: Performance comparison (3 panels: bars, loc errors, ROC)
    # ──────────────────────────────────────────────────────────────────────────
    print("Plotting performance comparison...")
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    plot_performance_bars(axes3[0], nn_res_dict, cusum_res_dict)
    plot_localization_errors(axes3[1],
                             nn_result.localization_errors.tolist(),
                             cusum_result.localization_errors.tolist())
    plot_roc(axes3[2], y_test, nn_probs, cusum_scores_norm)
    fig3.suptitle(f"Neural Network vs CUSUM — {args.experiment}", fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig3_performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure 4: Localization demo (long series)
    # ──────────────────────────────────────────────────────────────────���───────
    print("Plotting localization demo...")
    fig4, axes4 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    plot_localization_demo(axes4[0], axes4[1], model, cfg, device, seed=args.seed)
    fig4.suptitle(f"Algorithm 1 — Sliding Window Localization Demo ({args.experiment})",
                  fontsize=13, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(out_dir / "fig4_localization_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    print(f"\nAll plots saved to: {out_dir}/")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
