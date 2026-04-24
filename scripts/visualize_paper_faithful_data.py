#!/usr/bin/env python
"""Visualize canonical paper-faithful datasets stored in NPZ files.

This script expects datasets under ``data/paper_faithful/`` with names:
    - s1_train.npz, s1_test.npz
    - s1prime_train.npz, s1prime_test.npz
    - s2_train.npz, s2_test.npz
    - s3_train.npz, s3_test.npz

Each NPZ file must contain:
    - X:   shape (N, n)
    - y:   shape (N,)
    - taus: shape (N,)

For each split (train/test), the script writes one overview figure with one row
per scenario and four columns:
    1. example sequence with a change point
    2. example sequence without a change point
    3. central-value histogram (change vs no-change)
    4. change-point location histogram
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "paper_faithful"
SCENARIO_ORDER = ["s1", "s1prime", "s2", "s3"]
SPLIT_ORDER = ["train", "test"]
SCENARIO_LABELS = {
    "s1": "S1: iid Gaussian",
    "s1prime": "S1': AR(1), rho=0.7",
    "s2": "S2: AR(1), rho_t~U[0,1]",
    "s3": "S3: Cauchy(0, 0.3)",
}


@dataclass
class DatasetBundle:
    name: str
    scenario: str
    split: str
    X: np.ndarray
    y: np.ndarray
    taus: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.X.shape[1])

    @property
    def positive_mask(self) -> np.ndarray:
        return self.y == 1

    @property
    def negative_mask(self) -> np.ndarray:
        return self.y == 0

    @property
    def positive_taus(self) -> np.ndarray:
        return self.taus[self.positive_mask]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize paper-faithful NPZ datasets")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing paper-faithful NPZ datasets",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <data_dir>/plots)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used to pick representative example sequences",
    )
    return parser.parse_args()


def load_bundle(path: Path) -> DatasetBundle:
    with np.load(path) as data:
        required = {"X", "y", "taus"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        taus = np.asarray(data["taus"])

    if X.ndim != 2:
        raise ValueError(f"{path}: X must be rank-2, got shape {X.shape}")
    if y.ndim != 1 or taus.ndim != 1:
        raise ValueError(f"{path}: y and taus must be rank-1")
    if len(X) != len(y) or len(X) != len(taus):
        raise ValueError(f"{path}: inconsistent sample counts")

    stem = path.stem
    scenario, split = stem.rsplit("_", 1)
    return DatasetBundle(
        name=stem,
        scenario=scenario,
        split=split,
        X=X,
        y=y,
        taus=taus,
    )


def load_all(data_dir: Path) -> dict[tuple[str, str], DatasetBundle]:
    bundles: dict[tuple[str, str], DatasetBundle] = {}
    for scenario in SCENARIO_ORDER:
        for split in SPLIT_ORDER:
            path = data_dir / f"{scenario}_{split}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Expected dataset not found: {path}")
            bundles[(scenario, split)] = load_bundle(path)
    return bundles


def choose_example_indices(bundle: DatasetBundle, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    pos_candidates = np.flatnonzero(bundle.positive_mask)
    neg_candidates = np.flatnonzero(bundle.negative_mask)
    if len(pos_candidates) == 0 or len(neg_candidates) == 0:
        raise ValueError(f"{bundle.name} must contain both change and no-change samples")
    pos_idx = int(rng.choice(pos_candidates))
    neg_idx = int(rng.choice(neg_candidates))
    return pos_idx, neg_idx


def format_stats(bundle: DatasetBundle) -> str:
    values = bundle.X
    tau_pos = bundle.positive_taus
    change_ratio = float(bundle.positive_mask.mean())
    q01, q99 = np.quantile(values, [0.01, 0.99])
    return (
        f"N={bundle.n_samples}, n={bundle.seq_len}, "
        f"change={change_ratio:.2f}, "
        f"tau_mean={tau_pos.mean():.1f}, "
        f"value_q01/q99=({q01:.2f}, {q99:.2f})"
    )


def plot_example_with_change(ax: plt.Axes, bundle: DatasetBundle, index: int) -> None:
    x = bundle.X[index]
    tau = int(bundle.taus[index])
    ax.plot(x, color="#1f77b4", linewidth=1.25)
    ax.axvline(tau, color="crimson", linestyle="--", linewidth=1.3)
    ax.set_title("Example with change", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.text(
        0.02,
        0.98,
        f"idx={index}\ntau={tau}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )


def plot_example_without_change(ax: plt.Axes, bundle: DatasetBundle, index: int) -> None:
    x = bundle.X[index]
    ax.plot(x, color="#2ca02c", linewidth=1.25)
    ax.set_title("Example without change", fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.text(
        0.02,
        0.98,
        f"idx={index}\ny=0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )


def plot_value_histogram(ax: plt.Axes, bundle: DatasetBundle) -> None:
    pos_values = bundle.X[bundle.positive_mask].reshape(-1)
    neg_values = bundle.X[bundle.negative_mask].reshape(-1)
    lo = float(np.quantile(bundle.X, 0.01))
    hi = float(np.quantile(bundle.X, 0.99))
    pos_values = pos_values[(pos_values >= lo) & (pos_values <= hi)]
    neg_values = neg_values[(neg_values >= lo) & (neg_values <= hi)]
    bins = np.linspace(lo, hi, 60)
    ax.hist(neg_values, bins=bins, density=True, alpha=0.55, color="#2ca02c", label="No change")
    ax.hist(pos_values, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="Change")
    ax.set_title("Central 98% value histogram", fontsize=10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper right")


def plot_tau_histogram(ax: plt.Axes, bundle: DatasetBundle) -> None:
    taus = bundle.positive_taus
    bins = np.arange(1.5, bundle.seq_len - 0.5, 1.0)
    ax.hist(taus, bins=bins, color="#ff7f0e", alpha=0.8, edgecolor="white", linewidth=0.25)
    ax.set_title("Change-point locations", fontsize=10)
    ax.set_xlabel("tau")
    ax.set_ylabel("Count")
    ax.set_xlim(1, bundle.seq_len - 1)


def plot_split_overview(
    split: str,
    bundles: dict[tuple[str, str], DatasetBundle],
    out_path: Path,
    seed: int,
) -> None:
    n_rows = len(SCENARIO_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(18, 3.8 * n_rows),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, scenario in enumerate(SCENARIO_ORDER):
        bundle = bundles[(scenario, split)]
        pos_idx, neg_idx = choose_example_indices(bundle, seed + row)
        row_axes = axes[row]

        plot_example_with_change(row_axes[0], bundle, pos_idx)
        plot_example_without_change(row_axes[1], bundle, neg_idx)
        plot_value_histogram(row_axes[2], bundle)
        plot_tau_histogram(row_axes[3], bundle)

        label = SCENARIO_LABELS[scenario]
        row_axes[0].text(
            -0.42,
            0.5,
            label,
            transform=row_axes[0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        row_axes[3].text(
            1.02,
            0.5,
            format_stats(bundle),
            transform=row_axes[3].transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
        )

    fig.suptitle(f"Paper-faithful data overview: {split}", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(bundles: dict[tuple[str, str], DatasetBundle], out_dir: Path) -> None:
    lines = [
        "dataset,n_samples,seq_len,change_fraction,tau_min,tau_max,tau_mean,value_min,value_max,value_std",
    ]
    for scenario in SCENARIO_ORDER:
        for split in SPLIT_ORDER:
            bundle = bundles[(scenario, split)]
            tau_pos = bundle.positive_taus
            lines.append(
                ",".join(
                    [
                        bundle.name,
                        str(bundle.n_samples),
                        str(bundle.seq_len),
                        f"{bundle.positive_mask.mean():.4f}",
                        str(int(tau_pos.min())),
                        str(int(tau_pos.max())),
                        f"{tau_pos.mean():.4f}",
                        f"{bundle.X.min():.4f}",
                        f"{bundle.X.max():.4f}",
                        f"{bundle.X.std():.4f}",
                    ]
                )
            )
    (out_dir / "paper_faithful_summary.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else data_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_all(data_dir)

    for split in SPLIT_ORDER:
        out_path = out_dir / f"paper_faithful_{split}_overview.png"
        plot_split_overview(split=split, bundles=bundles, out_path=out_path, seed=args.seed)
        print(f"saved {out_path}")

    write_summary(bundles, out_dir)
    print(f"saved {out_dir / 'paper_faithful_summary.csv'}")


if __name__ == "__main__":
    main()
