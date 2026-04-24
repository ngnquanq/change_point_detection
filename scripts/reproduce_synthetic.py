#!/usr/bin/env python
"""Canonical one-command synthetic reproducibility workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import ExperimentConfig

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "synthetic"
CANONICAL_EXPERIMENTS = {
    "mlp_s1": "configs/mlp_s1.yaml",
    "rescnn_s1_paper": "configs/rescnn_s1_paper.yaml",
    "mlp_s1prime": "configs/mlp_s1prime.yaml",
    "rescnn_s1prime_paper": "configs/rescnn_s1prime_paper.yaml",
    "mlp_s2": "configs/mlp_s2.yaml",
    "rescnn_s2_paper": "configs/rescnn_s2_paper.yaml",
    "mlp_s3": "configs/mlp_s3.yaml",
    "rescnn_s3_paper": "configs/rescnn_s3_paper.yaml",
}
PLOT_EXPERIMENT = "mlp_s1"
CANONICAL_DATASETS = [
    "s1_train.npz",
    "s1_test.npz",
    "s1prime_train.npz",
    "s1prime_test.npz",
    "s2_train.npz",
    "s2_test.npz",
    "s3_train.npz",
    "s3_test.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce canonical synthetic artifacts")
    parser.add_argument(
        "--step",
        choices=["all", "data", "train", "eval", "plots", "manifest", "verify"],
        default="all",
        help="Which portion of the reproducibility workflow to run",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training/evaluation device for canonical runs (default: cpu)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke child scripts",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def expected_checkpoint_dir(experiment: str) -> Path:
    return REPO_ROOT / "models" / experiment


def expected_plot_paths() -> list[Path]:
    return [
        expected_checkpoint_dir(PLOT_EXPERIMENT) / "plots" / "fig1_simulated_sequences.png",
        expected_checkpoint_dir(PLOT_EXPERIMENT) / "plots" / "fig2_training_curves.png",
        expected_checkpoint_dir(PLOT_EXPERIMENT) / "plots" / "fig3_performance_comparison.png",
        expected_checkpoint_dir(PLOT_EXPERIMENT) / "plots" / "fig4_localization_demo.png",
        REPO_ROOT / "data" / "paper_faithful" / "plots" / "paper_faithful_train_overview.png",
        REPO_ROOT / "data" / "paper_faithful" / "plots" / "paper_faithful_test_overview.png",
        REPO_ROOT / "data" / "paper_faithful" / "plots" / "paper_faithful_summary.csv",
    ]


def run_data_step(python_bin: str) -> None:
    run_command([python_bin, "scripts/generate_reproducible_data.py"])


def run_train_step(python_bin: str, device: str) -> None:
    for config_path in CANONICAL_EXPERIMENTS.values():
        run_command([python_bin, "scripts/train.py", "--config", config_path, "--device", device])


def run_eval_step(python_bin: str, device: str) -> None:
    for experiment in CANONICAL_EXPERIMENTS:
        run_command([python_bin, "scripts/evaluate.py", "--experiment", experiment, "--device", device])


def run_plots_step(python_bin: str, device: str) -> None:
    run_command([python_bin, "scripts/visualize.py", "--experiment", PLOT_EXPERIMENT, "--device", device])
    run_command([python_bin, "scripts/visualize_paper_faithful_data.py"])


def verify_artifacts() -> None:
    run_command([sys.executable, "scripts/generate_reproducible_data.py", "--verify"])

    for experiment in CANONICAL_EXPERIMENTS:
        checkpoint_dir = expected_checkpoint_dir(experiment)
        required = [
            checkpoint_dir / "best_model.pt",
            checkpoint_dir / "config.yaml",
            checkpoint_dir / "history.json",
            checkpoint_dir / "eval_results.json",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Missing required artifact: {path}")
        ExperimentConfig.from_yaml(checkpoint_dir / "config.yaml")
        with (checkpoint_dir / "eval_results.json").open() as handle:
            json.load(handle)

    for path in expected_plot_paths():
        if not path.exists():
            raise FileNotFoundError(f"Missing required plot artifact: {path}")


def build_manifest() -> dict:
    datasets = []
    for filename in CANONICAL_DATASETS:
        path = REPO_ROOT / "data" / "paper_faithful" / filename
        sidecar = path.with_suffix(".hash.txt")
        datasets.append(
            {
                "name": filename,
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "declared_hash": sidecar.read_text().strip() if sidecar.exists() else None,
            }
        )

    experiments = []
    for experiment in CANONICAL_EXPERIMENTS:
        checkpoint_dir = expected_checkpoint_dir(experiment)
        eval_path = checkpoint_dir / "eval_results.json"
        history_path = checkpoint_dir / "history.json"
        config_path = checkpoint_dir / "config.yaml"
        with eval_path.open() as handle:
            results = json.load(handle)
        with history_path.open() as handle:
            history = json.load(handle)
        experiments.append(
            {
                "experiment": experiment,
                "checkpoint_dir": str(checkpoint_dir.relative_to(REPO_ROOT)),
                "config_sha256": sha256_file(config_path),
                "history_sha256": sha256_file(history_path),
                "eval_sha256": sha256_file(eval_path),
                "best_val_acc": max(history.get("val_acc", [0.0])),
                "nn": results["nn"],
                "cusum": results["cusum"],
            }
        )

    plots = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
        }
        for path in expected_plot_paths()
    ]

    return {
        "canonical_experiments": list(CANONICAL_EXPERIMENTS),
        "plot_experiment": PLOT_EXPERIMENT,
        "datasets": datasets,
        "experiments": experiments,
        "plots": plots,
        "notes": {
            "device_default": "cpu",
            "localization_benchmark": "README-facing synthetic summary reports fixed-window detection metrics; localization remains a deterministic demo plot generated by locate.py logic.",
            "canonical_models": ["mlp", "rescnn"],
        },
    }


def write_manifest_files(manifest: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    summary_path = ARTIFACTS_DIR / "summary.md"

    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    lines = [
        "# Synthetic Reproducibility Summary",
        "",
        "| Experiment | NN Acc | NN Power | NN FPR | CUSUM Acc | CUSUM Power | CUSUM FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in manifest["experiments"]:
        nn = exp["nn"]
        cs = exp["cusum"]
        lines.append(
            "| {experiment} | {nn_acc:.4f} | {nn_power:.4f} | {nn_fpr:.4f} | {cs_acc:.4f} | {cs_power:.4f} | {cs_fpr:.4f} |".format(
                experiment=exp["experiment"],
                nn_acc=nn["detection_accuracy"],
                nn_power=nn["power"],
                nn_fpr=nn["type1_error"],
                cs_acc=cs["detection_accuracy"],
                cs_power=cs["power"],
                cs_fpr=cs["type1_error"],
            )
        )
    lines.extend(
        [
            "",
            "Localization is intentionally not part of the fixed-window summary table.",
            "The deterministic localization demo is regenerated in `models/mlp_s1/plots/fig4_localization_demo.png`.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"saved {manifest_path}")
    print(f"saved {summary_path}")


def main() -> None:
    args = parse_args()

    if args.step in ("all", "data"):
        run_data_step(args.python)
    if args.step in ("all", "train"):
        run_train_step(args.python, args.device)
    if args.step in ("all", "eval"):
        run_eval_step(args.python, args.device)
    if args.step in ("all", "plots"):
        run_plots_step(args.python, args.device)
    if args.step in ("all", "verify"):
        verify_artifacts()
    if args.step in ("all", "manifest", "verify"):
        manifest = build_manifest()
        write_manifest_files(manifest)


if __name__ == "__main__":
    main()
