#!/usr/bin/env python
"""Evaluate a trained model against a held-out test set and the CUSUM baseline.

Usage:
    python scripts/evaluate.py --experiment mlp_s1
    python scripts/evaluate.py --experiment rescnn_s1 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, MODELS_DIR
from src.data.simulator import simulate_dataset
from src.data.transforms import build_preprocessing_pipeline
from src.models.mlp import MLPDetector
from src.models.rescnn import ResidualCNN
from src.evaluation.metrics import evaluate_detector
from src.evaluation.baselines import run_cusum_on_dataset


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    n_input = cfg.input_length()
    if cfg.model.architecture == "mlp":
        return MLPDetector(n=n_input, variant=cfg.model.mlp_variant)
    elif cfg.model.architecture == "rescnn":
        return ResidualCNN(
            n=n_input,
            n_blocks=cfg.model.n_blocks,
            base_channels=cfg.model.base_channels,
            kernel_size=cfg.model.kernel_size,
            in_channels=1,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.model.architecture!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained change-point model")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_test", type=int, default=2000, help="Number of test sequences")
    parser.add_argument("--seed", type=int, default=123, help="Test data seed (different from train)")
    args = parser.parse_args()

    checkpoint_dir = MODELS_DIR / args.experiment
    cfg = ExperimentConfig.from_yaml(checkpoint_dir / "config.yaml")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Generate test set (different seed)
    print(f"Generating {args.n_test} test sequences...")
    X_test, y_test, taus_test = simulate_dataset(
        N=args.n_test,
        n=cfg.simulation.n,
        noise_type=cfg.simulation.noise_type,
        rho=cfg.simulation.rho,
        mu_range=cfg.simulation.mu_range,
        sigma=cfg.simulation.sigma,
        seed=args.seed,
    )

    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.simulation.noise_type,
        use_squared=cfg.model.use_squared,
        use_cross_product=cfg.model.use_cross_product,
    )
    X_proc = preprocess(X_test)

    # Load model
    model = build_model(cfg)
    ckpt = checkpoint_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Inference
    flatten = cfg.model.architecture == "mlp"
    x_tensor = torch.from_numpy(X_proc.astype(np.float32))
    if not flatten:
        x_tensor = x_tensor.unsqueeze(1)

    batch_size = 256
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i : i + batch_size].to(device)
            logits = model(batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
    y_pred = np.concatenate(all_preds)
    probs_all = np.concatenate(all_probs)

    # For localization, use a simple heuristic: tau_hat = n/2 for predicted positives
    # (proper localization needs the Localizer on a full series)
    taus_pred = np.where(y_pred == 1, cfg.simulation.n // 2, 0)

    # Neural network evaluation
    nn_result = evaluate_detector(y_test, y_pred, taus_test, taus_pred)
    print("\n=== Neural Network ===")
    print(nn_result)

    # CUSUM baseline
    print("\n=== CUSUM Baseline ===")
    cusum_preds, cusum_taus = run_cusum_on_dataset(X_test)
    cusum_result = evaluate_detector(y_test, cusum_preds, taus_test, cusum_taus)
    print(cusum_result)

    # Save results
    results = {
        "nn": {
            "detection_accuracy": nn_result.detection_accuracy,
            "power": nn_result.power,
            "type1_error": nn_result.type1_error,
            "mean_localization_error": nn_result.mean_localization_error,
            "median_localization_error": nn_result.median_localization_error,
        },
        "cusum": {
            "detection_accuracy": cusum_result.detection_accuracy,
            "power": cusum_result.power,
            "type1_error": cusum_result.type1_error,
            "mean_localization_error": cusum_result.mean_localization_error,
            "median_localization_error": cusum_result.median_localization_error,
        },
    }
    out_path = checkpoint_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
