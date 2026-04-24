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

from src.config import ExperimentConfig, PROJECT_ROOT
from src.registry import MODEL_REGISTRY
from src.data.paper_faithful import maybe_load_split
from src.data.simulator import simulate_dataset
from src.data.transforms import build_preprocessing_pipeline
from src.evaluation.metrics import evaluate_detector
from src.evaluation.baselines import run_cusum_on_dataset

# Import to trigger registration
import src.models.rescnn  # noqa: F401
import src.models.mlp     # noqa: F401


def auto_detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    return {
        "detection_accuracy": float((y_true == y_pred).mean()),
        "power": float(y_pred[pos_mask].mean()) if pos_mask.any() else 0.0,
        "type1_error": float(y_pred[neg_mask].mean()) if neg_mask.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained change-point model")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_test", type=int, default=2000, help="Number of test sequences")
    parser.add_argument("--seed", type=int, default=123, help="Test data seed (different from train)")
    args = parser.parse_args()

    config_dir = PROJECT_ROOT / "models" / args.experiment
    cfg = ExperimentConfig.from_yaml(config_dir / "config.yaml")
    checkpoint_dir = cfg.models_path / args.experiment
    device = auto_detect_device(args.device)

    loaded = maybe_load_split(cfg.dataset.data_dir, cfg.dataset.noise_type, split="test")
    if loaded is not None:
        X_test, y_test, taus_test, dataset_path = loaded
        print(f"Loading canonical test split from {dataset_path}...")
        test_source = "canonical_npz"
    else:
        print(f"Generating {args.n_test} test sequences...")
        X_test, y_test, taus_test = simulate_dataset(
            N=args.n_test,
            n=cfg.dataset.n,
            noise_type=cfg.dataset.noise_type,
            rho=cfg.dataset.rho,
            sigma=cfg.dataset.sigma,
            cauchy_scale=cfg.dataset.cauchy_scale,
            snr_based_mu=cfg.dataset.snr_based_mu,
            seed=args.seed,
        )
        dataset_path = None
        test_source = "simulated"

    preprocess = build_preprocessing_pipeline(
        noise_type=cfg.dataset.noise_type,
        use_squared=cfg.model.use_squared,
        use_cross_product=cfg.model.use_cross_product,
    )
    X_proc = preprocess(X_test)

    # Load model via registry — no if/else
    model = MODEL_REGISTRY.build(cfg.model.architecture, cfg=cfg)
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

    # Neural-network classifier metrics on fixed windows.
    nn_result = summarize_predictions(y_test, y_pred)
    print("\n=== Neural Network ===")
    print(nn_result)

    # CUSUM baseline
    print("\n=== CUSUM Baseline ===")
    cusum_preds, cusum_taus = run_cusum_on_dataset(X_test)
    cusum_eval = evaluate_detector(y_test, cusum_preds, taus_test, cusum_taus)
    cusum_result = {
        "detection_accuracy": cusum_eval.detection_accuracy,
        "power": cusum_eval.power,
        "type1_error": cusum_eval.type1_error,
    }
    print(cusum_result)

    # Save results
    results = {
        "experiment": args.experiment,
        "test_source": test_source,
        "test_dataset_path": str(dataset_path) if dataset_path is not None else None,
        "n_test": int(len(X_test)),
        "nn": {
            **nn_result,
        },
        "cusum": {
            **cusum_result,
        },
    }
    out_path = checkpoint_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
