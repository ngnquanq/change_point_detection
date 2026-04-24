"""Simulated dataset for change-point detection training."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from src.registry import DATASET_REGISTRY
from src.data.paper_faithful import maybe_load_split
from src.data.simulator import simulate_dataset
from src.data.transforms import augment_reversed, build_preprocessing_pipeline


@DATASET_REGISTRY.register("simulated")
class SimulatedDataset:
    """Generate synthetic time series with optional change points.

    Wraps simulate_dataset + augmentation + preprocessing into a single class
    that can be instantiated from config.
    """

    def __init__(self, cfg) -> None:
        self.dataset_cfg = cfg.dataset
        self.model_cfg = cfg.model
        self.training_cfg = cfg.training

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate, augment, and preprocess data.

        Returns:
            (X, y, taus) — preprocessed sequences, labels, change-point locations
        """
        cfg = self.dataset_cfg

        loaded = maybe_load_split(cfg.data_dir, cfg.noise_type, split="train")
        if loaded is not None:
            X, y, taus, path = loaded
            print(f"Loading canonical train split from {path}...")
        else:
            print(f"Simulating {cfg.N} sequences of length {cfg.n} "
                  f"({cfg.noise_type} noise)...")
            X, y, taus = simulate_dataset(
                N=cfg.N,
                n=cfg.n,
                noise_type=cfg.noise_type,
                rho=cfg.rho,
                sigma=cfg.sigma,
                cauchy_scale=cfg.cauchy_scale,
                snr_based_mu=cfg.snr_based_mu,
                seed=cfg.seed,
            )

        # Augment with reversed sequences
        if self.training_cfg.augment_reversed:
            X, y, taus = augment_reversed(X, y, taus)
            print(f"After augmentation: {len(X)} sequences")

        # Preprocess
        preprocess = build_preprocessing_pipeline(
            noise_type=cfg.noise_type,
            use_squared=self.model_cfg.use_squared,
            use_cross_product=self.model_cfg.use_cross_product,
        )
        X = preprocess(X)

        return X, y, taus
