import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SimulationConfig, ModelConfig, TrainingConfig, LocalizationConfig, ExperimentConfig
from src.data.simulator import simulate_dataset
from src.models.mlp import MLPDetector
from src.models.rescnn import ResidualCNN


@pytest.fixture(scope="session")
def small_n():
    return 20


@pytest.fixture(scope="session")
def small_sim_config(small_n):
    return SimulationConfig(n=small_n, N=200, noise_type="S1", rho=0.0, seed=0)


@pytest.fixture(scope="session")
def small_dataset(small_sim_config):
    X, y, taus = simulate_dataset(
        N=small_sim_config.N,
        n=small_sim_config.n,
        noise_type=small_sim_config.noise_type,
        seed=small_sim_config.seed,
    )
    return X, y, taus


@pytest.fixture(scope="session")
def tiny_mlp(small_n):
    return MLPDetector(n=small_n, variant="pruned")


@pytest.fixture(scope="session")
def tiny_rescnn(small_n):
    return ResidualCNN(n=small_n, n_blocks=3, base_channels=4, kernel_size=3)
