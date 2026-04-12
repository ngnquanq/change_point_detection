from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class SimulationConfig:
    n: int = 100
    N: int = 10_000
    noise_type: Literal["S1", "S1_prime", "S2", "S3"] = "S1"
    rho: float = 0.0
    mu_range: Tuple[float, float] = (-2.0, 2.0)
    sigma: float = 1.0
    seed: int = 42


@dataclass
class ModelConfig:
    architecture: Literal["mlp", "rescnn"] = "mlp"
    mlp_variant: Literal["full", "pruned"] = "pruned"
    n_blocks: int = 21
    base_channels: int = 32
    kernel_size: int = 8
    use_squared: bool = False
    use_cross_product: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    val_fraction: float = 0.1
    augment_reversed: bool = True


@dataclass
class LocalizationConfig:
    window_size: int = 100
    step_size: int = 1
    rolling_window: int = 10
    gamma: float = 0.5


@dataclass
class ExperimentConfig:
    experiment_name: str = "default"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        sim = SimulationConfig(**raw.get("simulation", {}))
        # mu_range comes from YAML as a list; convert to tuple
        if isinstance(sim.mu_range, list):
            sim.mu_range = tuple(sim.mu_range)
        mdl = ModelConfig(**raw.get("model", {}))
        trn = TrainingConfig(**raw.get("training", {}))
        loc = LocalizationConfig(**raw.get("localization", {}))
        return cls(
            experiment_name=raw.get("experiment_name", "default"),
            simulation=sim,
            model=mdl,
            training=trn,
            localization=loc,
        )

    def save_yaml(self, path: str | Path) -> None:
        d = asdict(self)
        # Convert tuples to lists for YAML serialization
        if "simulation" in d and "mu_range" in d["simulation"]:
            d["simulation"]["mu_range"] = list(d["simulation"]["mu_range"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)

    def input_length(self) -> int:
        """Effective input length after pre-transforms."""
        n = self.simulation.n
        extra = 0
        if self.model.use_squared:
            extra += n
        if self.model.use_cross_product:
            extra += n  # cross-product is length n-1 padded to n
        return n + extra

    def mlp_hidden_size(self) -> int:
        n = self.input_length()
        if self.model.mlp_variant == "full":
            return 2 * n - 2
        return 4 * int(math.floor(math.log2(n)))
