from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Tuple

import yaml
from yaml.constructor import ConstructorError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _to_yaml_safe(value):
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_to_yaml_safe(v) for v in value]
    return value


class _TupleSafeLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


_TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


@dataclass
class DatasetConfig:
    """Unified dataset configuration for both simulated and HASC data."""
    source: Literal["simulated", "hasc"] = "simulated"
    data_dir: str = "data"

    # --- Simulated data params ---
    n: int = 100
    N: int = 10_000
    noise_type: Literal["S1", "S1_prime", "S2", "S3"] = "S1"
    rho: float = 0.0
    mu_range: Tuple[float, float] = (-2.0, 2.0)
    sigma: float = 1.0
    cauchy_scale: float = 0.3       # Paper S3: Cauchy(0, 0.3)
    snr_based_mu: bool = False      # Use paper's SNR-based μ_R formula (Section 5)

    # --- HASC data params ---
    hasc_dir: str = "data/hasc"
    channel: Literal["magnitude", "x", "y", "z"] = "magnitude"
    window_size: int = 100
    stride: int = 10

    # --- Common ---
    seed: int = 42

    @property
    def data_path(self) -> Path:
        """Resolved absolute path to data directory."""
        p = Path(self.data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


# Backward compatibility alias
SimulationConfig = DatasetConfig


@dataclass
class ModelConfig:
    architecture: Literal["mlp", "rescnn"] = "mlp"
    mlp_variant: Literal["full", "pruned"] = "pruned"
    n_blocks: int = 21
    base_channels: int = 32
    kernel_size: int = 8
    dropout: float = 0.3               # Paper Supp. C.3: dropout rate in FC layers
    use_squared: bool = False
    use_cross_product: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer parameters."""
    name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9          # SGD only
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam/AdamW


@dataclass
class SchedulerConfig:
    """Learning rate scheduler parameters."""
    name: Literal["none", "cosine", "step", "onecycle"] = "none"
    lrf: float = 0.01              # Final LR factor (lr * lrf = final lr)
    step_size: int = 50            # StepLR: decay every N epochs
    gamma: float = 0.1             # StepLR: multiply lr by gamma


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    patience: int = 20
    val_fraction: float = 0.1
    augment_reversed: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LocalizationConfig:
    window_size: int = 100
    step_size: int = 1
    rolling_window: int = 10
    gamma: float = 0.5


@dataclass
class ExperimentConfig:
    experiment_name: str = "default"
    models_dir: str = "models"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)

    # Backward compatibility: expose dataset as simulation
    @property
    def simulation(self) -> DatasetConfig:
        return self.dataset

    @property
    def models_path(self) -> Path:
        p = Path(self.models_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            try:
                raw = yaml.safe_load(f)
            except ConstructorError:
                f.seek(0)
                raw = yaml.load(f, Loader=_TupleSafeLoader)

        # Backward compat: accept "simulation" key as "dataset"
        dataset_raw = raw.get("dataset", raw.get("simulation", {}))
        ds = DatasetConfig(**dataset_raw)
        # mu_range comes from YAML as a list; convert to tuple
        if isinstance(ds.mu_range, list):
            ds.mu_range = tuple(ds.mu_range)

        mdl = ModelConfig(**raw.get("model", {}))

        # Parse training with nested optimizer/scheduler
        training_raw = dict(raw.get("training", {}))
        opt_raw = training_raw.pop("optimizer", {})
        sched_raw = training_raw.pop("scheduler", {})
        opt = OptimizerConfig(**opt_raw)
        if isinstance(opt.betas, list):
            opt.betas = tuple(opt.betas)
        sched = SchedulerConfig(**sched_raw)
        trn = TrainingConfig(**training_raw, optimizer=opt, scheduler=sched)

        loc = LocalizationConfig(**raw.get("localization", {}))
        return cls(
            experiment_name=raw.get("experiment_name", "default"),
            models_dir=raw.get("models_dir", "models"),
            dataset=ds,
            model=mdl,
            training=trn,
            localization=loc,
        )

    def save_yaml(self, path: str | Path) -> None:
        d = _to_yaml_safe(asdict(self))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(d, f, default_flow_style=False, sort_keys=False)

    def input_length(self) -> int:
        """Effective input length after pre-transforms."""
        n = self.dataset.n if self.dataset.source == "simulated" else self.dataset.window_size
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
