from pathlib import Path

from src.config import ExperimentConfig, PROJECT_ROOT


def test_config_round_trip_is_safe_yaml(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / "configs" / "mlp_s1.yaml")
    out_path = tmp_path / "config.yaml"

    cfg.save_yaml(out_path)
    text = out_path.read_text()

    assert "!!python" not in text

    reloaded = ExperimentConfig.from_yaml(out_path)
    assert reloaded.training.optimizer.betas == (0.9, 0.999)
    assert reloaded.models_path == PROJECT_ROOT / "models"
