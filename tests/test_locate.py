import numpy as np

from scripts.locate import generate_long_series


def test_generate_long_series_supports_all_synthetic_scenarios() -> None:
    scenarios = [
        ("S1", 0.0, 1.0, 0.3),
        ("S1_prime", 0.7, 1.0, 0.3),
        ("S2", 0.0, 1.0, 0.3),
        ("S3", 0.0, 1.0, 0.3),
    ]

    for noise_type, rho, sigma, cauchy_scale in scenarios:
        series, taus = generate_long_series(
            total_length=500,
            n_changes=3,
            noise_type=noise_type,
            rho=rho,
            sigma=sigma,
            cauchy_scale=cauchy_scale,
            seed=7,
        )
        assert series.shape == (500,)
        assert len(taus) == 3
        assert np.isfinite(series).all()
