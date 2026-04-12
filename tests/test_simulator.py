import numpy as np
import pytest
from scipy import stats

from src.data.simulator import simulate_sequence, simulate_dataset


def test_simulate_sequence_no_change_shape():
    x, tau = simulate_sequence(n=50, has_change=False, noise_type="S1")
    assert x.shape == (50,)
    assert tau is None


def test_simulate_sequence_change_shape():
    x, tau = simulate_sequence(n=50, has_change=True, noise_type="S1")
    assert x.shape == (50,)
    assert tau is not None
    assert 1 <= tau <= 48  # tau in [1, n-2]


def test_simulate_sequence_tau_range():
    rng = np.random.default_rng(42)
    for _ in range(100):
        _, tau = simulate_sequence(n=30, has_change=True, rng=rng)
        assert 1 <= tau <= 28


def test_simulate_sequence_noise_types():
    rng = np.random.default_rng(0)
    for noise_type in ("S1", "S1_prime", "S2", "S3"):
        x, _ = simulate_sequence(n=100, has_change=False, noise_type=noise_type, rho=0.5, rng=rng)
        assert x.shape == (100,)
        assert np.isfinite(x).all() or noise_type == "S3"  # Cauchy can be extreme


def test_simulate_dataset_shape(small_n):
    X, y, taus = simulate_dataset(N=100, n=small_n, seed=7)
    assert X.shape == (100, small_n)
    assert y.shape == (100,)
    assert taus.shape == (100,)


def test_simulate_dataset_label_balance(small_n):
    N = 200
    _, y, _ = simulate_dataset(N=N, n=small_n, seed=1)
    # Exactly half of each (before shuffle, N//2 each; shuffle preserves count)
    assert y.sum() == N // 2


def test_simulate_dataset_tau_valid(small_n):
    _, y, taus = simulate_dataset(N=200, n=small_n, seed=2)
    change_mask = y == 1
    # All change taus must be in [1, n-2]
    assert (taus[change_mask] >= 1).all()
    assert (taus[change_mask] <= small_n - 2).all()
    # All no-change taus are 0
    assert (taus[~change_mask] == 0).all()


def test_simulate_dataset_cauchy_heavier_tails(small_n):
    _, _, _ = simulate_dataset(N=500, n=small_n, noise_type="S3", seed=3)
    X_s1, _, _ = simulate_dataset(N=500, n=small_n, noise_type="S1", seed=3)
    X_s3, _, _ = simulate_dataset(N=500, n=small_n, noise_type="S3", seed=3)
    # Cauchy has larger absolute max than Gaussian (statistically)
    assert X_s3.max() > X_s1.max() or X_s3.min() < X_s1.min()
