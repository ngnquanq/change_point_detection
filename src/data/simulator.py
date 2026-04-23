from __future__ import annotations

import numpy as np


def _generate_gaussian_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, sigma, size=n)


def _generate_ar1_noise(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1): x_t = rho * x_{t-1} + eps_t, eps ~ N(0, 1 - rho^2).

    Variance of the stationary process is 1 regardless of rho.
    """
    sigma_eps = np.sqrt(max(1.0 - rho ** 2, 1e-8))
    eps = rng.normal(0.0, sigma_eps, size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def _generate_cauchy_noise(n: int, rng: np.random.Generator, scale: float = 0.3) -> np.ndarray:
    """Cauchy noise with given scale. Paper S3 uses Cauchy(0, 0.3)."""
    u = rng.uniform(0.0, 1.0, size=n)
    return scale * np.tan(np.pi * (u - 0.5))


def simulate_sequence(
    n: int,
    has_change: bool,
    noise_type: str = "S1",
    rho: float = 0.0,
    mu_range: tuple[float, float] = (-2.0, 2.0),
    sigma: float = 1.0,
    cauchy_scale: float = 0.3,
    snr_based_mu: bool = False,
    mu_scale_range: tuple[float, float] = (0.5, 1.5),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int | None]:
    """Generate a single time series of length n.

    Args:
        n: sequence length
        has_change: whether the sequence contains a change point
        noise_type: "S1" (iid Gaussian), "S1_prime"/"S2" (AR(1)), "S3" (Cauchy)
        rho: AR(1) coefficient (used for S1_prime/S2)
        mu_range: (low, high) for drawing mu_L and mu_R (when snr_based_mu=False)
        sigma: noise std for Gaussian cases
        cauchy_scale: scale parameter for Cauchy noise (paper uses 0.3)
        snr_based_mu: if True, use paper's SNR-based formula for mu_R (Section 5)
        rng: numpy random generator; created if None

    Returns:
        x: np.ndarray shape (n,)
        tau: change-point index (1-indexed, in [1, n-2]) or None if no change
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw noise
    if noise_type == "S1":
        noise = _generate_gaussian_noise(n, sigma, rng)
    elif noise_type in ("S1_prime", "S2"):
        noise = _generate_ar1_noise(n, rho, rng)
    elif noise_type == "S3":
        noise = _generate_cauchy_noise(n, rng, scale=cauchy_scale)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    mu_low, mu_high = mu_range

    if not has_change:
        mu = rng.uniform(mu_low, mu_high)
        x = mu + noise
        return x, None
    else:
        # tau is 1-indexed; change occurs after position tau (0-indexed: tau)
        # tau in {1, ..., n-2} so both segments have length >= 1
        tau = int(rng.integers(1, n - 1))  # [1, n-2] inclusive

        mu_l = rng.uniform(mu_low, mu_high)

        if snr_based_mu:
            # Paper Section 5: b = sqrt(8n*log(20n) / (tau*(n-tau)))
            # Training range: mu_R | tau ~ Unif([-1.5b, -0.5b] ∪ [0.5b, 1.5b])
            # Test range:     mu_R | tau ~ Unif([-1.75b, -0.25b] ∪ [0.25b, 1.75b])
            # Controlled by mu_scale_range = (s_lo, s_hi); default is training range.
            b = np.sqrt(8 * n * np.log(20 * n) / (tau * (n - tau)))
            s_lo, s_hi = mu_scale_range
            if rng.random() < 0.5:
                mu_r = mu_l + rng.uniform(s_lo * b, s_hi * b)
            else:
                mu_r = mu_l + rng.uniform(-s_hi * b, -s_lo * b)
        else:
            # Simple: draw mu_r from uniform range
            mu_r = rng.uniform(mu_low, mu_high)

        x = np.empty(n)
        x[:tau] = mu_l + noise[:tau]
        x[tau:] = mu_r + noise[tau:]
        return x, tau


def simulate_dataset(
    N: int,
    n: int,
    noise_type: str = "S1",
    rho: float = 0.0,
    mu_range: tuple[float, float] = (-2.0, 2.0),
    sigma: float = 1.0,
    cauchy_scale: float = 0.3,
    snr_based_mu: bool = False,
    mu_scale_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a balanced dataset: N/2 with change, N/2 without.

    Returns:
        X: np.ndarray shape (N, n)
        y: np.ndarray shape (N,) dtype int8  — 1=change, 0=no change
        taus: np.ndarray shape (N,) dtype int  — change locations (0 if y=0)
    """
    rng = np.random.default_rng(seed)
    half = N // 2

    X = np.empty((N, n), dtype=np.float32)
    y = np.zeros(N, dtype=np.int8)
    taus = np.zeros(N, dtype=np.int64)

    # First half: sequences with change
    for i in range(half):
        x, tau = simulate_sequence(
            n,
            has_change=True,
            noise_type=noise_type,
            rho=rho,
            mu_range=mu_range,
            sigma=sigma,
            cauchy_scale=cauchy_scale,
            snr_based_mu=snr_based_mu,
            mu_scale_range=mu_scale_range,
            rng=rng,
        )
        X[i] = x
        y[i] = 1
        taus[i] = tau

    # Second half: sequences without change
    for i in range(half, N):
        x, _ = simulate_sequence(
            n,
            has_change=False,
            noise_type=noise_type,
            rho=rho,
            mu_range=mu_range,
            sigma=sigma,
            cauchy_scale=cauchy_scale,
            snr_based_mu=snr_based_mu,
            mu_scale_range=mu_scale_range,
            rng=rng,
        )
        X[i] = x
        y[i] = 0
        taus[i] = 0

    # Shuffle
    perm = rng.permutation(N)
    return X[perm], y[perm], taus[perm]
