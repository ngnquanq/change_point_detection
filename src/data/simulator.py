from __future__ import annotations

import numpy as np


def _generate_gaussian_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, sigma, size=n)


def _generate_ar1_noise(
    n: int,
    rho: float | np.ndarray,
    innovation_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) noise matching paper Section 5.

    Paper formulation (line 440):
        ε_t = ξ_1               for t = 1
        ε_t = ρ_t · ε_{t-1} + ξ_t   for t ≥ 2

    Innovation ξ_t ~ N(0, innovation_std²).
    rho can be a scalar (S1') or array of length n (S2 time-varying).
    """
    xi = rng.normal(0.0, innovation_std, size=n)
    eps = np.empty(n)
    eps[0] = xi[0]

    if np.isscalar(rho):
        for t in range(1, n):
            eps[t] = rho * eps[t - 1] + xi[t]
    else:
        # Time-varying rho (S2)
        for t in range(1, n):
            eps[t] = rho[t] * eps[t - 1] + xi[t]
    return eps


def _generate_cauchy_noise(n: int, rng: np.random.Generator, scale: float = 0.3) -> np.ndarray:
    """Cauchy noise with given scale. Paper S3 uses Cauchy(0, 0.3)."""
    u = rng.uniform(0.0, 1.0, size=n)
    return scale * np.tan(np.pi * (u - 0.5))


def simulate_sequence(
    n: int,
    has_change: bool,
    noise_type: str = "S1",
    rho: float = 0.0,
    sigma: float = 1.0,
    cauchy_scale: float = 0.3,
    snr_based_mu: bool = False,
    mu_scale_range: tuple[float, float] = (0.5, 1.5),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int | None]:
    """Generate a single time series of length n following paper Section 5.

    Paper data generation (line 428-451):
        - τ ~ Unif{2, ..., n-2}
        - μ_L = 0
        - μ_R|τ ~ Unif([-1.5b, -0.5b] ∪ [0.5b, 1.5b])
          where b = sqrt(8n·log(20n) / (τ(n-τ)))
        - No-change: μ_R = μ_L = 0
        - Noise scenarios S1, S1', S2, S3

    Args:
        n: sequence length
        has_change: whether the sequence contains a change point
        noise_type: "S1" (iid Gaussian), "S1_prime" (AR(1) fixed rho),
                    "S2" (AR(1) time-varying rho), "S3" (Cauchy)
        rho: AR(1) coefficient (used for S1_prime only; S2 draws rho_t per step)
        sigma: noise std for Gaussian S1 case
        cauchy_scale: scale parameter for Cauchy noise (paper uses 0.3)
        snr_based_mu: if True, use paper's SNR-based formula for mu_R (Section 5)
        rng: numpy random generator; created if None

    Returns:
        x: np.ndarray shape (n,)
        tau: change-point index (1-indexed, in [2, n-2]) or None if no change
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw noise according to scenario
    if noise_type == "S1":
        # S1: ξ_t ~ N(0, 1), no autocorrelation
        noise = _generate_gaussian_noise(n, sigma, rng)
    elif noise_type == "S1_prime":
        # S1': ρ_t = rho (fixed), ξ_t ~ N(0, 1)
        noise = _generate_ar1_noise(n, rho, innovation_std=1.0, rng=rng)
    elif noise_type == "S2":
        # S2: ρ_t ~ Unif([0, 1]) per time step, ξ_t ~ N(0, sqrt(2))
        # Paper line 445: ρ_t ~ Unif([0,1]), ξ_t ~ N(0, 2) means variance=2 → std=sqrt(2)
        rho_t = rng.uniform(0.0, 1.0, size=n)
        noise = _generate_ar1_noise(n, rho_t, innovation_std=np.sqrt(2.0), rng=rng)
    elif noise_type == "S3":
        # S3: ρ_t = 0, ξ_t ~ Cauchy(0, 0.3)
        noise = _generate_cauchy_noise(n, rng, scale=cauchy_scale)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    # Paper Section 5: μ_L = 0 (fixed)
    mu_l = 0.0

    if not has_change:
        # No change: μ_R = μ_L = 0
        x = mu_l + noise
        return x, None
    else:
        # Paper: τ ~ Unif{2, ..., n-2} (line 428)
        tau = int(rng.integers(2, n - 1))  # [2, n-2] inclusive

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
            # Simple fallback: draw mu_r != 0
            # Use SNR-based formula as default since paper always uses it
            b = np.sqrt(8 * n * np.log(20 * n) / (tau * (n - tau)))
            if rng.random() < 0.5:
                mu_r = rng.uniform(0.5 * b, 1.5 * b)
            else:
                mu_r = rng.uniform(-1.5 * b, -0.5 * b)

        x = np.empty(n)
        x[:tau] = mu_l + noise[:tau]
        x[tau:] = mu_r + noise[tau:]
        return x, tau


def simulate_dataset(
    N: int,
    n: int,
    noise_type: str = "S1",
    rho: float = 0.0,
    sigma: float = 1.0,
    cauchy_scale: float = 0.3,
    snr_based_mu: bool = False,
    mu_scale_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 42,
    **kwargs,
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
