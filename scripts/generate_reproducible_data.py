"""Generate all paper-faithful datasets with locked seeds, save to disk.

Produces for each noise scenario (S1, S1', S2, S3):
  data/paper_faithful/{scenario}_train.npz     — N=10000, seed=42, mu_scale=[0.5, 1.5]
  data/paper_faithful/{scenario}_test.npz      — N=2000,  seed=2024, mu_scale=[0.25, 1.75]

Both use snr_based_mu=True per paper §5. These files are the canonical data for
reproducing every result in the report. Any run that loads them will see
exactly the same sequences.

Usage:
    python scripts/generate_reproducible_data.py
    python scripts/generate_reproducible_data.py --verify   # assert determinism
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.simulator import simulate_dataset


SCENARIOS = {
    "s1":      dict(noise_type="S1",       rho=0.0, sigma=1.0, cauchy_scale=0.3),
    "s1prime": dict(noise_type="S1_prime", rho=0.7, sigma=1.0, cauchy_scale=0.3),
    "s2":      dict(noise_type="S2",       rho=0.0, sigma=1.0, cauchy_scale=0.3),
    "s3":      dict(noise_type="S3",       rho=0.0, sigma=1.0, cauchy_scale=0.3),
}

SPLITS = {
    "train": dict(N=10000, seed=42,   mu_scale_range=(0.5, 1.5)),
    "test":  dict(N=2000,  seed=2024, mu_scale_range=(0.25, 1.75)),
}

OUT_DIR = REPO_ROOT / "data" / "paper_faithful"


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def generate_one(scenario: str, split: str, n: int = 100) -> dict:
    cfg = SCENARIOS[scenario]
    sp = SPLITS[split]
    X, y, taus = simulate_dataset(
        N=sp["N"], n=n,
        snr_based_mu=True,
        mu_scale_range=sp["mu_scale_range"],
        seed=sp["seed"],
        **cfg,
    )
    return {"X": X, "y": y, "taus": taus, "seed": sp["seed"],
            "n": n, "scenario": scenario, "split": split,
            "hash": _hash(X)}


def save(data: dict, path: Path) -> None:
    np.savez(path, X=data["X"], y=data["y"], taus=data["taus"])
    # Record the hash in a sidecar so we can verify integrity later.
    path.with_suffix(".hash.txt").write_text(data["hash"] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Regenerate and verify saved files byte-identical")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        for split in SPLITS:
            out = OUT_DIR / f"{scenario}_{split}.npz"
            data = generate_one(scenario, split)

            if args.verify and out.exists():
                prev = np.load(out)
                same = np.array_equal(prev["X"], data["X"]) and \
                       np.array_equal(prev["y"], data["y"]) and \
                       np.array_equal(prev["taus"], data["taus"])
                mark = "✓" if same else "✗ DIFFERS"
                print(f"{mark} {out.name}  hash={data['hash']}")
            else:
                save(data, out)
                X = data["X"]
                print(f"saved {out.name}  shape={X.shape}  "
                      f"range=[{X.min():.2f},{X.max():.2f}]  "
                      f"std={X.std():.2f}  hash={data['hash']}")


if __name__ == "__main__":
    main()
