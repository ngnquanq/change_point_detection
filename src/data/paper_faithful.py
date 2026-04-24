from __future__ import annotations

from pathlib import Path

import numpy as np


SCENARIO_TO_STEM = {
    "S1": "s1",
    "S1_prime": "s1prime",
    "S2": "s2",
    "S3": "s3",
}


def scenario_stem(noise_type: str) -> str:
    try:
        return SCENARIO_TO_STEM[noise_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported canonical scenario: {noise_type!r}") from exc


def candidate_split_paths(data_dir: str | Path, noise_type: str, split: str) -> list[Path]:
    stem = scenario_stem(noise_type)
    base = Path(data_dir)
    filename = f"{stem}_{split}.npz"
    return [
        base / filename,
        base / "paper_faithful" / filename,
    ]


def resolve_split_path(data_dir: str | Path, noise_type: str, split: str) -> Path | None:
    for path in candidate_split_paths(data_dir, noise_type, split):
        if path.exists():
            return path
    return None


def load_npz_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    with np.load(path) as data:
        required = {"X", "y", "taus"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
        taus = np.asarray(data["taus"])

    if X.ndim != 2:
        raise ValueError(f"{path}: X must be rank-2, got shape {X.shape}")
    if y.ndim != 1 or taus.ndim != 1:
        raise ValueError(f"{path}: y and taus must be rank-1")
    if len(X) != len(y) or len(X) != len(taus):
        raise ValueError(f"{path}: inconsistent sample counts")
    return X, y, taus


def maybe_load_split(
    data_dir: str | Path,
    noise_type: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path] | None:
    path = resolve_split_path(data_dir, noise_type, split)
    if path is None:
        return None
    X, y, taus = load_npz_dataset(path)
    return X, y, taus, path
