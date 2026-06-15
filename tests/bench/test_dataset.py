from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cusr.bench.dataset import Dataset
from cusr.bench.transform import apply_transform


def _write_parquet(path: Path, X: np.ndarray, y: np.ndarray) -> str:
    df = pd.DataFrame({f"x{i}": X[:, i] for i in range(X.shape[1])} | {"y": y})
    df.to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parquet_roundtrip(tmp_path: Path):
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = X[:, 0] ** 2
    parquet = tmp_path / "d.parquet"
    sha = _write_parquet(parquet, X, y)
    reg = tmp_path / "registry.yaml"
    with open(reg, "w") as f:
        yaml.safe_dump(
            {
                "datasets": {
                    "toy/1": {
                        "path": "d.parquet",
                        "ground_truth": "x0**2",
                        "n_samples": 20,
                        "n_features": 1,
                        "sha256": sha,
                    }
                }
            },
            f,
        )
    ds = Dataset.from_registry("toy/1", reg)
    Xr, yr = ds.load()
    np.testing.assert_allclose(Xr, X)
    np.testing.assert_allclose(yr, y)


def test_gaussian_noise_reproducibility():
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = np.zeros(10)
    t = {"type": "gaussian_noise", "sigma": 0.1, "seed": 42}
    _, y1 = apply_transform(X, y, t)
    _, y2 = apply_transform(X, y, t)
    np.testing.assert_allclose(y1, y2)
    assert np.any(y1 != 0.0)


def test_relative_gaussian_noise_scales_with_y_std():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 1))
    # Two targets with very different magnitudes but identical shape.
    y_small = rng.normal(size=500) * 1e-6
    y_big = rng.normal(size=500) * 1e6
    t = {"type": "relative_gaussian_noise", "sigma": 0.01, "seed": 42}
    _, y_small_noisy = apply_transform(X, y_small, t)
    _, y_big_noisy = apply_transform(X, y_big, t)
    # Noise magnitudes must track their y_std (1% of std, roughly).
    noise_small = np.std(y_small_noisy - y_small)
    noise_big = np.std(y_big_noisy - y_big)
    np.testing.assert_allclose(noise_small, 0.01 * np.std(y_small), rtol=0.15)
    np.testing.assert_allclose(noise_big, 0.01 * np.std(y_big), rtol=0.15)
    # 12-orders-of-magnitude ratio preserved (± ~30% from sampling noise).
    np.testing.assert_allclose(noise_big / noise_small, 1e12, rtol=0.3)


def test_relative_gaussian_noise_constant_y_falls_back():
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = np.full(10, 5.0)  # std(y) == 0 — degenerate
    t = {"type": "relative_gaussian_noise", "sigma": 0.1, "seed": 7}
    _, y_noisy = apply_transform(X, y, t)
    # Should still be reproducible and non-zero (fallback uses sigma as absolute).
    _, y_noisy2 = apply_transform(X, y, t)
    np.testing.assert_allclose(y_noisy, y_noisy2)
    assert np.any(y_noisy != 5.0)


def test_registry_transform_via_base(tmp_path: Path):
    X = np.linspace(-1, 1, 15).reshape(-1, 1)
    y = X[:, 0] ** 3
    parquet = tmp_path / "d.parquet"
    sha = _write_parquet(parquet, X, y)
    reg = tmp_path / "registry.yaml"
    with open(reg, "w") as f:
        yaml.safe_dump(
            {
                "datasets": {
                    "toy/1": {
                        "path": "d.parquet",
                        "ground_truth": "x0**3",
                        "n_samples": 15,
                        "n_features": 1,
                        "sha256": sha,
                    },
                    "toy/1/noise": {
                        "base": "toy/1",
                        "transform": {"type": "gaussian_noise", "sigma": 0.05, "seed": 7},
                    },
                }
            },
            f,
        )
    ds = Dataset.from_registry("toy/1/noise", reg)
    _, y_noisy = ds.load()
    assert y_noisy.shape == y.shape
    assert not np.allclose(y_noisy, y)


def test_unknown_transform_raises():
    X = np.zeros((3, 1))
    y = np.zeros(3)
    with pytest.raises(ValueError):
        apply_transform(X, y, {"type": "mystery"})
