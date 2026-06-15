from __future__ import annotations

import numpy as np


def apply_transform(X: np.ndarray, y: np.ndarray, transform: dict) -> tuple[np.ndarray, np.ndarray]:
    t = transform.get("type")
    if t == "gaussian_noise":
        sigma = float(transform["sigma"])
        seed = int(transform["seed"])
        rng = np.random.default_rng(seed)
        y_new = y + rng.normal(0.0, sigma, size=y.shape)
        return X, y_new
    if t == "relative_gaussian_noise":
        # SRBench-style: sigma is a fraction of std(y). Needed for Feynman
        # because y magnitudes span 30+ orders (n·ℏ vs m·g·z), making an
        # absolute sigma meaningless across formulas.
        sigma_rel = float(transform["sigma"])
        seed = int(transform["seed"])
        y_std = float(np.std(y))
        if not np.isfinite(y_std) or y_std == 0.0:
            # Degenerate target (constant y) — fall back to absolute sigma so
            # the variant stays reproducible instead of silently becoming clean.
            scale = sigma_rel
        else:
            scale = sigma_rel * y_std
        rng = np.random.default_rng(seed)
        y_new = y + rng.normal(0.0, scale, size=y.shape)
        return X, y_new
    raise ValueError(f"unknown transform type: {t!r}")
