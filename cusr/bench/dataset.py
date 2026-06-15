from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from cusr.bench.transform import apply_transform


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Dataset:
    id: str
    path: Path
    ground_truth: Optional[str]
    n_samples: int
    n_features: int
    transform: Optional[dict] = None
    sha256: Optional[str] = None
    _cache: Optional[tuple] = field(default=None, repr=False, compare=False)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cache is not None:
            return self._cache
        if self.sha256 is not None:
            got = _sha256_file(self.path)
            if got != self.sha256:
                raise ValueError(f"sha256 mismatch for {self.path}: {got} != {self.sha256}")
        df = pd.read_parquet(self.path)
        feat_cols = [c for c in df.columns if c != "y"]
        feat_cols.sort(key=lambda c: int(c[1:]) if c.startswith("x") and c[1:].isdigit() else 10_000)
        X = df[feat_cols].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float).ravel()
        if self.transform is not None:
            X, y = apply_transform(X, y, self.transform)
        self._cache = (X, y)
        return self._cache

    @classmethod
    def from_registry(cls, dataset_id: str, registry_path: Path) -> "Dataset":
        registry_path = Path(registry_path)
        with open(registry_path) as f:
            reg = yaml.safe_load(f) or {}
        datasets = reg.get("datasets") or {}
        if dataset_id not in datasets:
            raise KeyError(f"dataset {dataset_id!r} not in {registry_path}")
        entry = datasets[dataset_id]
        base_dir = registry_path.parent.resolve()
        if "base" in entry:
            base_entry = datasets[entry["base"]]
            path = (base_dir / base_entry["path"]).resolve()
            gt = base_entry.get("ground_truth")
            n_samples = base_entry["n_samples"]
            n_features = base_entry["n_features"]
            sha = base_entry.get("sha256")
            transform = entry.get("transform")
        else:
            path = (base_dir / entry["path"]).resolve()
            gt = entry.get("ground_truth")
            n_samples = entry["n_samples"]
            n_features = entry["n_features"]
            sha = entry.get("sha256")
            transform = entry.get("transform")
        # Path-escape guard: prevent a malicious registry from reading arbitrary
        # files (`path: "../../../etc/passwd"`).
        if not path.is_relative_to(base_dir):
            raise ValueError(
                f"dataset {dataset_id!r} path {path} escapes registry root {base_dir}"
            )
        return cls(
            id=dataset_id,
            path=path,
            ground_truth=gt,
            n_samples=n_samples,
            n_features=n_features,
            transform=transform,
            sha256=sha,
        )
