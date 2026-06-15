from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import yaml

from cusr.bench.backends.scipy_common import ScipyLSBackend, ScipyMinimizeBackend
from cusr.bench.dataset import Dataset
from cusr.bench.skeleton import FitRequest, Skeleton


def _make_linear_request(tmp_path: Path) -> FitRequest:
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    skel = Skeleton(expr=c0 * x0 + c1, variables=(x0,), constants=(c0, c1))
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(30, 1))
    y = 2.5 * X[:, 0] + 1.3
    parquet = tmp_path / "lin.parquet"
    df = pd.DataFrame({"x0": X[:, 0], "y": y})
    df.to_parquet(parquet, index=False)
    sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
    reg = tmp_path / "reg.yaml"
    with open(reg, "w") as f:
        yaml.safe_dump(
            {
                "datasets": {
                    "lin/1": {
                        "path": "lin.parquet",
                        "ground_truth": "2.5*x0 + 1.3",
                        "n_samples": 30,
                        "n_features": 1,
                        "sha256": sha,
                    }
                }
            },
            f,
        )
    dataset = Dataset.from_registry("lin/1", reg)
    return FitRequest(
        skeleton=skel,
        init_constants=np.array([0.0, 0.0]),
        dataset=dataset,
        source={"origin": "test"},
    )


def test_ls_lm_converges(tmp_path: Path):
    req = _make_linear_request(tmp_path)
    backend = ScipyLSBackend(method="lm")
    rec = backend.fit(req, seed=0)
    assert rec.converged is True
    assert rec.final_loss < 1e-8
    np.testing.assert_allclose(rec.final_constants, [2.5, 1.3], atol=1e-6)


def test_ls_lm_nonconvergent_records_failure(tmp_path: Path):
    # Use a skeleton that can't match the data (constant-only vs linear data).
    c0 = sp.Symbol("c0")
    x0 = sp.Symbol("x0")
    skel = Skeleton(expr=c0 + 0 * x0, variables=(x0,), constants=(c0,))
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(30, 1))
    y = 2.5 * X[:, 0] + 1.3
    parquet = tmp_path / "lin.parquet"
    df = pd.DataFrame({"x0": X[:, 0], "y": y})
    df.to_parquet(parquet, index=False)
    sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
    reg = tmp_path / "reg.yaml"
    with open(reg, "w") as f:
        yaml.safe_dump(
            {
                "datasets": {
                    "lin/2": {
                        "path": "lin.parquet",
                        "ground_truth": "2.5*x0 + 1.3",
                        "n_samples": 30,
                        "n_features": 1,
                        "sha256": sha,
                    }
                }
            },
            f,
        )
    dataset = Dataset.from_registry("lin/2", reg)
    req = FitRequest(skeleton=skel, init_constants=np.array([0.0]), dataset=dataset, source={})
    backend = ScipyLSBackend(method="lm", default_kwargs={"max_nfev": 5})
    rec = backend.fit(req, seed=0)
    # The constant-only best fit has nonzero residual (mean-y offset). Even if
    # scipy calls it "converged" at that mean, final_loss >> 1e-8.
    assert rec.final_loss > 1e-2


def test_minimize_bfgs_converges(tmp_path: Path):
    req = _make_linear_request(tmp_path)
    backend = ScipyMinimizeBackend(method="BFGS")
    rec = backend.fit(req, seed=0)
    assert rec.final_loss < 1e-6
