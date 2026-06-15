from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import sympy as sp
import yaml

from cusr.bench.dataset import Dataset
from cusr.bench.skeleton import FitRequest, Skeleton


@dataclass(frozen=True)
class _NguyenProblem:
    name: str
    skeleton_expr: sp.Expr
    variables: tuple[sp.Symbol, ...]
    constants: tuple[sp.Symbol, ...]
    ground_truth_constants: np.ndarray
    sampling_range: tuple[float, float]
    n_samples: int

    @property
    def dataset_id(self) -> str:
        return f"nguyen/{self.name}"

    @property
    def ground_truth_expr(self) -> sp.Expr:
        subs = {c: float(v) for c, v in zip(self.constants, self.ground_truth_constants)}
        return sp.simplify(self.skeleton_expr.subs(subs))


def _cs(n: int) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(f"c{i}") for i in range(n))


def _build_nguyen_problems() -> dict[str, _NguyenProblem]:
    x0 = sp.Symbol("x0")
    x1 = sp.Symbol("x1")
    probs: dict[str, _NguyenProblem] = {}

    c = _cs(3)
    probs["1"] = _NguyenProblem(
        name="1",
        skeleton_expr=c[0] * x0**3 + c[1] * x0**2 + c[2] * x0,
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0, 1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=20,
    )

    c = _cs(4)
    probs["2"] = _NguyenProblem(
        name="2",
        skeleton_expr=c[0] * x0**4 + c[1] * x0**3 + c[2] * x0**2 + c[3] * x0,
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0, 1.0, 1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=20,
    )

    c = _cs(5)
    probs["3"] = _NguyenProblem(
        name="3",
        skeleton_expr=c[0] * x0**5 + c[1] * x0**4 + c[2] * x0**3 + c[3] * x0**2 + c[4] * x0,
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=20,
    )

    c = _cs(6)
    probs["4"] = _NguyenProblem(
        name="4",
        skeleton_expr=(
            c[0] * x0**6 + c[1] * x0**5 + c[2] * x0**4
            + c[3] * x0**3 + c[4] * x0**2 + c[5] * x0
        ),
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0] * 6),
        sampling_range=(-1.0, 1.0),
        n_samples=20,
    )

    # Nguyen 5: sin(x²) * cos(x) − 1
    c = _cs(4)
    probs["5"] = _NguyenProblem(
        name="5",
        skeleton_expr=c[0] * sp.sin(c[1] * x0**2) * sp.cos(c[2] * x0) + c[3],
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0, 1.0, -1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=20,
    )

    # Nguyen 6: sin(x) + sin(x + x²)
    c = _cs(3)
    probs["6"] = _NguyenProblem(
        name="6",
        skeleton_expr=c[0] * sp.sin(x0) + c[1] * sp.sin(x0 + c[2] * x0**2),
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0, 1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=20,
    )

    # Nguyen 7: log(x+1) + log(x²+1)
    c = _cs(4)
    probs["7"] = _NguyenProblem(
        name="7",
        skeleton_expr=c[0] * sp.log(x0 + c[1]) + c[2] * sp.log(x0**2 + c[3]),
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0, 1.0, 1.0]),
        sampling_range=(0.0, 2.0),
        n_samples=20,
    )

    # Nguyen 8: sqrt(x)
    c = _cs(1)
    probs["8"] = _NguyenProblem(
        name="8",
        skeleton_expr=c[0] * sp.sqrt(x0),
        variables=(x0,),
        constants=c,
        ground_truth_constants=np.array([1.0]),
        sampling_range=(0.0, 4.0),
        n_samples=20,
    )

    # Nguyen 9: sin(x) + sin(y²), two vars
    c = _cs(2)
    probs["9"] = _NguyenProblem(
        name="9",
        skeleton_expr=c[0] * sp.sin(x0) + c[1] * sp.sin(x1**2),
        variables=(x0, x1),
        constants=c,
        ground_truth_constants=np.array([1.0, 1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=100,
    )

    # Nguyen 10: 2 * sin(x) * cos(y), two vars
    c = _cs(1)
    probs["10"] = _NguyenProblem(
        name="10",
        skeleton_expr=c[0] * sp.sin(x0) * sp.cos(x1),
        variables=(x0, x1),
        constants=c,
        ground_truth_constants=np.array([2.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=100,
    )

    # Nguyen 11: x^y, two vars, range [0,1]. GT c0=1 matches original literal x^y.
    # At x0=0, np.power(0, y)=0 for y>0 and =1 for y=0; no NaN in sampling range.
    c = _cs(1)
    probs["11"] = _NguyenProblem(
        name="11",
        skeleton_expr=c[0] * x0**x1,
        variables=(x0, x1),
        constants=c,
        ground_truth_constants=np.array([1.0]),
        sampling_range=(0.0, 1.0),
        n_samples=100,
    )

    # Nguyen 12: x⁴ − x³ + y²/2 − y, two vars
    c = _cs(4)
    probs["12"] = _NguyenProblem(
        name="12",
        skeleton_expr=c[0] * x0**4 + c[1] * x0**3 + c[2] * x1**2 + c[3] * x1,
        variables=(x0, x1),
        constants=c,
        ground_truth_constants=np.array([1.0, -1.0, 0.5, -1.0]),
        sampling_range=(-1.0, 1.0),
        n_samples=100,
    )

    return probs


NGUYEN = _build_nguyen_problems()


def _materialize_parquet(problem: _NguyenProblem, data_root: Path) -> tuple[Path, str]:
    out = data_root / "nguyen" / f"{problem.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=int(hashlib.sha256(problem.dataset_id.encode()).hexdigest()[:8], 16))
    lo, hi = problem.sampling_range
    X = rng.uniform(lo, hi, size=(problem.n_samples, len(problem.variables)))
    skel = Skeleton(
        expr=problem.skeleton_expr,
        variables=problem.variables,
        constants=problem.constants,
    )
    y = skel.evaluate(problem.ground_truth_constants, X)
    df = pd.DataFrame(
        {f"x{i}": X[:, i] for i in range(X.shape[1])} | {"y": y}
    )
    if out.exists():
        # If already written, reuse (idempotent).
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        return out, sha
    df.to_parquet(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return out, sha


def _sigma_label(sigma: float) -> str:
    """Variant label for a sigma value.

    Integer-percent sigmas (Nguyen: 0.01 → ``01pct``, 0.05 → ``05pct``) keep
    their existing format. Sub-percent sigmas (Feynman: 0.001 → ``0p1pct``)
    use ``p`` in place of the decimal point to stay filename-safe.
    """
    pct = sigma * 100
    if pct >= 1.0 and pct == int(pct):
        return f"{int(pct):02d}pct"
    return f"{pct:g}pct".replace(".", "p")


def register_noisy_variants(
    registry_path: Path,
    problems: Iterable[str],
    sigmas: Iterable[float] = (0.01, 0.05),
    seed: int = 42,
    base_prefix: str = "nguyen",
    transform_type: str = "gaussian_noise",
) -> list[str]:
    """Register noisy ``base + transform`` derived entries for a set of problems.

    Defaults target Nguyen (absolute gaussian noise). Feynman uses
    ``base_prefix="feynman"`` + ``transform_type="relative_gaussian_noise"``
    because y magnitudes span orders across formulas.
    """
    entries: dict[str, dict] = {}
    added: list[str] = []
    for p_name in problems:
        base_id = f"{base_prefix}/{p_name}"
        for sigma in sigmas:
            variant_id = f"{base_id}-noisy-{_sigma_label(sigma)}"
            entries[variant_id] = {
                "base": base_id,
                "transform": {
                    "type": transform_type,
                    "sigma": float(sigma),
                    "seed": int(seed),
                },
            }
            added.append(variant_id)
    _update_registry(Path(registry_path), entries)
    return added


def _update_registry(registry_path: Path, entries: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        with open(registry_path) as f:
            reg = yaml.safe_load(f) or {}
    else:
        reg = {}
    datasets = reg.get("datasets") or {}
    if not isinstance(datasets, dict):
        datasets = {}
    datasets.update(entries)
    reg["datasets"] = datasets
    with open(registry_path, "w") as f:
        yaml.safe_dump(reg, f, sort_keys=True)


@dataclass
class SyntheticSource:
    data_root: Path
    registry_path: Path
    problems: tuple[str, ...] = ("1", "2", "3")
    n_perturbed_per_problem: int = 3
    init_noise_sigma: float = 0.3
    seed: int = 0
    name: str = "synthetic"
    _ensured: bool = field(default=False, init=False, repr=False)

    def _ensure(self) -> None:
        if self._ensured:
            return
        entries: dict[str, dict] = {}
        for p_name in self.problems:
            prob = NGUYEN[p_name]
            path, sha = _materialize_parquet(prob, Path(self.data_root))
            rel = path.relative_to(Path(self.data_root))
            entries[prob.dataset_id] = {
                "path": str(rel),
                "ground_truth": str(prob.ground_truth_expr),
                "n_samples": int(prob.n_samples),
                "n_features": int(len(prob.variables)),
                "sha256": sha,
            }
        _update_registry(Path(self.registry_path), entries)
        self._ensured = True

    def iter_requests(self) -> Iterable[FitRequest]:
        self._ensure()
        rng = np.random.default_rng(self.seed)
        for p_name in self.problems:
            prob = NGUYEN[p_name]
            dataset = Dataset.from_registry(prob.dataset_id, Path(self.registry_path))
            skel = Skeleton(
                expr=prob.skeleton_expr,
                variables=prob.variables,
                constants=prob.constants,
            )
            gt = prob.ground_truth_constants
            for k in range(self.n_perturbed_per_problem):
                noise = rng.normal(0.0, self.init_noise_sigma, size=gt.shape)
                init = gt + noise
                yield FitRequest(
                    skeleton=skel,
                    init_constants=init,
                    dataset=dataset,
                    source={
                        "origin": "synthetic",
                        "problem": prob.dataset_id,
                        "perturbation_idx": k,
                        "ground_truth_constants": [float(v) for v in gt],
                    },
                )
