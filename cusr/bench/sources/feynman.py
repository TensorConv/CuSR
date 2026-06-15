"""Feynman source for NLS-for-constants bench.

Data definitions live in ``experiments/003_nls_bench/data/feynman/formulas.yaml``,
generated from the FSRD CSV (``scripts/build_feynman_formulas.py``) with:

- **R2 hard filter** — 16-op alphabet (drops arcsin/arccos/arctan)
- **R4' hard filter** — K ≤ 16 after parameterization
- **Scheme C** — literal→c + leading-c=1 fallback

Applied at materialize time → ~95 problems in the parquet set.

``FeynmanSource`` applies a *runtime* filter on top, so different experiments
can pick subsets without regenerating parquets:

- ``max_vars`` — reject skeletons with more than this many variables. Default
  ``None`` (no cap) — LM cost is driven by K, not n_vars, so there's no bench
  reason to gate on dimensionality. Reports should segment statistics by
  n_vars rather than pre-filter.
- ``max_constants`` — cap K. ``None`` = no cap (R4' already enforced at gen time).
- ``exclude_ids`` — explicit drop list (e.g. I.12.1 is a trivial pure product
  with GT c0=1, linear fit — skip by default).

Noise variants use ``relative_gaussian_noise`` (sigma = fraction of std(y));
Feynman y magnitudes span orders of magnitude across formulas so absolute
sigma would be meaningless.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import sympy as sp
import yaml

from cusr.bench.dataset import Dataset
from cusr.bench.skeleton import FitRequest, Skeleton
from cusr.bench.sources.synthetic import _update_registry


FORMULAS_YAML_DEFAULT = Path(__file__).resolve().parents[3] / "data/feynman/formulas.yaml"

# Gate subset (Phase 0 go/no-go): max_vars=4 filters the tractable subset.
# Additional exclusions are problems that offer no learning signal — pure
# products with no literal constants after parameterization reduce to
# c0·Π(x_i) with GT c0=1, which is a trivial linear fit.
GATE_EXCLUDE_IDS = ("I.12.1",)  # μ·Nn — trivial; Gate skips, full run keeps.


@dataclass(frozen=True)
class _FeynmanProblem:
    name: str
    skeleton_expr: sp.Expr
    variables: tuple[sp.Symbol, ...]
    constants: tuple[sp.Symbol, ...]
    ground_truth_constants: np.ndarray
    sampling_ranges: tuple[tuple[float, float], ...]
    physical_vars: str
    formula_original: str
    n_samples: int = 1000

    @property
    def dataset_id(self) -> str:
        return f"feynman/{self.name}"

    @property
    def ground_truth_expr(self) -> sp.Expr:
        subs = {c: float(v) for c, v in zip(self.constants, self.ground_truth_constants)}
        return self.skeleton_expr.subs(subs)


@cache
def _load_formulas(yaml_path: Path) -> dict[str, _FeynmanProblem]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    probs: dict[str, _FeynmanProblem] = {}
    for entry in data["formulas"]:
        var_syms = tuple(sp.Symbol(v) for v in entry["variables"])
        const_syms = tuple(sp.Symbol(c) for c in entry["constants"])
        locals_ = {str(s): s for s in (*var_syms, *const_syms)}
        skel = sp.sympify(entry["skeleton_expr"], locals=locals_)
        probs[entry["id"]] = _FeynmanProblem(
            name=entry["id"],
            skeleton_expr=skel,
            variables=var_syms,
            constants=const_syms,
            ground_truth_constants=np.array(entry["ground_truth_constants"]),
            sampling_ranges=tuple((float(lo), float(hi)) for lo, hi in entry["sampling_ranges"]),
            physical_vars=entry["physical_vars"],
            formula_original=entry["formula_original"],
            n_samples=int(entry["n_samples"]),
        )
    return probs


def load_feynman(yaml_path: Optional[Path] = None) -> dict[str, _FeynmanProblem]:
    """Load the Feynman problem catalogue (cached)."""
    return _load_formulas(Path(yaml_path or FORMULAS_YAML_DEFAULT))


def _materialize_parquet(problem: _FeynmanProblem, data_root: Path) -> tuple[Path, str]:
    out = data_root / "feynman" / f"{problem.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out, hashlib.sha256(out.read_bytes()).hexdigest()
    seed = int(hashlib.sha256(problem.dataset_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed=seed)
    n = problem.n_samples
    X = np.empty((n, len(problem.variables)))
    for i, (lo, hi) in enumerate(problem.sampling_ranges):
        X[:, i] = rng.uniform(lo, hi, size=n)
    skel = Skeleton(
        expr=problem.skeleton_expr,
        variables=problem.variables,
        constants=problem.constants,
    )
    y = skel.evaluate(problem.ground_truth_constants, X)
    if not np.all(np.isfinite(y)):
        n_bad = int((~np.isfinite(y)).sum())
        raise ValueError(
            f"{problem.dataset_id}: {n_bad}/{n} non-finite y values — check sampling ranges"
        )
    df = pd.DataFrame({f"x{i}": X[:, i] for i in range(X.shape[1])} | {"y": y})
    df.to_parquet(out, index=False)
    return out, hashlib.sha256(out.read_bytes()).hexdigest()


def select_problems(
    max_vars: Optional[int] = None,
    max_constants: Optional[int] = None,
    exclude_ids: Iterable[str] = (),
    include_ids: Optional[Iterable[str]] = None,
    yaml_path: Optional[Path] = None,
) -> tuple[str, ...]:
    """Apply runtime filter to the full catalogue, return ordered IDs."""
    catalogue = load_feynman(yaml_path)
    excluded = set(exclude_ids)
    if include_ids is not None:
        include_set = set(include_ids)
        return tuple(k for k in catalogue if k in include_set and k not in excluded)
    out: list[str] = []
    for name, prob in catalogue.items():
        if name in excluded:
            continue
        if max_vars is not None and len(prob.variables) > max_vars:
            continue
        if max_constants is not None and len(prob.constants) > max_constants:
            continue
        out.append(name)
    return tuple(out)


@dataclass
class FeynmanSource:
    data_root: Path
    registry_path: Path
    max_vars: Optional[int] = None
    max_constants: Optional[int] = None
    exclude_ids: tuple[str, ...] = GATE_EXCLUDE_IDS
    include_ids: Optional[tuple[str, ...]] = None
    problems: Optional[tuple[str, ...]] = None  # explicit override; skips filter
    yaml_path: Optional[Path] = None
    n_perturbed_per_problem: int = 3
    init_noise_sigma: float = 0.3
    seed: int = 0
    name: str = "feynman"
    _ensured: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.problems is None:
            self.problems = select_problems(
                max_vars=self.max_vars,
                max_constants=self.max_constants,
                exclude_ids=self.exclude_ids,
                include_ids=self.include_ids,
                yaml_path=self.yaml_path,
            )

    def _catalogue(self) -> dict[str, _FeynmanProblem]:
        return load_feynman(self.yaml_path)

    def _ensure(self) -> None:
        if self._ensured:
            return
        catalogue = self._catalogue()
        entries: dict[str, dict] = {}
        for p_name in self.problems:
            prob = catalogue[p_name]
            path, sha = _materialize_parquet(prob, Path(self.data_root))
            rel = path.relative_to(Path(self.data_root))
            entries[prob.dataset_id] = {
                "path": str(rel),
                "ground_truth": str(prob.ground_truth_expr),
                "n_samples": int(prob.n_samples),
                "n_features": int(len(prob.variables)),
                "sha256": sha,
                "physical_vars": prob.physical_vars,
            }
        _update_registry(Path(self.registry_path), entries)
        self._ensured = True

    def iter_requests(self) -> Iterable[FitRequest]:
        self._ensure()
        catalogue = self._catalogue()
        rng = np.random.default_rng(self.seed)
        for p_name in self.problems:
            prob = catalogue[p_name]
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
                        "origin": "feynman",
                        "problem": prob.dataset_id,
                        "perturbation_idx": k,
                        "ground_truth_constants": [float(v) for v in gt],
                    },
                )


