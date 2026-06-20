"""Skeleton-level problem adapter for the admit criterion.

The catalogue `Problem` record (seed_bench.Problem) stores only the BAKED
true_expr (constants substituted to literals), but the criterion needs the
SKELETON (free c-symbols) + ground-truth constant values to (a) locate inner
constants, (b) fit the outer-affine OLS at fixed references, and (c) run the
full-CO certification. This module reuses the existing skeleton sources
(load_feynman, synthetic.NGUYEN) and the authored Korns skeletons (incl. the
deliberately-dropped korns_8 absorbable anchor) and exposes them uniformly.

Ground truth is fp64 + independent of any GPU kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from . import prereg as P


@dataclass(frozen=True)
class SkelProblem:
    """A problem at skeleton resolution: skeleton with free constants + GT values."""
    id: str
    source: str
    skeleton: sp.Expr
    variables: tuple[sp.Symbol, ...]
    constants: tuple[sp.Symbol, ...]
    gt: np.ndarray                       # true constant values, aligned to `constants`
    domain: tuple[tuple[float, float], ...]

    @property
    def baked(self) -> sp.Expr:
        return self.skeleton.subs({c: float(v) for c, v in zip(self.constants, self.gt)})


def materialize(prob: SkelProblem, *, n: int = P.N_SAMPLES, seed: int = P.DATA_SEED):
    """Deterministic (X, y) by sampling `domain` and evaluating the baked skeleton.
    Raises if any y is non-finite (a sign the domain is wrong for the formula)."""
    rng = np.random.default_rng(seed)
    X = np.empty((n, len(prob.variables)))
    for i, (lo, hi) in enumerate(prob.domain):
        X[:, i] = rng.uniform(lo, hi, n)
    f = sp.lambdify(prob.variables, prob.baked, "numpy")
    cols = [X[:, i] for i in range(len(prob.variables))]
    y = np.asarray(f(*cols), dtype=float)
    if y.ndim == 0:
        y = np.full(n, float(y))
    if not np.all(np.isfinite(y)):
        n_bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"{prob.id}: {n_bad}/{n} non-finite y — check domain")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Feynman (reuse bench/sources/feynman.py)
# ─────────────────────────────────────────────────────────────────────────────

def feynman_problem(name: str) -> SkelProblem:
    from cusr.bench.sources.feynman import load_feynman
    p = load_feynman()[name]
    return SkelProblem(
        id=f"feynman_{name}",
        source="feynman",
        skeleton=p.skeleton_expr,
        variables=tuple(p.variables),
        constants=tuple(p.constants),
        gt=np.asarray(p.ground_truth_constants, dtype=float),
        domain=tuple((float(lo), float(hi)) for lo, hi in p.sampling_ranges),
    )


def feynman_inner_problems() -> list[SkelProblem]:
    """All positional-inner Feynman problems (the 34 — the §2.5 audit set)."""
    from cusr.bench.sources.feynman import load_feynman
    from cusr.demonstrator.taxonomy import count_inner_consts
    out = []
    for name, p in load_feynman().items():
        if count_inner_consts(p.skeleton_expr, p.constants) > 0:
            out.append(feynman_problem(name))
    return out


def feynman_outer_only(limit: int | None = None) -> list[SkelProblem]:
    """Outer-only Feynman (n_inner_consts == 0) — negative controls."""
    from cusr.bench.sources.feynman import load_feynman
    from cusr.demonstrator.taxonomy import count_inner_consts
    out = []
    for name, p in load_feynman().items():
        if count_inner_consts(p.skeleton_expr, p.constants) == 0:
            out.append(feynman_problem(name))
    return out[:limit] if limit is not None else out


# ─────────────────────────────────────────────────────────────────────────────
# Nguyen (reuse bench/sources/synthetic.py) — negative controls
# ─────────────────────────────────────────────────────────────────────────────

def nguyen_problem(name: str) -> SkelProblem:
    from cusr.bench.sources.synthetic import NGUYEN
    p = NGUYEN[name]
    return SkelProblem(
        id=f"nguyen_{name}",
        source="nguyen",
        skeleton=p.skeleton_expr,
        variables=tuple(p.variables),
        constants=tuple(p.constants),
        gt=np.asarray(p.ground_truth_constants, dtype=float),
        domain=tuple((float(p.sampling_range[0]), float(p.sampling_range[1]))
                     for _ in p.variables),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Korns anchors (authored; korns_8 reconstructed — it is dropped from the catalogue)
# ─────────────────────────────────────────────────────────────────────────────
_x = sp.symbols("x0 x1 x2 x3 x4")
_c = sp.symbols("c0 c1 c2 c3 c4")


def _korns(id_, skel, consts, gt, domain) -> SkelProblem:
    variables = tuple(sorted(skel.free_symbols - set(consts), key=lambda s: s.name))
    return SkelProblem(id_, "korns", skel, variables, tuple(consts),
                       np.asarray(gt, dtype=float), tuple(domain))


def korns_problems() -> dict[str, SkelProblem]:
    x0, x1, x2 = _x[0], _x[1], _x[2]
    c0, c1, c2, c3 = _c[0], _c[1], _c[2], _c[3]
    return {
        "korns_7":  _korns("korns_7", c0 * (1 - sp.exp(c1 * x0)), (c0, c1),
                           (213.80940889, -0.54723748542), ((0.1, 10.0),)),
        # Absorbable anchor — dropped from the live catalogue, reconstructed here:
        # sqrt(c2·prod) = sqrt(c2)·sqrt(prod), so c2 folds into the outer scale.
        "korns_8":  _korns("korns_8", c0 + c1 * sp.sqrt(c2 * x0 * x1 * x2),
                           (c0, c1, c2), (6.87, 11.0, 7.23), ((0.1, 10.0),) * 3),
        "korns_11": _korns("korns_11", c0 + c1 * sp.cos(c2 * x0**3), (c0, c1, c2),
                           (6.87, 11.0, 7.23), ((-5.0, 5.0),)),
        "korns_12": _korns("korns_12", c0 + c1 * sp.cos(c2 * x0) * sp.sin(c3 * x1),
                           (c0, c1, c2, c3), (2.0, -2.1, 9.8, 1.3),
                           ((-5.0, 5.0), (-5.0, 5.0))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constructed problems (experiment-2)
# ─────────────────────────────────────────────────────────────────────────────

def constructed_problem(template: str, values: tuple[float, ...], *,
                        outer: float = 1.7) -> SkelProblem:
    """Materialise one problem from a CONSTRUCTION_GRID template + inner values.
    Outer constant(s) get a fixed non-trivial nominal value (default 1.7)."""
    spec = P.CONSTRUCTION_GRID[template]
    skel = sp.sympify(spec["skeleton"])
    consts = tuple(sorted([s for s in skel.free_symbols if s.name.startswith("c")],
                          key=lambda s: s.name))
    variables = tuple(sorted([s for s in skel.free_symbols if s.name.startswith("x")],
                             key=lambda s: s.name))
    inner_names = [n.strip() for n in spec["inner"].split(",")]
    innermap = dict(zip(inner_names, values))
    gt = np.array([float(innermap[c.name]) if c.name in innermap else float(outer)
                   for c in consts])
    vstr = "_".join(f"{v:g}" for v in values)
    return SkelProblem(f"constructed_{template}_{vstr}", "constructed", skel,
                       variables, consts, gt, tuple(spec["domain"]))


def construction_candidates() -> list[SkelProblem]:
    """Every (template × inner-value) candidate from the frozen construction grid."""
    out = []
    for template, spec in P.CONSTRUCTION_GRID.items():
        if "values_2d" in spec:
            for vs in spec["values_2d"]:
                out.append(constructed_problem(template, tuple(vs)))
        else:
            for v in spec["values"]:
                out.append(constructed_problem(template, (v,)))
    return out
