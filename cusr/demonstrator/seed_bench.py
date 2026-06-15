"""009 SR-recovery benchmark — seed set (first brick).

Self-contained: the manifest + data generator live here; only the *recovery
judge* is reused (imported) from `bench.sources.evogp`, which is already
TDD-covered in `tests/bench/test_recovery.py`. We do not duplicate or move it.

A manifest entry carries everything needed to (a) generate (X, y) from the
ground-truth expression and (b) score a candidate against it. The field that
makes 009 different from the existing NLS-optimizer bench is `n_inner_consts`
— constants nested inside a nonlinear function (a frequency in cos(w*x), a
decay rate in exp(-x/tau)). That is the axis the recovery story rests on; the
judge tolerates *outer* additive/multiplicative constants but discriminates
hard on *inner* ones.

This is the seed (2 problems, 2 families). It exists to prove the pipe; the
full problem set + manifest file format come after the pipe is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Problem:
    """One ground-truth recovery problem.

    Constants are baked into `true_expr` as numeric literals (this is a
    ground-truth catalogue, not a fit-the-skeleton task), so `true_expr`'s only
    free symbols are x0..x{n_vars-1}.
    """
    id: str
    source: str                       # nguyen | korns | srsd-feynman | strogatz | ...
    true_expr: str                    # sympy syntax, vars x0..x{n_vars-1}
    n_vars: int
    var_ranges: tuple[tuple[float, float], ...]
    n_consts: int                     # total numeric constants in true_expr
    n_inner_consts: int               # constants nested inside a nonlinear fn (the key axis)
    difficulty: str                   # easy | medium | hard
    is_control: bool                  # negative control (CO should not help)
    range_variant: str = "realistic"  # easy | realistic
    known_ceiling: bool = False       # "nobody solves this" reference (e.g. Korns f11/f12)
    notes: str = ""


SEED: list[Problem] = [
    # Negative control: integer-coefficient polynomial, no free constants.
    # If constant optimization "helps" here, that's a false signal.
    Problem(
        id="nguyen_1",
        source="nguyen",
        true_expr="x0**3 + x0**2 + x0",
        n_vars=1,
        var_ranges=((-1.0, 1.0),),
        n_consts=0,
        n_inner_consts=0,
        difficulty="easy",
        is_control=True,
        notes="Nguyen-1; classic GP target, no free constants.",
    ),
    # Inner-constant magnifier: two frequencies (9.8, 1.3) nested inside
    # cos/sin; plus an outer additive (2.0) and outer multiplicative (-2.1).
    # Korns-12 uses two active vars (original is 5-var; we model the 2 active).
    Problem(
        id="korns_12",
        source="korns",
        true_expr="2.0 - 2.1*cos(9.8*x0)*sin(1.3*x1)",
        n_vars=2,
        var_ranges=((-50.0, 50.0), (-50.0, 50.0)),
        n_consts=4,
        n_inner_consts=2,
        difficulty="hard",
        is_control=False,
        notes="Korns-12 (active vars only). Inner freqs 9.8 & 1.3; outer 2.0 add, -2.1 mul.",
    ),
]


def generate(problem: Problem, seed: int = 0, n_samples: int = 1000):
    """Materialize (X, y) for a problem by sampling its variable ranges and
    evaluating `true_expr`. Deterministic given `seed`. Raises if any y is
    non-finite (a sign the ranges are wrong for the formula's domain).
    """
    syms = [sp.Symbol(f"x{i}") for i in range(problem.n_vars)]
    expr = sp.sympify(problem.true_expr, locals={s.name: s for s in syms})
    f = sp.lambdify(syms, expr, modules="numpy")

    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, problem.n_vars), dtype=float)
    for i, (lo, hi) in enumerate(problem.var_ranges):
        X[:, i] = rng.uniform(lo, hi, size=n_samples)

    cols = [X[:, i] for i in range(problem.n_vars)]
    y = np.asarray(f(*cols), dtype=float)
    if y.ndim == 0:  # constant expression
        y = np.full(n_samples, float(y))
    if not np.all(np.isfinite(y)):
        n_bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"{problem.id}: {n_bad}/{n_samples} non-finite y — check var_ranges")
    return X, y


def score(candidate, problem: Problem, X: np.ndarray, *, outer: str = "lenient") -> bool:
    """Is `candidate` (str or sympy expr) a symbolic recovery of `problem`?

    Uses 009's calibrated judge (uniform relative inner-constant tolerance, see
    judge.py / judge_calibration.md). Defaults to `outer="lenient"`
    (competitor-comparable linear scaling). Wrong inner constants are rejected.
    """
    from . import judge

    return judge.recovered(candidate, problem.true_expr, outer=outer, X=X)


def score_columns(candidate, problem: Problem, X: np.ndarray) -> dict:
    """All three reported recovery columns (strict / lenient / srbench)."""
    from . import judge

    return judge.recovery_columns(candidate, problem.true_expr, X=X)
