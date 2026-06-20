from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import sympy as sp

from cusr.bench.canonicalize import DEFAULT as DEFAULT_CANONICALIZE

if TYPE_CHECKING:
    from cusr.bench.dataset import Dataset


_VAR_RE = re.compile(r"^x\d+$")
_CONST_RE = re.compile(r"^c\d+$")


@dataclass(frozen=True)
class Skeleton:
    expr: sp.Expr
    variables: tuple[sp.Symbol, ...]
    constants: tuple[sp.Symbol, ...]
    canonicalize_fn: Optional[Callable[[sp.Expr], str]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for v in self.variables:
            if not _VAR_RE.match(v.name):
                raise ValueError(f"variable {v.name!r} must match x\\d+")
        for c in self.constants:
            if not _CONST_RE.match(c.name):
                raise ValueError(f"constant {c.name!r} must match c\\d+")
        allowed = set(self.variables) | set(self.constants)
        free = self.expr.free_symbols
        extra = free - allowed
        if extra:
            raise ValueError(f"expr has free symbols not in vars∪consts: {extra}")

    @cached_property
    def n_vars(self) -> int:
        return len(self.variables)

    @cached_property
    def n_constants(self) -> int:
        return len(self.constants)

    @cached_property
    def n_outputs(self) -> int:
        return 1

    @cached_property
    def structural_hash(self) -> str:
        fn = self.canonicalize_fn or DEFAULT_CANONICALIZE
        return fn(self.expr)

    @cached_property
    def _is_pathological(self) -> bool:
        # A tree whose SYMBOLIC form folds to ComplexInfinity (zoo) / ±oo / nan
        # — e.g. a GP subtree c0/(x0-x0) — cannot be lambdified: sympy's
        # NumPyPrinter raises KeyError('ComplexInfinity') at BUILD time, before
        # residual's runtime 1e10 guard can ever act. Detect it once here so
        # _eval_fn/_jac_fn demote it to the existing sentinels instead of raising
        # (which, caught and re-raised in a backend's except handler, double-faults
        # and crashes the whole CO batch).
        return self.expr.has(sp.zoo, sp.oo, sp.S.NegativeInfinity, sp.nan)

    @cached_property
    def _eval_fn(self) -> Callable:
        args = (self.constants, self.variables)
        if self._is_pathological:
            return lambda c, v: np.inf  # residual's nan_to_num -> 1e10 sentinel
        return sp.lambdify(args, self.expr, modules="numpy")

    @cached_property
    def _jac_fn(self) -> Callable:
        if self.n_constants == 0:
            return lambda c, X: np.zeros((X.shape[0], 0))
        n = self.n_constants
        if self._is_pathological:  # zero Jacobian -> LM avoids the direction
            return lambda c, X: np.zeros((np.asarray(X).shape[0], n))
        jac_exprs = [sp.diff(self.expr, c) for c in self.constants]
        # Differentiation can introduce a ComplexInfinity/±oo/nan atom even when
        # self.expr was finite (so _is_pathological missed it) — same lambdify
        # KeyError. Guard the derivative exprs too: zero Jacobian, never raise.
        if any(je.has(sp.zoo, sp.oo, sp.S.NegativeInfinity, sp.nan) for je in jac_exprs):
            return lambda c, X: np.zeros((np.asarray(X).shape[0], n))
        args = (self.constants, self.variables)
        f = sp.lambdify(args, jac_exprs, modules="numpy")

        def _jac(c, X):
            cols = f(tuple(c), tuple(X[:, i] for i in range(X.shape[1])))
            n = X.shape[0]
            out = np.empty((n, len(self.constants)))
            for j, col in enumerate(cols):
                arr = np.asarray(col)
                if arr.ndim == 0:
                    out[:, j] = float(arr)
                else:
                    out[:, j] = arr
            return out

        return _jac

    def evaluate(self, constants, X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        c = tuple(np.asarray(constants, dtype=float))
        var_cols = tuple(X[:, i] for i in range(X.shape[1]))
        out = self._eval_fn(c, var_cols)
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 0:
            arr = np.full(X.shape[0], float(arr))
        return arr

    def residual(self, constants, X, y) -> np.ndarray:
        pred = self.evaluate(constants, X)
        y = np.asarray(y, dtype=float).ravel()
        r = pred - y
        return np.nan_to_num(r, nan=1e10, posinf=1e10, neginf=-1e10)

    def jacobian(self, constants, X) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        c = tuple(np.asarray(constants, dtype=float))
        J = self._jac_fn(c, X)
        # NaN guard must match residual's: zero-gradient signals LM/BFGS to
        # avoid that direction. Without this, residual=1e10 + jacobian=NaN
        # fools scipy into reporting bogus "tol converged" with no progress.
        return np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class FitRequest:
    skeleton: Skeleton
    init_constants: np.ndarray
    dataset: "Dataset"
    source: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.dataset.load()[0]

    @property
    def y(self) -> np.ndarray:
        return self.dataset.load()[1]


@dataclass
class FitRecord:
    skeleton_id: str
    skeleton_repr: str
    n_constants: int
    init_constants: list
    final_constants: list
    initial_loss: float
    final_loss: float
    n_fn_evals: int
    n_grad_evals: Optional[int]
    wall_time_s: float
    converged: bool
    stop_reason: str
    backend: str
    optimizer_kwargs: dict
    seed: int
    dataset_id: str
    source: dict
    error_msg: Optional[str] = None
    restart_idx: int = 0
    restart_strategy: str = "no_restart"

    def to_dict(self) -> dict:
        return {
            "skeleton_id": self.skeleton_id,
            "skeleton_repr": self.skeleton_repr,
            "n_constants": self.n_constants,
            "init_constants": list(self.init_constants),
            "final_constants": list(self.final_constants),
            "initial_loss": float(self.initial_loss),
            "final_loss": float(self.final_loss),
            "n_fn_evals": int(self.n_fn_evals),
            "n_grad_evals": None if self.n_grad_evals is None else int(self.n_grad_evals),
            "wall_time_s": float(self.wall_time_s),
            "converged": bool(self.converged),
            "stop_reason": self.stop_reason,
            "backend": self.backend,
            "optimizer_kwargs": self.optimizer_kwargs,
            "seed": int(self.seed),
            "dataset_id": self.dataset_id,
            "source": self.source,
            "error_msg": self.error_msg,
            "restart_idx": int(self.restart_idx),
            "restart_strategy": self.restart_strategy,
        }
