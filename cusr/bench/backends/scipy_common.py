from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sympy as sp
from scipy.optimize import least_squares, minimize

from cusr.bench.skeleton import FitRecord, FitRequest


def _loss_from_residuals(r: np.ndarray) -> float:
    """Mean squared residual — dataset-size invariant."""
    return float(np.mean(r * r))


def _initial_loss(request: FitRequest) -> float:
    r = request.skeleton.residual(request.init_constants, request.X, request.y)
    return _loss_from_residuals(r)


_LS_STATUS_MAP = {
    -1: "error",
    0: "maxiter",
    1: "tol",
    2: "tol",
    3: "tol",
    4: "tol",
}

# When Skeleton.residual's NaN guard fires, each residual becomes ±1e10, so
# mean(r²) ≈ 1e20. A "descent" that stays within the sentinel regime hasn't
# actually escaped the NaN region — treat it as no progress.
_NAN_GUARD_LOSS_CEILING = 1e18


@dataclass
class ScipyLSBackend:
    method: str = "lm"
    default_kwargs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"scipy_ls_{self.method}"

    def fit(self, request: FitRequest, seed: int, **kwargs) -> FitRecord:
        skel = request.skeleton
        X, y = request.X, request.y
        init = np.asarray(request.init_constants, dtype=float)

        opt_kwargs = {**self.default_kwargs, **kwargs}
        if self.method == "lm" and "bounds" in opt_kwargs:
            opt_kwargs.pop("bounds")

        def _fun(c):
            return skel.residual(c, X, y)

        def _jac(c):
            return skel.jacobian(c, X)

        jac_arg = _jac if skel.n_constants > 0 else "2-point"
        initial_loss = _initial_loss(request)
        t0 = time.perf_counter()
        err_msg: Optional[str] = None
        try:
            res = least_squares(_fun, init, jac=jac_arg, method=self.method, **opt_kwargs)
            wall = time.perf_counter() - t0
            final = np.asarray(res.x, dtype=float)
            final_loss = _loss_from_residuals(res.fun)
            # Require both scipy success and actual loss decrease; avoids the
            # NaN-Jacobian "pretend we fit" trap where status=1 at step 0.
            # The < NaN ceiling check guards against "descent" that never
            # escapes the residual-sentinel regime.
            progressed = (
                np.isfinite(final_loss)
                and final_loss < initial_loss
                and final_loss < _NAN_GUARD_LOSS_CEILING
            )
            converged = bool(int(res.status) > 0 and progressed)
            if not np.isfinite(final_loss):
                stop_reason = "nan"
            elif not progressed:
                stop_reason = "no_progress"
            else:
                stop_reason = _LS_STATUS_MAP.get(int(res.status), "unknown")
            n_fn = int(res.nfev)
            n_jac = int(getattr(res, "njev", 0)) or None
        except Exception as e:
            wall = time.perf_counter() - t0
            final = init.copy()
            final_loss = float("inf")
            converged = False
            stop_reason = "error"
            n_fn = 0
            n_jac = None
            err_msg = f"{type(e).__name__}: {e}"

        return FitRecord(
            skeleton_id=skel.structural_hash,
            skeleton_repr=sp.srepr(skel.expr),
            n_constants=skel.n_constants,
            init_constants=list(init),
            final_constants=list(final),
            initial_loss=initial_loss,
            final_loss=final_loss,
            n_fn_evals=n_fn,
            n_grad_evals=n_jac,
            wall_time_s=wall,
            converged=converged,
            stop_reason=stop_reason,
            backend=self.name,
            optimizer_kwargs={"method": self.method, **opt_kwargs},
            seed=seed,
            dataset_id=request.dataset.id,
            source=dict(request.source),
            error_msg=err_msg,
        )


@dataclass
class ScipyMinimizeBackend:
    method: str = "BFGS"
    default_kwargs: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"scipy_minimize_{self.method.lower()}"

    def fit(self, request: FitRequest, seed: int, **kwargs) -> FitRecord:
        skel = request.skeleton
        X, y = request.X, request.y
        init = np.asarray(request.init_constants, dtype=float)
        opt_kwargs = {**self.default_kwargs, **kwargs}

        n_samples = int(np.asarray(y).size)
        inv_n = 1.0 / max(n_samples, 1)

        def _loss(c):
            r = skel.residual(c, X, y)
            return float(np.mean(r * r))

        def _grad(c):
            # ∂/∂c_k of mean(r²) = (2/n) * J^T @ r
            r = skel.residual(c, X, y)
            J = skel.jacobian(c, X)
            return (2.0 * inv_n) * (J.T @ r)

        initial_loss = _initial_loss(request)
        t0 = time.perf_counter()
        err_msg: Optional[str] = None
        use_jac = skel.n_constants > 0 and self.method in {
            "BFGS", "L-BFGS-B", "Newton-CG", "CG", "TNC", "SLSQP", "trust-ncg", "trust-krylov", "trust-exact"
        }
        try:
            res = minimize(
                _loss,
                init,
                jac=_grad if use_jac else None,
                method=self.method,
                **opt_kwargs,
            )
            wall = time.perf_counter() - t0
            final = np.asarray(res.x, dtype=float)
            final_loss = float(res.fun)
            progressed = (
                np.isfinite(final_loss)
                and final_loss < initial_loss
                and final_loss < _NAN_GUARD_LOSS_CEILING
            )
            converged = bool(bool(res.success) and progressed)
            if not np.isfinite(final_loss):
                stop_reason = "nan"
            elif not progressed:
                stop_reason = "no_progress"
            elif res.success:
                stop_reason = "tol"
            elif "maximum" in str(getattr(res, "message", "")).lower():
                stop_reason = "maxiter"
            else:
                stop_reason = "diverged"
            n_fn = int(getattr(res, "nfev", 0))
            n_jac = int(getattr(res, "njev", 0)) or None
        except Exception as e:
            wall = time.perf_counter() - t0
            final = init.copy()
            final_loss = float("inf")
            converged = False
            stop_reason = "error"
            n_fn = 0
            n_jac = None
            err_msg = f"{type(e).__name__}: {e}"

        return FitRecord(
            skeleton_id=skel.structural_hash,
            skeleton_repr=sp.srepr(skel.expr),
            n_constants=skel.n_constants,
            init_constants=list(init),
            final_constants=list(final),
            initial_loss=initial_loss,
            final_loss=final_loss,
            n_fn_evals=n_fn,
            n_grad_evals=n_jac,
            wall_time_s=wall,
            converged=converged,
            stop_reason=stop_reason,
            backend=self.name,
            optimizer_kwargs={"method": self.method, **opt_kwargs},
            seed=seed,
            dataset_id=request.dataset.id,
            source=dict(request.source),
            error_msg=err_msg,
        )
