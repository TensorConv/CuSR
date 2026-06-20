"""Pluggable constant-optimization (CO) backends for the 009 memetic loop.

A CO backend takes a batch of candidate skeletons that share one problem's
(X, y) and fits each skeleton's free constants. The common currency is
`bench.skeleton.Skeleton` (sympy expr + c-symbols + x-vars) — the memetic loop
already produces these via forest_member_to_skeleton.

Backends (all behind the same `fit_batch` contract, so they're directly
comparable and the memetic pipeline is backend-agnostic):
  - ScipyLM       : per-candidate scipy least_squares (CPU) — the reference,
                    mirrors what bench's _MemeticTopKPipeline does today.
  - TorchLM       : GPU-native LM, autograd Jacobian + torch.linalg solve.
  - CudaKernelLM  : FUTURE — wrap the 008 CUDA kernel (Skeleton -> tree bytecode).

See project memory project_009_co_backend_interface for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class COResult:
    constants: np.ndarray   # fitted c*  (shape (n_consts,), () -> shape (0,))
    n_iter: int
    converged: bool
    final_loss: float       # mean(residual**2)


@runtime_checkable
class ConstantOptimizer(Protocol):
    name: str

    def fit_batch(
        self,
        skeletons: Sequence,
        inits: Sequence[np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        *,
        max_iter: int = 100,
        native: Sequence | None = None,
    ) -> list[COResult]:
        ...


# ---------------------------------------------------------------------------
# ScipyLM — reference backend (CPU), mirrors bench's existing per-member LM
# ---------------------------------------------------------------------------

class ScipyLM:
    name = "scipy_lm"

    def fit_batch(self, skeletons, inits, X, y, *, max_iter=100, native=None) -> list[COResult]:
        from scipy.optimize import least_squares

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        out: list[COResult] = []
        for skel, c0 in zip(skeletons, inits):
            c0 = np.asarray(c0, dtype=float).ravel()
            if skel.n_constants == 0:
                loss = float(np.mean(skel.residual(c0, X, y) ** 2))
                out.append(COResult(c0.copy(), 0, True, loss))
                continue
            try:
                r = least_squares(
                    lambda c: skel.residual(c, X, y),
                    c0,
                    jac=lambda c: skel.jacobian(c, X),
                    method="lm",
                    max_nfev=max_iter,
                )
                loss = float(np.mean(r.fun ** 2))
                out.append(COResult(np.asarray(r.x, dtype=float), int(r.nfev),
                                    bool(r.success), loss))
            except Exception:  # noqa: BLE001 — never raise into the GP loop
                # The fallback must not double-fault: if skel.residual itself
                # raises (e.g. a zoo expr whose lambdify build throws), catching
                # here is the difference between demoting one tree and crashing
                # the whole CO batch to R2=0.
                try:
                    loss = float(np.mean(skel.residual(c0, X, y) ** 2))
                except Exception:  # noqa: BLE001
                    loss = float("inf")
                out.append(COResult(c0.copy(), 0, False, loss))
        return out


# ---------------------------------------------------------------------------
# TorchLM — GPU-native Levenberg-Marquardt, autograd Jacobian
# ---------------------------------------------------------------------------

_TORCHMAP = None  # built lazily so importing this module doesn't require torch


def _torchmap():
    global _TORCHMAP
    if _TORCHMAP is None:
        import torch
        _TORCHMAP = {
            "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
            "asin": torch.asin, "acos": torch.acos, "atan": torch.atan,
            "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
            "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt,
            "Abs": torch.abs,
        }
    return _TORCHMAP


class TorchLM:
    """Levenberg-Marquardt on GPU. Per-candidate loop; each residual/Jacobian
    eval is vectorized over the N data points on the device. (Fusing across
    candidates is the future perf lever — and the literal 008 CUDA kernel.)"""

    name = "torch_lm"

    def __init__(self, device: str | None = None, dtype: str = "float64",
                 lambda0: float = 1e-3, grad_tol: float = 1e-10,
                 step_tol: float = 1e-12):
        self.device = device
        self.dtype = dtype
        self.lambda0 = lambda0
        self.grad_tol = grad_tol
        self.step_tol = step_tol

    def _resolve_device(self):
        import torch
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit_batch(self, skeletons, inits, X, y, *, max_iter=100, native=None) -> list[COResult]:
        import torch

        dev = self._resolve_device()
        dt = getattr(torch, self.dtype)
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        Xt = torch.tensor(Xa, dtype=dt, device=dev)
        yt = torch.tensor(np.asarray(y, dtype=float).ravel(), dtype=dt, device=dev)
        varcols = tuple(Xt[:, i] for i in range(Xt.shape[1]))
        out: list[COResult] = []
        for skel, c0 in zip(skeletons, inits):
            c0a = np.asarray(c0, dtype=float).ravel()
            # Graceful degrade on any failure (e.g. an operator not in _TORCHMAP,
            # singular system) — mirror ScipyLM's defensive path, never crash the
            # batch. inf loss => the GP loop won't prefer this candidate.
            try:
                out.append(self._fit_one(skel, c0a, varcols, yt, dev, dt, max_iter))
            except Exception:  # noqa: BLE001
                out.append(COResult(c0a.copy(), 0, False, float("inf")))
        return out

    def _fit_one(self, skel, c0, varcols, yt, dev, dt, max_iter) -> COResult:
        import sympy as sp
        import torch

        nc = skel.n_constants
        f = sp.lambdify((tuple(skel.constants), tuple(skel.variables)),
                        skel.expr, modules=[_torchmap()])

        def model(c):
            pred = f(tuple(c[i] for i in range(nc)), varcols)
            if not torch.is_tensor(pred):
                pred = torch.as_tensor(pred, dtype=dt, device=dev)
            return pred.reshape(-1)

        def resid(c):
            # Match bench.Skeleton.residual's NaN/inf guard (1e10 sentinel) so
            # TorchLM backs off pathological trees the SAME way ScipyLM does —
            # otherwise scipy-vs-torch at equal compute compares blow-up policies,
            # not optimizers. autograd through nan_to_num gives 0 grad at sentinels
            # (matches bench's jacobian->0 guard).
            import torch as _t
            r = model(c) - yt
            return _t.nan_to_num(r, nan=1e10, posinf=1e10, neginf=-1e10)

        if nc == 0:
            r = resid(torch.zeros(0, dtype=dt, device=dev))
            return COResult(np.zeros(0), 0, True, float(torch.mean(r ** 2).item()))

        c = torch.tensor(c0, dtype=dt, device=dev)
        r = resid(c)
        loss = float(torch.mean(r ** 2).item())
        lam = self.lambda0
        eye = torch.eye(nc, dtype=dt, device=dev)
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            try:
                J = torch.autograd.functional.jacobian(resid, c, vectorize=True)
            except Exception:  # noqa: BLE001
                break
            J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            g = J.T @ r                      # (nc,)
            H = J.T @ J                      # (nc, nc)
            if float(torch.max(torch.abs(g)).item()) < self.grad_tol:
                converged = True
                break
            # LM damping search: grow lambda until the step reduces loss.
            accepted = False
            for _ in range(30):
                try:
                    delta = torch.linalg.solve(H + lam * eye, -g)
                except Exception:  # noqa: BLE001 — singular; damp harder
                    lam *= 10.0
                    continue
                c_new = c + delta
                r_new = resid(c_new)
                loss_new = float(torch.mean(r_new ** 2).item())
                if np.isfinite(loss_new) and loss_new < loss:
                    step = float(torch.max(torch.abs(delta)).item())
                    c, r, loss = c_new, r_new, loss_new
                    lam = max(lam / 10.0, 1e-12)
                    accepted = True
                    if step < self.step_tol:
                        converged = True
                    break
                lam *= 10.0
                if lam > 1e12:
                    break
            if not accepted or converged:
                if not accepted:
                    converged = converged or (float(torch.max(torch.abs(g)).item()) < self.grad_tol)
                break
        return COResult(c.detach().cpu().numpy().astype(float), int(it),
                        bool(converged), float(loss))


# ---------------------------------------------------------------------------
# CudaKernelLM — wrap the 008 CUDA kernel via the tree -> pop.bin bridge.
# Marshaling + per-tree fallback live in kernel_bridge; this is the thin face.
# ---------------------------------------------------------------------------

class CudaKernelLM:
    """008 CUDA LM kernel as a CO backend (native EvoGP tree -> pop.bin -> kernel).

    Unlike scipy/torch, this needs the live EvoGP tree per candidate, supplied
    via `native=` (the memetic pipeline passes `forest[mid]`). Trees the kernel
    can't take (TFUNC, K>32, stack>64) fall back to `fallback` (scipy by default)
    so no candidate is dropped. The kernel budget is `max_iter` *LM iterations*
    set at construction (a different currency from scipy's nfev); per-candidate
    marshaling stats land in `self.last_stats` for honest reporting.
    """
    name = "cuda_kernel_lm"

    def __init__(self, binary=None, max_iter: int = 50, fallback=None,
                 inproc: bool = False, device_id: int = 0):
        self.binary = binary
        self.max_iter = max_iter
        self.fallback = fallback
        # inproc=True routes through the persistent in-process ctypes drop-in
        # (co_inproc.get_inproc_co) instead of the per-gen subprocess, paying the
        # CUDA primary-context init once per process (P1). device_id selects the
        # GPU for that handle (0 under CUDA_VISIBLE_DEVICES pinning).
        self.inproc = bool(inproc)
        self.device_id = int(device_id)
        self.last_stats: dict = {}
        self.cum_stats: dict = {}  # accumulated across fit_batch calls (one run)

    def fit_batch(self, skeletons, inits, X, y, *, max_iter=100, native=None) -> list[COResult]:
        from .kernel_bridge import DEFAULT_BINARY, fit_natives

        if native is None:
            raise ValueError(
                "CudaKernelLM needs the native EvoGP trees via native= — the "
                "memetic pipeline passes forest[mid] at the CO call site."
            )
        results, stats = fit_natives(
            skeletons, inits, native, X, y,
            binary=self.binary or DEFAULT_BINARY,
            max_iter=self.max_iter,
            fallback=self.fallback,
            inproc=self.inproc,
            device_id=self.device_id,
        )
        self.last_stats = stats
        for k, v in stats.items():
            self.cum_stats[k] = self.cum_stats.get(k, 0) + v
        return results
