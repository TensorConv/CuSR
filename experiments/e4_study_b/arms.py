"""The four Study B arms — identical EvoGP config, differ ONLY in the CO knob.

The ladder (strategy doc §4.4): no-CO < sparse-CO < CPU-CO-every-gen < GPU-CO.

  no_co       co_probability=0          : honest stock EvoGP, selection on raw
                                           fitness (EvoGP's own discrete const
                                           mutation still runs). The floor.
  sparse_gpu  GPU kernel, co_every=K     : CO every K gens — the frequency rung
                                           between no-CO and every-gen (optimizer
                                           held = the kernel, so only FREQUENCY
                                           varies along no_co -> sparse_gpu ->
                                           gpu_every).
  cpu_every   scipy LM (fp64), every gen : the reference CPU optimizer. Under
                                           FIXED-GENERATION this is a
                                           QUALITY-PARITY arm (does the fp32-guard
                                           kernel reach the same end-to-end
                                           recovery as fp64 scipy?), NOT a speed
                                           claim — speed is wallclock-only (phase 2).
  gpu_every   GPU kernel, every gen      : our in-process kernel, CO every gen.

HONESTY (advisor): under fixed-generation NO hardware claim is made, so scipy
not being numerically matched to the kernel is fine and is labelled "reference
CPU optimizer (scipy LM, fp64)". The clean same-LM cpu-vs-gpu micro-check and
Operon ceiling are deferred to the wallclock phase. The CO BUDGET differs in
currency (scipy max_nfev vs kernel LM iterations) — reported as a named factor,
not silently equated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cusr.demonstrator.co_backend import COResult, CudaKernelLM, ScipyLM


class CappedCO:
    """Wrap a CO backend; SKIP (return init unchanged) skeletons that are too
    complex (sympy count_ops > max_ops, or n_constants > max_k). The SAME cap is
    applied to EVERY arm, so all arms see an IDENTICAL CO-eligible set (fair).

    Why: under fixed-generation, cpu_every (scipy) on bloated GP trees is
    catastrophically slow (a single problem/seed cell ran >14 min on a damped
    multi-inner tree). Bloated trees are non-recoveries anyway, so capping CO to
    the recoverable-complexity regime bounds cost without losing signal. The
    kernel's "can afford much bigger trees" edge is the WALLCLOCK-phase story and
    is deliberately held out here, so capping both equally is the apples-to-apples
    quality-parity setup. n_capped is reported (not hidden)."""

    def __init__(self, inner, *, max_ops: int = 40, max_k: int = 32):
        self.inner = inner
        self.max_ops = max_ops
        self.max_k = max_k
        self.name = f"capped[{getattr(inner, 'name', '?')}]"
        self._cum_capped = 0

    @property
    def cum_stats(self) -> dict:
        s = dict(getattr(self.inner, "cum_stats", {}) or {})
        s["cum_capped"] = self._cum_capped
        return s

    def _eligible(self, skel) -> bool:
        if skel.n_constants > self.max_k:
            return False
        try:
            return int(skel.expr.count_ops()) <= self.max_ops
        except Exception:  # noqa: BLE001
            return False

    def fit_batch(self, skeletons, inits, X, y, *, max_iter=100, native=None):
        idx = [i for i, s in enumerate(skeletons) if self._eligible(s)]
        out = [COResult(np.asarray(inits[i], dtype=float).ravel(), 0, False, float("inf"))
               for i in range(len(skeletons))]
        if idx:
            sub_sk = [skeletons[i] for i in idx]
            sub_in = [inits[i] for i in idx]
            sub_nat = ([native[i] for i in idx] if native is not None else None)
            res = self.inner.fit_batch(sub_sk, sub_in, X, y, max_iter=max_iter, native=sub_nat)
            for j, i in enumerate(idx):
                out[i] = res[j]
        self._cum_capped += len(skeletons) - len(idx)
        return out


@dataclass(frozen=True)
class Arm:
    name: str
    make_backend: Callable      # () -> a fresh CO backend (one per run)
    co_probability: float
    co_every: int
    uses_kernel: bool           # True => the run MUST route through the GPU kernel
    label: str


def make_arms(
    *,
    device_id: int = 0,
    kernel_max_iter: int = 50,
    scipy_max_nfev: int = 50,
    sparse_every: int = 5,
    max_ops: int = 40,
    max_k: int = 32,
) -> list[Arm]:
    # Every backend is wrapped in the SAME complexity cap so all arms optimize an
    # identical CO-eligible set (fair) and the CPU arm is bounded on bloated trees.
    def kernel():
        return CappedCO(CudaKernelLM(inproc=True, device_id=device_id,
                                     max_iter=kernel_max_iter, fallback=ScipyLM()),
                        max_ops=max_ops, max_k=max_k)

    def scipy():
        return CappedCO(ScipyLM(), max_ops=max_ops, max_k=max_k)

    # scipy_max_nfev is carried for honest budget reporting; ScipyLM reads
    # max_iter at fit_batch time (the pipeline passes co_max_iter), so the two
    # budgets are recorded side by side, not equated.
    return [
        Arm("no_co",      scipy,  0.0, 1,            False,
            "no CO (stock EvoGP)"),
        Arm("sparse_gpu", kernel, 1.0, sparse_every, True,
            f"sparse GPU-CO (every {sparse_every} gens)"),
        Arm("cpu_every",  scipy,  1.0, 1,            False,
            "CPU-CO every gen (scipy LM, fp64)"),
        Arm("gpu_every",  kernel, 1.0, 1,            True,
            "GPU-CO every gen (kernel, fp32+guard)"),
    ]
