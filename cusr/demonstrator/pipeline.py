"""009 memetic GP pipeline driven by a pluggable CO backend.

Same shape as bench's _MemeticTopKPipeline (extract -> fit constants -> write
back -> re-evaluate -> rollback regressions -> selection), but the per-member
CPU scipy call is replaced by ONE batched `co_backend.fit_batch(...)` over all
candidates that share this problem's (X, y). Swapping ScipyLM / TorchLM /
(future) CudaKernelLM changes nothing else — that's the point of the interface.

We do NOT modify bench's pipeline (exp007 imports it); this is 009-owned and
reuses only the bridges forest_member_to_skeleton / _writeback_constants /
_clone_tree.

`full_pop=True` runs CO over the whole population (the aggressive regime the
"capability-unlock" story is about); default is top-K. dtype of the backend is
the caller's choice — note EvoGP fitness + _writeback_constants are float32, so
a float64 fit is truncated to f32 on writeback (the rollback guard catches any
regression that causes).
"""
from __future__ import annotations

import numpy as np
import torch

from cusr.bench.sources.evogp import (
    _clone_tree, _writeback_constants, forest_member_to_skeleton,
)
from evogp.pipeline import StandardPipeline


class MemeticPipeline(StandardPipeline):
    def __init__(
        self,
        algorithm,
        problem,
        co_backend,
        *,
        problem_n_vars: int,
        X_np: np.ndarray,
        y_np: np.ndarray,
        top_k: int = 16,
        full_pop: bool = False,
        co_max_iter: int = 100,
        co_every: int = 1,
        co_probability: float = 1.0,
        co_rng_seed: int = 0,
        generation_limit: int = 50,
        is_show_details: bool = False,
        **kwargs,
    ):
        super().__init__(
            algorithm=algorithm, problem=problem,
            generation_limit=generation_limit,
            is_show_details=is_show_details, **kwargs,
        )
        self.co = co_backend
        self.problem_n_vars = problem_n_vars
        self.X_np = np.asarray(X_np, dtype=float)
        self.y_np = np.asarray(y_np, dtype=float).ravel()
        self.top_k = top_k
        self.full_pop = full_pop
        self.co_max_iter = co_max_iter
        self.co_every = max(1, co_every)
        # E2 main axis: probability a selected candidate gets CO. p=0 -> honest
        # stock EvoGP (no CO, selection sees raw fitness, EvoGP's own discrete
        # const mutation still runs); p=1 -> all selected candidates (default).
        self.co_probability = float(co_probability)
        self._co_rng = np.random.default_rng(co_rng_seed)
        self._gen = 0
        # diagnostics for compute accounting / fair comparison
        self.n_co_fits = 0       # total skeletons sent to the CO backend
        self.n_rollbacks = 0
        self.n_evals = 0         # problem.evaluate() calls

    def _evaluate(self, forest):
        f = self.problem.evaluate(forest)
        f[torch.isnan(f)] = -torch.inf
        self.n_evals += 1
        return f

    def _update_best(self, forest, cpu_fit):
        bi = int(torch.argmax(cpu_fit))
        bf = torch.max(cpu_fit)
        if bf > self.best_fitness:
            self.best_fitness = bf
            self.best_tree = _clone_tree(forest[bi])  # decouple from live forest

    def step(self):  # noqa: D401
        forest = self.algorithm.forest
        fit = self._evaluate(forest)
        cpu_before = fit.cpu().clone()
        self._update_best(forest, cpu_before)

        # Pure-GP generation: no CO this step.
        if self._gen % self.co_every != 0:
            self.algorithm.step(fit)
            self._gen += 1
            return cpu_before

        # p=0: honest stock EvoGP — identical to a pure-GP step (no CO call, no
        # writeback, selection on raw fitness; EvoGP's discrete const mutation
        # still runs inside algorithm.step). This is the E2 baseline arm.
        if self.co_probability <= 0.0:
            self.algorithm.step(fit)
            self._gen += 1
            return cpu_before

        # Build the candidate pool.
        if self.full_pop:
            pool = list(range(cpu_before.numel()))
        else:
            k = min(self.top_k, cpu_before.numel())
            pool = torch.topk(cpu_before, k).indices.tolist()

        # Probabilistic CO (E2 p-sweep): keep each candidate with prob co_probability.
        if self.co_probability < 1.0:
            pool = [m for m in pool if self._co_rng.random() < self.co_probability]

        skels, inits, ids = [], [], []
        for idx in pool:
            mid = int(idx)
            try:
                skel, init, _ = forest_member_to_skeleton(forest[mid], self.problem_n_vars)
            except Exception:  # noqa: BLE001 — one bad tree must not crash the gen
                continue
            if skel.n_constants == 0:
                continue
            skels.append(skel)
            inits.append(init)
            ids.append(mid)

        if not skels:
            self._update_best(forest, cpu_before)
            self.algorithm.step(fit)
            self._gen += 1
            return cpu_before

        # ONE batched CO call over all candidates (the swap point). `native` =
        # the live EvoGP trees, which scipy/torch ignore but CudaKernelLM needs
        # for the tree -> pop.bin bridge (read-only; extracted before writeback).
        results = self.co.fit_batch(skels, inits, self.X_np, self.y_np,
                                    max_iter=self.co_max_iter,
                                    native=[forest[m] for m in ids])
        self.n_co_fits += len(skels)

        for mid, res in zip(ids, results):
            c = np.asarray(res.constants, dtype=float)
            if c.size and np.all(np.isfinite(c)):
                try:
                    _writeback_constants(forest[mid], c)
                except Exception:  # noqa: BLE001 — size-mismatch bug; skip member
                    pass

        # Re-evaluate; roll back any member whose fitness regressed (f32 safety net).
        cpu_after = self._evaluate(forest).cpu()
        for mid, init in zip(ids, inits):
            if cpu_after[mid] < cpu_before[mid]:
                try:
                    _writeback_constants(forest[mid], np.asarray(init, dtype=float))
                    self.n_rollbacks += 1
                except Exception:  # noqa: BLE001
                    pass

        # Final fitness AFTER rollback so selection sees the true current state.
        fit_final = self._evaluate(forest)
        cpu_final = fit_final.cpu()
        self._update_best(forest, cpu_final)
        self.algorithm.step(fit_final)
        self._gen += 1
        return cpu_final
