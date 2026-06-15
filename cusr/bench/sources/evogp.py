"""EvoGP skeleton source.

Streams top-K members of EvoGP's per-generation population as
`FitRequest` objects. Walks the prefix-order tree tensors to build a
parametrized skeleton (`c0, c1, ...` symbols instead of float literals),
records the raw constant values as warm-start init values, and notes any
`loose_*` ops that were degraded to their standard sympy equivalents.

Implementation notes:
- `pipeline.step()` replaces the forest with next-gen BEFORE returning
  (see `algorithm/genetic_programming.py:118`). So top-K must be captured
  INSIDE a pipeline override, before `self.algorithm.step(fitnesses)`.
- `Forest[int]` returns a `Tree` view; the tensors are live on CUDA. We
  deep-clone at capture time to avoid aliasing into the next generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import sympy as sp
import torch
from scipy.optimize import least_squares

# All EvoGP imports kept inside module to surface CUDA-related errors early
# but only when this file is imported — not when `bench` is imported.
from evogp.algorithm import (
    DefaultCrossover,
    DefaultMutation,
    DefaultSelection,
    GeneticProgramming,
)
from evogp.pipeline import StandardPipeline
from evogp.problem import SymbolicRegression
from evogp.tree import Forest, GenerateDescriptor, Tree
from evogp.tree.utils import Func, NType, SYMPY_MAP

from cusr.bench.dataset import Dataset
from cusr.bench.skeleton import FitRequest, Skeleton


# Map of loose-op enum int -> replacement sympy callable. EvoGP's `SYMPY_MAP`
# points these at custom `sp.Function` subclasses (LooseDiv/Log/Inv) which
# lack `.fdiff()` and aren't recognised by `sp.lambdify(..., modules='numpy')`.
# We substitute standard ops and rely on `Skeleton.residual`'s NaN guard.
# LOOSE_SQRT and LOOSE_POW are NOT degraded — SYMPY_MAP already uses native
# sp.sqrt(sp.Abs(x)) / sp.Pow(sp.Abs(x), y) which lambdify handles fine.
DEGRADE_MAP = {
    Func.LOOSE_DIV: lambda x, y: x / y,
    Func.LOOSE_LOG: sp.log,
    Func.LOOSE_INV: lambda x: 1 / x,
}


# Human-readable names for degraded ops (for metadata['degraded_ops']).
_DEGRADE_NAMES = {
    Func.LOOSE_DIV: "LooseDiv",
    Func.LOOSE_LOG: "LooseLog",
    Func.LOOSE_INV: "LooseInv",
}


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def forest_member_to_skeleton(
    tree: Tree, problem_n_vars: int
) -> tuple[Skeleton, np.ndarray, list[str]]:
    """Convert an EvoGP `Tree` into a `(Skeleton, init_constants, degraded_ops)` triple.

    - Variables become `x0..xn-1` symbols (match paper-level datasets).
    - CONSTs are replaced with fresh `c_i` symbols in encounter order.
    - Loose ops are rewritten per `DEGRADE_MAP`; each degradation is logged.
    """
    n = int(tree.subtree_size[0].item())
    node_type = _to_numpy(tree.node_type[:n])
    node_value = _to_numpy(tree.node_value[:n])

    x_syms = sp.symbols(f"x0:{problem_n_vars}", real=True)
    # sp.symbols("x0:1") returns a single Symbol, not a tuple; normalise.
    if problem_n_vars == 1 and not isinstance(x_syms, tuple):
        x_syms = (x_syms,)

    # First pass: collect CONSTs in forward prefix order so `c0` is the first
    # CONST in natural reading order (matches T-01 note §3 spec).
    const_positions: list[int] = []
    const_values: list[float] = []
    for i in range(n):
        if int(node_type[i]) == NType.CONST:
            const_positions.append(i)
            const_values.append(float(node_value[i]))
    n_consts = len(const_values)
    if n_consts > 0:
        c_syms = sp.symbols(f"c0:{n_consts}", real=True)
        if n_consts == 1 and not isinstance(c_syms, tuple):
            c_syms = (c_syms,)
    else:
        c_syms = ()
    # Map tree-node-index -> c_i symbol (stable, deterministic).
    pos_to_cidx = {p: k for k, p in enumerate(const_positions)}

    degraded_ops: list[str] = []
    stack: list[sp.Expr] = []

    # Walk reverse prefix — mirrors Tree.to_sympy_expr single-output branch.
    for i in reversed(range(n)):
        t = int(node_type[i])
        v = node_value[i]
        if t == NType.VAR:
            stack.append(x_syms[int(v)])
        elif t == NType.CONST:
            stack.append(c_syms[pos_to_cidx[i]])
        elif t == NType.UFUNC:
            fid = int(v)
            if fid in DEGRADE_MAP:
                fn = DEGRADE_MAP[fid]
                degraded_ops.append(f"{_DEGRADE_NAMES[fid]}@node{i}")
            else:
                fn = SYMPY_MAP[fid]
            stack.append(fn(stack.pop()))
        elif t == NType.BFUNC:
            fid = int(v)
            if fid in DEGRADE_MAP:
                fn = DEGRADE_MAP[fid]
                degraded_ops.append(f"{_DEGRADE_NAMES[fid]}@node{i}")
            else:
                fn = SYMPY_MAP[fid]
            left = stack.pop()
            right = stack.pop()
            stack.append(fn(left, right))
        elif t == NType.TFUNC:
            fid = int(v)
            if fid in DEGRADE_MAP:
                fn = DEGRADE_MAP[fid]
                degraded_ops.append(f"{_DEGRADE_NAMES[fid]}@node{i}")
            else:
                fn = SYMPY_MAP[fid]
            left = stack.pop()
            mid = stack.pop()
            right = stack.pop()
            stack.append(fn(left, mid, right))
        else:
            raise ValueError(f"unknown node_type {t} at index {i}")

    if len(stack) != 1:
        raise RuntimeError(f"malformed tree: stack depth {len(stack)} != 1 at end of walk")

    expr = stack[0]
    skel = Skeleton(
        expr=expr,
        variables=x_syms,
        constants=c_syms,
        metadata={"degraded_ops": list(degraded_ops)},
    )
    init = np.asarray(const_values, dtype=float)
    return skel, init, degraded_ops


def _clone_tree(tree: Tree) -> Tree:
    """Deep-clone a Tree view — decouples captured top-K from the live forest."""
    return Tree(
        tree.input_len,
        tree.output_len,
        tree.node_value.clone(),
        tree.node_type.clone(),
        tree.subtree_size.clone(),
    )


class _TopKPipeline(StandardPipeline):
    """StandardPipeline override that snapshots top-K members before
    `algorithm.step()` mutates the forest.

    After each `step()`, `self.captured` holds a list of
    `(member_id: int, fitness: float, cloned_tree: Tree)` tuples for the
    current generation's top-K members, ordered descending by fitness.
    """

    def __init__(self, *args, top_k: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.captured: list[tuple[int, float, Tree]] = []

    def step(self):  # noqa: D401
        fitnesses = self.problem.evaluate(self.algorithm.forest)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf
        cpu_fitness = fitnesses.cpu()

        # Snapshot top-K BEFORE algorithm mutates the forest.
        k = min(self.top_k, cpu_fitness.numel())
        _, topk_idx = torch.topk(cpu_fitness, k)
        captured: list[tuple[int, float, Tree]] = []
        for idx in topk_idx.tolist():
            tree = self.algorithm.forest[int(idx)]  # view
            captured.append((int(idx), float(cpu_fitness[idx]), _clone_tree(tree)))
        self.captured = captured

        # Preserve upstream best_tree bookkeeping (for pipeline.run() parity).
        best_idx = int(torch.argmax(cpu_fitness))
        best_fitness = torch.max(cpu_fitness)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = self.algorithm.forest[best_idx]

        self.algorithm.step(fitnesses)
        return cpu_fitness


def _writeback_constants(tree: Tree, c_star: np.ndarray) -> bool:
    """Write `c_star` (numpy f64) into `tree`'s CONST node positions, in-place via view aliasing.

    Distinguishes runtime-failure (NaN/inf in c_star → returns False, log as
    `skipped_nan`) from programming-bug (size mismatch → raises ValueError, log
    as `extract_error` at caller level).

    Returns:
        True  if write succeeded (or there were no constants — no-op).
        False if `c_star` contained NaN / inf — original tensor untouched.

    Raises:
        ValueError: when `c_star.size != n_consts` for n_consts > 0. This is a
            programming bug — fail loud rather than silently corrupt the forest.
    """
    arr = np.asarray(c_star, dtype=np.float64)
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        return False

    n = int(tree.subtree_size[0].item())
    node_type = tree.node_type[:n]
    mask = node_type == NType.CONST
    n_consts = int(mask.sum().item())

    if n_consts == 0:
        # Treat zero-const write-back as success (nothing to do); tolerate
        # empty c_star or any trailing length (caller shouldn't pass content,
        # but enforcing here is stricter than warranted).
        return True
    if arr.size != n_consts:
        # Programming bug: caller passed an array whose length doesn't match
        # the number of CONST nodes. Raise rather than corrupt the forest.
        raise ValueError(
            f"c_star size mismatch: got {arr.size}, expected {n_consts}"
        )

    # Cast to f32 for forest dtype contract.
    new_vals = torch.from_numpy(arr.astype(np.float32))
    # Place in the same device/dtype as forest.
    new_vals = new_vals.to(device=tree.node_value.device, dtype=tree.node_value.dtype)

    # Use nonzero+absolute index for robust write-through across torch versions.
    pos = mask.nonzero(as_tuple=False).squeeze(-1)
    tree.node_value[pos] = new_vals
    return True


def _zero_tiny_floats(expr, tol: float):
    """Replace Float atoms below `tol` in absolute value with 0.

    NLS-converged expressions carry tiny float residuals (e.g. `5.30e-9*x`
    coefficient on an otherwise-correct skeleton). These are numerically
    far below our r²>0.9999 fitness floor, but `sympy.simplify` won't drop
    them on its own. After this substitution, a follow-up `simplify()`
    collapses `0*x` → `0` etc, so `is_constant()` works as intended.
    """
    subs = {f: sp.S.Zero for f in expr.atoms(sp.Float) if abs(float(f)) < tol}
    return expr.xreplace(subs) if subs else expr


def check_recovery(f_hat, f_star, tol: float = 1e-6) -> bool:
    """SRBench-style recovery check (NeurIPS 2021 SRBench, La Cava et al.).

    Returns True iff after sympy simplification + sub-tolerance noise absorption:
      - `f_hat / f_star` reduces to a NON-ZERO constant (scaling-equivalent
        `f̂ = a·f*`, a != 0), OR
      - `f_hat - f_star` reduces to a constant (offset-equivalent `f̂ = f* + b`).

    BOTH holding (`f̂ = a·f* + b` with a≠1 AND b≠0) is NOT recovered — affine
    transform doesn't count as structural match.

    `tol` (default 1e-6) controls how small a numeric tail must be before it's
    absorbed as 0. Aligns with our r²>0.9999 early-stop threshold (relative
    error ~1e-4..6).

    Returns False rather than raising on sympy failure (timeout / parse error /
    divide-by-symbolic-zero) — recovery is a best-effort check; we'd rather
    miss a recovery than crash a 50-gen run.

    Args:
        f_hat: Hypothesis formula. `sympy.Expr` or sympy-parseable string.
        f_star: Ground-truth formula. Same.
        tol: Numeric tolerance for "this tiny float is just NLS noise".
    """
    try:
        # Normalize via str → sympify so both sides use Symbols with identical
        # default assumptions ({commutative: True}). Tree.to_sympy_expr emits
        # Symbols with {real, finite, hermitian, ...} assumptions; without
        # normalization sympy treats `Symbol('x', real=True)` and
        # `Symbol('x')` as distinct objects → `(x*y)/(x*y)` doesn't reduce
        # to 1, breaking is_constant() check.
        e_hat = sp.sympify(str(f_hat))
        e_star = sp.sympify(str(f_star))

        # Test 1: ratio is a non-zero constant → scaling-equivalent.
        ratio = sp.simplify(e_hat / e_star)
        ratio = sp.simplify(_zero_tiny_floats(ratio, tol))
        if ratio.is_constant() and not ratio.is_zero:
            return True

        # Test 2: difference is a constant → offset-equivalent.
        diff = sp.simplify(e_hat - e_star)
        diff = sp.simplify(_zero_tiny_floats(diff, tol))
        if diff.is_constant():
            return True

        return False
    except Exception:
        return False


def check_recovery_numeric(
    f_hat,
    f_star,
    X: np.ndarray,
    *,
    tol_rel: float = 1e-4,
) -> bool:
    """Numeric proxy for SRBench recovery — `f̂ = a·f*` OR `f̂ = f* + b`.

    Samples both expressions on `X` via `sympy.lambdify`, then runs two
    SEPARATE 1-parameter fits (NOT a 2-param affine fit, which would
    falsely accept `2x+3` ≈ x):
      - scaling-only `a`: `a = (y_hat·y_star) / (y_star·y_star)`, check
        residual `y_hat - a·y_star`
      - offset-only `b`: `b = mean(y_hat - y_star)`, check residual
        `y_hat - y_star - b`

    Pass if EITHER residual's max-abs is below `tol_rel · max|y_hat|`
    (relative tolerance against y_hat scale). a≈0 (collapsed-to-zero
    candidate) rejected outright in the scaling fit.

    Used as fallback for sympy `check_recovery` when the equivalence is
    sympy-opaque (huge coefficient algebra, transcendental rewrites like
    `sin(α+π/2) → cos(α)` that sympy.simplify won't reach).

    Returns False rather than raising on lambdify/eval errors — recovery
    is best-effort.

    Args:
        f_hat: Hypothesis. `sympy.Expr` or sympy-parseable string.
        f_star: Ground truth. Same.
        X: 2D array shape (n, n_vars). Variable order x0, x1, ... matches columns.
        tol_rel: Relative tolerance against `max|y_hat|` (default 1e-4 ≈
            r²>0.9999 fitness-floor analog).
    """
    n_vars = int(np.asarray(X).shape[1])
    syms = sp.symbols(f"x0:{n_vars}", real=True)
    if n_vars == 1 and not isinstance(syms, tuple):
        syms = (syms,)
    try:
        e_hat = sp.sympify(str(f_hat))
        e_star = sp.sympify(str(f_star))
        f_h = sp.lambdify(syms, e_hat, modules="numpy")
        f_s = sp.lambdify(syms, e_star, modules="numpy")
        y_hat = np.asarray(f_h(*np.asarray(X).T), dtype=float).reshape(-1)
        y_star = np.asarray(f_s(*np.asarray(X).T), dtype=float).reshape(-1)
        # Broadcast scalar return (e.g. f_hat = constant) to y_star shape.
        if y_hat.shape != y_star.shape:
            y_hat = np.broadcast_to(y_hat, y_star.shape).copy()
        if not np.all(np.isfinite(y_hat)) or not np.all(np.isfinite(y_star)):
            return False
    except Exception:  # noqa: BLE001
        return False

    scale = max(1.0, float(np.max(np.abs(y_hat))))
    abs_tol = tol_rel * scale

    # Pure scaling fit: a = <y_hat, y_star> / <y_star, y_star>
    denom = float(np.dot(y_star, y_star))
    if denom > 0.0:
        a = float(np.dot(y_hat, y_star) / denom)
        if abs(a) >= 1e-9:  # reject a≈0 collapse (degenerate)
            res_scale = float(np.max(np.abs(y_hat - a * y_star)))
            if res_scale < abs_tol:
                return True

    # Pure offset fit: b = mean(y_hat - y_star)
    b = float(np.mean(y_hat - y_star))
    res_offset = float(np.max(np.abs(y_hat - y_star - b)))
    if res_offset < abs_tol:
        return True

    return False


def check_recovery_composite(
    f_hat,
    f_star,
    X: np.ndarray,
    *,
    tol_sym: float = 1e-6,
    tol_rel: float = 1e-4,
) -> bool:
    """SRBench recovery — `sympy check OR numeric proxy`.

    Used in driver code for sympy-opaque expressions (extreme coefficient
    algebra, transcendental rewrites). 007 panel re-eval (gate, 2026-05-10)
    found 3/200 recoveries (1.5%) that sympy alone missed but numeric
    catches.
    """
    if check_recovery(f_hat, f_star, tol=tol_sym):
        return True
    return check_recovery_numeric(f_hat, f_star, X, tol_rel=tol_rel)


class _MemeticTopKPipeline(StandardPipeline):
    """Memetic GP+local-search pipeline.

    Each `step()`:
      1. evaluate fitness on the current forest;
      2. pick top-K by fitness;
      3. for each top-K member with constants: extract skeleton, run scipy LM,
         and write c* back into the LIVE forest tree (via view aliasing);
      4. recompute fitness once after write-back (single batch evaluate);
      5. for any member whose post-NLS fitness regressed, restore original c0;
      6. call `algorithm.step(final_fitnesses)` so selection sees the
         constant-refined fitness.

    NaN/inf in `c*` → write-back is skipped for that member (defensive).
    f32 round-trip safety net: if fitness drops, restore original constants.
    """

    def __init__(
        self,
        algorithm,
        problem,
        *,
        top_k: int,
        problem_n_vars: int,
        X_np: np.ndarray,
        y_np: np.ndarray,
        nls_max_nfev: int = 100,
        nls_every: int = 5,
        skeleton_dedup: bool = False,
        generation_limit: int = 50,
        is_show_details: bool = False,
        **kwargs,
    ):
        if nls_every < 1:
            raise ValueError(f"nls_every must be >= 1, got {nls_every}")
        super().__init__(
            algorithm=algorithm,
            problem=problem,
            generation_limit=generation_limit,
            is_show_details=is_show_details,
            **kwargs,
        )
        self.top_k = top_k
        self.problem_n_vars = problem_n_vars
        self.X_np = np.asarray(X_np, dtype=np.float64)
        self.y_np = np.asarray(y_np, dtype=np.float64)
        self.nls_max_nfev = nls_max_nfev
        self.nls_every = nls_every
        self.skeleton_dedup = skeleton_dedup
        # Generation counter — incremented at END of step(). First step emits gen=0.
        self._gen: int = 0
        # Diagnostics: per-top-K-member-per-step record. Each entry has
        #   {gen, member_id, rank, fitness_before, fitness_after, n_iter,
        #    converged, status}
        # status enum:
        #   "written"            — NLS ran, write-back applied, no rollback
        #   "rolled_back"        — written, but post-NLS fitness regressed and
        #                          original constants were restored
        #   "skipped_nan"        — NLS returned NaN/inf c_star; no write-back
        #   "skipped_no_consts"  — tree had zero CONST nodes
        #   "skipped_dup_skel"   — skeleton_dedup=True and this tree's skeleton
        #                          string already seen earlier in this batch
        #                          (lower-ranked dup of a higher-fit tree)
        #   "extract_error"      — forest_member_to_skeleton raised, or
        #                          _writeback_constants raised ValueError
        #                          (size-mismatch programming bug)
        #   "nls_error"          — scipy least_squares itself raised
        # For non-"written"/"rolled_back" paths fitness_after is None
        # ("not measured"). n_iter and converged are None when NLS didn't
        # complete (extract_error / nls_error).
        self.nls_records: list[dict] = []

    def step(self):  # noqa: D401
        forest = self.algorithm.forest

        # 1. Evaluate fitness BEFORE NLS.
        fitnesses = self.problem.evaluate(forest)
        fitnesses[torch.isnan(fitnesses)] = -torch.inf
        cpu_fitness_before = fitnesses.cpu().clone()

        # 1.5. Skip NLS this generation if not on the trigger schedule.
        # Pure GP step: no top-K extraction, no second evaluate, no nls_records.
        if self._gen % self.nls_every != 0:
            best_idx = int(torch.argmax(cpu_fitness_before))
            best_fitness = torch.max(cpu_fitness_before)
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_tree = forest[best_idx]
            self.algorithm.step(fitnesses)
            self._gen += 1
            return cpu_fitness_before

        # 2. Build iteration pool. Without dedup: top_k by fitness (current
        # behavior). With dedup: full forest sorted DESC, iterate until we hit
        # top_k unique-skeleton NLS attempts OR pool exhausts.
        k = min(self.top_k, cpu_fitness_before.numel())
        if self.skeleton_dedup:
            iter_pool = torch.argsort(cpu_fitness_before, descending=True).tolist()
            seen_skeletons: set[str] | None = set()
        else:
            _, topk_idx = torch.topk(cpu_fitness_before, k)
            iter_pool = topk_idx.tolist()
            seen_skeletons = None
        n_nls_attempted = 0  # increments only when a tree clears dedup AND has consts

        # 3. NLS → write-back per pool member. Track originals + record refs.
        rollback: dict[int, torch.Tensor] = {}
        # member_id -> the dict object in self.nls_records for "written" entries.
        # Holding the reference lets us mutate fields (status, fitness_after)
        # in-place during the rollback pass without searching by (gen, member_id).
        written_records: dict[int, dict] = {}
        any_writeback = False

        for rank, idx in enumerate(iter_pool):
            if n_nls_attempted >= k:
                break
            member_id = int(idx)
            fit_before = float(cpu_fitness_before[member_id].item())
            tree = forest[member_id]  # view

            try:
                skel, c0_init, _ = forest_member_to_skeleton(tree, self.problem_n_vars)
            except Exception as e:  # noqa: BLE001 — never let one tree crash the loop
                self.nls_records.append({
                    "gen": self._gen, "member_id": member_id, "rank": rank,
                    "fitness_before": fit_before, "fitness_after": None,
                    "n_iter": None, "converged": None,
                    "status": "extract_error", "error": str(e),
                })
                continue

            n_c = int(skel.n_constants)
            if n_c == 0:
                self.nls_records.append({
                    "gen": self._gen, "member_id": member_id, "rank": rank,
                    "fitness_before": fit_before, "fitness_after": None,
                    "n_iter": None, "converged": None,
                    "status": "skipped_no_consts",
                })
                continue

            # Dedup gate (after no-consts check, before scipy LM).
            # Hash key = str(skel.expr). c-symbols are encounter-order canonical
            # in forest_member_to_skeleton, so two structurally identical trees
            # with different float constants produce identical strings.
            if seen_skeletons is not None:
                skel_key = str(skel.expr)
                if skel_key in seen_skeletons:
                    self.nls_records.append({
                        "gen": self._gen, "member_id": member_id, "rank": rank,
                        "fitness_before": fit_before, "fitness_after": None,
                        "n_iter": None, "converged": None,
                        "status": "skipped_dup_skel",
                    })
                    continue
                seen_skeletons.add(skel_key)

            n_nls_attempted += 1

            # Snapshot originals BEFORE write-back for rollback.
            n_pref = int(tree.subtree_size[0].item())
            mask = tree.node_type[:n_pref] == NType.CONST
            pos = mask.nonzero(as_tuple=False).squeeze(-1)
            orig_const = tree.node_value[pos].detach().clone()

            # Run scipy LM directly. Defensive try/except — never raise into pipeline.
            try:
                res = least_squares(
                    fun=lambda c: skel.residual(c, self.X_np, self.y_np),
                    x0=np.asarray(c0_init, dtype=np.float64),
                    jac=lambda c: skel.jacobian(c, self.X_np),
                    method='lm',
                    max_nfev=self.nls_max_nfev,
                )
                c_star = np.asarray(res.x, dtype=np.float64)
            except Exception as e:  # noqa: BLE001
                self.nls_records.append({
                    "gen": self._gen, "member_id": member_id, "rank": rank,
                    "fitness_before": fit_before, "fitness_after": None,
                    "n_iter": None, "converged": None,
                    "status": "nls_error", "error": str(e),
                })
                continue

            n_iter = int(res.nfev)
            converged = bool(res.status > 0)

            try:
                ok = _writeback_constants(tree, c_star)
            except ValueError as e:
                # Programming-bug path (c_star size != n_consts). The forest
                # is untouched (raise-before-assign in _writeback_constants).
                self.nls_records.append({
                    "gen": self._gen, "member_id": member_id, "rank": rank,
                    "fitness_before": fit_before, "fitness_after": None,
                    "n_iter": n_iter, "converged": converged,
                    "status": "extract_error", "error": str(e),
                })
                continue

            if not ok:
                self.nls_records.append({
                    "gen": self._gen, "member_id": member_id, "rank": rank,
                    "fitness_before": fit_before, "fitness_after": None,
                    "n_iter": n_iter, "converged": converged,
                    "status": "skipped_nan",
                })
                continue

            # write-back succeeded — defer fitness_after until we re-evaluate.
            rec = {
                "gen": self._gen, "member_id": member_id, "rank": rank,
                "fitness_before": fit_before, "fitness_after": None,  # filled in step 4
                "n_iter": n_iter, "converged": converged,
                "status": "written",
            }
            self.nls_records.append(rec)
            written_records[member_id] = rec
            rollback[member_id] = orig_const
            any_writeback = True

        # 4. Recompute fitness — only if at least one write-back happened.
        if any_writeback:
            fitnesses_after = self.problem.evaluate(forest)
            fitnesses_after[torch.isnan(fitnesses_after)] = -torch.inf
            cpu_fitness_after = fitnesses_after.cpu().clone()

            # Patch each "written" record's fitness_after now that we have it.
            for member_id, rec in written_records.items():
                rec["fitness_after"] = float(cpu_fitness_after[member_id].item())

            # 5. Safety net: for each member we wrote back to, if post-NLS
            # fitness regressed (lower = worse since higher = better),
            # restore original c0 and patch fitness back to pre-NLS value.
            # The constant-evaluation count stays at exactly 2 evaluates per
            # step regardless of how many rollbacks happen.
            final_cpu = cpu_fitness_after.clone()
            fitnesses_for_step = fitnesses_after.clone()
            for member_id, orig_const in rollback.items():
                fb = cpu_fitness_before[member_id].item()
                fa = cpu_fitness_after[member_id].item()
                if fa < fb:
                    # Restore original constants in the live forest.
                    tree = forest[member_id]
                    n_pref = int(tree.subtree_size[0].item())
                    mask = tree.node_type[:n_pref] == NType.CONST
                    pos = mask.nonzero(as_tuple=False).squeeze(-1)
                    tree.node_value[pos] = orig_const.to(
                        device=tree.node_value.device, dtype=tree.node_value.dtype,
                    )
                    final_cpu[member_id] = cpu_fitness_before[member_id]
                    fitnesses_for_step[member_id] = fitnesses[member_id]
                    # Mutate the same dict already in self.nls_records.
                    written_records[member_id]["status"] = "rolled_back"

            cpu_final = final_cpu
        else:
            fitnesses_for_step = fitnesses
            cpu_final = cpu_fitness_before

        # 6. best_tree bookkeeping (mirrors StandardPipeline + _TopKPipeline).
        best_idx = int(torch.argmax(cpu_final))
        best_fitness = torch.max(cpu_final)
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_tree = forest[best_idx]

        # 7. Hand off to algorithm.
        self.algorithm.step(fitnesses_for_step)

        # 8. Advance generation counter (post-step convention: first step emits gen=0).
        self._gen += 1
        return cpu_final


@dataclass
class EvoGPSource:
    """SkeletonSource that runs EvoGP for N generations and streams top-K per gen.

    The EvoGP problem and descriptor are constructed by the caller; we just
    drive the pipeline and convert captured trees to FitRequests. Does not
    modify `upstream/evogp/` — all hooks are via subclassing.
    """

    problem: SymbolicRegression
    descriptor: GenerateDescriptor
    dataset: Dataset
    n_generations: int
    top_k: int
    pop_size: int = 50
    algorithm_kwargs: Optional[dict] = None
    seed: int = 0
    name: str = "evogp"

    def iter_requests(self) -> Iterable[FitRequest]:
        # Deterministic seeding for reproducibility.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        kwargs = dict(self.algorithm_kwargs or {})
        crossover = kwargs.pop("crossover", None) or DefaultCrossover()
        mutation = kwargs.pop("mutation", None) or DefaultMutation(
            mutation_rate=0.2, descriptor=self.descriptor.update(max_layer_cnt=3)
        )
        selection = kwargs.pop("selection", None) or DefaultSelection(
            survival_rate=0.3, elite_rate=0.01
        )

        algorithm = GeneticProgramming(
            initial_forest=Forest.random_generate(
                pop_size=self.pop_size, descriptor=self.descriptor
            ),
            crossover=crossover,
            mutation=mutation,
            selection=selection,
            **kwargs,
        )

        pipeline = _TopKPipeline(
            algorithm=algorithm,
            problem=self.problem,
            top_k=self.top_k,
            generation_limit=self.n_generations,
            is_show_details=False,
        )

        problem_n_vars = self.descriptor.input_len

        for gen in range(self.n_generations):
            pipeline.step()
            for rank, (member_id, fit, tree) in enumerate(pipeline.captured):
                skel, init, degraded = forest_member_to_skeleton(tree, problem_n_vars)
                yield FitRequest(
                    skeleton=skel,
                    init_constants=init,
                    dataset=self.dataset,
                    source={
                        "origin": "evogp",
                        "gen": gen,
                        "member_id": member_id,
                        "evogp_fitness": fit,
                        "rank": rank,
                        "degraded_ops": list(degraded),
                    },
                )
