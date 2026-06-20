"""test_inproc_bridge.py — END-TO-END wiring of the in-process CO drop-in (P1 step 5).

This is the step that puts the persistent ctypes handle (cusr.kernel.co_inproc) on
the REAL EvoGP CO path: kernel_bridge.fit_natives(inproc=True) /
co_backend.CudaKernelLM(inproc=True). See docs/kernel/INPROCESS_CO_PLAN.md step 7 +
MUST-FIX #7.

Two gates (TDD, written before the wiring existed):

  (a) END-TO-END equivalence — fit_natives on the SAME (skeletons, natives, X, y)
      with inproc=True vs inproc=False produces byte/tight-fp identical
      COResult.constants AND matching status/converged. We PROVE the inproc path
      actually routes through co_inproc (not silently the subprocess) by spying on
      kernel_bridge._run_kernel: it must be called 0× when inproc=True and exactly
      1× when inproc=False (the variant tracks the `binary` argument — FD binary ->
      FD .so -> byte-identical to the FD subprocess).

  (b) MUST-FIX #7 MIXED-BATCH — one batch mixing kernel-eligible trees with
      ineligible ones (TFUNC absent here, so K>32 over-K + over-stack), inproc=True.
      Assert NO None remains in results[M], each result lands in its correct
      orig_idx slot, and the scipy-fallback routing stays intact (n_kernel>=2 AND
      n_fallback>=2 so the batch genuinely mixes, and the fallback constants match a
      direct scipy fit on those exact skeletons).

Build prerequisite (run once):
    source scripts/env.sh
    cd cusr/kernel && make batch_lm libcusr_co_fd.so
Run (GPU pinned, device must be idle):
    source scripts/env.sh
    export CUDA_VISIBLE_DEVICES=2
    uv run pytest experiments/e2_demonstrator/test_inproc_bridge.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cusr.benchmark import interp, popio
from cusr.demonstrator import kernel_bridge as kb
from cusr.demonstrator.co_backend import CudaKernelLM, ScipyLM

# FD .so must be built for the inproc path; the subprocess path needs batch_lm.
_FD_SO = Path(kb.__file__).resolve().parents[1] / "kernel" / "libcusr_co_fd.so"
_have_so = _FD_SO.exists()
_have_binary = kb.DEFAULT_BINARY.exists()

inproc_gate = pytest.mark.skipif(
    not (_have_so and _have_binary),
    reason="needs libcusr_co_fd.so AND batch_lm (make batch_lm libcusr_co_fd.so)",
)


# --------------------------------------------------------------------------- helpers


class _ArrayTree:
    """Minimal EvoGP-Tree stand-in `_extract_tree` can read (re-injects c_init into
    the CONST node_value slots, as a live tree carries them)."""

    def __init__(self, nt, nv, ci, c_init):
        import torch

        nt = np.asarray(nt, np.int32)
        nv = np.asarray(nv, np.float32).copy()
        const_pos = np.where(nt == popio.NTYPE_CONST)[0]
        nv[const_pos] = np.asarray(c_init, np.float32)
        n = len(nt)
        self.subtree_size = torch.tensor([n], dtype=torch.int32)
        self.node_type = torch.tensor(nt, dtype=torch.int32)
        self.node_value = torch.tensor(nv, dtype=torch.float32)


def _linear_tree(c0, c1):
    """y = c0 + c1*x0 : prefix ADD(CONST, MUL(CONST, VAR)). Eligible (K=2).
    Prefix-CONST order -> (c0=intercept, c1=slope)."""
    nt = [popio.NTYPE_BFUNC, popio.NTYPE_CONST, popio.NTYPE_BFUNC,
          popio.NTYPE_CONST, popio.NTYPE_VAR]
    nv = [interp.F.ADD, 0.0, interp.F.MUL, 0.0, 0.0]
    ci = [-1, 0, -1, 1, -1]
    c_init = np.array([c0, c1], np.float32)
    return nt, nv, ci, c_init


def _affine_tree(c0, c1):
    """y = c0*x0 + c1 : prefix ADD(MUL(CONST, VAR), CONST). Eligible (K=2).
    Same K=2 as _linear_tree but the prefix-CONST order is REVERSED relative to
    the line params -> (c0=slope, c1=intercept). Fit to the SAME shared y as a
    _linear_tree, this converges to the SWAPPED constant pair, so the two same-K
    eligible trees have DISTINCT, well-separated ground truths. That makes a
    kernel-side reorder / c_final off-by-one on the COMPACTED writeback path
    (which would land one eligible tree's constants in the other's slot) VISIBLE
    to a per-slot value check — invisible when both same-K trees share a GT."""
    nt = [popio.NTYPE_BFUNC, popio.NTYPE_BFUNC, popio.NTYPE_CONST,
          popio.NTYPE_VAR, popio.NTYPE_CONST]
    nv = [interp.F.ADD, interp.F.MUL, 0.0, 0.0, 0.0]
    ci = [-1, -1, 0, -1, 1]
    c_init = np.array([c0, c1], np.float32)
    return nt, nv, ci, c_init


def _sin_tree(c0):
    """y = sin(c0*x0) : prefix SIN(MUL(CONST, VAR)). Eligible (K=1)."""
    nt = [popio.NTYPE_UFUNC, popio.NTYPE_BFUNC, popio.NTYPE_CONST, popio.NTYPE_VAR]
    nv = [interp.F.SIN, interp.F.MUL, 0.0, 0.0]
    ci = [-1, -1, 0, -1]
    return nt, nv, ci, np.array([c0], np.float32)


def _overk_tree(n):
    """Left-deep add chain with n constants (n > MAX_K) -> ineligible (over-K)."""
    nt_arr, nv_arr, ci_arr = [], [], []

    def emit(k):
        if k < 0:
            nt_arr.append(popio.NTYPE_VAR); nv_arr.append(0.0); ci_arr.append(-1)
            return
        nt_arr.append(popio.NTYPE_BFUNC); nv_arr.append(interp.F.ADD); ci_arr.append(-1)
        emit(k - 1)
        nt_arr.append(popio.NTYPE_CONST); nv_arr.append(0.0); ci_arr.append(k)

    emit(n - 1)
    return nt_arr, nv_arr, ci_arr, np.ones(n, np.float32)


def _skeleton_from_arrays(nt, nv, ci, n_vars):
    return kb.arrays_to_skeleton(np.asarray(nt, np.int32),
                                 np.asarray(nv, np.float32),
                                 np.asarray(ci, np.int32), n_vars)


# =========================================================================== (a)
@inproc_gate
def test_end_to_end_inproc_equals_subprocess(monkeypatch):
    """fit_natives(inproc=True) == fit_natives(inproc=False) on identical input,
    AND the inproc path never spawns the subprocess (spy on _run_kernel)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, (64, 1)).astype(float)
    x = X[:, 0]
    # all-eligible batch with their own targets baked into one shared y is wrong
    # (fit_natives fits every tree to ONE y); use one y and several skeletons that
    # all see the same y — the constants still come out deterministically.
    y = (1.7 - 0.4 * x).astype(float)

    specs = [_linear_tree(1.0, -0.2), _sin_tree(0.5), _linear_tree(-0.3, 0.9),
             _sin_tree(1.1)]
    skels, inits, natives = [], [], []
    for nt, nv, ci, c0 in specs:
        skels.append(_skeleton_from_arrays(nt, nv, ci, 1))
        inits.append(np.asarray(c0, float))
        natives.append(_ArrayTree(nt, nv, ci, c0))

    # spy: count subprocess invocations without disabling them.
    real_run_kernel = kb._run_kernel
    calls = {"n": 0}

    def spy(pop, binary, max_iter):
        calls["n"] += 1
        return real_run_kernel(pop, binary, max_iter)

    monkeypatch.setattr(kb, "_run_kernel", spy)

    # subprocess baseline (inproc=False)
    calls["n"] = 0
    res_sub, stats_sub = kb.fit_natives(
        skels, inits, natives, X, y, max_iter=50, inproc=False)
    n_sub = calls["n"]

    # inproc drop-in (inproc=True)
    calls["n"] = 0
    res_in, stats_in = kb.fit_natives(
        skels, inits, natives, X, y, max_iter=50, inproc=True)
    n_in = calls["n"]

    # routing proof: subprocess path hits _run_kernel once; inproc path NEVER does.
    assert n_sub == 1, f"subprocess path should call _run_kernel once, got {n_sub}"
    assert n_in == 0, (
        f"inproc path spawned the subprocess {n_in}× — it is NOT routing through "
        "co_inproc (the whole point of P1)"
    )

    # all trees here are kernel-eligible -> all go through the kernel both ways.
    assert stats_sub["n_kernel"] == len(specs) == stats_in["n_kernel"], \
        (stats_sub, stats_in)
    assert stats_sub["n_fallback"] == 0 and stats_in["n_fallback"] == 0

    # equivalence: constants byte/tight-fp identical, status/converged match.
    assert len(res_in) == len(res_sub) == len(specs)
    for j, (a, b) in enumerate(zip(res_in, res_sub)):
        assert a is not None and b is not None, j
        assert a.constants.shape == b.constants.shape, (j, a.constants, b.constants)
        # FD .so single-sources the FD device kernels -> bit-identical to FD subproc.
        assert np.array_equal(a.constants, b.constants), (
            f"tree {j}: inproc constants {a.constants} != subprocess {b.constants}")
        assert a.converged == b.converged, (j, a.converged, b.converged)
        assert a.final_loss == b.final_loss or (
            np.isfinite(a.final_loss) and np.isfinite(b.final_loss)
            and abs(a.final_loss - b.final_loss) <= 1e-12 * (abs(b.final_loss) + 1e-30)
        ), (j, a.final_loss, b.final_loss)


# =========================================================================== (b)
@inproc_gate
def test_mixed_batch_inproc_no_dropped_individual():
    """MUST-FIX #7 + COMPACTED-WRITEBACK value correctness: a batch mixing
    eligible + ineligible trees through inproc=True leaves NO None in results,
    routes every ineligible tree to scipy in its own slot, AND every ELIGIBLE
    result lands in its correct orig_idx slot with the correct VALUE.

    The eligible trees are filtered + COMPACTED into `kernel_slots` (kernel slot
    index != orig_idx) and written back via a per-slot `c_final[c_off:c_off+K]`
    slice. A kernel-side reorder or a c_final / c_off off-by-one on that compacted
    path would land one eligible tree's constants in another's slot. To make that
    VISIBLE, two of the eligible trees have the SAME K (=2) but DISTINCT, well-
    separated ground truths: _linear_tree(GT (intercept,slope)=(0.8,1.3)) vs
    _affine_tree(GT (slope,intercept)=(1.3,0.8)) — the reversed prefix-CONST order
    fits the same shared y to the SWAPPED pair. Each eligible slot is value-checked
    against HAND-COMPUTED ground truth (external to the shared writeback loop, so
    the check is not blind to a bug that hits inproc and subprocess identically).
    A swap maps (0.8,1.3) <-> (1.3,0.8): off by ~0.5 >> tol, while LM converges to
    ~1e-7."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.5, 1.5, (48, 1)).astype(float)
    x = X[:, 0]
    # Shared target line: intercept 0.8, slope 1.3. Both same-K eligible forms are
    # linear-in-params -> LM converges crisply/exactly to this line (distinct
    # prefix-CONST orderings -> distinct, swapped converged constants).
    INTERCEPT, SLOPE = 0.8, 1.3
    y = (INTERCEPT + SLOPE * x).astype(float)

    # Interleave eligible and ineligible so a naive slot-misalignment is caught.
    # Per-eligible hand-computed ground truth in prefix-CONST order (the order the
    # compacted writeback fills). None for the ineligible (scipy) slots.
    n_over = kb.MAX_K + 3  # K=35 > 32 -> over-K AND deep stack -> ineligible
    specs = [
        # _linear_tree: prefix CONST order = (intercept, slope)
        ("elig", _linear_tree(1.0, -0.2), np.array([INTERCEPT, SLOPE])),
        ("inelig", _overk_tree(n_over), None),
        # _sin_tree (K=1): sin(c0*x) fit to a line -> c0~0 (small-angle ~ linear);
        # value-checked against a single-tree (M=1) inproc fit, which exercises the
        # SAME kernel but with no compaction (kernel slot 0 == orig_idx 0), so it is
        # an independent reference for THIS compacted slot.
        ("elig_sin", _sin_tree(0.6), None),
        ("inelig", _overk_tree(n_over + 1), None),
        # _affine_tree: prefix CONST order = (slope, intercept) -> SWAPPED vs linear
        ("elig", _affine_tree(0.2, 1.4), np.array([SLOPE, INTERCEPT])),
    ]
    skels, inits, natives, kinds, gts = [], [], [], [], []
    for kind, (nt, nv, ci, c0), gt in specs:
        skels.append(_skeleton_from_arrays(nt, nv, ci, 1))
        inits.append(np.asarray(c0, float))
        natives.append(_ArrayTree(nt, nv, ci, c0))
        kinds.append(kind)
        gts.append(gt)

    fb = ScipyLM()
    results, stats = kb.fit_natives(
        skels, inits, natives, X, y, max_iter=50, inproc=True, fallback=fb)

    # the batch genuinely mixes (else the test is vacuous w.r.t. the audit gap)
    assert stats["n_kernel"] >= 2, stats
    assert stats["n_fallback"] >= 2, stats
    assert stats["n_overflow"] == sum(k == "inelig" for k in kinds), stats

    # MUST-FIX #7: no candidate silently dropped.
    assert all(r is not None for r in results), \
        [i for i, r in enumerate(results) if r is None]
    assert len(results) == len(specs)

    # The two same-K eligible trees MUST have distinct ground truths, else a swap
    # is invisible and this whole strengthening is vacuous. (Guards the test, not
    # the code: if someone later edits the GT constants to coincide.)
    elig_gts = [g for k, g in zip(kinds, gts) if k == "elig"]
    assert len(elig_gts) == 2 and not np.allclose(elig_gts[0], elig_gts[1]), \
        ("same-K eligible ground truths must differ to catch a compacted-writeback "
         f"swap; got {elig_gts}")

    # Independent reference for the nonlinear K=1 eligible slot: a single-tree
    # inproc fit (no compaction, kernel slot 0 == orig 0) of the SAME skeleton.
    j_sin = kinds.index("elig_sin")
    sin_ref = kb.fit_natives(
        [skels[j_sin]], [inits[j_sin]], [natives[j_sin]], X, y,
        max_iter=50, inproc=True)[0][0]

    # Per-slot value check for EVERY result (orig_idx slot correctness):
    #   - eligible same-K trees: against HAND-COMPUTED ground truth (tol << ~0.5
    #     separation; LM converges to ~1e-7) -> catches reorder AND off-by-one;
    #   - eligible nonlinear tree: against the M=1 (uncompacted) inproc reference;
    #   - ineligible (scipy) trees: against a direct scipy fit on THOSE skeletons.
    for j, kind in enumerate(kinds):
        assert results[j].constants.shape == (skels[j].n_constants,), (j, kind)
        if kind == "elig":
            assert np.allclose(results[j].constants, gts[j], atol=1e-3), (
                f"slot {j}: COMPACTED-writeback eligible constants "
                f"{results[j].constants} != ground truth {gts[j]} "
                "(kernel-side reorder / c_final off-by-one on the compacted path)")
        elif kind == "elig_sin":
            assert np.array_equal(results[j].constants, sin_ref.constants), (
                f"slot {j}: eligible (nonlinear) constants {results[j].constants} "
                f"!= single-tree inproc reference {sin_ref.constants} "
                "(compacted-writeback misroute)")
        else:  # inelig
            ref = fb.fit_batch([skels[j]], [inits[j]], X, y, max_iter=50)[0]
            assert np.array_equal(results[j].constants, ref.constants), (
                f"slot {j}: scipy-fallback constants misrouted "
                f"({results[j].constants} != {ref.constants})")
            assert results[j].final_loss == ref.final_loss, (j, kind)


# =========================================================================== backend face
@inproc_gate
def test_cuda_kernel_lm_inproc_flag_routes(monkeypatch):
    """CudaKernelLM(inproc=True) threads inproc/device_id into fit_natives and does
    NOT spawn the subprocess; inproc=False (default) still uses it."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-2.0, 2.0, (40, 1)).astype(float)
    y = (1.0 + 0.5 * X[:, 0]).astype(float)
    nt, nv, ci, c0 = _linear_tree(0.9, 0.4)
    skel = _skeleton_from_arrays(nt, nv, ci, 1)
    native = _ArrayTree(nt, nv, ci, c0)

    real_run_kernel = kb._run_kernel
    calls = {"n": 0}

    def spy(pop, binary, max_iter):
        calls["n"] += 1
        return real_run_kernel(pop, binary, max_iter)

    monkeypatch.setattr(kb, "_run_kernel", spy)

    calls["n"] = 0
    co_in = CudaKernelLM(inproc=True, device_id=0, max_iter=50)
    r_in = co_in.fit_batch([skel], [np.asarray(c0, float)], X, y, native=[native])
    assert calls["n"] == 0, f"inproc CudaKernelLM spawned subprocess {calls['n']}×"
    assert r_in[0] is not None and r_in[0].constants.shape == (2,)

    calls["n"] = 0
    co_sub = CudaKernelLM(inproc=False, max_iter=50)
    r_sub = co_sub.fit_batch([skel], [np.asarray(c0, float)], X, y, native=[native])
    assert calls["n"] == 1, f"subprocess CudaKernelLM should call _run_kernel once, got {calls['n']}"
    assert np.array_equal(r_in[0].constants, r_sub[0].constants)
