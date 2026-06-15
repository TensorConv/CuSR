"""Tests + parity gate for the W5 tree -> pop.bin -> batch_lm bridge.

Two layers:
  - GPU-free: arrays <-> sympy equivalence (so kernel-vs-scipy is a true
    same-problem comparison) and extract round-trip.
  - GPU gate (skipped if no batch_lm binary): kernel final loss within tier-B
    of scipy on real W4 snapshot trees AND a realistic-range (~1e-5 inner
    constant) case — the advisor's must-do before trusting any pilot number.

Run `pytest -s test_kernel_bridge.py` for assertions, or
`python test_kernel_bridge.py` to print the parity table for the record.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from cusr.benchmark import interp, popio
from cusr.demonstrator import kernel_bridge as kb
from cusr.demonstrator.co_backend import ScipyLM

_SYNTH = kb._DIR_012 / "workload" / "synth" / "synth_inner-const-heavy_M4000_N1000_seed0.bin"
_TIER_B = 1.05
_LOSS_FLOOR = 1e-12  # absolute floor so near-zero (recoverable) losses don't blow up the ratio

_have_binary = kb.DEFAULT_BINARY.exists()
gpu_gate = pytest.mark.skipif(not _have_binary, reason="batch_lm binary not built")


def _have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


_HAS_CUDA = _have_cuda()
evogp_gate = pytest.mark.skipif(not _HAS_CUDA, reason="EvoGP needs CUDA")


# --------------------------------------------------------------------------- helpers

class _ArrayTree:
    """Minimal stand-in for an EvoGP Tree that `_extract_tree` can read.

    pop.bin zeroes a CONST node's node_value (the value lives in c_init); a real
    EvoGP tree keeps the value *in* node_value. So we re-inject c_init into the
    CONST slots, mirroring what _extract_tree reads off a live tree.
    """

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


def _iter_trees(pop, limit):
    for m in range(min(limit, pop["M"])):
        nt, nv, ci, _ym, K = interp.tree_view(pop, m)
        _, _, c_off, _ = pop["metas"][m].tolist()
        c_init = pop["c_init"][c_off:c_off + K]
        yield nt, nv, ci, c_init


def _mean_sq_loss(skel, c, X, y):
    return float(np.mean(skel.residual(np.asarray(c, float), X, y) ** 2))


# --------------------------------------------------------------------------- GPU-free

@pytest.mark.skipif(not _SYNTH.exists(), reason="W4 synth snapshot missing")
def test_arrays_to_skeleton_matches_interp():
    """skel.evaluate(c) == interp.eval_tree_arrays(arrays, c) for random c."""
    # Gate on the bulk (median): a structural bug (wrong opcode, swapped SUB/DIV
    # operands, wrong const index) shifts the WHOLE curve by O(1). The 95th pct
    # tolerates the thin tail of near-singular div/tan points where two float64
    # eval orders legitimately diverge though both stay finite.
    pop = popio.load_pop_bin(_SYNTH)
    rng = np.random.default_rng(0)
    n_checked = 0
    for nt, nv, ci, _c_init in _iter_trees(pop, 40):
        skel = kb.arrays_to_skeleton(nt, nv, ci, pop["n_vars"])
        for _ in range(5):
            c = rng.uniform(-2, 2, skel.n_constants)
            ref = interp.eval_tree_arrays(nt, nv, ci, pop["xs"], c)
            got = skel.evaluate(c, pop["xs"])
            m = np.isfinite(ref) & np.isfinite(got)
            if m.sum() == 0:
                continue
            rel = np.abs(got[m] - ref[m]) / (np.abs(ref[m]) + 1e-9)
            assert np.median(rel) < 1e-5 and np.quantile(rel, 0.95) < 1e-2, \
                f"arrays/sympy mismatch (med={np.median(rel):.1e}): " \
                f"{interp.to_infix(nt, nv, ci)}"
        n_checked += 1
    assert n_checked >= 20


@pytest.mark.skipif(not _SYNTH.exists(), reason="W4 synth snapshot missing")
def test_extract_roundtrip():
    """_ArrayTree -> _extract_tree recovers nt/ci and the injected c_init."""
    pop = popio.load_pop_bin(_SYNTH)
    for nt, nv, ci, c_init in _iter_trees(pop, 20):
        native = _ArrayTree(nt, nv, ci, c_init)
        nt2, nv2, ci2, c2 = kb.extract_native(native, pop["n_vars"])
        assert np.array_equal(np.asarray(nt), nt2)
        assert np.array_equal(np.asarray(ci), ci2)
        assert np.allclose(c2, np.asarray(c_init, np.float32), rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- GPU gate

def _parity_on_pop(pop, limit, max_iter=50):
    """Run kernel + scipy on the same trees; return (loss_kernel, loss_scipy, loss_init) rows."""
    sub = popio.slice_pop(pop, limit)
    c_final, _status = kb._run_kernel(sub, kb.DEFAULT_BINARY, max_iter)
    scipy = ScipyLM()
    rows = []
    for m in range(sub["M"]):
        nt, nv, ci, ym, K = interp.tree_view(sub, m)
        _, _, c_off, _ = sub["metas"][m].tolist()
        c_init = sub["c_init"][c_off:c_off + K]
        skel = kb.arrays_to_skeleton(nt, nv, ci, sub["n_vars"])
        loss_k = _mean_sq_loss(skel, c_final[c_off:c_off + K], sub["xs"], ym)
        loss_init = _mean_sq_loss(skel, c_init, sub["xs"], ym)
        res = scipy.fit_batch([skel], [np.asarray(c_init, float)], sub["xs"], ym,
                              max_iter=max_iter)[0]
        rows.append((loss_k, res.final_loss, loss_init))
    return rows


@gpu_gate
@pytest.mark.skipif(not _SYNTH.exists(), reason="W4 synth snapshot missing")
def test_kernel_parity_synth():
    rows = _parity_on_pop(popio.load_pop_bin(_SYNTH), 24)
    # Direction-correctness (the "not garbage" gate): LM never accepts a
    # loss-increasing step, so a kernel optimizing the WRONG function (encoding
    # bug) would push the loss measured via the CORRECT skeleton UP from init.
    # This must hold for EVERY tree; fp32 stalls only fail tier-B, never this.
    for lk, _ls, li in rows:
        assert lk <= 1.01 * li + 1e-9, \
            f"kernel raised loss vs init ({li:.3e} -> {lk:.3e}) — encoding bug"
    # Quality parity: the fp32 kernel matches fp64 scipy within tier-B on the
    # bulk; the tail is the honest fp32 plateau (see RESULTS_kernel_bridge.md).
    n_ok = sum(lk <= _TIER_B * ls + _LOSS_FLOOR for lk, ls, _li in rows)
    assert n_ok / len(rows) >= 0.90, \
        f"tier-B parity {n_ok}/{len(rows)} < 90% — fp32 stall rate higher than expected"


@gpu_gate
def test_kernel_parity_realistic_range():
    """sin(c0*x0) with a tiny inner frequency (~1e-5): exercises relative eps_fd.

    Without W0's relative FD step a 1e-5 constant gives a degenerate Jacobian;
    the gate is that the kernel tracks scipy from the same (off) init.
    """
    c_true = 1.0e-5
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 2 * np.pi / c_true, 400).astype(np.float32).reshape(-1, 1)
    y = np.sin(c_true * x[:, 0]).astype(np.float32)
    # prefix sin(mul(c0, x0)): UFUNC SIN, BFUNC MUL, CONST, VAR
    nt = np.array([popio.NTYPE_UFUNC, popio.NTYPE_BFUNC, popio.NTYPE_CONST, popio.NTYPE_VAR], np.int32)
    nv = np.array([interp.F.SIN, interp.F.MUL, 0.0, 0.0], np.float32)
    ci = np.array([-1, -1, 0, -1], np.int32)
    c_init = np.array([1.2e-5], np.float32)  # 20% off true
    pop = popio.build_pop([(nt, nv, ci, c_init)], x, y.reshape(1, -1))

    c_final, _ = kb._run_kernel(pop, kb.DEFAULT_BINARY, 100)
    skel = kb.arrays_to_skeleton(nt, nv, ci, 1)
    loss_k = _mean_sq_loss(skel, c_final[:1], x, y)
    loss_s = ScipyLM().fit_batch([skel], [c_init.astype(float)], x, y, max_iter=100)[0].final_loss
    assert loss_k <= _TIER_B * loss_s + _LOSS_FLOOR, \
        f"realistic-range parity fail: kernel={loss_k:.3e} scipy={loss_s:.3e}"


@gpu_gate
@pytest.mark.skipif(not _SYNTH.exists(), reason="W4 synth snapshot missing")
def test_fit_natives_alignment():
    """fit_natives via _ArrayTree natives: aligned, K matches, losses finite."""
    pop = popio.load_pop_bin(_SYNTH)
    skels, inits, natives = [], [], []
    for nt, nv, ci, c_init in _iter_trees(pop, 16):
        skels.append(kb.arrays_to_skeleton(nt, nv, ci, pop["n_vars"]))
        inits.append(np.asarray(c_init, float))
        natives.append(_ArrayTree(nt, nv, ci, c_init))
    # Each synth tree has its own target (ym row); fit per-tree to keep it honest.
    res_all = []
    for j, (skel, init, nat) in enumerate(zip(skels, inits, natives)):
        r, _stats = kb.fit_natives([skel], [init], [nat], pop["xs"], pop["ym"][j],
                                   binary=kb.DEFAULT_BINARY, max_iter=50)
        res_all.append(r[0])
    assert len(res_all) == len(skels)
    for skel, r in zip(skels, res_all):
        assert r.constants.shape == (skel.n_constants,)
        assert np.isfinite(r.final_loss)


def test_fit_natives_fallback_on_overflow():
    """A K>MAX_K tree routes to the fallback (no kernel call needed)."""
    import sympy as sp
    n = MAX = kb.MAX_K + 2
    # linear chain c0 + c1 + ... (K = MAX_K+2 > 32) -> overflow -> fallback
    x0 = sp.Symbol("x0", real=True)
    cs = sp.symbols(f"c0:{n}", real=True)
    expr = x0
    for c in cs:
        expr = expr + c
    from cusr.bench.skeleton import Skeleton
    skel = Skeleton(expr=expr, variables=(x0,), constants=tuple(cs))
    # build matching prefix arrays: ((...((x0 + c0) + c1) ...) + c_{n-1})
    nt = [popio.NTYPE_BFUNC] * n
    nv = [interp.F.ADD] * n
    ci = [-1] * n
    # left-deep: emit ADD nodes then the x0 leaf, then consts as right children.
    # simplest valid prefix for left-deep add chain:
    nt_arr, nv_arr, ci_arr = [], [], []

    def emit(k):
        if k < 0:
            nt_arr.append(popio.NTYPE_VAR); nv_arr.append(0.0); ci_arr.append(-1)
            return
        nt_arr.append(popio.NTYPE_BFUNC); nv_arr.append(interp.F.ADD); ci_arr.append(-1)
        emit(k - 1)
        nt_arr.append(popio.NTYPE_CONST); nv_arr.append(0.0); ci_arr.append(k)
    emit(n - 1)
    native = _ArrayTree(np.array(nt_arr, np.int32), np.array(nv_arr, np.float32),
                        np.array(ci_arr, np.int32), np.ones(n, np.float32))
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = np.zeros(20)
    results, stats = kb.fit_natives([skel], [np.ones(n)], [native], X, y,
                                    binary=kb.DEFAULT_BINARY, max_iter=10,
                                    fallback=ScipyLM())
    assert stats["n_fallback"] == 1 and stats["n_kernel"] == 0
    assert stats["n_overflow"] == 1
    assert results[0] is not None


@evogp_gate
def test_extract_vs_skeleton_function_agreement():
    """THE load-bearing invariant: the kernel's input view (`_extract_tree`) and the
    scipy/writeback/judge view (`forest_member_to_skeleton`) compute the SAME function
    for every live EvoGP tree. They are two separate walks with separate degrade maps;
    any drift means the kernel optimizes fn A while writeback/judge assume fn B → a
    p=1.0 false-null. Random forest, full 009 FUNCS, SIGNED inputs (exposes any Abs(x)
    divergence). Loose sqrt/pow (the one known divergence) are routed to the fallback
    by the bridge, so they must not reach the kernel — asserted via _has_divergent_loose.
    """
    import torch
    from cusr.bench.sources.evogp import forest_member_to_skeleton
    from evogp.tree import Forest, GenerateDescriptor

    n_vars = 2
    FUNCS = {"+": 1., "-": 1., "*": 1., "/": 1., "sin": .5, "cos": .5,
             "exp": .3, "log": .3, "sqrt": .3, "tanh": .2}
    desc = GenerateDescriptor(max_tree_len=64, input_len=n_vars, output_len=1,
                              using_funcs=FUNCS, max_layer_cnt=6,
                              const_range=[-5., 5.], sample_cnt=10000, layer_leaf_prob=0.3)
    torch.manual_seed(0)
    forest = Forest.random_generate(pop_size=512, descriptor=desc)
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (64, n_vars))  # signed -> exposes sqrt(|x|)/|x|^y divergence
    n_checked = 0
    for m in range(len(forest)):
        nat = forest[m]
        if kb._has_divergent_loose(nat):
            continue  # bridge fallbacks these; not a kernel tree
        ext = kb.extract_native(nat, n_vars)
        if ext is None:
            continue
        try:
            skel, _init, _deg = forest_member_to_skeleton(nat, n_vars)
        except Exception:  # noqa: BLE001
            continue
        if skel.n_constants == 0:
            continue
        nt, nv, ci, c_init = ext
        assert len(c_init) == skel.n_constants, "const-count drift: extract vs skeleton"
        for _ in range(3):
            c = rng.uniform(-3, 3, skel.n_constants)
            try:
                with np.errstate(all="ignore"):  # NaN on sqrt/log of negatives is expected & masked
                    a = interp.eval_tree_arrays(nt, nv, ci, X, c)
                    b = skel.evaluate(c, X)
            except Exception:  # noqa: BLE001 — degenerate (zoo) skeletons; pipeline guards these too
                continue
            mf = np.isfinite(a) & np.isfinite(b)
            if mf.sum() < 8:
                continue
            rel = np.abs(a[mf] - b[mf]) / (np.abs(b[mf]) + 1e-9)
            assert np.median(rel) < 1e-5, \
                f"FUNCTION DRIFT extract vs skeleton (med={np.median(rel):.1e}): " \
                f"{interp.to_infix(nt, nv, ci)}"
        n_checked += 1
    assert n_checked >= 100, f"only {n_checked} trees checked — sample too small to gate"


# --------------------------------------------------------------------------- record

if __name__ == "__main__":
    if not _have_binary:
        print("batch_lm binary not built — skipping parity table")
        sys.exit(0)
    pop = popio.load_pop_bin(_SYNTH)
    rows = _parity_on_pop(pop, 24)
    n_ok = sum(lk <= _TIER_B * ls + _LOSS_FLOOR for lk, ls, _li in rows)
    n_dir = sum(lk <= 1.01 * li + 1e-9 for lk, _ls, li in rows)
    print(f"\nsynth inner-const-heavy, first 24 trees, kernel(fp32) vs scipy(fp64), mean r^2:")
    print(f"  direction-correct (loss<=init): {n_dir}/{len(rows)}")
    print(f"  tier-B (<=1.05x scipy):         {n_ok}/{len(rows)}")
    worst = sorted(rows, key=lambda r: -(r[0] / (r[1] + _LOSS_FLOOR)))[:5]
    for lk, ls, li in worst:
        print(f"    init={li:.3e}  kernel={lk:.3e}  scipy={ls:.3e}  ratio={lk/(ls+_LOSS_FLOOR):.2f}")
