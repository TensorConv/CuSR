"""kernel_bridge.py — marshal native EvoGP trees -> 008 pop.bin -> batch_lm kernel.

W5 (E2 demonstrator) needs the 008 CUDA LM kernel as an in-loop CO backend. The
009 memetic pipeline already holds the live EvoGP `Tree` at the CO call site, so
we take the **tree -> bytecode** path (not sympy Skeleton -> bytecode):

  - reuse `dump_evogp._extract_tree` — the SAME extraction that produced every
    pop.bin the 012 E1 harness benchmarks, so E2 runs the identical kernel path
    (the "single frozen kernel" discipline of the plan, for free);
  - reuse `popio.build_pop` / `save_pop_bin` — the validated byte format;
  - only the subprocess call + per-tree slice/align is new here.

EvoGP trees carry only VAR / CONST / FUNC nodes (no literal-number node), so the
structural-literal problem of the sympy path does not arise. Per-tree the CONST
count equals the skeleton's `n_constants` (both prefix-forward over the same
nodes) and `ci` rank == prefix-DFS order == `_writeback_constants` order, so the
kernel's `c_final` slots back with no remapping.

Trees the kernel cannot take (TFUNC, K>MAX_K, stack>MAX_STACK) fall back to the
provided backend (scipy by default) so no candidate is silently dropped; the
fallback count is returned for honest reporting.
"""
from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

import cusr.kernel
from cusr.benchmark import interp  # vectorized oracle + opcodes
from cusr.benchmark import popio  # validated pop.bin writer
from cusr.kernel.dump_evogp import _extract_tree  # validated tree extractor

from .co_backend import COResult, ScipyLM

# Repo-root data dir. Kept as `_DIR_012` for back-compat with consumers that
# join `_DIR_012 / "workload" / "synth" / ...` (e.g. test_kernel_bridge.py).
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DIR_012 = _ROOT / "data"

# Must match the batch_lm binary's compile-time flags (batch_lm.cu:48-49). The
# binary hard-errors if a batch exceeds these, so we filter per-tree first.
MAX_K = 32
MAX_STACK = 64
DEFAULT_BINARY = pathlib.Path(cusr.kernel.__file__).parent / "batch_lm"


class KernelRunError(RuntimeError):
    pass


def extract_native(native, n_vars: int):
    """Native EvoGP `Tree` (or array-shim) -> (nt, nv, ci, c_init) or None.

    None means the tree is not kernel-encodable (TFUNC present, or extraction
    failed) -> caller routes it to the fallback backend.
    """
    try:
        return _extract_tree(native, n_vars)
    except Exception:  # noqa: BLE001 — a malformed tree must not crash the gen
        return None


def _kernel_eligible(nt) -> bool:
    """K and interpreter stack within the binary's compile-time limits."""
    K = int(np.sum(np.asarray(nt) == popio.NTYPE_CONST))
    if K == 0 or K > MAX_K:
        return False
    return popio._sim_stack_depth(nt) <= MAX_STACK


# LOOSE_SQRT/LOOSE_POW are the one place the two tree-walks DISAGREE: dump_evogp's
# DEGRADE_OP sends them to plain SQRT/POW (kernel), but forest_member_to_skeleton
# keeps EvoGP's SYMPY_MAP semantics sqrt(|x|) / |x|^y (the function scipy fits,
# writeback assumes, and the judge sees). On negative args those differ, so the
# kernel would optimize the wrong function -> route such trees to the fallback.
# (Empirically these ops don't appear with the 009 FUNCS set; this is defence in
# depth so a config/mutation change can't silently produce a false-null.)
_DIVERGENT_LOOSE = {int(interp.F.LOOSE_SQRT), int(interp.F.LOOSE_POW)}


def _has_divergent_loose(native) -> bool:
    try:
        n = int(native.subtree_size[0].item())
        nv = native.node_value[:n].detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        return False
    return bool(_DIVERGENT_LOOSE.intersection(int(v) for v in nv))


def _run_kernel(pop: dict, binary, max_iter):
    """pop dict -> (c_final float64 (total_c,), raw_status int32 (M,))."""
    binary = pathlib.Path(binary)
    if not binary.exists():
        raise KernelRunError(f"batch_lm binary not found: {binary}")
    with tempfile.TemporaryDirectory(prefix="co_kernel_") as td:
        td = pathlib.Path(td)
        pop_path = td / "pop.bin"
        popio.save_pop_bin(pop, pop_path)
        cmd = [str(binary), str(pop_path), str(td), "--quiet"]
        if max_iter is not None:
            cmd += ["--max-iter", str(int(max_iter))]
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8")
        if r.returncode != 0:
            raise KernelRunError(f"exit={r.returncode}: {(r.stderr or r.stdout)[-500:]}")
        raw_status = np.fromfile(td / "status.bin", dtype=np.int32)
        c_final = np.fromfile(td / "c_final.bin", dtype=np.float32).astype(np.float64)
    return c_final, raw_status


def _variant_for_binary(binary) -> str:
    """The in-process .so variant that matches the standalone `binary`.

    Byte-parity is per-variant (the FD .so single-sources the FD device kernels;
    the AD .so the AD ones). The inproc drop-in must use the SAME variant as the
    subprocess it replaces, so `fit_natives(inproc=True)` is byte-identical to
    `inproc=False` for whatever `binary` the caller passed. `batch_lm_ad` -> 'ad';
    everything else (incl. the default `batch_lm`) -> 'fd'."""
    return "ad" if pathlib.Path(binary).name.endswith("_ad") else "fd"


def _run_kernel_inproc(pop: dict, binary, max_iter, device_id):
    """In-process twin of `_run_kernel`: same (c_final float64, raw_status int32)
    contract, but via the persistent ctypes handle (no subprocess, no disk). The
    CUDA primary context is paid ONCE per process (get_inproc_co singleton)."""
    from cusr.kernel.co_inproc import get_inproc_co

    variant = _variant_for_binary(binary)
    co = get_inproc_co(device_id=device_id, variant=variant)
    out = co.optimize(pop, max_iter=max_iter)
    # optimize() returns float32 c_final; match _run_kernel's float64 cast so the
    # downstream slot/loss math is bit-identical to the subprocess path.
    c_final = np.asarray(out["c_final"], dtype=np.float32).astype(np.float64)
    raw_status = np.asarray(out["status"], dtype=np.int32)
    return c_final, raw_status


def fit_natives(
    skeletons,
    inits,
    natives,
    X,
    y,
    *,
    binary=DEFAULT_BINARY,
    max_iter: int = 50,
    fallback=None,
    inproc: bool = False,
    device_id: int = 0,
) -> tuple[list[COResult], dict]:
    """Fit each skeleton's constants via the 008 kernel; return (results, stats).

    `natives[j]` is the live EvoGP `Tree` for `skeletons[j]` (read-only here —
    extracted before any writeback). Results are aligned with `skeletons`; the
    kernel-final loss is recomputed fp64 via `skel.residual` to match the scipy /
    torch loss convention exactly (apples-to-apples). `stats` reports how many
    trees went to the kernel vs the fallback (and why).

    `inproc=True` replaces the per-gen subprocess (`_run_kernel`) with the
    persistent in-process ctypes handle (`co_inproc.get_inproc_co(device_id)`),
    paying the CUDA primary-context init ONCE per process (P1). The variant tracks
    `binary` (FD/AD), so the inproc result is byte-identical to the subprocess it
    replaces. `inproc=False` keeps the subprocess baseline path. The per-tree
    eligibility filtering + scipy-fallback routing is identical on both paths, so
    no candidate is dropped either way (MUST-FIX #7)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_vars = X.shape[1]
    M = len(skeletons)
    results: list = [None] * M

    kernel_slots: list[tuple[int, tuple]] = []  # (orig_idx, extracted arrays)
    fb_idx: list[int] = []
    n_tfunc = n_overflow = n_kmismatch = n_loose_div = 0
    for j, (skel, nat) in enumerate(zip(skeletons, natives)):
        ext = extract_native(nat, n_vars)
        if ext is None:
            n_tfunc += 1
            fb_idx.append(j)
            continue
        if _has_divergent_loose(nat):
            # kernel-vs-skeleton semantics diverge on these (see _DIVERGENT_LOOSE)
            n_loose_div += 1
            fb_idx.append(j)
            continue
        nt, nv, ci, c_init = ext
        if not _kernel_eligible(nt):
            n_overflow += 1
            fb_idx.append(j)
            continue
        if len(c_init) != skel.n_constants:
            # Invariant guard: prefix CONST count must equal the skeleton's K.
            n_kmismatch += 1
            fb_idx.append(j)
            continue
        kernel_slots.append((j, ext))

    if kernel_slots:
        trees = [ext for _, ext in kernel_slots]
        ym = np.tile(y, (len(trees), 1))
        pop = popio.build_pop(trees, X, ym)
        if inproc:
            c_final, raw_status = _run_kernel_inproc(pop, binary, max_iter, device_id)
        else:
            c_final, raw_status = _run_kernel(pop, binary, max_iter)
        for slot, (j, _ext) in enumerate(kernel_slots):
            _, _, c_off, K = pop["metas"][slot].tolist()
            c = np.asarray(c_final[c_off:c_off + K], dtype=float)
            skel = skeletons[j]
            try:
                loss = float(np.mean(skel.residual(c, X, y) ** 2))
            except Exception:  # noqa: BLE001
                loss = float("inf")
            converged = bool(int(raw_status[slot]) == 0) if slot < len(raw_status) else False
            results[j] = COResult(c, -1, converged, loss)

    if fb_idx:
        fb = fallback if fallback is not None else ScipyLM()
        fb_res = fb.fit_batch(
            [skeletons[j] for j in fb_idx],
            [inits[j] for j in fb_idx],
            X, y, max_iter=max_iter,
        )
        for j, res in zip(fb_idx, fb_res):
            results[j] = res

    stats = dict(
        n_total=M,
        n_kernel=len(kernel_slots),
        n_fallback=len(fb_idx),
        n_tfunc=n_tfunc,
        n_overflow=n_overflow,
        n_kmismatch=n_kmismatch,
        n_loose_div=n_loose_div,
    )
    return results, stats


# ---------------------------------------------------------------------------
# arrays_to_skeleton — inverse view (bytecode -> sympy Skeleton). Used by the
# parity gate to build a scipy-fittable skeleton from the same pop.bin arrays
# the kernel sees, so kernel-vs-scipy is a true same-problem comparison.
# ---------------------------------------------------------------------------

def arrays_to_skeleton(nt, nv, ci, n_vars: int):
    """Per-tree (nt, nv, ci) prefix arrays -> bench.skeleton.Skeleton.

    Reverse-prefix walk mirroring interp.eval_tree_arrays / forest_member_to_skeleton,
    building a sympy expr with x0.. vars and c0.. constants (in prefix-forward
    CONST order, matching the kernel's `ci` ranks).
    """
    import sympy as sp
    from cusr.bench.skeleton import Skeleton

    F = interp.F
    unary = {
        F.SIN: sp.sin, F.COS: sp.cos, F.TAN: sp.tan,
        F.SINH: sp.sinh, F.COSH: sp.cosh, F.TANH: sp.tanh,
        F.LOG: sp.log, F.EXP: sp.exp,
        F.INV: lambda a: 1 / a, F.NEG: lambda a: -a, F.ABS: sp.Abs,
        F.SQRT: sp.sqrt,
    }
    binary = {
        F.ADD: lambda l, r: l + r, F.SUB: lambda l, r: l - r,
        F.MUL: lambda l, r: l * r, F.DIV: lambda l, r: l / r,
        F.POW: lambda l, r: l ** r, F.MAX: sp.Max, F.MIN: sp.Min,
    }
    x_syms = sp.symbols(f"x0:{n_vars}", real=True)
    if n_vars == 1 and not isinstance(x_syms, tuple):
        x_syms = (x_syms,)
    K = int(np.sum(np.asarray(nt) == popio.NTYPE_CONST))
    c_syms = sp.symbols(f"c0:{K}", real=True) if K else ()
    if K == 1 and not isinstance(c_syms, tuple):
        c_syms = (c_syms,)

    stack: list = []
    for i in reversed(range(len(nt))):
        t = int(nt[i])
        if t == popio.NTYPE_VAR:
            stack.append(x_syms[int(nv[i])])
        elif t == popio.NTYPE_CONST:
            stack.append(c_syms[int(ci[i])])
        elif t == popio.NTYPE_UFUNC:
            stack.append(unary[int(nv[i])](stack.pop()))
        elif t == popio.NTYPE_BFUNC:
            l = stack.pop(); r = stack.pop()
            stack.append(binary[int(nv[i])](l, r))
        else:
            raise ValueError(f"unsupported node_type {t}")
    if len(stack) != 1:
        raise RuntimeError(f"malformed tree: stack depth {len(stack)}")
    return Skeleton(expr=stack[0], variables=x_syms, constants=tuple(c_syms))
