"""lib.py — shared primitives for the Operon-vs-kernel CO baseline.

THE NEUTRAL ARBITER. Every engine (Operon LMOptimizer, kernel FD, kernel AD)
starts from the SAME harvested ``c_init`` and returns optimized coefficients; we
score each one with the *identical* fp64 objective the kernel minimizes:

    loss = 0.5 * sum_i (f(x_i) - y_i)^2          # raw half-SSE, fp64

computed by ``half_sse`` below, on the SAME (X, y), with the SAME reverse-prefix
evaluator. Operon's own ``OptimizerSummary.FinalCost`` is NOT used for any quality
claim (it carries a reporting-scale quirk — InitialCost is 0.5*SSE but FinalCost
tracked ~SSE over train+test in probing); the neutral fp64 recompute sidesteps all
engine-internal cost conventions and is engine-agnostic by construction.

Pure numpy + stdlib so it imports under BOTH the operon venv and the main venv.
The evaluator mirrors ``cusr/benchmark/workload/pop_ref.ref_eval`` and the kernel
``eval_tree_d`` (reverse iteration, stack machine) but in fp64.
"""
from __future__ import annotations

import numpy as np

_UFUNC = {14: np.sin, 15: np.cos, 16: np.tan}


def eval_fp64(nt, nv, ci, c, X):
    """fp64 reverse-prefix stack eval of ONE tree. ``c`` is the per-tree K-vector;
    ``ci[i]`` is the per-tree-local const index (0..K-1). Mirrors the kernel."""
    X = np.asarray(X, dtype=np.float64)
    st = []
    for i in range(len(nt) - 1, -1, -1):
        t = int(nt[i]); v = nv[i]
        if t == 0:      # VAR
            st.append(X[:, int(v)].astype(np.float64))
        elif t == 1:    # CONST
            st.append(np.full(X.shape[0], float(c[int(ci[i])]), np.float64))
        elif t == 2:    # UFUNC
            st.append(_UFUNC[int(v)](st.pop()))
        elif t == 3:    # BFUNC: stored [op, A, B]; pops l=A then rv=B; computes A op B
            l = st.pop(); rv = st.pop(); f = int(v)
            if f == 1:   st.append(l + rv)
            elif f == 2: st.append(l - rv)
            elif f == 3: st.append(l * rv)
            elif f == 4: st.append(l / rv)
            else: raise ValueError(f"bad BFUNC id {f}")
        else:
            raise ValueError(f"bad node type {t} at {i}")
    return st[0]


def half_sse(nt, nv, ci, c, X, y):
    """The neutral objective: 0.5 * sum((f(x)-y)^2) in fp64. Returns float, or
    ``inf`` if the evaluation is non-finite (a tree pushed to a singularity)."""
    with np.errstate(all="ignore"):     # div-by-zero / overflow on singular trees -> inf (intended)
        pred = eval_fp64(nt, nv, ci, c, X)
        if not np.all(np.isfinite(pred)):
            return float("inf")
        return 0.5 * float(np.sum((pred - np.asarray(y, np.float64)) ** 2))


def iter_trees(pop):
    """Yield ``(m, nt_m, nv_m, ci_m, K_m, c_off, n_off)`` per tree from a
    ``read_pop_bin`` dict, slicing the flat arrays by ``metas``."""
    metas = pop["metas"]
    for m in range(pop["M_prob"]):
        n_off, n_nodes, c_off, K = (int(metas[m, 0]), int(metas[m, 1]),
                                    int(metas[m, 2]), int(metas[m, 3]))
        sl = slice(n_off, n_off + n_nodes)
        yield m, pop["nt"][sl], pop["nv"][sl], pop["ci"][sl], K, c_off, n_off


# ---- the matched experiment grid (mirrors cross_kernel.py for continuity) -------
PROBLEMS = ["feynman/I.12.1", "feynman/I.18.12", "feynman/I.27.6", "feynman/I.6.2",
            "feynman/I.12.2", "feynman/I.13.12", "feynman/II.3.24",
            "nguyen/1", "nguyen/2", "nguyen/3", "nguyen/4", "nguyen/5", "nguyen/6",
            "nguyen/7", "nguyen/8", "nguyen/9", "nguyen/10"]
GENS = [0, 4, 16, 64, 100]
CAP, SEED, NOISE, POP, N = 32, 0, 0.0, 4000, 1000


def committed_cell_dir(snap_root, prob):
    """Committed Operon corpus cell dir for (prob, cap32, noise0, seed0)."""
    from pathlib import Path
    safe = prob.replace("/", "_")
    return Path(snap_root) / f"operon_{safe}_pop4000_noise0_len32_seed0"
