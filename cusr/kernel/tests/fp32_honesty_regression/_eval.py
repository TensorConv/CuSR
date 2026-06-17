"""_eval.py — self-contained fp64 reference evaluator for the fp32-honesty regression.

Mirrors the kernel eval_tree_d / pop_ref.ref_eval (reverse-prefix stack machine) but in
EXACT fp64 — this is the neutral arbiter the regression uses to judge whether a kernel's
returned coefficients are genuinely good (vs. only good in the kernel's own fast-math fp32).
Pure numpy so the regression has no dependency on the results/ analysis tree.
"""
from __future__ import annotations

import numpy as np

_UFUNC = {14: np.sin, 15: np.cos, 16: np.tan}


def eval_fp64(nt, nv, ci, c, X):
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
        elif t == 3:    # BFUNC [op, A, B] -> A op B (A = left/first popped)
            l = st.pop(); rv = st.pop(); f = int(v)
            if f == 1:   st.append(l + rv)
            elif f == 2: st.append(l - rv)
            elif f == 3: st.append(l * rv)
            elif f == 4: st.append(l / rv)
            else: raise ValueError(f"bad BFUNC id {f}")
        else:
            raise ValueError(f"bad node type {t}")
    return st[0]


def half_sse(nt, nv, ci, c, X, y):
    """The neutral objective: 0.5 * sum((f(x)-y)^2), fp64; inf if non-finite."""
    with np.errstate(all="ignore"):
        pred = eval_fp64(nt, nv, ci, c, X)
        if not np.all(np.isfinite(pred)):
            return float("inf")
        return 0.5 * float(np.sum((pred - np.asarray(y, np.float64)) ** 2))
