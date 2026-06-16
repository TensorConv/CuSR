"""pop_ref.py — host reference interpreter, mirrors the kernel exactly.

``ref_eval(nt, nv, ci, c_init, X)`` evaluates a single prefix-stored tree the
same way the CUDA kernel ``eval_tree_d`` (batch_lm.cu) does: iterate
``i = n-1 .. 0`` with a stack. A binary node stored prefix as ``[op, A, B]`` is
processed with B's subtree first (lands on stack), then A's (on top); the op pops
``l = A`` (left/first operand), ``rv = B`` (right/second) and computes ``A OP B``.
So A is the numerator/minuend.

Supported Func ids: ADD=1, SUB=2, MUL=3, DIV=4 (binary); SIN=14, COS=15, TAN=16
(unary) — the restricted grammar (spec §4/§6). This is the golden oracle the
round-trip adapter test compares Operon's own evaluator against.
"""
from __future__ import annotations

import numpy as np

# NType tags (spec §2)
N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3

_UFUNC = {14: np.sin, 15: np.cos, 16: np.tan}


def ref_eval(nt, nv, ci, c_init, X):
    """Evaluate one prefix tree on ``X`` (shape ``(N, n_vars)``, float32).

    Reverse-iteration stack machine, byte-faithful to the kernel. Returns a
    float32 array of length ``N``.
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(nt)
    st = []
    for i in range(n - 1, -1, -1):
        t = int(nt[i])
        v = nv[i]
        if t == N_VAR:
            st.append(X[:, int(v)].astype(np.float32))
        elif t == N_CONST:
            st.append(np.full(X.shape[0], c_init[int(ci[i])], np.float32))
        elif t == N_UFUNC:
            a = st.pop()
            st.append(_UFUNC[int(v)](a).astype(np.float32))
        elif t == N_BFUNC:
            l = st.pop()
            rv = st.pop()
            f = int(v)
            if f == 1:
                st.append(l + rv)
            elif f == 2:
                st.append(l - rv)
            elif f == 3:
                st.append(l * rv)
            elif f == 4:
                st.append(l / rv)
            else:
                raise ValueError(f"ref_eval: unsupported BFUNC id {f}")
        else:
            raise ValueError(f"ref_eval: unsupported node type {t} at index {i}")
    return st[0]
