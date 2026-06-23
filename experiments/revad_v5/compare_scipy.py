#!/usr/bin/env python3
"""Independent fp64 oracle for the reverse-AD vs forward-AD divergence.

Reads jac_sample.jsonl (dumped from the REAL pop by _dump_jac_sample.cu, carrying the
ACTUAL forward-AD `fwd[]` and reverse-AD `rev[]` gradients) and, for every element,
computes the ground-truth Jacobian with scipy.optimize._numdiff.approx_derivative in
fp64 (an interpreter reimplemented independently in numpy fp64). For each (tree, point,
column) it decides which of fwd / rev matches scipy, and — crucially — whether the tree
VALUE at that point is finite (so a finite gradient is meaningful) or NaN/Inf (so the
function is genuinely undefined and any finite gradient is spurious).

Verdict logic for the disputed elements (where fwd and rev disagree on finiteness):
  - value finite & scipy finite & rev~=scipy & fwd nonfinite  -> REVERSE correct (fwd over-NaNs)
  - value nonfinite (NaN/Inf)                                  -> function undefined; fwd's NaN defensible
"""
import json, sys, warnings
import numpy as np
from scipy.optimize._numdiff import approx_derivative

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_MAX, F_MIN, F_LT, F_GT, F_LE, F_GE = 8, 9, 10, 11, 12, 13
F_SIN, F_COS, F_TAN, F_SINH, F_COSH, F_TANH = 14, 15, 16, 17, 18, 19
F_LOG, F_EXP, F_INV, F_NEG, F_ABS, F_SQRT = 20, 22, 23, 25, 26, 27

def eval_tree(nt, nv, ci, x, c):
    """fp64 reverse-scan prefix stack machine; mirrors eval_tree_val_host semantics."""
    st = []
    for i in range(len(nt) - 1, -1, -1):
        t = nt[i]; v = nv[i]
        if t == N_VAR:
            st.append(np.float64(x[int(v)]))
        elif t == N_CONST:
            st.append(np.float64(c[ci[i]]))
        elif t == N_UFUNC:
            a = st.pop(); fid = int(v)
            if   fid == F_SIN:  r = np.sin(a)
            elif fid == F_COS:  r = np.cos(a)
            elif fid == F_TAN:  r = np.tan(a)
            elif fid == F_SINH: r = np.sinh(a)
            elif fid == F_COSH: r = np.cosh(a)
            elif fid == F_TANH: r = np.tanh(a)
            elif fid == F_LOG:  r = np.log(a)
            elif fid == F_EXP:  r = np.exp(a)
            elif fid == F_INV:  r = np.float64(1.0) / a
            elif fid == F_NEG:  r = -a
            elif fid == F_ABS:  r = np.abs(a)
            elif fid == F_SQRT: r = np.sqrt(a)
            else: r = np.float64(0.0)
            st.append(r)
        elif t == N_BFUNC:
            l = st.pop(); rv = st.pop(); fid = int(v)
            if   fid == F_ADD: o = l + rv
            elif fid == F_SUB: o = l - rv
            elif fid == F_MUL: o = l * rv
            elif fid == F_DIV: o = l / rv
            elif fid == F_POW: o = np.power(l, rv)
            elif fid == F_MAX: o = np.maximum(l, rv)
            elif fid == F_MIN: o = np.minimum(l, rv)
            elif fid == F_LT:  o = np.float64(1.0 if l <  rv else 0.0)
            elif fid == F_GT:  o = np.float64(1.0 if l >  rv else 0.0)
            elif fid == F_LE:  o = np.float64(1.0 if l <= rv else 0.0)
            elif fid == F_GE:  o = np.float64(1.0 if l >= rv else 0.0)
            else: o = np.float64(0.0)
            st.append(o)
    return st[0]

def scipy_grad(nt, nv, ci, x, c):
    c = np.asarray(c, dtype=np.float64)
    fun = lambda cc: np.array([eval_tree(nt, nv, ci, x, cc)], dtype=np.float64)
    try:
        J = approx_derivative(fun, c, method="3-point")
        return J.reshape(-1)
    except Exception as e:
        return np.full(len(c), np.nan)

def close(a, b, rtol=2e-2, atol=1e-4):
    # fp32 AD vs fp64 numerical central diff -> loose-ish tol; both-nonfinite counts as equal.
    if not np.isfinite(a) and not np.isfinite(b): return True
    if np.isfinite(a) != np.isfinite(b): return False
    return abs(a - b) <= atol + rtol * abs(b)

def classify(path):
    """Compare dumped forward/reverse-AD gradients against the scipy fp64 oracle.

    Pure: reads the jsonl sample, returns a dict of counters (no printing) so
    tests can assert on it. Keys: n_trees, agreed_n/agreed_rev_ok/agreed_fwd_ok,
    n_disputed, rev_matches_scipy, fwd_matches_scipy, neither,
    rev_right_valfinite, fwd_right_valnan, examples.
    """
    trees = [json.loads(line) for line in open(path) if line.strip()]
    r = dict(
        n_trees=len(trees), path=str(path),
        agreed_n=0, agreed_rev_ok=0, agreed_fwd_ok=0,
        n_disputed=0, rev_matches_scipy=0, fwd_matches_scipy=0, neither=0,
        rev_right_valfinite=0,   # value finite, scipy finite, rev=scipy, fwd nonfinite -> reverse correct
        fwd_right_valnan=0,      # value nonfinite -> function undefined, fwd's NaN defensible
        disp_fwd_fin_rev_nan=0,  # disputed where FORWARD finite but reverse NaN (reverse worse — should be 0)
        disp_fwd_nan_rev_fin=0,  # disputed where forward NaN but reverse finite (reverse finite-safe; the v5 win)
        examples=[],
    )
    for T in trees:
        nt, nv, ci, c = T["nt"], T["nv"], T["ci"], np.array(T["c"], dtype=np.float64)
        K = T["K"]
        for P in T["pts"]:
            x = P["x"]
            fwd = np.array(P["fwd"], dtype=np.float64)
            rev = np.array(P["rev"], dtype=np.float64)
            val = float(eval_tree(nt, nv, ci, x, c))
            ref = scipy_grad(nt, nv, ci, x, c)
            val_finite = np.isfinite(val)
            for k in range(K):
                f, rv, g = fwd[k], rev[k], ref[k]
                disputed = (np.isfinite(f) != np.isfinite(rv))
                if disputed:
                    r["n_disputed"] += 1
                    if np.isfinite(f) and not np.isfinite(rv):
                        r["disp_fwd_fin_rev_nan"] += 1
                    else:
                        r["disp_fwd_nan_rev_fin"] += 1
                    rok, fok = close(rv, g), close(f, g)
                    if rok: r["rev_matches_scipy"] += 1
                    if fok: r["fwd_matches_scipy"] += 1
                    if not rok and not fok: r["neither"] += 1
                    if val_finite and np.isfinite(g) and rok and not np.isfinite(f):
                        r["rev_right_valfinite"] += 1
                    if not val_finite:
                        r["fwd_right_valnan"] += 1
                    if len(r["examples"]) < 14:
                        r["examples"].append((T["m"], P["i"], k, val, f, rv, g, val_finite))
                else:
                    if np.isfinite(f) and np.isfinite(rv):
                        r["agreed_n"] += 1
                        if close(rv, g): r["agreed_rev_ok"] += 1
                        if close(f, g): r["agreed_fwd_ok"] += 1
    return r


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/weish/hao/CuSR/experiments/revad_v5/jac_sample.jsonl"
    r = classify(path)
    print(f"loaded {r['n_trees']} sample trees from {path}\n")

    print("=== AGREED elements (fwd & rev both finite) — sanity vs scipy ===")
    print(f"  n={r['agreed_n']}  rev~scipy={r['agreed_rev_ok']}  fwd~scipy={r['agreed_fwd_ok']}\n")

    print("=== DISPUTED elements (fwd vs rev differ in finiteness) ===")
    print(f"  n_disputed                 = {r['n_disputed']}")
    print(f"  rev matches scipy          = {r['rev_matches_scipy']}")
    print(f"  fwd matches scipy          = {r['fwd_matches_scipy']}")
    print(f"  neither matches scipy      = {r['neither']}")
    print(f"  -> value FINITE & rev=scipy & fwd nonfinite (REVERSE correct) = {r['rev_right_valfinite']}")
    print(f"  -> value NON-finite (function undefined; fwd NaN defensible)  = {r['fwd_right_valnan']}\n")

    print("=== sample disputed elements  [m,i,k] value  fwd  rev  scipy  (val_finite) ===")
    for (m, i, k, val, f, rv, g, vf) in r["examples"]:
        print(f"  m={m:<5} i={i:<4} k={k}  val={val:<12.5g} fwd={f:<10.5g} rev={rv:<12.5g} scipy={g:<12.5g} valfin={vf}")

    print()
    nd = r["n_disputed"]
    if nd == 0:
        print("VERDICT: no disputed elements in sample.")
    elif r["rev_matches_scipy"] >= r["fwd_matches_scipy"] and r["rev_right_valfinite"] > 0 and r["fwd_right_valnan"] == 0:
        print("VERDICT: REVERSE-AD matches the scipy fp64 oracle; forward-AD over-NaNs (NaN-contamination). Reverse is MORE correct.")
    elif r["fwd_right_valnan"] == nd:
        print("VERDICT: all disputes are at NON-finite tree values (function undefined); forward's NaN is defensible -> reverse should replicate (poison fix).")
    else:
        print("VERDICT: MIXED — see breakdown above; decide per-category.")

if __name__ == "__main__":
    main()
