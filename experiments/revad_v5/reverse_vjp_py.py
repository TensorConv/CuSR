#!/usr/bin/env python3
"""fp32 Python mirror of eval_tree_vjp_d (revad_interp.cuh) — to PROTOTYPE the NaN-safety
fix before touching CUDA. Runs the reverse VJP with SAFE_MUL on/off over jac_sample.jsonl,
classifies each residual's mechanism, and reports whether safe-multiply zeroes ALL 36
shared-NaN-with-finite-truth elements (the GPU re-dump is the final arbiter).

Mirrors the C EXACTLY: float32 throughout, value stack reverse-scan i=n-1..0 then adjoint
sweep i=0..n-1, d1/d2 tape, same op rules. SAFE_MUL replaces `adj * partial` with a
0-annihilating multiply.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_scipy import eval_tree as eval_tree_f64, scipy_grad

f32 = np.float32
N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_MAX, F_MIN, F_LT, F_GT, F_LE, F_GE = 8, 9, 10, 11, 12, 13
F_SIN, F_COS, F_TAN, F_SINH, F_COSH, F_TANH = 14, 15, 16, 17, 18, 19
F_LOG, F_EXP, F_INV, F_NEG, F_ABS, F_SQRT = 20, 22, 23, 25, 26, 27

def smul(a, b, safe):
    a = f32(a); b = f32(b)
    if safe and (a == f32(0.0) or b == f32(0.0)):
        return f32(0.0)
    return f32(a * b)

def vjp(nt, nv, ci, x, c, K, safe):
    """Returns out_grad[K] in float32, mirroring eval_tree_vjp_d."""
    n = len(nt)
    sv = []
    d1 = [f32(0.0)] * n
    d2 = [f32(0.0)] * n
    # forward: values + tape
    for i in range(n - 1, -1, -1):
        t = nt[i]; v = nv[i]
        if t == N_VAR:
            sv.append(f32(x[int(v)]))
        elif t == N_CONST:
            sv.append(f32(c[ci[i]]))
        elif t == N_UFUNC:
            a = sv.pop(); fid = int(v); a = f32(a)
            if   fid == F_SIN:  r = np.sin(a);  d = np.cos(a)
            elif fid == F_COS:  r = np.cos(a);  d = -np.sin(a)
            elif fid == F_TAN:  r = np.tan(a);  d = f32(1.0) + r*r
            elif fid == F_SINH: r = np.sinh(a); d = np.cosh(a)
            elif fid == F_COSH: r = np.cosh(a); d = np.sinh(a)
            elif fid == F_TANH: r = np.tanh(a); d = f32(1.0) - r*r
            elif fid == F_LOG:  r = np.log(a);  d = f32(1.0)/a
            elif fid == F_EXP:  r = np.exp(a);  d = r
            elif fid == F_INV:  r = f32(1.0)/a; d = -r*r
            elif fid == F_NEG:  r = -a;         d = f32(-1.0)
            elif fid == F_ABS:  r = np.abs(a);  d = f32(1.0) if a>0 else (f32(-1.0) if a<0 else f32(0.0))
            elif fid == F_SQRT: r = np.sqrt(a); d = f32(1.0)/(f32(2.0)*r)
            else: r = f32(0.0); d = f32(0.0)
            d1[i] = f32(d); sv.append(f32(r))
        elif t == N_BFUNC:
            l = f32(sv.pop()); rv = f32(sv.pop()); fid = int(v)
            if   fid == F_ADD: o=l+rv; dl=f32(1.0);   dr=f32(1.0)
            elif fid == F_SUB: o=l-rv; dl=f32(1.0);   dr=f32(-1.0)
            elif fid == F_MUL: o=l*rv; dl=rv;          dr=l
            elif fid == F_DIV: o=l/rv; dl=f32(1.0)/rv; dr=-o/rv
            elif fid == F_POW: o=np.power(l,rv); dl=rv*np.power(l,rv-f32(1.0)); dr=o*np.log(l)
            elif fid == F_MAX: o=np.maximum(l,rv); dl=f32(1.0) if l>=rv else f32(0.0); dr=f32(0.0) if l>=rv else f32(1.0)
            elif fid == F_MIN: o=np.minimum(l,rv); dl=f32(1.0) if l<=rv else f32(0.0); dr=f32(0.0) if l<=rv else f32(1.0)
            elif fid == F_LT:  o=f32(1.0 if l<rv else 0.0);  dl=f32(0.0); dr=f32(0.0)
            elif fid == F_GT:  o=f32(1.0 if l>rv else 0.0);  dl=f32(0.0); dr=f32(0.0)
            elif fid == F_LE:  o=f32(1.0 if l<=rv else 0.0); dl=f32(0.0); dr=f32(0.0)
            elif fid == F_GE:  o=f32(1.0 if l>=rv else 0.0); dl=f32(0.0); dr=f32(0.0)
            else: o=f32(0.0); dl=f32(0.0); dr=f32(0.0)
            d1[i]=f32(dl); d2[i]=f32(dr); sv.append(f32(o))
    # backward
    out = [f32(0.0)] * K
    sa = [f32(1.0)]
    for i in range(n):
        t = nt[i]
        if t == N_VAR:
            sa.pop()
        elif t == N_CONST:
            adj = sa.pop(); out[ci[i]] = f32(out[ci[i]] + adj)
        elif t == N_UFUNC:
            adj = sa.pop(); sa.append(smul(adj, d1[i], safe))
        elif t == N_BFUNC:
            adj = sa.pop()
            sa.append(smul(adj, d2[i], safe))
            sa.append(smul(adj, d1[i], safe))
    return out

def run(safe):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jac_sample.jsonl")
    trees = [json.loads(line) for line in open(path) if line.strip()]
    residual = 0; fixed = 0; broke = 0; total_finite_truth = 0; rev_ok = 0
    per_tree = {}
    for T in trees:
        nt, nv, ci, c = T["nt"], T["nv"], T["ci"], np.array(T["c"], dtype=np.float64)
        K = T["K"]
        for P in T["pts"]:
            x = P["x"]
            old_rev = np.array(P["rev"], dtype=np.float64)
            new_rev = vjp(nt, nv, ci, x, c, K, safe)
            val = float(eval_tree_f64(nt, nv, ci, x, c))
            ref = scipy_grad(nt, nv, ci, x, c)
            for k in range(K):
                g = ref[k]
                if np.isfinite(val) and np.isfinite(g):
                    total_finite_truth += 1
                    nr = float(new_rev[k])
                    ok = (np.isfinite(nr) and abs(nr - g) <= 1e-4 + 2e-2*abs(g))
                    if ok: rev_ok += 1
                    else:
                        residual += 1
                        per_tree[T["m"]] = per_tree.get(T["m"], 0) + 1
                    # did safe_mul FIX a previously-nan element?
                    if not np.isfinite(old_rev[k]) and ok:
                        fixed += 1
                    if np.isfinite(old_rev[k]) and not ok:
                        broke += 1
    return residual, fixed, broke, total_finite_truth, rev_ok, per_tree

def main():
    for safe in (False, True):
        r, fx, bk, tot, ok, pt = run(safe)
        tag = "SAFE_MUL ON " if safe else "SAFE_MUL OFF"
        print(f"[{tag}] finite-truth elements={tot}  rev_ok={ok}  RESIDUAL(rev wrong)={r}"
              f"  fixed_vs_old={fx}  broke_vs_old={bk}")
        if r:
            print(f"            residual by tree m: {pt}")
    print("\n(If SAFE_MUL ON -> RESIDUAL=0 and broke=0, the masked multiply is sufficient & non-regressive on the sample.)")

if __name__ == "__main__":
    main()
