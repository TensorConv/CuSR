"""probe_lm2.py — disambiguate Operon LMOptimizer cost/param semantics.

For a few trees: print InitialParameters/FinalParameters, the coeffs actually
carried by the returned tree, and raw 0.5*SSE evaluated at each, vs summary
Initial/Final cost. Determines the correct way to extract "the loss Operon achieved".
"""
import sys
from pathlib import Path
import numpy as np
import pyoperon as op

ROOT = Path("/home/weish/hao/CuSR"); sys.path.insert(0, str(ROOT))
from cusr.benchmark.workload.operon_dump import run_cell, _grammar_int
from cusr.benchmark.workload.operon_adapter import build_hash2idx, convert

_UF = {14: np.sin, 15: np.cos, 16: np.tan}
def eval_fp64(nt, nv, ci, c, X):
    X = np.asarray(X, np.float64); st = []
    for i in range(len(nt) - 1, -1, -1):
        t = int(nt[i]); v = nv[i]
        if t == 0:   st.append(X[:, int(v)].astype(np.float64))
        elif t == 1: st.append(np.full(X.shape[0], float(c[int(ci[i])]), np.float64))
        elif t == 2: st.append(_UF[int(v)](st.pop()))
        elif t == 3:
            l = st.pop(); rv = st.pop(); f = int(v)
            st.append(l+rv if f==1 else l-rv if f==2 else l*rv if f==3 else l/rv)
    return st[0]
def half_sse(nt, nv, ci, c, X, y):
    return 0.5 * float(np.sum((eval_fp64(nt,nv,ci,c,X) - np.asarray(y,np.float64))**2))

_, _, intern = run_cell(dataset="feynman/I.18.12", pop=4000, N=1000, seed=0, noise=0.0,
                        cap=32, checkpoint_gens=[0], out=Path("/tmp/probe_regen/pop.bin"),
                        threads=1, _return_internals=True)
snaps, hash2idx, X, y = intern["snaps"], intern["hash2idx"], intern["X"], intern["y"]
n_vars = X.shape[1]
ds = op.Dataset(np.asfortranarray(np.column_stack([X, y]).astype(np.float64)))
V = ds.Variables; inputs = [v.Hash for v in V[:n_vars]]
pr = op.Problem(ds); pr.TrainingRange = op.Range(0,1000); pr.TestRange = op.Range(0,1000)
pr.Target = V[n_vars]; pr.InputHashes = inputs; pr.ConfigurePrimitiveSet(_grammar_int())
dt = op.DispatchTable()
lm = op.LMOptimizer(dt, pr, max_iter=50); co = op.CoefficientOptimizer(lm)

print("Tree methods:", [m for m in dir(snaps[0][0]) if not m.startswith("_")])

picks = []
for t in snaps[0]:
    try: nt, nv, ci, c0 = convert(t, hash2idx)
    except Exception: continue
    K = len(c0)
    if K in (1, 3) and not any(p[0]==K for p in picks):
        picks.append((K, t, (nt,nv,ci,c0)))
    if len(picks) >= 2: break

for K, tree, (nt,nv,ci,c0) in picks:
    print(f"\n===== K={K} =====")
    raw0 = half_sse(nt,nv,ci,c0,X,y)
    otree, summ = co(op.RandomGenerator(0), tree)
    IP = [float(x) for x in summ.InitialParameters]
    FP = [float(x) for x in summ.FinalParameters]
    GC_after = [float(x) for x in otree.GetCoefficients()]
    # raw at otree's own coeffs (re-convert)
    nt1,nv1,ci1,c1 = convert(otree, hash2idx)
    raw_otree = half_sse(nt1,nv1,ci1,c1,X,y)
    # raw at FinalParameters forced into the tree
    try:
        otree.SetCoefficients(FP)
        ntf,nvf,cif,cf = convert(otree, hash2idx)
        raw_FP = half_sse(ntf,nvf,cif,cf,X,y)
        set_ok = True
    except Exception as e:
        raw_FP = float("nan"); set_ok = False; print("  SetCoefficients failed:", repr(e))
    print(f"  IP            = {IP}")
    print(f"  FP            = {FP}")
    print(f"  otree coeffs  = {GC_after}")
    print(f"  InitialCost   = {summ.InitialCost:.6g}   raw0(0.5SSE@c0)      = {raw0:.6g}")
    print(f"  FinalCost     = {summ.FinalCost:.6g}   raw@otree           = {raw_otree:.6g}")
    if set_ok:
        print(f"                              raw@FinalParameters  = {raw_FP:.6g}")
    print(f"  Iterations={summ.Iterations} Success={summ.Success} "
          f"FuncEvals={summ.FunctionEvaluations} JacEvals={summ.JacobianEvaluations}")
