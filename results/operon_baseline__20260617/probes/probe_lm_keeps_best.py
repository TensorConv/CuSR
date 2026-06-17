"""probe_lm3.py — confirm Operon CO returns the BEST tree (never worse than start),
so neutral loss @ convert(otree) is a fair 'achieved loss'. 40 trees, varied K."""
import sys
from pathlib import Path
import numpy as np
import pyoperon as op
ROOT = Path("/home/weish/hao/CuSR"); sys.path.insert(0, str(ROOT))
from cusr.benchmark.workload.operon_dump import run_cell, _grammar_int
from cusr.benchmark.workload.operon_adapter import build_hash2idx, convert
_UF = {14: np.sin, 15: np.cos, 16: np.tan}
def ev(nt,nv,ci,c,X):
    X=np.asarray(X,np.float64); st=[]
    for i in range(len(nt)-1,-1,-1):
        t=int(nt[i]); v=nv[i]
        if t==0: st.append(X[:,int(v)].astype(np.float64))
        elif t==1: st.append(np.full(X.shape[0],float(c[int(ci[i])]),np.float64))
        elif t==2: st.append(_UF[int(v)](st.pop()))
        elif t==3:
            l=st.pop(); rv=st.pop(); f=int(v)
            st.append(l+rv if f==1 else l-rv if f==2 else l*rv if f==3 else l/rv)
    return st[0]
def hs(nt,nv,ci,c,X,y): return 0.5*float(np.sum((ev(nt,nv,ci,c,X)-np.asarray(y,np.float64))**2))

_,_,it = run_cell(dataset="feynman/I.18.12", pop=4000, N=1000, seed=0, noise=0.0, cap=32,
                  checkpoint_gens=[0], out=Path("/tmp/probe_regen/pop.bin"), threads=1,
                  _return_internals=True)
snaps,h2i,X,y = it["snaps"],it["hash2idx"],it["X"],it["y"]; nv_=X.shape[1]
ds=op.Dataset(np.asfortranarray(np.column_stack([X,y]).astype(np.float64))); V=ds.Variables
pr=op.Problem(ds); pr.TrainingRange=op.Range(0,1000); pr.TestRange=op.Range(0,1000)
pr.Target=V[nv_]; pr.InputHashes=[v.Hash for v in V[:nv_]]; pr.ConfigurePrimitiveSet(_grammar_int())
dt=op.DispatchTable()  # MUST hold a named ref (temporary -> GC -> dangling ptr -> segfault)
lm=op.LMOptimizer(dt,pr,max_iter=50); co=op.CoefficientOptimizer(lm)

worse=0; improved=0; same=0; n=0
for t in snaps[0][:200]:
    try: nt,nv,ci,c0=convert(t,h2i)
    except Exception: continue
    if len(c0)==0: continue
    raw0=hs(nt,nv,ci,c0,X,y)
    ot,su=co(op.RandomGenerator(0),t)
    n1,v1,i1,c1=convert(ot,h2i); raw1=hs(n1,v1,i1,c1,X,y)
    n+=1
    if raw1 > raw0*(1+1e-6)+1e-9: worse+=1
    elif raw1 < raw0*(1-1e-9): improved+=1
    else: same+=1
    if n>=40: break
print(f"trees={n}: improved={improved} unchanged={same} WORSE_than_start={worse}")
print("=> if WORSE==0, otree=best, neutral loss@otree is a fair achieved-loss for Operon")
