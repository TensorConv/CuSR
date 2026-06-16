"""mechanism_proof.py — algebraic proof that Operon and evogp pre-CO populations
reach rank-deficient JtJ by DIFFERENT mechanisms (advisor Guard 2: not a fake
"both show it -> confirmed").

Builds the EXACT pop.bin trees the adapter/dumpers produce, computes the kernel's
finite-difference Jacobian J = d m / d c (the kernel's fd_jacobian), forms JtJ, and
reports rank / smallest singular value / condition number. Pure numpy; mirrors the
kernel's reverse-eval via the committed pop_ref.ref_eval.

Mechanism, proven on minimal trees (no GP, no bloat):
  * Operon: every leaf is a weighted coefficient, so its NATURAL form for x_a*x_b is
    (w_a x_a)(w_b x_b): a 7-node tree, K=2, whose two Jacobian columns are exact
    scalar multiples (dm/dw_a = w_b x_a x_b, dm/dw_b = w_a x_a x_b) -> JtJ rank 1,
    SINGULAR at zero bloat. Structural over-parameterization.
  * evogp: bare variables, so c0*x_a*x_b is K=1 -> JtJ 1x1, FULL RANK at minimal
    size. evogp only goes singular when BLOAT introduces redundant constants, e.g.
    c0*c1*x_a*x_b (K=2): dm/dc0=c1 x..., dm/dc1=c0 x... -> proportional -> singular.
"""
import sys; sys.path.insert(0, "/home/weish/hao/CuSR")
import json
import numpy as np
from cusr.benchmark.workload.pop_ref import ref_eval

VAR, CONST, UF, BF = 0, 1, 2, 3
MUL = 3

def jtj_stats(nt, nv, ci, c_init, X, eps=1e-3):
    """Kernel-style finite-difference J (d residual / d c) and JtJ conditioning."""
    nt=np.array(nt,np.int32); nv=np.array(nv,np.float32); ci=np.array(ci,np.int32)
    c=np.array(c_init,np.float64); K=len(c)
    base=ref_eval(nt,nv,ci,c.astype(np.float32),X).astype(np.float64)
    J=np.empty((X.shape[0],K))
    for k in range(K):
        cp=c.copy(); h=eps*max(1.0,abs(c[k])); cp[k]+=h
        J[:,k]=(ref_eval(nt,nv,ci,cp.astype(np.float32),X).astype(np.float64)-base)/h
    JtJ=J.T@J
    sv=np.linalg.svd(JtJ, compute_uv=False)
    cond=float(sv.max()/sv.min()) if sv.min()>0 else float("inf")
    # fp32-effective rank: the kernel's Cholesky runs in fp32 (eps~6e-8), so a JtJ
    # whose smin/smax falls below ~fp32-eps is effectively singular *to the kernel*
    # even when exact-arithmetic SVD would call it full rank. cond>1e6 => the fp32
    # Cholesky of (JtJ + small*I) hits a non-positive pivot -> fail_cholesky.
    rank_fp32=int(np.sum(sv > sv.max()*1e-6)) if sv.max()>0 else 0
    return dict(K=K, n_nodes=len(nt), rank_fp32=rank_fp32, fp32_full_rank=(rank_fp32==K),
                effectively_singular_fp32=bool(cond>1e6),
                smin=float(sv.min()), smax=float(sv.max()), cond=cond)

rng=np.random.default_rng(0)
N=256
xa=rng.uniform(1,5,N); xb=rng.uniform(1,5,N)
X=np.stack([xa,xb],1).astype(np.float32)   # col0=x_a, col1=x_b

cases={}
# --- Operon natural form: (w_a x_a)*(w_b x_b) = MUL(MUL(Cwa,Va), MUL(Cwb,Vb)) ---
# prefix: [MUL, MUL, C, V0, MUL, C, V1]
cases["operon_xa*xb (2 weighted leaves, K=2, 7 nodes, ZERO bloat)"]=jtj_stats(
    [BF,BF,CONST,VAR,BF,CONST,VAR],[MUL,MUL,0,0,MUL,0,1],[-1,-1,0,-1,-1,1,-1],[1.3,0.7],X)
# --- evogp natural form: c0*x_a*x_b = MUL(MUL(C,V0),V1), K=1, 5 nodes ---
cases["evogp  c0*xa*xb (1 const, K=1, 5 nodes, minimal)"]=jtj_stats(
    [BF,BF,CONST,VAR,VAR],[MUL,MUL,0,0,1],[-1,-1,0,-1,-1],[1.3],X)
# --- evogp WITH BLOAT: c0*c1*x_a*x_b = MUL(C0, MUL(C1, MUL(V0,V1))), K=2, 7 nodes ---
cases["evogp  c0*c1*xa*xb (redundant consts via BLOAT, K=2, 7 nodes)"]=jtj_stats(
    [BF,CONST,BF,CONST,BF,VAR,VAR],[MUL,0,MUL,0,MUL,0,1],[-1,0,-1,1,-1,-1,-1],[1.3,0.7],X)
# --- Operon single weighted leaf: w*x = MUL(C,V), K=1 -> full rank (control) ---
cases["operon w*x (1 weighted leaf, K=1, control)"]=jtj_stats(
    [BF,CONST,VAR],[MUL,0,0],[-1,0,-1],[1.3],X)

print(f"{'tree':<58}{'K':>3}{'nodes':>6}{'rkfp32':>7}{'sing?':>6}{'cond':>12}")
for name,s in cases.items():
    print(f"{name:<58}{s['K']:>3}{s['n_nodes']:>6}{s['rank_fp32']:>7}"
          f"{('YES' if s['effectively_singular_fp32'] else 'no'):>6}{s['cond']:>12.2e}")

json.dump(cases, open("results/operon_corpus__20260617/data/mechanism_proof.json","w"), indent=1)
print("\nKEY: Operon's natural x_a*x_b is rank-deficient at 7 nodes/zero-bloat (structural);")
print("evogp's c0*x_a*x_b is full-rank K=1 and only goes singular after bloat adds a 2nd const.")
