"""roofline_model.py — ANALYTICAL roofline for batch_lm_fusedfd (no profiler; perms-blocked).

ncu is unavailable (ERR_NVGPUCTRPERM, no sudo) -> we MODEL FLOPs + DRAM bytes from the
algorithm and combine with the MEASURED end-to-end wall time. All assumptions explicit.
ncu measured-roofline is a TODO (needs admin: NVreg_RestrictProfilingToAdminUsers=0),
which would separate per-kernel compute time from the host-orchestration/launch overhead
that the wall time here includes.

Per LM iter, per tree (K consts, n nodes, N pts) — fusedfd device work:
  fd_jacobian_fused: K tree-evals over N + K*N diff(sub+div); writes J (K*N)
  build_jtj_jtr    : JtR K dotprods(N) + JtJ K(K+1)/2 dotprods(N); reads J(K*N)+r(N)
  solve            : Cholesky ~K^3/3 + 2 tri-solves K^2 (tiny)
  eval(try)+eval(G): 2 tree-evals over N; residual x2 (N); loss x2 (2N)
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cusr.benchmark import popio

# per-op FLOP-equivalent (DISCLOSED; ncu would measure real SASS). arith=1, div/sqrt~4, transc~8.
OPFLOP = {1:1,2:1,3:1,4:4,6:8, 14:8,15:8,16:8,17:8,18:8,19:8, 20:8,22:8,23:4,25:1,26:1,27:4}
N_UFUNC, N_BFUNC = 2, 3
POP = "data/workload/synth/synth_early-gen_M16000_N1000_seed0.bin"
# fusedfd measured points (Tier0 clean): (M, trees/s)
POINTS = [(16000, 17372.0), (64000, 36551.0), (256000, 50394.0)]
CURVE = {0:1713,5:5543,10:8878,15:12028,20:13670,25:14502,30:14959,35:15221,40:15375,45:15462}
MAX_ITER = 50
PEAK_FP32, PEAK_BW = 19.5e12, 2039e9       # A100-SXM4-80GB: fp32 peak, HBM2e BW

def per_tree_stats(pop):
    nt, nv, metas = pop["nt"], pop["nv"], pop["metas"]
    M, N = pop["M"], pop["N"]
    Feval = np.zeros(M)
    for m in range(M):
        off, n = int(metas[m,0]), int(metas[m,1])
        t = nt[off:off+n]; v = nv[off:off+n]
        Feval[m] = sum(OPFLOP.get(int(round(vi)),1) for ti,vi in zip(t,v) if ti in (N_UFUNC,N_BFUNC))
    K = metas[:,3].astype(float); n = metas[:,1].astype(float)
    # mean iters from convergence curve (interp; tree finishing at iter it ran it+1; unfinished -> MAX_ITER)
    xs = np.array(sorted(CURVE)); ys = np.array([CURVE[i] for i in sorted(CURVE)], float)
    done = np.interp(np.arange(MAX_ITER), xs, ys, right=ys[-1])
    running = np.clip(M - np.concatenate([[0], done[:-1]]), 0, M)
    mean_it = running.sum()/M
    flops_iter = N*(Feval*(K+2) + 2*K + 2*K + K*(K+1) + 6) + (K**3)/3.0
    bytes_iter = 4*K*N + 4*K*N + 2*4*N + 12*n + 4*N*pop["n_vars"]/M
    return dict(N=N, mean_nodes=float(n.mean()), mean_K=float(K.mean()),
                mean_Feval=float(Feval.mean()), mean_iters=float(mean_it),
                flops_per_tree=float(flops_iter.mean()*mean_it),
                bytes_per_tree=float(bytes_iter.mean()*mean_it))

def main():
    outdir = Path(sys.argv[1]) if len(sys.argv)>1 else Path(".")
    s = per_tree_stats(popio.load_pop_bin(POP))
    ai = s["flops_per_tree"]/s["bytes_per_tree"]
    ridge = PEAK_FP32/PEAK_BW
    pts = []
    for M, tps in POINTS:
        ach = s["flops_per_tree"]*tps          # achieved FLOP/s = (FLOPs/tree)*(trees/s)
        bw  = s["bytes_per_tree"]*tps
        pts.append(dict(M=M, trees_per_s=tps, achieved_TFLOPs=ach/1e12,
                        pct_fp32_peak=100*ach/PEAK_FP32, eff_BW_GBps=bw/1e9,
                        pct_HBM_peak=100*bw/PEAK_BW))
    res = dict(pop=POP, stats=s, arithmetic_intensity=round(ai,3),
               A100_ridge=round(ridge,2), bound_if_efficient=("memory" if ai<ridge else "compute"),
               points=pts, peak_fp32_TFLOPs=PEAK_FP32/1e12, peak_BW_GBps=PEAK_BW/1e9)
    (outdir/"roofline_data.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ---- plot ----
    plt.figure(figsize=(8,5.6))
    I = np.logspace(-1, 3, 400)
    roof = np.minimum(PEAK_FP32/1e12, (PEAK_BW*I)/1e12)   # TFLOP/s
    plt.loglog(I, roof, 'k-', lw=2, label="A100 roofline (19.5 TFLOP/s fp32, 2039 GB/s)")
    plt.axvline(ridge, ls=':', color='gray', lw=1); plt.text(ridge*1.05, 0.02, f"ridge {ridge:.1f}", color='gray', fontsize=9)
    colors=['tab:blue','tab:orange','tab:green']
    for p,c in zip(pts, colors):
        plt.loglog([ai],[p["achieved_TFLOPs"]], 'o', color=c, ms=9,
                   label=f"fusedfd M={p['M']//1000}k: {p['achieved_TFLOPs']*1000:.0f} GFLOP/s ({p['pct_fp32_peak']:.2f}% peak)")
    # ceiling at this AI = memory-bound limit
    mb = PEAK_BW*ai/1e12
    plt.loglog([ai],[mb],'kx',ms=10); plt.text(ai*0.5, mb*1.2, f"BW-bound ceiling @AI\n{mb:.1f} TFLOP/s", fontsize=8, color='dimgray')
    plt.xlabel("arithmetic intensity (FLOP / byte, analytical)"); plt.ylabel("performance (TFLOP/s)")
    plt.title("batch_lm_fusedfd — ANALYTICAL roofline (A100, fp32, end-to-end wall)\nkernel sits ~1% below ceilings: bound by execution efficiency / overhead, not compute or BW")
    plt.grid(True, which='both', alpha=0.3); plt.legend(fontsize=8, loc='lower right'); plt.ylim(1e-2, 50)
    plt.tight_layout(); plt.savefig(outdir/"plots"/"roofline.png", dpi=130)
    print("wrote plots/roofline.png")

if __name__ == "__main__":
    main()
