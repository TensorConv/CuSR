"""coverage_probe.py — is one engine's convergence-count lead a real fit-quality
edge, or just a labeling difference?

The stop-criterion fix (commit 8b9234b) made "CONVERGED" strict: a tree is only
labeled converged if it reached xtol via a non-worsening step. An engine can thus
hold a perfectly good fit but be labeled MAXITER. So a convergence-COUNT gap
(e.g. FD 2051 vs AD 1393 on high-K feyn) does not by itself mean better fits.

This probe takes the trees ENGINE-A converged but ENGINE-B did NOT, and
fp64-recomputes BOTH engines' loss at their own final constants (scipy-free).
  ratio = loss_B / loss_A  on those trees:
    ~1   => B matched A's fit, just didn't earn the CONVERGED label (labeling gap)
    >>1  => A genuinely fits better there (real coverage edge)

Same env interface as setcontrolled.py:
    SETCTRL_POP / SETCTRL_AD_OUT / SETCTRL_FD_OUT  (default = pop.bin run)
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/weish/hao/CuSR')
from cusr.kernel.verify import load_pop_bin
from cusr.kernel.tree_interpreter import eval_batch

POP = os.environ.get('SETCTRL_POP', '/home/weish/hao/CuSR/data/fixtures/pop.bin')
AD_OUT = os.environ.get('SETCTRL_AD_OUT', '/tmp/cmp/ad_out')
FD_OUT = os.environ.get('SETCTRL_FD_OUT', '/tmp/cmp/fd_out')
pop = load_pop_bin(POP); M, xs = pop['M'], pop['xs']
ad_c = np.fromfile(f'{AD_OUT}/c_final.bin', dtype=np.float32)
ad_st = np.fromfile(f'{AD_OUT}/status.bin', dtype=np.int32)
fd_c = np.fromfile(f'{FD_OUT}/c_final.bin', dtype=np.float32)
fd_st = np.fromfile(f'{FD_OUT}/status.bin', dtype=np.int32)
c_init = pop['c_init']


def gpu_loss(nt, nv, nn, c, ym):
    ss = np.array([nn], dtype=np.int32)
    try:
        yp = eval_batch(nt, nv, ss, xs, c)
        if np.all(np.isfinite(yp)):
            return 0.5 * float(np.sum((yp - ym) ** 2))
    except Exception:
        pass
    return float('inf')


def probe(label, a_st, a_c, b_st, b_c):
    """Trees A converged but B did NOT. Compare B's loss to A's loss."""
    tgt = [m for m in range(M)
           if a_st[m] == 0 and b_st[m] != 0 and int(pop['metas'][m][3]) > 0]
    la, lb, li = {}, {}, {}
    for m in tgt:
        no, nn, co, K = pop['metas'][m].tolist()
        nt, nv, ym = pop['nt'][no:no+nn], pop['nv'][no:no+nn], pop['ym'][m]
        la[m] = gpu_loss(nt, nv, nn, a_c[co:co+K], ym)
        lb[m] = gpu_loss(nt, nv, nn, b_c[co:co+K], ym)
        li[m] = gpu_loss(nt, nv, nn, c_init[co:co+K], ym)
    fin = [m for m in tgt if np.isfinite(la[m]) and np.isfinite(lb[m]) and la[m] > 0]
    print(f"\n== {label}: {len(tgt)} trees (A converged, B did not); {len(fin)} fp64-comparable ==")
    if not fin:
        return
    ratio = np.array([lb[m] / la[m] for m in fin])     # B/A; ~1 => B matched A
    matched = int((ratio <= 1.05).sum())
    b_down = sum(1 for m in fin if np.isfinite(li[m]) and lb[m] <= li[m])
    print(f"  B (non-converged engine) matched A within 1.05x: {matched}/{len(fin)} ({100*matched/len(fin):.1f}%)")
    print(f"  ratio loss_B/loss_A: p50={np.percentile(ratio,50):.4g} p90={np.percentile(ratio,90):.4g} "
          f"p99={np.percentile(ratio,99):.4g} max={ratio.max():.4g}")
    print(f"  B still loss-down vs init on these: {b_down}/{len(fin)} ({100*b_down/len(fin):.1f}%)")


print(f"corpus={POP}")
probe("FD-converged / AD-not  (A=FD, B=AD)", fd_st, fd_c, ad_st, ad_c)
probe("AD-converged / FD-not  (A=AD, B=FD)", ad_st, ad_c, fd_st, fd_c)
print("\nDONE")
