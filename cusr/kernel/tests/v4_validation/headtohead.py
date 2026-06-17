"""Decisive AD-vs-FD head-to-head on COMMON converged trees.
If AD ~= FD on common trees -> AD Jacobian is fine (FD passes the gate); the
within_1.05x gate regression is then explained by AD converging extra marginal
trees. If AD is systematically worse on common trees -> real derivative bug.
No scipy needed; reuses the same loss recompute path as verify.py.
"""
import sys
import numpy as np
sys.path.insert(0, '/home/weish/hao/CuSR')
from cusr.kernel.verify import load_pop_bin
from cusr.kernel.tree_interpreter import eval_batch

POP = '/home/weish/hao/CuSR/data/fixtures/pop.bin'
pop = load_pop_bin(POP)
M, xs = pop['M'], pop['xs']
ad_c = np.fromfile('/tmp/cmp/ad_out/c_final.bin', dtype=np.float32)
ad_st = np.fromfile('/tmp/cmp/ad_out/status.bin', dtype=np.int32)
fd_c = np.fromfile('/tmp/cmp/fd_out/c_final.bin', dtype=np.float32)
fd_st = np.fromfile('/tmp/cmp/fd_out/status.bin', dtype=np.int32)


def loss(nt, nv, n_nodes, c, ym):
    ss = np.array([n_nodes], dtype=np.int32)
    try:
        yp = eval_batch(nt, nv, ss, xs, c)
        if np.all(np.isfinite(yp)):
            return 0.5 * float(np.sum((yp - ym) ** 2))
    except Exception:
        pass
    return float('inf')


rows = []
for m in range(M):
    no, nn, co, K = pop['metas'][m].tolist()
    if K == 0:
        continue
    nt = pop['nt'][no:no+nn]; nv = pop['nv'][no:no+nn]; ym = pop['ym'][m]
    li = loss(nt, nv, nn, pop['c_init'][co:co+K], ym)
    la = loss(nt, nv, nn, ad_c[co:co+K], ym)
    lf = loss(nt, nv, nn, fd_c[co:co+K], ym)
    rows.append((m, K, nn, int(ad_st[m]), int(fd_st[m]), li, la, lf))

print(f"K>0 trees: {len(rows)}")
print(f"AD converged(status0): {sum(1 for r in rows if r[3]==0)}   FD converged: {sum(1 for r in rows if r[4]==0)}")

common = [r for r in rows if r[3] == 0 and r[4] == 0]
print(f"COMMON (both converged): {len(common)}")

ad_worse = fd_worse = tie = 0
ratios = []
ad_much = []
for (m, K, nn, a, f, li, la, lf) in common:
    if la > 1.05 * lf and la > 1e-12:
        ad_worse += 1
        if la > 2 * lf and lf > 0:
            ad_much.append((m, K, nn, li, la, lf, la/lf))
    elif lf > 1.05 * la and lf > 1e-12:
        fd_worse += 1
    else:
        tie += 1
    if lf > 1e-15 and np.isfinite(la) and np.isfinite(lf):
        ratios.append(la / lf)

print(f"\n== HEAD-TO-HEAD on COMMON (loss_AD vs loss_FD) ==")
print(f"  AD~=FD (within 1.05x either way): {tie}")
print(f"  AD strictly worse (>1.05x FD):    {ad_worse}")
print(f"  FD strictly worse (>1.05x AD):    {fd_worse}")
r = np.array(ratios)
if len(r):
    print(f"  loss_AD/loss_FD ratio: p10={np.percentile(r,10):.4g} p50={np.median(r):.4g} "
          f"p90={np.percentile(r,90):.4g} p99={np.percentile(r,99):.4g} max={r.max():.4g}")
print(f"  AD much-worse (>2x FD): {len(ad_much)}  (top by ratio:)")
for z in sorted(ad_much, key=lambda z: -z[6])[:15]:
    print(f"    m={z[0]:4d} K={z[1]} nodes={z[2]:2d} init={z[3]:.3e} AD={z[4]:.3e} FD={z[5]:.3e} ratio={z[6]:.3g}")

ad_only = [r for r in rows if r[3] == 0 and r[4] != 0]
print(f"\n== AD-only-converged (FD failed/maxiter; the extra trees) ==")
print(f"  count: {len(ad_only)}")
dn = sum(1 for r in ad_only if r[6] < r[5])
print(f"  loss-down (loss_AD < loss_init): {dn}/{len(ad_only)}")
print("DONE")
