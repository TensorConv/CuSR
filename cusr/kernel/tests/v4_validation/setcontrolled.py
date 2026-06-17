"""Set-controlled within_1.05x vs scipy: prove AD's gate 'regression' is a
denominator/set-composition artifact, not worse quality.

Reproduces the gate's per-binary within_1.05x (should match 80.8% AD / 92.4% FD),
then recomputes it on the STABLE set common to AD&FD&scipy. If AD~=FD there, the
gate gap is purely that AD converges extra hard trees (good), not worse minima.
"""
import sys, time, numpy as np, multiprocessing as mp
sys.path.insert(0, '/home/weish/hao/CuSR')
from cusr.kernel.verify import load_pop_bin, _scipy_worker
from cusr.kernel.tree_interpreter import eval_batch

POP = '/home/weish/hao/CuSR/data/fixtures/pop.bin'
pop = load_pop_bin(POP); M, xs = pop['M'], pop['xs']
ad_c = np.fromfile('/tmp/cmp/ad_out/c_final.bin', dtype=np.float32)
ad_st = np.fromfile('/tmp/cmp/ad_out/status.bin', dtype=np.int32)
fd_c = np.fromfile('/tmp/cmp/fd_out/c_final.bin', dtype=np.float32)
fd_st = np.fromfile('/tmp/cmp/fd_out/status.bin', dtype=np.int32)


def gpu_loss(nt, nv, nn, c, ym):
    ss = np.array([nn], dtype=np.int32)
    try:
        yp = eval_batch(nt, nv, ss, xs, c)
        if np.all(np.isfinite(yp)):
            return 0.5 * float(np.sum((yp - ym) ** 2))
    except Exception:
        pass
    return float('inf')


payloads, meta = [], {}
for m in range(M):
    no, nn, co, K = pop['metas'][m].tolist()
    meta[m] = (no, nn, co, K)
    if K > 0:
        payloads.append((m, pop['nt'][no:no+nn].copy(), pop['nv'][no:no+nn].copy(),
                         xs, pop['ym'][m], pop['c_init'][co:co+K].copy()))

t = time.time()
sc = {}
with mp.Pool(64) as pool:
    for (mi, c_final, loss, status, nit, il) in pool.imap_unordered(_scipy_worker, payloads, chunksize=4):
        sc[mi] = (loss, status)
print(f"scipy done {time.time()-t:.1f}s  ({len(sc)} trees)")

la, lf = {}, {}
for m in range(M):
    no, nn, co, K = meta[m]
    if K == 0:
        continue
    nt, nv, ym = pop['nt'][no:no+nn], pop['nv'][no:no+nn], pop['ym'][m]
    la[m] = gpu_loss(nt, nv, nn, ad_c[co:co+K], ym)
    lf[m] = gpu_loss(nt, nv, nn, fd_c[co:co+K], ym)


def w105(ids, lg):
    ids = [m for m in ids if np.isfinite(lg[m]) and np.isfinite(sc[m][0])]
    if not ids:
        return float('nan'), 0
    hit = sum(1 for m in ids if lg[m] <= 1.05 * sc[m][0] + 1e-30)
    return 100.0 * hit / len(ids), len(ids)


ad_conv = {m for m in la if ad_st[m] == 0}
fd_conv = {m for m in lf if fd_st[m] == 0}
sc_conv = {m for m in sc if sc[m][1] == 'ok'}

ad_both = ad_conv & sc_conv          # AD gate denominator
fd_both = fd_conv & sc_conv          # FD gate denominator
stable = ad_conv & fd_conv & sc_conv  # common to ALL THREE

print("\n== reproduce gate within_1.05x (per-binary both_conv vs scipy) ==")
r, n = w105(ad_both, la); print(f"  AD : {r:.1f}%  (n={n})   [gate said 80.8%]")
r, n = w105(fd_both, lf); print(f"  FD : {r:.1f}%  (n={n})   [gate said 92.4%]")

print("\n== SET-CONTROLLED: same STABLE tree set (AD&FD&scipy all converged) ==")
ra, na = w105(stable, la)
rf, nf = w105(stable, lf)
print(f"  stable set size: {len(stable)}")
print(f"  AD within_1.05x on stable: {ra:.1f}%")
print(f"  FD within_1.05x on stable: {rf:.1f}%")
print(f"  => gap on identical trees: {ra-rf:+.1f}pp  (≈0 means NO quality regression; gate gap is set composition)")

extra = ad_conv - fd_conv  # AD converged, FD did not
print(f"\n== extra trees AD converged that FD did not: {len(extra)} ==")
ex_sc = [m for m in extra if m in sc_conv]
if ex_sc:
    r, n = w105(ex_sc, la)
    print(f"  of those also scipy-converged ({n}): AD within_1.05x = {r:.1f}%  (these drag AD's denominator)")
print("DONE")
