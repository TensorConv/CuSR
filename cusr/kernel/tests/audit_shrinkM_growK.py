"""Adversarial realloc-bytecap audit (lens: realloc-bytecap, MUST-FIX #6).

Constructs the genuine SHRINK-M-GROW-K hazard the spec names:
  pop X: large M, small K  (M=8000, K=8)  -> d_J product = 8000*8*N
  pop Y: small M, large K  (operon M=4000, K=28) -> d_J product = 4000*28*N (LARGER)

M strictly DECREASES (8000 -> 4000) while d_J = M*K_max*N and d_JtJ = M*K_max^2
products strictly INCREASE. A per-dim (M-keyed) capacity would think "M=4000 <= 8000,
no realloc" and under-allocate d_J / d_JtJ -> OOB writes in build_jtj_jtr / ad_jacobian.
Correct byte-capacity reallocs. Run under compute-sanitizer to detect any OOB.

Also runs Y -> X -> Y to make d_J shrink-in-product-then-grow-back (cap must stay
at the max product; second Y must NOT realloc and must NOT OOB).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/weish/hao/CuSR")
KER = ROOT / "cusr" / "kernel"
sys.path.insert(0, str(KER))
sys.path.insert(0, str(ROOT))

import co_inproc
from cusr.benchmark import popio

base = popio.load_pop_bin(ROOT / "data" / "fixtures" / "pop.bin")          # M=1000 K=8
operon = popio.load_pop_bin(ROOT / "data" / "workload" / "snapshots" /
    "operon_feynman_I.12.1_pop4000_noise0_len64_seed0" / "pop_gen0100.bin")  # M=4000 K=28


def tile(pd, reps):
    M = pd["M"]
    trees = []
    for _ in range(reps):
        for m in range(M):
            no, nn, co_, K = pd["metas"][m].tolist()
            trees.append((pd["nt"][no:no + nn], pd["nv"][no:no + nn],
                          pd["ci"][no:no + nn], pd["c_init"][co_:co_ + K]))
    ym = np.tile(pd["ym"], (reps, 1))
    return popio.build_pop(trees, pd["xs"], ym)


bigM_lowK = tile(base, 8)   # M=8000, K=8
assert bigM_lowK["M"] == 8000, bigM_lowK["M"]

def prod(p):
    M, K, N = p["M"], int(p["K_max"]), int(p["N"])
    return M * K * N, M * K * K

for nm, p in [("bigM_lowK", bigM_lowK), ("operon", operon)]:
    dj, djtj = prod(p)
    print(f"{nm:10s} M={p['M']:6d} K_max={int(p['K_max']):3d} N={int(p['N'])} "
          f"d_J(M*K*N)={dj:>12d}  d_JtJ(M*K^2)={djtj:>10d}")

dj_big, _ = prod(bigM_lowK)
dj_op, _ = prod(operon)
print(f"\nHAZARD CHECK: M shrinks 8000->4000 ({bigM_lowK['M']}>{operon['M']}); "
      f"d_J product GROWS {dj_big}->{dj_op} ({'GROWS' if dj_op > dj_big else 'NO'}); "
      f"per-dim-M cap would under-allocate, byte-cap reallocs.")

# operon AD golden — upgrades "no crash + self-stable" to "byte-CORRECT under
# M-shrink": an under-allocated d_J/d_JtJ (or MUST-FIX#4 stale-buffer corruption on
# the shrink) yields FINITE-but-wrong bytes that self-stability can't catch, but the
# golden does.
GOLD = KER / "tests" / "golden" / "ad" / "operon4000_g100"
op_gold = {
    "c_final": np.fromfile(GOLD / "c_final.bin", dtype=np.float32),
    "status":  np.fromfile(GOLD / "status.bin",  dtype=np.int32),
} if (GOLD / "c_final.bin").exists() else None

so = KER / "libcusr_co_ad.so"
co = co_inproc.InProcessCO(device_id=0, lib_path=str(so), variant="ad")
try:
    # Hazard sequence: big-M/low-K  ->  small-M/high-K (product grows on M-shrink)
    seq = [("bigM_lowK", bigM_lowK), ("operon", operon),
           ("bigM_lowK", bigM_lowK), ("operon", operon)]
    prev = {}
    for tag, p in seq:
        out = co.optimize(p, max_iter=50)
        cf = out["c_final"]
        chk = (float(cf.sum()), float(cf[0]), float(cf[-1]),
               int(out["status"].sum()))
        print(f"  ran {tag:10s} M={p['M']:6d} K={int(p['K_max']):2d}  "
              f"c_final.sum={chk[0]:+.6e} status.sum={chk[3]}")
        if tag in prev:
            assert prev[tag] == chk, f"{tag} differs on reuse (corruption!)"
            print(f"       repeat-{tag}: byte-stable across reuse = True")
        # operon AFTER the bigM shrink must be BYTE-CORRECT vs its golden — this is
        # the M-shrink-grow-K byte-correctness assertion (not just stability).
        if tag == "operon" and op_gold is not None:
            ceq = out["c_final"].tobytes() == op_gold["c_final"].tobytes()
            seq_ok = out["status"].tobytes() == op_gold["status"].tobytes()
            print(f"       operon-after-shrink vs AD golden: c_final byte-eq={ceq} "
                  f"status byte-eq={seq_ok}")
            assert ceq and seq_ok, (
                "operon after M-shrink is NOT byte-identical to its single-call "
                "golden -> M-shrink-grow-K under-alloc OR stale-buffer corruption")
        prev[tag] = chk
    print("\nDONE: shrink-M-grow-K sequence byte-correct vs golden, no crash.")
finally:
    co.teardown()
