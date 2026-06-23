#!/usr/bin/env python3
"""Does the AD-vs-FD loss gap affect SELECTION RANKING? (the metric CO actually cares about)

CO's job inside GP is to rank candidates and keep the top x% — not to reproduce scipy's exact
loss. So a small per-tree loss difference between the AD path and the finite-difference path is
only a problem if it changes the *ranking*. This recomputes per-tree fp64 loss for forward-AD /
reverse-AD / fused-FD on one population and asks:

  PRIMARY     Spearman(ad, fd) + top-10%/25% selection overlap  -> does the gap move the ranking?
  ONE-SIDED   is "ad worse than fd" really one-directional (the advisor's claim) or ~symmetric?
  MECHANISM   are the ad-worse trees the high-K (structurally rank-deficient) ones?
              (cf. docs/MIXED_PRECISION_PROBE.md: high-K GP trees are rank-deficient)
  SANITY      revad vs ad should be numerically identical.

Usage:
  python analyze_ad_vs_fd_ranking.py <pop.bin> <c_ad.bin> <c_fd.bin> [c_revad.bin]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cusr.benchmark import popio, interp


def load_c(path, total_c):
    c = np.fromfile(path, dtype=np.float32)
    assert c.size == total_c, f"{path}: got {c.size} floats, want {total_c}"
    return c


def topk_set(loss, frac, idx):
    """Indices (subset of idx) of the best (lowest-loss) `frac` fraction."""
    n = max(1, int(round(len(idx) * frac)))
    order = idx[np.argsort(loss[idx])]
    return set(order[:n].tolist())


def main():
    pop_path, c_ad_path, c_fd_path = sys.argv[1], sys.argv[2], sys.argv[3]
    c_revad_path = sys.argv[4] if len(sys.argv) > 4 else None

    pop = popio.load_pop_bin(Path(pop_path))
    total_c = pop["c_init"].size
    K = pop["metas"][:, 3]

    L_ad = interp.loss_pop(pop, load_c(c_ad_path, total_c))
    L_fd = interp.loss_pop(pop, load_c(c_fd_path, total_c))
    L_re = interp.loss_pop(pop, load_c(c_revad_path, total_c)) if c_revad_path else None

    kpos = np.where(K > 0)[0]
    both = kpos[np.isfinite(L_ad[kpos]) & np.isfinite(L_fd[kpos])]
    print(f"pop M={pop['M']}  K>0 trees={len(kpos)}  both-finite(ad&fd)={len(both)}")

    # ---- PRIMARY: does the loss gap move the selection ranking? ----
    rho, _ = spearmanr(L_ad[both], L_fd[both])
    print(f"\n[RANKING] Spearman(ad_loss, fd_loss) over both-finite K>0 = {rho:.4f}")
    for frac in (0.10, 0.25):
        sa, sf = topk_set(L_ad, frac, both), topk_set(L_fd, frac, both)
        print(f"  top-{int(frac*100):>2}% best: overlap={len(sa & sf)/len(sa):.3f}  "
              f"jaccard={len(sa & sf)/len(sa | sf):.3f}  (|top|={len(sa)})")

    print(f"  K over both-finite: mean={K[both].mean():.2f} median={int(np.median(K[both]))} max={int(K[both].max())}")

    # ---- SANITY: revad vs ad (rank-equivalent, but NOT bitwise) ----
    if L_re is not None:
        bre = kpos[np.isfinite(L_ad[kpos]) & np.isfinite(L_re[kpos])]
        rr, _ = spearmanr(L_ad[bre], L_re[bre])
        div = bre[np.abs(L_ad[bre] - L_re[bre]) > 0.01 * np.maximum(np.abs(L_ad[bre]), 1e-12)]
        print(f"\n[revad vs ad] Spearman={rr:.6f}  >1%-diff trees={len(div)}/{len(bre)}  "
              f"max|diff|={np.max(np.abs(L_ad[bre]-L_re[bre])):.2e}")
        if len(div):
            print(f"   K on the divergent trees: mean={K[div].mean():.2f} median={int(np.median(K[div]))} "
                  f"max={int(K[div].max())}  (high-K => the singular/NaN-fix trees; near-overall => fp32 path noise)")

    # ---- ONE-SIDED? with a MATERIALITY FLOOR (exclude near-perfect-tie trees) ----
    FLOOR = 1e-6  # below this both are essentially perfect fits; a >1.05x ratio is tie-noise, not a real gap
    cmp_ = both[np.maximum(L_ad[both], L_fd[both]) > FLOOR]
    worse_ad = cmp_[L_ad[cmp_] > 1.05 * L_fd[cmp_]]
    worse_fd = cmp_[L_fd[cmp_] > 1.05 * L_ad[cmp_]]
    nt = max(len(cmp_), 1)
    print(f"\n[ONE-SIDED] {len(cmp_)} 'material' trees (max loss > {FLOOR:.0e}; "
          f"dropped {len(both)-len(cmp_)} near-perfect ties):")
    print(f"   ad worse than fd (>1.05x): {len(worse_ad)} = {len(worse_ad)/nt*100:.1f}%")
    print(f"   fd worse than ad (>1.05x): {len(worse_fd)} = {len(worse_fd)/nt*100:.1f}%")
    if len(worse_ad):
        print(f"   [mechanism] K on ad-worse trees: mean={K[worse_ad].mean():.2f} "
              f"median={int(np.median(K[worse_ad]))} max={int(K[worse_ad].max())} (vs all {K[both].mean():.2f})")


if __name__ == "__main__":
    main()
