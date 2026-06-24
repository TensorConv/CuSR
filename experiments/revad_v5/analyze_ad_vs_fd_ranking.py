#!/usr/bin/env python3
"""Does the differentiation-method loss gap affect SELECTION RANKING? (the metric CO cares about)

CO's job inside GP is to rank candidates and keep the top x% — not to reproduce scipy's exact
loss. So a small per-tree loss difference between an AD path and the finite-difference path is
only a problem if it changes the *ranking*. This recomputes per-tree fp64 loss for forward-AD /
reverse-AD / fused-FD on one population and, for each pair, asks:

  PRIMARY     Spearman + top-10%/25% selection overlap  -> does the gap move the ranking?
  ONE-SIDED   is one path systematically worse, or ~symmetric?
  MECHANISM   are the worse trees the high-K (structurally rank-deficient) ones?
              (cf. docs/MIXED_PRECISION_PROBE.md: high-K GP trees are rank-deficient)

Pairs reported (all via the SAME `rank_pair` function, so they are directly comparable):
  ad   vs fd   — forward-AD baseline (the original audited comparison)
  revad vs fd  — the DEPLOYED kernel vs FD  (added 2026-06-24: the paper headline uses revad,
                 so the de-risking ranking number must be revad-vs-fd, not routed through ad)
  revad vs ad  — SANITY: drop-in should be rank-equivalent (near-identical)

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


def rank_pair(name_a, name_b, L_a, L_b, kpos, K):
    """PRIMARY (Spearman + top-k overlap) + ONE-SIDED — identical method for any (a, b) pair."""
    both = kpos[np.isfinite(L_a[kpos]) & np.isfinite(L_b[kpos])]
    print(f"\n=== {name_a} vs {name_b} ===  both-finite K>0 = {len(both)}")

    # ---- PRIMARY: does the loss gap move the selection ranking? ----
    rho, _ = spearmanr(L_a[both], L_b[both])
    print(f"[RANKING] Spearman({name_a}_loss, {name_b}_loss) over both-finite K>0 = {rho:.4f}")
    for frac in (0.10, 0.25):
        sa, sb = topk_set(L_a, frac, both), topk_set(L_b, frac, both)
        print(f"  top-{int(frac*100):>2}% best: overlap={len(sa & sb)/len(sa):.3f}  "
              f"jaccard={len(sa & sb)/len(sa | sb):.3f}  (|top|={len(sa)})")
    print(f"  K over both-finite: mean={K[both].mean():.2f} median={int(np.median(K[both]))} max={int(K[both].max())}")

    # ---- ONE-SIDED? with a MATERIALITY FLOOR (exclude near-perfect-tie trees) ----
    FLOOR = 1e-6  # below this both are essentially perfect fits; a >1.05x ratio is tie-noise, not a real gap
    cmp_ = both[np.maximum(L_a[both], L_b[both]) > FLOOR]
    worse_a = cmp_[L_a[cmp_] > 1.05 * L_b[cmp_]]
    worse_b = cmp_[L_b[cmp_] > 1.05 * L_a[cmp_]]
    nt = max(len(cmp_), 1)
    print(f"[ONE-SIDED] {len(cmp_)} 'material' trees (max loss > {FLOOR:.0e}; "
          f"dropped {len(both)-len(cmp_)} near-perfect ties):")
    print(f"   {name_a} worse than {name_b} (>1.05x): {len(worse_a)} = {len(worse_a)/nt*100:.1f}%")
    print(f"   {name_b} worse than {name_a} (>1.05x): {len(worse_b)} = {len(worse_b)/nt*100:.1f}%")
    if len(worse_a):
        print(f"   [mechanism] K on {name_a}-worse trees: mean={K[worse_a].mean():.2f} "
              f"median={int(np.median(K[worse_a]))} max={int(K[worse_a].max())} (vs all {K[both].mean():.2f})")
    return rho


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
    print(f"pop M={pop['M']}  K>0 trees={len(kpos)}")

    # forward-AD vs FD (the original audited comparison; kept verbatim for regression continuity)
    rank_pair("ad", "fd", L_ad, L_fd, kpos, K)

    if L_re is not None:
        # the DEPLOYED kernel vs FD — same method as ad-vs-fd, so directly comparable
        rank_pair("revad", "fd", L_re, L_fd, kpos, K)

        # ---- SANITY: revad vs ad (rank-equivalent drop-in, but NOT bitwise) ----
        bre = kpos[np.isfinite(L_ad[kpos]) & np.isfinite(L_re[kpos])]
        rr, _ = spearmanr(L_ad[bre], L_re[bre])
        div = bre[np.abs(L_ad[bre] - L_re[bre]) > 0.01 * np.maximum(np.abs(L_ad[bre]), 1e-12)]
        print(f"\n[revad vs ad SANITY] Spearman={rr:.6f}  >1%-diff trees={len(div)}/{len(bre)}  "
              f"max|diff|={np.max(np.abs(L_ad[bre]-L_re[bre])):.2e}")
        if len(div):
            print(f"   K on the divergent trees: mean={K[div].mean():.2f} median={int(np.median(K[div]))} "
                  f"max={int(K[div].max())}  (high-K => the singular/NaN-fix trees; near-overall => fp32 path noise)")


if __name__ == "__main__":
    main()
