"""probe_fp32_lie.py — recompute & classify the kernel's "converged-but-worsened" trees,
backing the report's fp32-honesty finding (review fix #3: the report must not cite a
verification that has no code). Run AFTER driver.py has produced the per-cell JSONs:

    python results/operon_baseline__20260617/probes/probe_fp32_lie.py

For each kernel (FD, AD) it recomputes, from the committed-pipeline per-cell JSONs
(neutral fp64 loss at start/final + the kernel's OWN fp32 final loss + status):
  * converged-but-worsened : status==0 yet neutral fp64 loss_final > loss_start
  * genuine fp32-LIE       : the subset whose kernel fp32 loss <= start (kernel is FOOLED)
  * converged-worse-own    : the rest — worse in the kernel's OWN fp32 objective too
  * serious >2x / >100x    : magnitude tail (cannot be fp32 noise)
and prints the >100x subset with start / fp64-final / kernel-fp32 so the "fast-math fooled
it" claim is inspectable, then asserts the totals match report_aggregates.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path("/home/weish/hao/CuSR/results/operon_baseline__20260617")
DATA = D / "data"
TOL = 1e-4   # kernel fp32 <= start*(1+TOL) == "kernel thinks it held/improved"


def scan(eng):
    tot = defaultdict(int); examples = []
    for f in sorted(DATA.glob(f"*__{eng}.json")):
        r = json.loads(f.read_text())
        for g, gd in r["gens"].items():
            if "error" in gd:
                continue
            st = np.array(gd["status"]); K = np.array(gd["K"])
            ls = np.array(gd["neutral_loss_start"]); lf = np.array(gd["neutral_loss_final"])
            kl = np.array(gd["kernel_fp32_loss"])
            opt = K > 0
            worse = opt & (st == 0) & ((~np.isfinite(lf)) | (lf > ls * (1 + 1e-6) + 1e-9))
            lie = worse & (kl <= ls * (1 + TOL))
            tot["convworse"] += int(worse.sum())
            tot["fp32_lie"] += int(lie.sum())
            tot["convworse_own"] += int((worse & ~lie).sum())
            tot["serious_2x"] += int((opt & (st == 0) & ((~np.isfinite(lf)) | (lf > ls * 2))).sum())
            cat = worse & np.isfinite(lf) & (ls > 0) & (lf > ls * 100)
            tot["over_100x"] += int(cat.sum())
            tot["over_100x_lie"] += int((cat & (kl <= ls * (1 + TOL))).sum())
            for i in np.where(cat)[0][:3]:
                examples.append((r["dataset"], g, float(ls[i]), float(lf[i]), float(kl[i]),
                                 "LIE" if kl[i] <= ls[i] * (1 + TOL) else "own-worse"))
    return tot, examples


def main():
    A = json.loads((DATA / "report_aggregates.json").read_text())
    BG = A["by_gen"]; G = [str(g) for g in A["gens"]]

    def gsum(k):
        return sum(BG[g].get(k, 0) for g in G)

    ok = True
    for eng in ("fd", "ad"):
        tot, examples = scan(eng)
        print(f"\n=== {eng.upper()} ===")
        print(f"  converged-but-worsened : {tot['convworse']}")
        print(f"    genuine fp32-lie     : {tot['fp32_lie']}  ({100*tot['fp32_lie']/max(tot['convworse'],1):.0f}% — kernel's own fp32 <= start, FOOLED)")
        print(f"    worse-in-own-fp32    : {tot['convworse_own']}  (accept/stop-criterion defect, not precision)")
        print(f"    serious >2x          : {tot['serious_2x']}")
        print(f"    >100x worse          : {tot['over_100x']}  (of which {tot['over_100x_lie']} genuine lies)")
        print(f"  >100x examples (start -> fp64_final | kernel_fp32):")
        for ds, g, s, lf, kl, tag in examples[:6]:
            print(f"    {ds:16} g{g:>3}: {s:.4g} -> {lf:.4g} | fp32={kl:.4g}  [{tag}]")
        # regression check vs the committed aggregates
        for k, agg in (("convworse", f"{eng}_convworse"), ("fp32_lie", f"{eng}_fp32lie"),
                       ("convworse_own", f"{eng}_convworse_ownmetric"), ("serious_2x", f"{eng}_convworse2x")):
            want = gsum(agg)
            if tot[k] != want:
                print(f"  MISMATCH {eng} {k}: probe {tot[k]} != aggregates {want}"); ok = False
    print("\n" + ("OK: probe totals match report_aggregates.json" if ok else "FAIL: mismatch (see above)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
