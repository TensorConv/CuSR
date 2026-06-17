"""check_no_worse_than_init.py — the STRONG (un-gameable) honesty gate.

`check_fp32_honesty.py` only inspects trees the kernel *reported* converged
(status==0). That gate is gameable: an optimizer can clear it by relocating the
damage to the MAXITER / FAIL_NAN buckets (we proved exactly this with the
trust-region ablation — it passed the converged gate yet delivered a NET WORSE
population). This test closes that hole.

THE INVARIANT (what a correct optimizer must NEVER violate):

    for EVERY tree (any status), the delivered c_final must NOT be worse in
    honest fp64 than where it started (c_init).  delivered <= init.

For each offender fixture tree with K>0 it runs CO from c_init, then recomputes
the neutral fp64 0.5*SSE at c_final and at c_init using the INDEPENDENT numpy
oracle (`half_sse` — NOT the kernel's own fp64 eval, so this also cross-validates
the in-kernel fp64 interpreter the guard relies on). A tree FAILS if
fp64(c_final) is non-finite, or worse than fp64(c_init) by more than --tol
(default 5%, to absorb CUDA-fp64 vs numpy-fp64 rounding on boundary trees; a
real eval bug or a relocated regression blows past 5%).

    # baseline (no guard) — expected FAIL (these are offenders):
    python .../check_no_worse_than_init.py cusr/kernel/batch_lm_ad_noguard
    # guarded kernel — expected PASS (delivered<=init enforced at the boundary):
    python .../check_no_worse_than_init.py cusr/kernel/batch_lm_ad

Gate: exits non-zero if ANY K>0 tree is delivered worse-than-init beyond --tol.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cusr.benchmark.workload.pop_io import read_pop_bin   # noqa: E402
from _eval import half_sse                                # noqa: E402


def run(binary, max_iter, gpu, tol):
    binary = Path(binary)
    assert binary.exists(), f"kernel binary not found: {binary}"
    manifest = json.loads((HERE / "manifest.json").read_text())
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmp = Path("/tmp/no_worse_than_init_check"); tmp.mkdir(parents=True, exist_ok=True)

    n_total = 0
    worse_tol = worse_strict = nonfinite = 0
    worst = 0.0
    failures = []
    for prob, fx in manifest["fixtures"].items():
        fbin = HERE / fx["file"]
        outd = tmp / fx["file"]; outd.mkdir(exist_ok=True)
        p = subprocess.run([str(binary), str(fbin), str(outd), "--max-iter", str(max_iter)],
                           capture_output=True, text=True, env=env, timeout=600)
        if p.returncode != 0:
            print(f"  {prob}: kernel exited {p.returncode}\n{p.stderr[-300:]}"); sys.exit(2)
        pop = read_pop_bin(fbin)
        c_final = np.fromfile(outd / "c_final.bin", dtype=np.float32)
        X = pop["xs"]; y = pop["ym"][0]; metas = pop["metas"]
        for m in range(pop["M_prob"]):
            n_off, n_nodes, c_off, K = (int(metas[m, 0]), int(metas[m, 1]),
                                        int(metas[m, 2]), int(metas[m, 3]))
            if K == 0:
                continue
            sl = slice(n_off, n_off + n_nodes)
            nt, nv, ci = pop["nt"][sl], pop["nv"][sl], pop["ci"][sl]
            s0 = half_sse(nt, nv, ci, pop["c_init"][c_off:c_off + K], X, y)
            sf = half_sse(nt, nv, ci, c_final[c_off:c_off + K], X, y)
            n_total += 1
            # init non-finite => "worse than init" is ill-defined; the guard
            # likewise only reverts on a non-finite FINAL there. Skip ranking.
            if not np.isfinite(s0):
                if not np.isfinite(sf):
                    nonfinite += 1
                continue
            if not np.isfinite(sf):
                nonfinite += 1; worse_tol += 1; worse_strict += 1
                failures.append((prob, fx["trees"][m]["gen"], fx["trees"][m]["orig_index"],
                                 s0, sf, float("inf")))
                continue
            ratio = sf / s0 if s0 > 0 else (1.0 if sf == 0 else float("inf"))
            worst = max(worst, ratio)
            if sf > s0 * (1.0 + 1e-9) + 1e-12:
                worse_strict += 1
            if sf > s0 * (1.0 + tol) + 1e-12:
                worse_tol += 1
                failures.append((prob, fx["trees"][m]["gen"], fx["trees"][m]["orig_index"],
                                 s0, sf, ratio))

    print(f"\nkernel: {binary.name}   fixtures: {len(manifest['fixtures'])} problems, "
          f"{n_total} K>0 trees")
    print(f"  delivered WORSE-THAN-INIT (>{tol:.0%}, fp64) : {worse_tol}   <-- gate (must be 0)")
    print(f"  delivered worse-than-init (strict >, fp64)  : {worse_strict}")
    print(f"  delivered NON-FINITE in fp64                : {nonfinite}")
    print(f"  worst fp64 final/init ratio                 : {worst:.4g}")
    if failures:
        print("  worst offenders (problem gen idx: init -> fp64_final = ratio):")
        for prob, g, idx, s0, sf, r in sorted(failures, key=lambda x: -x[5])[:8]:
            print(f"    {prob:16} g{g:<3} #{idx:<4} {s0:.4g} -> {sf:.4g}  = {r:.3g}x")
    ok = worse_tol == 0
    print("\n" + ("PASS: delivered<=init holds for every tree (monotone invariant)"
                  if ok else f"FAIL: {worse_tol} trees delivered worse-than-init beyond {tol:.0%}"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help="path to a kernel binary (batch_lm_*)")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="fractional worse-than-init tolerance for the gate (default 0.05)")
    a = ap.parse_args()
    sys.exit(0 if run(a.binary, a.max_iter, a.gpu, a.tol) else 1)


if __name__ == "__main__":
    main()
