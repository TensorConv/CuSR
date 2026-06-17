"""check_fp32_honesty.py — regression test for the kernel fp32-honesty gap.

Runs a kernel binary on the committed offender fixtures and checks the HONESTY PROPERTY:

    a tree reported converged (status==0) must NOT be worse in exact fp64 than its start.

For each fixture tree it runs CO from the original c_init, then recomputes the neutral
fp64 0.5·SSE at the returned c_final and at c_init. A tree FAILS if status==0 yet
fp64(c_final) > fp64(c_init). The fixtures are pre-selected offenders for the ESTABLISHED
kernels, so FD/AD FAIL here today (documenting the bug); a fixed kernel must PASS.

    # establish the baseline (expected: FAIL — these are the offenders):
    python cusr/kernel/tests/fp32_honesty_regression/check_fp32_honesty.py cusr/kernel/batch_lm_fusedfd
    python cusr/kernel/tests/fp32_honesty_regression/check_fp32_honesty.py cusr/kernel/batch_lm_ad
    # after an optimization, point it at the new binary; PASS == serious offenders cleared:
    python .../check_fp32_honesty.py path/to/optimized_kernel [--max-iter 1000] [--gpu 0]

Gate: exits non-zero if ANY serious (>2× worse) converged tree remains. Also reports the
softer counts (all converged-but-worse, catastrophic >100×) and how many of the manifest's
baseline-serious trees this binary still fails.
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

SERIOUS = 2.0
CATASTROPHIC = 100.0


def run(binary, max_iter, gpu):
    binary = Path(binary)
    assert binary.exists(), f"kernel binary not found: {binary}"
    manifest = json.loads((HERE / "manifest.json").read_text())
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tmp = Path(f"/tmp/fp32_honesty_check"); tmp.mkdir(parents=True, exist_ok=True)

    n_total = 0
    conv_worse = serious = catastrophic = 0
    base_serious = still_serious = 0
    failures = []
    for prob, fx in manifest["fixtures"].items():
        fbin = HERE / fx["file"]
        outd = tmp / fx["file"]; outd.mkdir(exist_ok=True)
        p = subprocess.run([str(binary), str(fbin), str(outd), "--max-iter", str(max_iter)],
                           capture_output=True, text=True, env=env, timeout=600)
        if p.returncode != 0:
            print(f"  {prob}: kernel exited {p.returncode}\n{p.stderr[-300:]}"); sys.exit(2)
        pop = read_pop_bin(fbin)
        status = np.fromfile(outd / "status.bin", dtype=np.int32)
        c_final = np.fromfile(outd / "c_final.bin", dtype=np.float32)
        X = pop["xs"]; y = pop["ym"][0]; metas = pop["metas"]
        for m in range(pop["M_prob"]):
            n_off, n_nodes, c_off, K = (int(metas[m, 0]), int(metas[m, 1]),
                                        int(metas[m, 2]), int(metas[m, 3]))
            sl = slice(n_off, n_off + n_nodes)
            nt, nv, ci = pop["nt"][sl], pop["nv"][sl], pop["ci"][sl]
            s0 = half_sse(nt, nv, ci, pop["c_init"][c_off:c_off + K], X, y)
            sf = half_sse(nt, nv, ci, c_final[c_off:c_off + K], X, y)
            n_total += 1
            rec = fx["trees"][m]
            if rec["fd"]["serious"] or rec["ad"]["serious"]:
                base_serious += 1
            if int(status[m]) == 0 and (not np.isfinite(sf) or sf > s0 * (1 + 1e-6) + 1e-9):
                conv_worse += 1
                ratio = float("inf") if not np.isfinite(sf) else sf / s0
                if ratio > CATASTROPHIC:
                    catastrophic += 1
                if ratio > SERIOUS:
                    serious += 1
                    if rec["fd"]["serious"] or rec["ad"]["serious"]:
                        still_serious += 1
                    failures.append((prob, rec["gen"], rec["orig_index"], s0, sf, ratio))

    print(f"\nkernel: {binary.name}   fixtures: {len(manifest['fixtures'])} problems, "
          f"{n_total} offender trees (baseline-serious {base_serious})")
    print(f"  converged-but-worse (status==0 & fp64>start) : {conv_worse}")
    print(f"  SERIOUS (>2x worse)                          : {serious}   <-- gate (must be 0)")
    print(f"  catastrophic (>100x worse)                   : {catastrophic}")
    print(f"  of the baseline-serious offenders, still serious here: {still_serious}/{base_serious}")
    if failures:
        print("  worst remaining (problem gen idx: start -> fp64_final = ratio):")
        for prob, g, idx, s0, sf, r in sorted(failures, key=lambda x: -x[5])[:8]:
            print(f"    {prob:16} g{g:<3} #{idx:<4} {s0:.4g} -> {sf:.4g}  = {r:.3g}x")
    ok = serious == 0
    print("\n" + ("PASS: no serious converged-but-worse trees (honesty gap cleared)"
                  if ok else f"FAIL: {serious} serious converged-but-worse trees remain"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help="path to a kernel binary (batch_lm_*)")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    sys.exit(0 if run(a.binary, a.max_iter, a.gpu) else 1)


if __name__ == "__main__":
    main()
