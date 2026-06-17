"""run_kernel.py — kernel CO side of the baseline, ONE (problem, kernel) over all gens.

    python results/operon_baseline__20260617/run_kernel.py \\
        --dataset feynman/I.18.12 --kernel fd --gpu 0 --max-iter 200 --out <dir>

Runs the chosen kernel binary on the COMMITTED corpus ``pop_gen{g}.bin`` (the exact
trees Operon optimized — Probe B proved byte-identity), then scores each returned
``c_final`` with the neutral fp64 0.5*SSE (``lib.half_sse``) — the SAME arbiter used
for Operon. Records per-tree status + neutral loss at c_init (start) and c_final, plus
the kernel's own fp32 loss (to detect fp32 'lying': status==converged but fp64 loss bad).

  fd -> cusr/kernel/batch_lm_fusedfd  (finite-difference Jacobian, established baseline)
  ad -> cusr/kernel/batch_lm_ad       (forward-mode AD Jacobian, candidate; commit 90b8d1f)

Both share the CLI ``<pop.bin> <out_dir> [--max-iter N]`` and write status.bin (int32
M), c_final.bin (float32 total_c, per-tree-K jagged in pop.bin order), loss_final.bin
(float32 M). Status: 0=converged 1=maxiter 2=FAIL_NAN 3=K0-skip 4=FAIL_CHOLESKY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/weish/hao/CuSR")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/operon_baseline__20260617"))
from cusr.benchmark.workload.pop_io import read_pop_bin
from lib import half_sse, iter_trees, GENS, committed_cell_dir

SNAP = ROOT / "data/workload/snapshots"
BINS = {"fd": ROOT / "cusr/kernel/batch_lm_fusedfd",
        "ad": ROOT / "cusr/kernel/batch_lm_ad"}
SUMMARY_RE = re.compile(
    r"converged=(\d+) \(([\d.]+)%\).*?maxiter=(\d+).*?fail_nan=(\d+).*?"
    r"fail_cholesky=(\d+).*?k0_skip=(\d+)")


def run(dataset, kernel, gpu, max_iter, out_dir, gens=GENS):
    binary = BINS[kernel]
    assert binary.exists(), f"missing kernel binary {binary}"
    safe = dataset.replace("/", "_")
    cell = committed_cell_dir(SNAP, dataset)
    tmp = Path(f"/tmp/baseline_kernel/{kernel}_{safe}"); tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()

    result = dict(dataset=dataset, engine=f"kernel_{kernel}", binary=str(binary),
                  max_iter=max_iter, gpu=gpu, gens={})
    for g in gens:
        pop_bin = cell / f"pop_gen{g:04d}.bin"
        if not pop_bin.exists():
            result["gens"][str(g)] = dict(error="missing_bin", file=str(pop_bin)); continue
        outd = tmp / f"g{g}"; outd.mkdir(exist_ok=True)
        p = subprocess.run([str(binary), str(pop_bin), str(outd), "--max-iter", str(max_iter)],
                           capture_output=True, text=True, env=env, timeout=1200)
        if p.returncode != 0:
            result["gens"][str(g)] = dict(error="nonzero_exit", rc=p.returncode,
                                          stderr=p.stderr[-500:]); continue

        pop = read_pop_bin(pop_bin)
        M = pop["M_prob"]
        status = np.fromfile(outd / "status.bin", dtype=np.int32)
        c_final = np.fromfile(outd / "c_final.bin", dtype=np.float32)
        kloss = np.fromfile(outd / "loss_final.bin", dtype=np.float32)  # kernel's own fp32 loss
        assert len(status) == M, f"status len {len(status)} != M {M}"
        assert len(c_final) == pop["total_c"], "c_final len != total_c"
        X = pop["xs"]; y = pop["ym"][0]   # all trees share one y

        neut_start, neut_final, Ks = [], [], []
        for m, nt, nv, ci, K, c_off, _ in iter_trees(pop):
            Ks.append(K)
            c0 = pop["c_init"][c_off:c_off + K]
            cf = c_final[c_off:c_off + K]
            neut_start.append(half_sse(nt, nv, ci, c0, X, y))   # K=0: well-defined fixed loss
            neut_final.append(half_sse(nt, nv, ci, cf, X, y))

        mt = SUMMARY_RE.search(p.stdout)
        summ = None
        if mt:
            conv, pct, mit, fnan, fchol, k0 = mt.groups()
            summ = dict(converged=int(conv), conv_pct=float(pct), maxiter=int(mit),
                        fail_nan=int(fnan), fail_cholesky=int(fchol), k0_skip=int(k0))
        result["gens"][str(g)] = dict(
            M=M, K=Ks, status=status.tolist(),
            neutral_loss_start=neut_start, neutral_loss_final=neut_final,
            kernel_fp32_loss=kloss.tolist(), summary=summ)
        nconv = int((status == 0).sum())
        print(f"[kernel {kernel} {dataset} g{g:>3}] M={M} conv={nconv} "
              f"chol={int((status==4).sum())} nan={int((status==2).sum())} "
              f"k0={int((status==3).sum())} maxit={int((status==1).sum())}", flush=True)

    result["wall_s"] = round(time.time() - t0, 1)
    out = Path(out_dir) / f"{safe}__{kernel}.json"
    out.write_text(json.dumps(result))
    print(f"[kernel {kernel} {dataset}] -> {out}  ({result['wall_s']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kernel", choices=["fd", "ad"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "results/operon_baseline__20260617/data"))
    ap.add_argument("--gens", default=",".join(map(str, GENS)))
    a = ap.parse_args()
    gens = [int(x) for x in a.gens.split(",") if x.strip()]
    run(a.dataset, a.kernel, a.gpu, a.max_iter, a.out, gens)


if __name__ == "__main__":
    main()
