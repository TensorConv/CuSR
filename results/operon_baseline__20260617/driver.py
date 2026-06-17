"""driver.py — run the full 17-problem baseline (Operon CO + kernel FD/AD).

    python results/operon_baseline__20260617/driver.py --phase operon   # CPU, ~6 min (mi=500)
    python results/operon_baseline__20260617/driver.py --phase kernel   # GPU, ~6 min (mi=1000)
    python results/operon_baseline__20260617/driver.py --phase all

Defaults REPRODUCE the committed data: Operon mi=500 (it converges within), kernel mi=1000
(generous, so the kernel is never understated). Override with --operon-max-iter /
--kernel-max-iter. Operon phase: one subprocess per problem (operon venv, threads=1 for
determinism), pooled. Kernel phase: 17 problems x {fd,ad} = 34 jobs sharded ONE-PER-GPU over
the free pool (standing rule: leave 2 GPUs free). Resumable: skips a (problem,engine) whose
JSON already exists and parses.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/weish/hao/CuSR")
HERE = ROOT / "results/operon_baseline__20260617"
DATA = HERE / "data"
OPVENV = "/home/weish/hao/operon-venv/bin/python"
PROBLEMS = ["feynman/I.12.1", "feynman/I.18.12", "feynman/I.27.6", "feynman/I.6.2",
            "feynman/I.12.2", "feynman/I.13.12", "feynman/II.3.24",
            "nguyen/1", "nguyen/2", "nguyen/3", "nguyen/4", "nguyen/5", "nguyen/6",
            "nguyen/7", "nguyen/8", "nguyen/9", "nguyen/10"]
GPUS = [0, 1, 2, 3, 4, 5]        # leave 6,7 free (standing rule)
# Run-to-convergence caps (advisor blocking item #1), measured separately per engine:
#   Operon converges within 500 corpus-wide (max 498 iters, 0.08% of trees near the cap).
#   The kernel needs more: at 500 the worst nguyen cell still leaves ~19.6% at maxiter; at
#   1000 the corpus-worst maxiter fraction is 11.1% (most cells <3%). The achieved-loss
#   headline is robust to residual maxiter, but the kernel gets the more generous cap so it
#   is never understated. These defaults REPRODUCE the committed data (Operon 500, kernel 1000).
OPERON_MAX_ITER = 500
KERNEL_MAX_ITER = 1000


def _done(safe, eng):
    p = DATA / f"{safe}__{eng}.json"
    if not p.exists():
        return False
    try:
        json.loads(p.read_text()); return True
    except Exception:
        return False


def operon_phase(force=False, pool=16, max_iter=OPERON_MAX_ITER):
    jobs = [p for p in PROBLEMS if force or not _done(p.replace("/", "_"), "operon")]
    print(f"[operon] {len(jobs)} problems to run (pool={pool})", flush=True)
    q = queue.Queue()
    for p in jobs:
        q.put(p)
    lock = threading.Lock(); done = []

    def worker():
        while True:
            try:
                prob = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            r = subprocess.run(
                [OPVENV, str(HERE / "run_operon.py"), "--dataset", prob,
                 "--max-iter", str(max_iter), "--out", str(DATA)],
                capture_output=True, text=True)
            ok = r.returncode == 0 and _done(prob.replace("/", "_"), "operon")
            with lock:
                done.append((prob, ok))
                print(f"[operon] {prob:18} {'OK' if ok else 'FAIL'} "
                      f"({time.time()-t0:.0f}s)  [{len(done)}/{len(jobs)}]", flush=True)
                if not ok:
                    print(f"         stderr: {r.stderr[-400:]}", flush=True)
            q.task_done()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(min(pool, len(jobs) or 1))]
    for t in ts: t.start()
    for t in ts: t.join()
    fails = [p for p, ok in done if not ok]
    print(f"[operon] complete; {len(fails)} failures: {fails}", flush=True)
    return not fails


def kernel_phase(force=False, max_iter=KERNEL_MAX_ITER):
    jobs = [(p, k) for p in PROBLEMS for k in ("fd", "ad")
            if force or not _done(p.replace("/", "_"), k)]
    print(f"[kernel] {len(jobs)} (problem,kernel) jobs over GPUs {GPUS}", flush=True)
    q = queue.Queue()
    for j in jobs: q.put(j)
    lock = threading.Lock(); done = []

    def worker(gpu):
        while True:
            try:
                prob, k = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            r = subprocess.run(
                [OPVENV, str(HERE / "run_kernel.py"), "--dataset", prob, "--kernel", k,
                 "--gpu", str(gpu), "--max-iter", str(max_iter), "--out", str(DATA)],
                capture_output=True, text=True)
            ok = r.returncode == 0 and _done(prob.replace("/", "_"), k)
            with lock:
                done.append((prob, k, ok))
                print(f"[kernel] gpu{gpu} {prob:18} {k} {'OK' if ok else 'FAIL'} "
                      f"({time.time()-t0:.0f}s)  [{len(done)}/{len(jobs)}]", flush=True)
                if not ok:
                    print(f"         stderr: {r.stderr[-400:]}", flush=True)
            q.task_done()

    ts = [threading.Thread(target=worker, args=(g,), daemon=True) for g in GPUS]
    for t in ts: t.start()
    for t in ts: t.join()
    fails = [(p, k) for p, k, ok in done if not ok]
    print(f"[kernel] complete; {len(fails)} failures: {fails}", flush=True)
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["operon", "kernel", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    # Defaults reproduce the committed data (Operon 500, kernel 1000); override per engine if needed.
    ap.add_argument("--operon-max-iter", type=int, default=OPERON_MAX_ITER)
    ap.add_argument("--kernel-max-iter", type=int, default=KERNEL_MAX_ITER)
    a = ap.parse_args()
    ok = True
    if a.phase in ("operon", "all"):
        ok &= operon_phase(a.force, max_iter=a.operon_max_iter)
    if a.phase in ("kernel", "all"):
        ok &= kernel_phase(a.force, max_iter=a.kernel_max_iter)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
