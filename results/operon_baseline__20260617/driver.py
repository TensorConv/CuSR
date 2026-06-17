"""driver.py — run the full 17-problem baseline (Operon CO + kernel FD/AD).

    python results/operon_baseline__20260617/driver.py --phase operon   # CPU, ~4 min
    python results/operon_baseline__20260617/driver.py --phase kernel   # GPU, ~2 min
    python results/operon_baseline__20260617/driver.py --phase all

Operon phase: one subprocess per problem (operon venv, threads=1 for determinism),
pooled. Kernel phase: 17 problems x {fd,ad} = 34 jobs sharded ONE-PER-GPU over the
free pool (standing rule: leave 2 GPUs free; one timed job per card). Resumable:
skips a (problem,engine) whose JSON already exists and parses.
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
# Generous cap so BOTH engines run to convergence (advisor blocking item #1). Sensitivity
# probe (I.18.12 g64): at 500 the kernel maxiter fraction is <3% (FD 91, AD 111 of 4000)
# and the Cholesky count is stable from 200->500 (rank-deficient trees aren't rescued by
# more iters). Operon also hits ~196 iters on a few trees, so it gets the same cap.
MAX_ITER = 500


def _done(safe, eng):
    p = DATA / f"{safe}__{eng}.json"
    if not p.exists():
        return False
    try:
        json.loads(p.read_text()); return True
    except Exception:
        return False


def operon_phase(force=False, pool=16):
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
                 "--max-iter", str(MAX_ITER), "--out", str(DATA)],
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


def kernel_phase(force=False):
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
                 "--gpu", str(gpu), "--max-iter", str(MAX_ITER), "--out", str(DATA)],
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
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    a = ap.parse_args()
    globals()["MAX_ITER"] = a.max_iter
    ok = True
    if a.phase in ("operon", "all"):
        ok &= operon_phase(a.force)
    if a.phase in ("kernel", "all"):
        ok &= kernel_phase(a.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
