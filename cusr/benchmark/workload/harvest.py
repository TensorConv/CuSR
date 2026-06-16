"""harvest.py — parallel GP-snapshot harvest across GPUs (Phase 1 corpus).

Sweeps problems x noise x max_tree_len x seed, dumping geometric-generation
snapshots via dump_evogp.py. Shards cells across GPUs via a shared queue with one
worker per GPU (one evogp process per GPU at a time -> no co-tenancy). Resumable:
skips cells whose manifest.json already exists. Run in background; appends a log
to the snapshots root. The .bin are gitignored; each cell's manifest.json is the
committed record.

Defaults reproduce the Phase 1 broad corpus (pop=4000). Reuse for the large-M set
(task #19) by overriding, e.g.:
  uv run python cusr/benchmark/workload/harvest.py --pop 65536 \\
    --problems feynman/I.18.12,nguyen/5,feynman/I.6.2 --gens 8,100 \\
    --noises 0.0 --caps 64 --seeds 0 --gpus 0,1,2

Dry-run first to see the cell count + est storage:
  uv run python cusr/benchmark/workload/harvest.py --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]            # repo root
DUMP = ROOT / "cusr" / "kernel" / "dump_evogp.py"
OUT_ROOT = ROOT / "data" / "workload" / "snapshots"

ALL_PROBLEMS = [
    "feynman/I.12.1", "feynman/I.18.12", "feynman/I.27.6", "feynman/I.6.2",
    "feynman/I.12.2", "feynman/I.13.12", "feynman/II.3.24",
    "nguyen/1", "nguyen/2", "nguyen/3", "nguyen/4", "nguyen/5",
    "nguyen/6", "nguyen/7", "nguyen/8", "nguyen/9", "nguyen/10",
]


def cell_dir(ds, pop, noise, cap, seed):
    safe = ds.replace("/", "_")
    ntag = "0" if float(noise) == 0.0 else str(noise)
    return OUT_ROOT / f"{safe}_pop{pop}_noise{ntag}_len{cap}_seed{seed}"


def log(logf, msg, lock=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if lock:
        with lock:
            print(line, flush=True)
            logf.open("a").write(line + "\n")
    else:
        print(line, flush=True)
        logf.open("a").write(line + "\n")


def free_gpus(requested, thresh_mib=500):
    """Filter requested GPU indices to those with used memory < thresh (exclusive)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used = {}
    for ln in out.stdout.strip().splitlines():
        idx, mem = (x.strip() for x in ln.split(","))
        used[int(idx)] = int(mem)
    free, busy = [], []
    for g in requested:
        (free if used.get(g, 0) < thresh_mib else busy).append(g)
    return free, busy, used


def run_cell(cell, gpu, args, logf, lock):
    ds, noise, cap, seed = cell
    d = cell_dir(ds, args.pop, noise, cap, seed)
    if (d / "manifest.json").exists():
        log(logf, f"SKIP gpu{gpu} {d.name} (manifest exists)", lock)
        return "skip"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, str(DUMP), f"--dataset={ds}", f"--pop={args.pop}",
           f"--N={args.N}", f"--seed={seed}", f"--noise={noise}",
           f"--max-tree-len={cap}", f"--checkpoint-gens={args.gens}",
           "-o", str(d / "pop.bin")]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    ok = p.returncode == 0 and (d / "manifest.json").exists()
    if ok:
        log(logf, f"OK   gpu{gpu} {d.name}  {dt:.1f}s", lock)
        return "ok"
    log(logf, f"FAIL gpu{gpu} {d.name} rc={p.returncode}\n--- stderr tail ---\n{p.stderr[-800:]}", lock)
    return "fail"


def worker(gpu, q, args, logf, lock, tally):
    while True:
        try:
            cell = q.get_nowait()
        except queue.Empty:
            return
        try:
            status = run_cell(cell, gpu, args, logf, lock)
        except Exception as e:  # never let one cell kill the worker
            status = "fail"
            log(logf, f"FAIL gpu{gpu} {cell} EXC {e}", lock)
        with lock:
            tally[status] = tally.get(status, 0) + 1
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=4000)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--gens", default="0,1,2,4,8,16,32,64,100")
    ap.add_argument("--problems", default="all", help="'all' or comma list of dataset ids")
    ap.add_argument("--noises", default="0.0,0.01")
    ap.add_argument("--caps", default="32,64")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    problems = ALL_PROBLEMS if args.problems == "all" else args.problems.split(",")
    noises = [float(x) for x in args.noises.split(",")]
    caps = [int(x) for x in args.caps.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    requested = [int(g) for g in args.gpus.split(",")]
    cells = list(itertools.product(problems, noises, caps, seeds))
    n_snaps = len(args.gens.split(","))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logf = OUT_ROOT / "harvest_log.txt"

    free, busy, used = free_gpus(requested)
    done = sum(1 for c in cells if cell_dir(c[0], args.pop, c[1], c[2], c[3]).joinpath("manifest.json").exists())
    print(f"matrix: {len(problems)} problems x {len(noises)} noise x {len(caps)} caps "
          f"x {len(seeds)} seeds = {len(cells)} cells; {n_snaps} snapshots/cell "
          f"=> {len(cells)*n_snaps} snapshots @ pop={args.pop}")
    print(f"already done (manifest exists): {done}/{len(cells)} cells")
    print(f"GPUs requested={requested}  free={free}  busy(foreign?)={busy}  used_MiB={used}")
    if args.dry_run:
        return
    if not free:
        sys.exit("no free GPUs among requested — aborting (someone else is using them).")
    if busy:
        log(logf, f"WARN dropping busy GPUs {busy}; using {free}", None)

    q = queue.Queue()
    for c in cells:
        q.put(c)
    lock = threading.Lock()
    tally = {}
    log(logf, f"=== harvest start: {len(cells)} cells, {len(free)} GPUs {free}, pop={args.pop} ===", None)
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(g, q, args, logf, lock, tally), daemon=True)
               for g in free]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0
    log(logf, f"=== harvest done in {dt/60:.1f} min: {tally} ===", None)


if __name__ == "__main__":
    main()
