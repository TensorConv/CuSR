"""operon_harvest.py — parallel Operon GP-snapshot harvest (Operon/CPU driver).

The Operon counterpart of ``cusr/benchmark/workload/harvest.py``. Sweeps
``problems x noise x cap x seed`` and dumps geometric-generation population
snapshots via ``operon_dump.py``. Operon is **CPU-only**, so cells are sharded
across **processes** (one ``operon_dump`` subprocess per cell, for crash
isolation — exactly as ``harvest.py`` shards across GPUs by spawning one
``dump_evogp`` process per cell). Concurrency is a worker count, not a GPU pool.

Resumable + idempotent: a cell is skipped iff its ``manifest.json`` already
exists **and parses as JSON**. ``operon_dump`` writes the manifest **last** and
**atomically** (temp + rename), so a crashed cell leaves ``.bin`` but no manifest
(or, in the rare torn-write case, an unparseable one) and is re-run cleanly — the
dump is deterministic, hence the overwrite is byte-identical. Appends a log to the
snapshots root. The ``.bin`` are gitignored; each cell's ``manifest.json`` is the
committed record.

Thread budget (honesty / spec §9): the box is 2x EPYC 7763 = 256 hardware
threads. Each cell pins ``--threads`` (default 1) for determinism, and the driver
refuses to launch when ``procs * threads > 256`` (oversubscription). The budget is
logged at start and ``threads`` is recorded in every per-cell manifest by
``operon_dump``.

Defaults reproduce the Phase-1 broad corpus comparable to the evogp corpus
(same ``ALL_PROBLEMS`` imported from ``harvest.py``, same noises/caps/seeds/gens,
pop=4000, N=1000). Override for a targeted run, e.g.::

    /home/weish/hao/operon-venv/bin/python cusr/benchmark/workload/operon_harvest.py \\
        --problems feynman/I.18.12,nguyen/5 --gens 8,100 --noises 0.0 \\
        --caps 64 --seeds 0 --procs 8

Dry-run first to see the cell matrix + count + est storage::

    /home/weish/hao/operon-venv/bin/python cusr/benchmark/workload/operon_harvest.py --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]                 # repo root = .../CuSR
# Allow `python cusr/benchmark/workload/operon_harvest.py` (run-by-path, like
# harvest.py) as well as `python -m ...`: put the repo root on sys.path before the
# `cusr.*` import below so it resolves regardless of cwd / invocation style.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same problem set as the evogp harvest (single source of truth — imported, not
# re-listed, so the two corpora can never drift). harvest.py has no torch/evogp
# import at module scope, so this resolves under the operon-venv python too.
from cusr.benchmark.workload.harvest import ALL_PROBLEMS  # noqa: E402
# operon_dump is imported as a MODULE (it imports cusr.benchmark.workload.*,
# which is the repo, not a pip-installed package) — a bare file-path call would
# fail to resolve those imports. Always launch it with `-m` and cwd=ROOT.
DUMP_MODULE = "cusr.benchmark.workload.operon_dump"
DEFAULT_OUT_ROOT = ROOT / "data" / "workload" / "snapshots"

MAX_THREADS = 256          # 2x EPYC 7763 = 256 hardware threads; never oversubscribe.
LOG_NAME = "operon_harvest_log.txt"
# Per-snapshot adapter drop fields recorded in each manifest's ``drop_totals`` (the
# anti-cheat / honesty record — every dropped tree is counted, never silent).
DROP_KEYS = ("n_in", "n_kept", "dropped_unsupported", "dropped_k_over",
             "dropped_bad_type", "dropped_nonfinite")


def cell_dir(out_root: Path, ds: str, pop: int, noise, cap: int, seed: int) -> Path:
    """Per-cell snapshot directory. Mirrors ``harvest.py::cell_dir`` byte-for-byte
    except for the ``operon_`` prefix (so the dir matches ``snapshots/operon_*`` and
    never collides with an evogp cell in the shared root)."""
    safe = ds.replace("/", "_")
    ntag = "0" if float(noise) == 0.0 else str(noise)
    return out_root / f"operon_{safe}_pop{pop}_noise{ntag}_len{cap}_seed{seed}"


def enumerate_cells(problems, noises, caps, seeds):
    """The cell matrix as a list of ``(ds, noise, cap, seed)`` tuples (cartesian
    product, same axis order as ``harvest.py``). Importable so tests assert the
    count/contents directly rather than scraping stdout."""
    return list(itertools.product(problems, noises, caps, seeds))


def est_cell_bytes(n_snaps: int, pop: int, N: int, n_vars: int,
                   mean_nodes: float = 50.0, mean_K: float = 22.0) -> int:
    """Approximate on-disk bytes for one cell's snapshots. The pop.bin layout
    (``pop_format.h``): per snapshot
    ``64 + total_nodes*12 + M*16 + total_c*4 + N*n_vars*4 + M*N*4``. The
    ``M*N*4`` ym-replication term (one shared y tiled M times) dominates. We can't
    know mean nodes/K before running, so we assume spec-§10 cap64/gen100 figures
    (~50 nodes, ~22 K) — the estimate is approximate and labelled as such."""
    total_nodes = pop * mean_nodes
    total_c = pop * mean_K
    per_snap = 64 + total_nodes * 12 + pop * 16 + total_c * 4 + N * n_vars * 4 + pop * N * 4
    return int(per_snap * n_snaps)


def _mean_n_vars(problems) -> float:
    """Mean variable count across the swept problems (for the storage estimate).
    Falls back to 2.0 if sr_problems can't resolve them (estimate-only path)."""
    try:
        from cusr.benchmark.workload.sr_problems import get_problem
        nv = [len(get_problem(p)["variables"]) for p in problems]
        return sum(nv) / len(nv) if nv else 2.0
    except Exception:
        return 2.0


def log(logf: Path, msg: str, lock: threading.Lock | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if lock is not None:
        with lock:
            print(line, flush=True)
            with logf.open("a") as fh:
                fh.write(line + "\n")
    else:
        print(line, flush=True)
        with logf.open("a") as fh:
            fh.write(line + "\n")


def run_cell(cell, args, logf: Path, lock: threading.Lock, python: str) -> str:
    """Run one cell as an ``operon_dump`` subprocess (crash isolation). Returns
    ``"skip"`` / ``"ok"`` / ``"fail"``. Skips iff the cell's manifest already
    exists (resume); otherwise the dump (re)writes every snapshot + the manifest."""
    ds, noise, cap, seed = cell
    d = cell_dir(args.out_root, ds, args.pop, noise, cap, seed)
    mf = d / "manifest.json"
    # Resume = skip iff the manifest exists AND parses as JSON. A truncated manifest from a
    # crash mid-write fails to parse, so it is re-run (regenerated) instead of being SKIPPED
    # forever and silently dropped from the corpus-wide honesty totals (the dump is
    # deterministic, so the re-run is byte-identical).
    if mf.exists():
        try:
            json.loads(mf.read_text())
            log(logf, f"SKIP {d.name} (manifest exists)", lock)
            return "skip"
        except Exception as e:
            log(logf, f"REDO {d.name} (manifest corrupt, re-running): {e!r}", lock)
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, "-m", DUMP_MODULE,
        f"--dataset={ds}", f"--pop={args.pop}", f"--N={args.N}",
        f"--seed={seed}", f"--noise={noise}", f"--cap={cap}",
        f"--checkpoint-gens={args.gens}", f"--threads={args.threads}",
        "-o", str(d / "pop.bin"),
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dt = time.time() - t0
    ok = p.returncode == 0 and (d / "manifest.json").exists()
    if ok:
        log(logf, f"OK   {d.name}  {dt:.1f}s", lock)
        return "ok"
    log(logf, f"FAIL {d.name} rc={p.returncode}\n--- stderr tail ---\n{p.stderr[-800:]}", lock)
    return "fail"


def worker(q: queue.Queue, args, logf: Path, lock: threading.Lock,
           tally: dict, python: str) -> None:
    while True:
        try:
            cell = q.get_nowait()
        except queue.Empty:
            return
        try:
            status = run_cell(cell, args, logf, lock, python)
        except Exception as e:               # never let one cell kill the worker
            status = "fail"
            log(logf, f"FAIL {cell} EXC {e!r}", lock)
        with lock:
            tally[status] = tally.get(status, 0) + 1
        q.task_done()


def aggregate_drops(cells, args) -> tuple[dict, int]:
    """Sum the per-cell ``drop_totals`` across every cell whose manifest exists, so
    the DRIVER (the only component that sees all problems/seeds) can surface the
    corpus-wide honesty record (spec §9): total kept/in + every dropped_* count,
    K-over expected ~0. Reads the committed manifests (robust — not stdout scraping)."""
    agg = {k: 0 for k in DROP_KEYS}
    n_manifests = 0
    for c in cells:
        mp = (cell_dir(args.out_root, c[0], args.pop, c[1], c[2], c[3]) / "manifest.json")
        if not mp.exists():
            continue
        try:
            dt_ = json.loads(mp.read_text()).get("drop_totals", {})
        except Exception:
            continue
        n_manifests += 1
        for k in DROP_KEYS:
            agg[k] += int(dt_.get(k, 0))
    return agg, n_manifests


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Operon/CPU harvest driver: shard GP cells across processes "
                    "(one operon_dump subprocess per cell), resumable, thread-budgeted.")
    ap.add_argument("--pop", type=int, default=4000)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--gens", default="0,1,2,4,8,16,32,64,100",
                    help="checkpoint generations passed through to operon_dump")
    ap.add_argument("--problems", default="all",
                    help="'all' (= harvest.py ALL_PROBLEMS) or comma list of dataset ids")
    ap.add_argument("--noises", default="0.0,0.01")
    ap.add_argument("--caps", default="32,64")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 32),
                    help="concurrent cells (worker processes). procs*threads must be <= 256.")
    ap.add_argument("--threads", type=int, default=1,
                    help="Operon threads PER cell (pinned to 1 for determinism; disclosed "
                         "in every manifest). procs*threads must not oversubscribe the 256-thread box.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="snapshots root (cells go under out_root/operon_*); overridable for tests")
    ap.add_argument("--python", default=sys.executable,
                    help="python used to launch operon_dump (default: the running interpreter; "
                         "must be the operon-venv python that has pyoperon)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cell matrix + count + est storage and exit (no work)")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    problems = ALL_PROBLEMS if args.problems == "all" else args.problems.split(",")
    noises = [float(x) for x in args.noises.split(",")]
    caps = [int(x) for x in args.caps.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    cells = enumerate_cells(problems, noises, caps, seeds)
    n_snaps = len([g for g in args.gens.split(",") if g.strip()])

    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    logf = args.out_root / LOG_NAME

    done = sum(1 for c in cells
               if cell_dir(args.out_root, c[0], args.pop, c[1], c[2], c[3])
               .joinpath("manifest.json").exists())

    n_vars_mean = _mean_n_vars(problems)
    per_cell = est_cell_bytes(n_snaps, args.pop, args.N, n_vars_mean)
    todo = len(cells) - done
    matrix_line = (
        f"matrix: {len(problems)} problems x {len(noises)} noise x {len(caps)} caps "
        f"x {len(seeds)} seeds = {len(cells)} cells; {n_snaps} snapshots/cell "
        f"=> {len(cells) * n_snaps} snapshots @ pop={args.pop}, N={args.N}")
    print(matrix_line)
    print(f"already done (manifest exists): {done}/{len(cells)} cells; {todo} to run")
    print(f"est storage: ~{_fmt_bytes(per_cell)}/cell (approx; assumes ~50 nodes/~22 K, "
          f"mean n_vars~{n_vars_mean:.1f}) => ~{_fmt_bytes(per_cell * len(cells))} full / "
          f"~{_fmt_bytes(per_cell * todo)} remaining")
    print(f"concurrency: procs={args.procs} x threads={args.threads} "
          f"= {args.procs * args.threads} threads (budget {MAX_THREADS}); python={args.python}")

    if args.dry_run:
        return 0

    # Thread budget guard (honesty §9 — never oversubscribe the box; disclose it).
    if args.procs * args.threads > MAX_THREADS:
        sys.exit(f"refusing to launch: procs({args.procs}) x threads({args.threads}) = "
                 f"{args.procs * args.threads} > {MAX_THREADS} (would oversubscribe the "
                 f"2x EPYC 7763 = 256-thread box). Lower --procs or --threads.")
    if not cells:
        sys.exit("empty cell matrix — nothing to do.")

    procs = max(1, min(args.procs, len(cells)))
    q: queue.Queue = queue.Queue()
    for c in cells:
        q.put(c)
    lock = threading.Lock()
    tally: dict = {}
    log(logf, f"=== operon harvest start: {len(cells)} cells ({done} already done), "
              f"procs={procs} x threads={args.threads} = {procs * args.threads} threads, "
              f"pop={args.pop}, N={args.N}, gens={args.gens} ===", None)
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(q, args, logf, lock, tally, args.python),
                                daemon=True) for _ in range(procs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0

    # Honesty (spec §9): the driver sees ALL problems/seeds, so it logs the
    # corpus-wide drop totals from the committed manifests — total kept/in + every
    # dropped_* count, and verifies K-over stays ~0 across the whole sweep.
    agg, n_manifests = aggregate_drops(cells, args)
    log(logf, f"=== drop totals over {n_manifests} cells: kept {agg['n_kept']}/{agg['n_in']} "
              f"trees; K-over={agg['dropped_k_over']} (expect ~0) "
              f"unsupported={agg['dropped_unsupported']} bad_type={agg['dropped_bad_type']} "
              f"nonfinite={agg['dropped_nonfinite']} ===", None)
    if agg["dropped_k_over"]:
        log(logf, f"WARN K-over={agg['dropped_k_over']} > 0 across the corpus — investigate "
                  f"(spec §10 measured ~0 on I.18.12; verify it holds for all problems)", None)

    log(logf, f"=== operon harvest done in {dt / 60:.1f} min: {tally} ===", None)
    return 0 if tally.get("fail", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
