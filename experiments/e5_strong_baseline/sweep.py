#!/usr/bin/env python
"""sweep.py — e5 strong-baseline THROUGHPUT sweep (DRAFT, honesty-first).

Primary figure: THROUGHPUT (trees/sec = (M - n_dropped)/wall_median) vs M for the
GPU CO kernel, with Operon 1-core / 32-core / 64-core points, and iso-quality
reported alongside (a kernel point that fails the iso-quality gate is NOT a clean
speedup and is flagged as such in the JSON + on every derived metric).

This generalises the M=4000 prelim (prelim.py) across M in {1k,4k,16k,64k} x 3
presets, and adds R2 + per-constant recovery, an iter-Pareto sub-sweep, parallel
efficiency / crossover-core estimates, and the 64k single-core project-or-omit
decision — all under strict incremental/resumable JSONL writes.

HONESTY RULES enforced (see test_sweep_honesty.py):
 1. SAME pop object to both backends per (preset,M) — get_pop() memoises it.
 2. loss/R2/recovery recomputed in fp64 from c_final + pop (interp.loss_pop); the
    iso-quality gate uses the COMMON-SET (paired, both-finite) median, not each
    backend's own surviving subset (audit#2 fix).
 3. NO silent skip/cap/projection: every dropped tree, capped rep, projected or
    omitted point is recorded in the JSON with a reason AND printed to stdout.
 4. Timed region includes marshaling: kernel .optimize re-coerces+H2D/D2H every
    call; Operon timing uses wall_e2e (includes the per-generation tree_view
    payload loop, audit#3 fix). wall_e2e - wall_core recorded.
 5. Incremental: each (preset,M,backend) result appended as a JSONL line as it
    completes; resume skips done configs; sweep.DONE only after a clean finish.
 6. throughput = (M - n_dropped)/wall_median, never M/wall (audit#4 fix).
 7. clocks UNLOCKED -> stamped DRAFT in meta.

Run (full sweep, ~1hr on unlocked clocks):
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/e5_strong_baseline/sweep.py
Smoke (tiny, separate jsonl):
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/e5_strong_baseline/sweep.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# ---- sweep config (the spec) ------------------------------------------------
MS = [1000, 4000, 16000, 64000]
ALL_PRESETS = ["early-gen", "late-gen-bloated", "inner-const-heavy"]
N = 1000
SEED = 0
KERNEL_MAX_ITER = 50            # headline operating point (batch runs to cap)
OPERON_MAX_ITER = 200
DEVICE_ID = 0
N_REP_KERNEL = 3                # >=3 reps, median (clocks unlocked)
N_REP_PARALLEL = 3             # >=3 reps for operon 32/64
N_REP_OP1 = 2                  # >=2 for single-core; 16k may be 1 (logged)
OP1_MS = [1000, 4000, 16000]   # single-core measured here; 64k projected/omitted
OP_PARALLEL_NPROC = [32, 64]
PARETO_M = 16000
PARETO_PRESET = "inner-const-heavy"
PARETO_ITERS = [25, 50, 100, 200]
QUALITY_TOL = 1.05              # iso-quality gate multiplier
FLAT_SPREAD_TOL = 0.25          # >25% spread => do not project 64k single-core

# kernel raw status -> unified (4=FAIL_CHOLESKY -> failed); see backends.py
_KERNEL_STATUS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2}


# ===========================================================================
# PURE HELPERS (unit-tested in test_sweep_honesty.py) — keep side-effect-free.
# ===========================================================================
def median(xs):
    return statistics.median(xs)


def quality_matched(med_kernel: float, med_operon: float,
                    tol: float = QUALITY_TOL) -> bool:
    """Exact iso-quality gate: kernel common-set median <= operon * tol.
    Non-finite either side => cannot certify => False."""
    if not (np.isfinite(med_kernel) and np.isfinite(med_operon)):
        return False
    return bool(med_kernel <= med_operon * tol)


def common_set_median(loss_k: np.ndarray, loss_op: np.ndarray):
    """Paired median over trees BOTH backends gave a finite loss on (audit#2).
    Returns (med_kernel, med_operon, n_common). Empty common set => (inf,inf,0)."""
    loss_k = np.asarray(loss_k, np.float64)
    loss_op = np.asarray(loss_op, np.float64)
    common = np.isfinite(loss_k) & np.isfinite(loss_op)
    n = int(common.sum())
    if n == 0:
        return float("inf"), float("inf"), 0
    return float(np.median(loss_k[common])), float(np.median(loss_op[common])), n


def throughput(M: int, n_dropped: int, wall: float) -> float:
    """trees/sec on the work the kernel actually did (audit#4): exclude dropped."""
    if wall <= 0:
        return float("nan")
    return (M - n_dropped) / wall


def decide_projection(throughputs: dict):
    """64k single-core project-or-omit (rule 3 / audit#8).
    throughputs: {M: per-core trees/sec} at the measured single-core M's.
    Returns (action, spread, reason, projected_tput):
      action 'projected' if max spread <= FLAT_SPREAD_TOL, else 'omitted'.
    spread = (max-min)/min over the measured per-core throughputs."""
    vals = [v for v in throughputs.values() if v is not None and np.isfinite(v) and v > 0]
    if len(vals) < 2:
        return ("omitted", float("nan"),
                "omitted (cannot assess flatness: <2 measured single-core points)",
                None)
    lo, hi = min(vals), max(vals)
    spread = (hi - lo) / lo
    if spread <= FLAT_SPREAD_TOL:
        proj = statistics.median(vals)
        reason = (f"projected from measured single-core throughput "
                  f"(per-core spread {spread:.1%} <= {FLAT_SPREAD_TOL:.0%}); "
                  f"projected per-core trees/sec={proj:.1f}")
        return ("projected", spread, reason, proj)
    reason = (f"omitted (throughput not flat: {spread:.1%} > {FLAT_SPREAD_TOL:.0%} "
              f"spread across measured single-core M)")
    return ("omitted", spread, reason, None)


def expected_configs(ms, presets):
    """Enumerate the asymmetric plan as (backend, preset, M, knob) keys.
    knob = kernel max_iter for kernel, nproc for operon."""
    plan = set()
    for preset in presets:
        for M in ms:
            plan.add(("kernel", preset, M, KERNEL_MAX_ITER))
            for nproc in OP_PARALLEL_NPROC:
                plan.add(("operon", preset, M, nproc))
            if M in OP1_MS:
                plan.add(("operon", preset, M, 1))
    # iter-Pareto sub-sweep: kernel at extra iters, 16k inner-const-heavy only
    if PARETO_M in ms and PARETO_PRESET in presets:
        for it in PARETO_ITERS:
            plan.add(("kernel", PARETO_PRESET, PARETO_M, it))
    return plan


def config_key(rec: dict):
    return (rec["backend"], rec["preset"], int(rec["M"]), int(rec["knob"]))


def missing_configs(done, plan):
    return sorted(set(plan) - set(done), key=lambda k: (k[1], k[2], k[0], k[3]))


def timed_reps(op, n_rep: int):
    """Call op() FRESH n_rep times (rule 4: per-gen marshaling re-incurred each
    rep). Returns (walls, results) with len==n_rep. No hoisting of the op."""
    walls, results = [], []
    for _ in range(n_rep):
        t = time.perf_counter()
        r = op()
        walls.append(time.perf_counter() - t)
        results.append(r)
    return walls, results


# ===========================================================================
# fp64 quality metrics (recomputed, never backend-reported)
# ===========================================================================
def per_tree_metrics(pop, c_final):
    """Returns (losses fp64, r2 per finite tree, recovery per constant).
    losses: (M,) fp64 (inf for non-finite trees).
    r2:     median over finite trees of 1 - sum(r^2)/sum((y-mean)^2).
    recovery: median over constants of |c_hat - c_true|/max(|c_true|,1e-9)."""
    c_final = np.asarray(c_final, np.float64)
    losses = interp.loss_pop(pop, c_final)
    M = pop["M"]
    r2s = []
    for m in range(M):
        _, _, c_off, K = pop["metas"][m].tolist()
        if K == 0:
            continue
        try:
            yhat = interp.eval_pop_tree(pop, m, c_final[c_off:c_off + K])
        except Exception:  # noqa: BLE001
            continue
        if not np.all(np.isfinite(yhat)):
            continue
        y = np.asarray(pop["ym"][m], np.float64)
        ss_res = float(np.sum((yhat - y) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0 and np.isfinite(ss_res):
            r2s.append(1.0 - ss_res / ss_tot)
    med_r2 = float(np.median(r2s)) if r2s else float("nan")
    c_true = np.asarray(pop["c_true"], np.float64)
    if c_true.size == c_final.size and c_true.size:
        rel = np.abs(c_final - c_true) / np.maximum(np.abs(c_true), 1e-9)
        med_recovery = float(np.median(rel[np.isfinite(rel)]))
    else:
        med_recovery = float("nan")
    return losses, med_r2, med_recovery


def med_finite(losses):
    finite = losses[np.isfinite(losses)]
    return float(np.median(finite)) if finite.size else float("inf")


def status_frac(status, code):
    return float(np.mean(status == code)) if len(status) else 0.0


# ===========================================================================
# pop memoisation (rule 1: identical object to both backends)
# ===========================================================================
_POP_CACHE: dict = {}


def get_pop(preset: str, M: int, n: int, seed: int):
    """Memoised so both backends at one (preset,M) get the IDENTICAL object —
    never regenerated between backends (rule 1)."""
    from cusr.benchmark.workload.gen_synth import gen_pop
    key = (preset, M, n, seed)
    pop = _POP_CACHE.get(key)
    if pop is None:
        pop = gen_pop(preset, M, n, seed)
        _POP_CACHE[key] = pop
    return pop


def evict_pop(preset, M, n, seed):
    """Free a pop after both backends finished it (64k pops are large)."""
    _POP_CACHE.pop((preset, M, n, seed), None)


# imported lazily so the pure helpers + tests don't require the GPU/.so at import
from cusr.benchmark import interp  # noqa: E402


# ===========================================================================
# TIMED BACKEND CALLS
# ===========================================================================
def run_kernel(pop, max_iter, n_rep):
    """In-process .so (deployment path), warm on the FULL pop (untimed), then
    median of n_rep timed reps. Timed region = .optimize (re-coerce + H2D/D2H =
    the real per-generation cost; do NOT hoist marshaling)."""
    from cusr.kernel.co_inproc import get_inproc_co
    co = get_inproc_co(device_id=DEVICE_ID, variant="fd")
    co.optimize(pop, max_iter=max_iter)                      # warm on FULL pop
    walls, results = timed_reps(lambda: co.optimize(pop, max_iter=max_iter), n_rep)
    res = results[-1]
    status = np.array([_KERNEL_STATUS_MAP.get(int(s), 2) for s in res["status"]],
                      np.int32)
    losses, med_r2, med_recovery = per_tree_metrics(pop, res["c_final"])
    n_dropped = int(np.sum(status == 3))                    # K0 trees kernel skips
    return dict(
        wall_median=median(walls), wall_reps=list(walls),
        losses=losses, med_loss=med_finite(losses),
        med_r2=med_r2, med_recovery=med_recovery,
        frac_converged=status_frac(status, 0), frac_failed=status_frac(status, 2),
        n_dropped=n_dropped, n_rep=n_rep,
    )


def run_operon(pop, nproc, n_rep):
    """Operon persistent pool. WARM on the FULL pop (untimed), median of n_rep
    timed reps. Timed metric = wall_e2e (includes the per-generation tree_view
    payload loop, audit#3); wall_e2e - wall_core recorded."""
    from cusr.benchmark.backends import OperonLM
    backend = OperonLM(nproc=nproc, max_iter=OPERON_MAX_ITER)
    backend.fit_pop(pop)                                     # warm on FULL pop
    walls, results = timed_reps(lambda: backend.fit_pop(pop), n_rep)
    # use the e2e wall (apples-to-apples with kernel marshaling) as the timed wall
    e2e = [r.wall_e2e for r in results]
    core = [r.wall_core for r in results]
    res = results[-1]
    losses, med_r2, med_recovery = per_tree_metrics(pop, res.c_final)
    at_limit = np.logical_and(res.status == 1, res.n_iter >= OPERON_MAX_ITER)
    n_dropped = int(np.sum(res.status == 3))
    return dict(
        wall_median=median(e2e), wall_reps=list(e2e),
        wall_core_median=median(core),
        wall_marshal_overhead=median(e2e) - median(core),
        losses=losses, med_loss=med_finite(losses),
        med_r2=med_r2, med_recovery=med_recovery,
        frac_converged=status_frac(res.status, 0), frac_failed=status_frac(res.status, 2),
        frac_at_iter_limit=float(np.mean(at_limit)) if len(res.status) else 0.0,
        n_dropped=n_dropped, n_rep=n_rep,
    )


# ===========================================================================
# JSONL incremental write + resume
# ===========================================================================
def load_done(jsonl_path: Path):
    """Return (done_keys, records). Tolerate a half-written trailing line."""
    done, recs = [], []
    if not jsonl_path.exists():
        return done, recs
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                                         # crash mid-write
        recs.append(rec)
        done.append(config_key(rec))
    return done, recs


def append_record(jsonl_path: Path, rec: dict):
    with jsonl_path.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _strip_losses(rec: dict):
    """Per-tree loss arrays are huge; keep them out of the persisted record but
    fold their summary in (medians/n_dropped already computed)."""
    return {k: v for k, v in rec.items() if k != "losses"}


# ===========================================================================
# MAIN SWEEP
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: M in {200,800}, inner-const only, nproc=8, 2 reps")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip configs already in the jsonl (default on)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore+overwrite existing jsonl (no resume)")
    args = ap.parse_args()

    ms, presets = MS, ALL_PRESETS
    op1_ms = OP1_MS
    parallel_nproc = OP_PARALLEL_NPROC
    n_rep_kernel, n_rep_parallel, n_rep_op1 = N_REP_KERNEL, N_REP_PARALLEL, N_REP_OP1
    pareto_iters = PARETO_ITERS
    tag = "sweep"
    if args.smoke:
        ms, presets = [200, 800], ["inner-const-heavy"]
        op1_ms = [200, 800]
        parallel_nproc = [8]
        n_rep_kernel, n_rep_parallel, n_rep_op1 = 2, 2, 2
        pareto_iters = []                                    # no pareto in smoke
        tag = "sweep_smoke"

    jsonl = OUT / f"{tag}.jsonl"
    done_marker = OUT / f"{tag}.DONE"
    done_marker.unlink(missing_ok=True)
    if args.fresh:
        jsonl.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(OUT / f"{tag}_losses", ignore_errors=True)  # no stale losses

    # build the expected plan FOR THIS RUN (smoke has its own asymmetry)
    plan = set()
    for preset in presets:
        for M in ms:
            plan.add(("kernel", preset, M, KERNEL_MAX_ITER))
            for nproc in parallel_nproc:
                plan.add(("operon", preset, M, nproc))
            if M in op1_ms:
                plan.add(("operon", preset, M, 1))
    if not args.smoke and PARETO_M in ms and PARETO_PRESET in presets:
        for it in pareto_iters:
            plan.add(("kernel", PARETO_PRESET, PARETO_M, it))

    done, _ = load_done(jsonl)
    done_set = set(done)
    n_cpu = os.cpu_count()
    clk = "UNLOCKED-draft"
    print(f"# e5 strong-baseline THROUGHPUT sweep  tag={tag}  N={N} seed={SEED}  "
          f"GPU={os.environ['CUDA_VISIBLE_DEVICES']}  clocks={clk}  cpu={n_cpu}")
    print(f"# plan={len(plan)} configs; resuming, {len(done_set)} already done")
    print(f"# kernel=in-process libcusr_co_fd.so it={KERNEL_MAX_ITER}; "
          f"operon it={OPERON_MAX_ITER}; quality_tol={QUALITY_TOL}\n")

    def already(key):
        return key in done_set

    def record(rec):
        # persist per-tree fp64 losses to a DURABLE sidecar (.npy) for the paired
        # common-set iso-quality gate (final pass), then strip from the jsonl line
        # (the arrays are huge; the medians/n_dropped are kept). Durable (not just
        # in-memory) so a crashed+resumed run still gets the PAIRED gate — the
        # resume path is the expected path for the 64k sweep on unlocked clocks,
        # and the per-backend-median fallback is the audit#2-forbidden basis.
        if "losses" in rec:
            key = config_key(rec)
            arr = np.asarray(rec["losses"], np.float64)
            _LOSS_SIDECAR[key] = arr
            save_losses(tag, key, arr)
        append_record(jsonl, _strip_losses(rec))
        done_set.add(config_key(rec))

    # ----- per-(preset,M): run kernel headline + operon points on the SAME pop
    for preset in presets:
        for M in ms:
            pop = get_pop(preset, M, N, SEED)
            K_mean = float(pop["metas"][:, 3].mean())
            noise_floor = med_finite(interp.loss_pop(pop, np.asarray(pop["c_true"],
                                                                     np.float64)))
            print(f"## {preset} M={M}  K_mean={K_mean:.2f}  "
                  f"noise_floor={noise_floor:.3e}")

            # kernel headline (max_iter=50)
            key = ("kernel", preset, M, KERNEL_MAX_ITER)
            if already(key):
                print(f"   skip (done): {key}")
            else:
                k = run_kernel(pop, KERNEL_MAX_ITER, n_rep_kernel)
                tput = throughput(M, k["n_dropped"], k["wall_median"])
                rec = dict(backend="kernel", preset=preset, M=M, knob=KERNEL_MAX_ITER,
                           max_iter=KERNEL_MAX_ITER, K_mean=K_mean,
                           noise_floor=noise_floor, throughput=tput, clocks=clk,
                           draft=True, **k)
                record(rec)
                print(f"   kernel@{KERNEL_MAX_ITER:<3d} wall={k['wall_median']:.3f}s "
                      f"tput={tput:.0f} tr/s med_loss={k['med_loss']:.3e} "
                      f"R2={k['med_r2']:.3f} rec={k['med_recovery']:.2e} "
                      f"drop={k['n_dropped']} fail={k['frac_failed']:.1%}")

            # operon single-core (only at op1_ms)
            if M in op1_ms:
                key = ("operon", preset, M, 1)
                nrep = n_rep_op1
                if already(key):
                    print(f"   skip (done): {key}")
                else:
                    op = run_operon(pop, 1, nrep)
                    if op["n_rep"] < 2:
                        print(f"   !! op1 M={M}: only {op['n_rep']} rep (LOGGED)")
                    tput = throughput(M, op["n_dropped"], op["wall_median"])
                    rec = dict(backend="operon", preset=preset, M=M, knob=1, nproc=1,
                               K_mean=K_mean, noise_floor=noise_floor,
                               throughput=tput, clocks=clk, draft=True,
                               capped_reps=(op["n_rep"] < 2), **op)
                    record(rec)
                    print(f"   operon n=1   wall={op['wall_median']:.3f}s "
                          f"(core={op['wall_core_median']:.3f} "
                          f"marshal+={op['wall_marshal_overhead']:.3f}) "
                          f"tput={tput:.0f} med_loss={op['med_loss']:.3e} "
                          f"atlim={op['frac_at_iter_limit']:.1%}")
            else:
                print(f"   operon n=1   NOT measured at M={M} "
                      f"(too slow; project/omit decided in final pass)")

            # operon parallel (every M)
            for nproc in parallel_nproc:
                key = ("operon", preset, M, nproc)
                if already(key):
                    print(f"   skip (done): {key}")
                    continue
                op = run_operon(pop, nproc, n_rep_parallel)
                tput = throughput(M, op["n_dropped"], op["wall_median"])
                rec = dict(backend="operon", preset=preset, M=M, knob=nproc,
                           nproc=nproc, K_mean=K_mean, noise_floor=noise_floor,
                           throughput=tput, clocks=clk, draft=True, **op)
                record(rec)
                print(f"   operon n={nproc:<3d} wall={op['wall_median']:.3f}s "
                      f"(core={op['wall_core_median']:.3f} "
                      f"marshal+={op['wall_marshal_overhead']:.3f}) "
                      f"tput={tput:.0f} med_loss={op['med_loss']:.3e} "
                      f"atlim={op['frac_at_iter_limit']:.1%}")
            print()
            evict_pop(preset, M, N, SEED)

    # ----- iter-Pareto sub-sweep (16k inner-const-heavy): kernel at extra iters
    if not args.smoke and PARETO_M in ms and PARETO_PRESET in presets:
        print(f"## PARETO sub-sweep: {PARETO_PRESET} M={PARETO_M} kernel iters")
        pop = get_pop(PARETO_PRESET, PARETO_M, N, SEED)
        for it in pareto_iters:
            key = ("kernel", PARETO_PRESET, PARETO_M, it)
            if already(key):
                print(f"   skip (done): {key}")
                continue
            k = run_kernel(pop, it, n_rep_kernel)
            tput = throughput(PARETO_M, k["n_dropped"], k["wall_median"])
            rec = dict(backend="kernel", preset=PARETO_PRESET, M=PARETO_M, knob=it,
                       max_iter=it, pareto=True, throughput=tput, clocks=clk,
                       draft=True, **k)
            record(rec)
            print(f"   kernel@{it:<3d} wall={k['wall_median']:.3f}s "
                  f"med_loss={k['med_loss']:.3e}")
        evict_pop(PARETO_PRESET, PARETO_M, N, SEED)
        print()

    # ----- final pass: re-read jsonl, compute cross-backend derived metrics ---
    finalize(jsonl, done_marker, plan, ms, presets, op1_ms, parallel_nproc,
             n_cpu, clk, tag, args.smoke)


def finalize(jsonl, done_marker, plan, ms, presets, op1_ms, parallel_nproc,
             n_cpu, clk, tag, smoke):
    """Re-read the jsonl and compute the cross-backend derived metrics (pure
    re-reduction so it is resumable and testable): iso-quality gate on the
    common solved set, speedups, parallel efficiency, crossover-core estimate,
    and the 64k single-core project-or-omit decision."""
    _, recs = load_done(jsonl)
    by_key = {config_key(r): r for r in recs}
    done_keys = list(by_key.keys())
    missing = missing_configs(done_keys, plan)

    derived = {"meta": dict(tag=tag, N=N, seed=SEED, clocks=clk, draft=True,
                            cpu_cores=n_cpu, quality_tol=QUALITY_TOL,
                            kernel_max_iter=KERNEL_MAX_ITER,
                            operon_max_iter=OPERON_MAX_ITER,
                            operon_timed_metric="wall_e2e (incl per-gen tree_view "
                                                "payload; audit#3)",
                            throughput_def="(M - n_dropped)/wall_median; n_dropped "
                                           "= K0 trees kernel skips (audit#4)",
                            quality_gate="common-set (paired both-finite) median; "
                                         "kernel <= operon * tol (audit#2)",
                            frac_converged_note="backend-specific stopping "
                                                "criterion; NOT cross-comparable "
                                                "(audit minor)",
                            n_configs=len(done_keys), expected=len(plan),
                            missing=[list(m) for m in missing]),
               "presets": {}}

    # The iso-quality common-set gate needs PAIRED per-tree losses; record()
    # persists them to a durable .npy sidecar per config, so this final pass can
    # compute the paired gate even on a crashed+resumed run (_gate_medians loads
    # from the sidecar; fallback fires only if a .npy is genuinely missing).

    for preset in presets:
        pblock = {}
        for M in ms:
            kkey = ("kernel", preset, M, KERNEL_MAX_ITER)
            krec = by_key.get(kkey)
            op1 = by_key.get(("operon", preset, M, 1))
            op_par = {n: by_key.get(("operon", preset, M, n)) for n in parallel_nproc}

            entry = dict(M=M)
            if krec:
                entry["kernel"] = krec
            if op1:
                entry["operon_1core"] = op1
            for n, r in op_par.items():
                if r:
                    entry[f"operon_{n}core"] = r

            # iso-quality gate: prefer common-set; fall back to per-backend median
            # (we recompute common-set from the loss sidecar if present).
            ref_op = op1
            ref_label = "operon_1core"
            if M == 64000:
                ref_op = op_par.get(64) or op_par.get(parallel_nproc[-1])
                ref_label = f"operon_{parallel_nproc[-1]}core"
            if krec and ref_op:
                med_k, med_op, n_common, gate_basis = _gate_medians(
                    tag, preset, M, krec, ref_op)
                qm = quality_matched(med_k, med_op)
                entry["quality_gate"] = dict(
                    matched=qm, basis=gate_basis, n_common=n_common,
                    med_loss_kernel=med_k, med_loss_operon_ref=med_op,
                    ref=ref_label,
                    over_pct=(100.0 * (med_k / med_op - 1.0)
                              if np.isfinite(med_k) and np.isfinite(med_op)
                              and med_op > 0 else float("nan")))
            # speedups (vs single-core measured) + parallel efficiency + crossover
            if krec and op1:
                kw = krec["wall_median"]
                entry["speedup_vs_1core"] = op1["wall_median"] / kw if kw > 0 else None
            for n, r in op_par.items():
                if krec and r and krec["wall_median"] > 0:
                    entry[f"speedup_vs_{n}core"] = r["wall_median"] / krec["wall_median"]
            if op1 and op_par.get(parallel_nproc[-1]):
                npx = parallel_nproc[-1]
                opP = op_par[npx]
                eff = ((op1["wall_median"] / opP["wall_median"]) / npx
                       if opP["wall_median"] > 0 else None)
                entry["parallel_efficiency_at_max_nproc"] = eff
                entry["parallel_efficiency_nproc"] = npx
                if krec and eff and eff > 0 and krec["wall_median"] > 0:
                    # crossover cores ~ (op1_wall/kernel_wall)/efficiency
                    entry["crossover_cores_est"] = (
                        (op1["wall_median"] / krec["wall_median"]) / eff)
            pblock[str(M)] = entry

        # verify op1 med_loss == op64 med_loss at measured M's (audit#5 precond)
        eq_checks = {}
        for M in op1_ms:
            o1 = by_key.get(("operon", preset, M, 1))
            o64 = by_key.get(("operon", preset, M, parallel_nproc[-1]))
            if o1 and o64:
                eq_checks[str(M)] = dict(
                    op1_med_loss=o1["med_loss"], op64_med_loss=o64["med_loss"],
                    equal=bool(abs(o1["med_loss"] - o64["med_loss"]) <= 1e-9
                               * max(1.0, abs(o1["med_loss"]))))
        pblock["op1_eq_op64_med_loss_check"] = eq_checks

        # 64k single-core project-or-omit decision (rule 3 / audit#8)
        percore = {}
        for M in op1_ms:
            r = by_key.get(("operon", preset, M, 1))
            if r and r["wall_median"] > 0:
                solved = M - r["n_dropped"]
                percore[M] = solved / r["wall_median"]            # nproc=1 => per-core
        action, spread, reason, proj = decide_projection(percore)
        proj_block = dict(action=action, spread=spread, reason=reason,
                          measured_percore_tput=percore)
        if action == "projected" and 64000 in ms:
            solved_64k = 64000
            kr64 = by_key.get(("kernel", preset, 64000, KERNEL_MAX_ITER))
            if kr64:
                solved_64k = 64000 - kr64["n_dropped"]
            proj_block["projected_op1_throughput_64k"] = proj
            proj_block["projected_op1_wall_64k"] = (solved_64k / proj
                                                    if proj > 0 else None)
        pblock["operon_1core_64k_projection"] = proj_block
        print(f"   [64k op1 {preset}] {action}: {reason}")

        derived["presets"][preset] = pblock

    report_path = OUT / f"{tag}_report.json"
    report_path.write_text(json.dumps(derived, indent=2, default=float))

    # config-count line (rule 6)
    print(f"\nDONE n_configs={len(done_keys)} expected={len(plan)} "
          f"missing={[list(m) for m in missing]}")
    if not missing:
        done_marker.write_text(f"clean finish n_configs={len(done_keys)}\n")
        print(f"-> wrote {done_marker}")
    else:
        print(f"!! {len(missing)} configs MISSING -> {done_marker.name} NOT written "
              f"(resumable: re-run to fill)")
    print(f"-> {report_path}")
    print(f"-> {jsonl}")


# loss sidecar: persist per-tree fp64 losses per config so the FINAL pass can
# compute the COMMON-SET (paired) median for the iso-quality gate without
# re-running the backends. DURABLE (one .npy per config) so a crashed+resumed
# run still gets the paired gate, not the audit#2-forbidden per-backend median.
_LOSS_SIDECAR: dict = {}


def _loss_dir(tag: str) -> Path:
    d = OUT / f"{tag}_losses"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _loss_file(tag: str, key) -> Path:
    backend, preset, M, knob = key
    return _loss_dir(tag) / f"{backend}_{preset}_{M}_{knob}.npy"


def save_losses(tag: str, key, arr: np.ndarray) -> None:
    np.save(_loss_file(tag, key), np.asarray(arr, np.float64))


def load_losses(tag: str, key):
    """In-memory sidecar first (this session); then durable .npy (prior session);
    else None."""
    arr = _LOSS_SIDECAR.get(key)
    if arr is not None:
        return arr
    f = _loss_file(tag, key)
    if f.exists():
        return np.load(f)
    return None


def _gate_medians(tag, preset, M, krec, oprec):
    """Common-set (paired) medians for the gate, from the durable loss sidecar
    (in-memory this session or .npy from a prior session). Falls back to
    per-backend medians ONLY if a loss file is genuinely missing — and flags the
    basis so the JSON never silently claims paired when it isn't."""
    kl = load_losses(tag, ("kernel", preset, M, krec.get("knob")))
    ol = load_losses(tag, (oprec["backend"], preset, M, oprec.get("knob")))
    if kl is not None and ol is not None and len(kl) == len(ol):
        med_k, med_op, n_common = common_set_median(kl, ol)
        return med_k, med_op, n_common, "common-set (paired, both-finite)"
    # fallback: a loss .npy is genuinely missing (e.g. hand-deleted)
    return (krec["med_loss"], oprec["med_loss"], -1,
            "per-backend median (loss sidecar file missing; "
            "NOT a paired gate — re-run --fresh to regenerate)")


if __name__ == "__main__":
    main()
