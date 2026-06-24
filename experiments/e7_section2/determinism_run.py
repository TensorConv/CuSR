#!/usr/bin/env python
"""determinism_run.py — C2 hardening, SERIAL measurement (铁律 §1).

Reuses the AUDITED sweep_e6 helpers (run_gpu_kernel / run_operon keep every
anti-fraud guard: device-Jacobian assertion, binary allowlist, fp64-independent
quality gate, pop-hash). Adds nothing to the timed path; only orchestrates:

  Phase L  linearity pre-check (advisor gate-1): wide --max-iter sweep; if the GPU
           LM loop early-stops, loop_ms saturates and intercept regression is void.
  Phase G  GPU CV: R reps/config of loop_ms at the two crossover cells.
  Phase O  Operon CV + clean baseline: R reps at 64c (node1, isolated) and 128c
           (full machine, overlaps tenants BY CONSTRUCTION -> disclosed).
  Phase F  IPC-floor control (Q1): homogeneous-replicated pop -> uniform work ->
           residual CV = scheduler/pool floor; excess over it = real load imbalance.
  Phase R  intercept regression (only on the linear regime from Phase L).
  Phase X  inner-const-heavy oversubscription spot-check (capped vs uncapped).
  Phase C  crossover_clean: recompute revad-vs-{64c,128c} from THIS session.

Every rep -> reps_raw.jsonl (durable, fsync). A background TenantSampler records
foreign (luoq/jinyb) thread overlap with our bound mask AND verifies our own
Operon workers stayed inside the mask (affinity-inheritance check). Honest failure:
a crash/timeout is logged status!=ok and the run continues.

Run (serial, on the clean card):
  source scripts/env.sh
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python experiments/e7_section2/determinism_run.py \
      --gpu 7 --locked-mhz 1410 --reps 20 --phases LGOFRXC
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e6_kernel_sweep import sweep_e6 as S
from experiments.e7_section2.determinism_analysis import (
    build_homogeneous_pop, check_linearity, compute_cv, fit_intercept_slope,
    recompute_crossover,
)

OUT = Path(__file__).resolve().parent / "out" / "determinism"

# crossover cells that BACK the headlines (FINDINGS Phase-3 table):
#   stable 9.6x  = revad vs 128c @ early-gen        M=16000 N=1000
#   noisy  28.2x = revad vs 128c @ late-gen-bloated M=64000 N=100  (the un-sentineled one)
CELLS = [
    dict(tag="stable", preset="early-gen", M=16000, N=1000),
    dict(tag="noisy", preset="late-gen-bloated", M=64000, N=100),
]
GPU_VARIANTS = ["revad", "ad", "fusedfd"]
NCORES = [64, 128]
ITER_SWEEP = [5, 10, 20, 40, 80]      # wide; Phase L decides the linear prefix
NODE1_PHYS = list(range(64, 128))     # node1 physical cores (tenants pinned to node0)
ALL_PHYS = list(range(0, 128))        # full machine (overlaps node0 tenants)
MASK = {64: NODE1_PHYS, 128: ALL_PHYS}

_REPS = None  # set from CLI


def append_rep(rec):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "reps_raw.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=float))
    print(f"[WROTE] {name}")


# --------------------------------------------------------------- tenant sampler
class TenantSampler(threading.Thread):
    """Samples `ps -eLo psr,user,comm` every `period`s during a CPU phase.
    Records foreign (luoq/jinyb) threads whose psr lands in our bound mask
    (contamination) AND our own python threads' psr (to confirm affinity
    inheritance actually pinned the Operon workers inside the mask)."""

    def __init__(self, mask_cores, period=0.5):
        super().__init__(daemon=True)
        self.mask = set(int(c) for c in mask_cores)
        self.period = period
        self.driver_pid = os.getpid()      # Operon Pool workers are our direct children
        self._stopev = threading.Event()   # NOT _stop: Thread reserves _stop()
        self.n_samples = 0
        self.foreign_overlap_samples = 0
        self.foreign_cores_hit = set()
        self.max_foreign_on_mask = 0
        self.my_psr_seen = set()
        self.my_outside_mask = set()

    def _sample(self):
        try:
            r = subprocess.run(["ps", "-eLo", "psr=,user=,ppid=,comm="],
                               capture_output=True, text=True, timeout=5)
        except Exception:
            return
        foreign_now = 0
        for ln in r.stdout.splitlines():
            parts = ln.split(None, 3)
            if len(parts) < 3:
                continue
            try:
                psr = int(parts[0])
                ppid = int(parts[2])
            except ValueError:
                continue
            user = parts[1]
            if user in ("luoq", "jinyb"):       # co-tenant contamination
                if psr in self.mask:
                    foreign_now += 1
                    self.foreign_cores_hit.add(psr)
            elif ppid == self.driver_pid:        # OUR Operon Pool workers (precise)
                self.my_psr_seen.add(psr)
                if psr not in self.mask:
                    self.my_outside_mask.add(psr)
        self.n_samples += 1
        if foreign_now:
            self.foreign_overlap_samples += 1
            self.max_foreign_on_mask = max(self.max_foreign_on_mask, foreign_now)

    def run(self):
        while not self._stopev.is_set():
            self._sample()
            self._stopev.wait(self.period)

    def stop_sampler(self):
        self._stopev.set()
        self.join(timeout=3)
        return dict(
            mask_size=len(self.mask), n_samples=self.n_samples,
            foreign_overlap_samples=self.foreign_overlap_samples,
            foreign_overlap_frac=(self.foreign_overlap_samples / self.n_samples
                                  if self.n_samples else float("nan")),
            max_foreign_threads_on_mask=self.max_foreign_on_mask,
            foreign_cores_hit=sorted(self.foreign_cores_hit),
            my_workers_outside_mask=sorted(self.my_outside_mask),
            affinity_inheritance_ok=(len(self.my_outside_mask) == 0))


def set_affinity(cores):
    os.sched_setaffinity(0, set(int(c) for c in cores))
    got = sorted(os.sched_getaffinity(0))
    return got


def _scrap_all_operon_pools():
    """Scrap EVERY cached pool so the next config spawns FRESH under its own
    affinity (codex fix #2). Otherwise an idle pool from a different nproc lingers
    on the wrong NUMA node and its parked workers trip the affinity-inheritance
    check (the noisy_64c affinity_ok=False artifact in the 2026-06-24 run)."""
    from cusr.benchmark.backends import _POOL_CACHE, _scrap_pool
    for nc in list(_POOL_CACHE.keys()):
        _scrap_pool(nc)


# ----------------------------------------------------------------- GPU phases
def gpu_reps(pop, variant, gpu_id, max_iter, reps, scratch):
    """R reps of one GPU config; returns the raw loop_ms list + quality + tput.
    run_gpu_kernel does n_rep launches internally and keeps the per-rep loop_ms
    in loop_ms_reps -> we read those raw (NOT the median) for CV."""
    k = S.run_gpu_kernel(pop, variant, max_iter, scratch, gpu_id, n_rep=reps)
    return k


def phase_L(gpu_id, scratch, locked_mhz):
    """Linearity pre-check (gate-1). For revad+fusedfd at both cells, sweep
    --max-iter wide and test whether loop_ms is linear in iters (no early-stop
    saturation). Records the per-iter loop_ms + the linearity verdict."""
    print("\n=== PHASE L: linearity pre-check (gate-1) ===")
    out = {}
    for cell in CELLS:
        pop = S.get_pop(cell["preset"], cell["M"], cell["N"], 0)
        for variant in ["revad", "fusedfd"]:
            iters, loops = [], []
            for mi in ITER_SWEEP:
                sd = scratch / f"L_{variant}_{cell['tag']}_{mi}"
                try:
                    k = S.run_gpu_kernel(pop, variant, mi, sd, gpu_id, n_rep=3)
                except Exception as e:  # noqa: BLE001
                    print(f"[L][FAIL] {variant} {cell['tag']} mi={mi}: {e}")
                    append_rep(dict(phase="L", variant=variant, cell=cell["tag"],
                                    max_iter=mi, status="fail", err=str(e)[:300]))
                    continue
                lm = float(np.median(k["loop_ms_reps"]))
                iters.append(mi)
                loops.append(lm)
                append_rep(dict(phase="L", variant=variant, cell=cell["tag"],
                                preset=cell["preset"], M=cell["M"], N=cell["N"],
                                max_iter=mi, loop_ms_median=lm,
                                loop_ms_reps=k["loop_ms_reps"], status="ok",
                                gpu_id=gpu_id, locked_mhz=locked_mhz))
                print(f"[L] {variant} {cell['tag']} mi={mi}: loop_ms={lm:.3f}")
            if len(iters) >= 2:
                lin = check_linearity(iters, loops)
                key = f"{variant}_{cell['tag']}"
                out[key] = dict(iters=iters, loop_ms=loops, **lin)
                print(f"[L] {key}: linear={lin['linear']} "
                      f"top_ratio={lin['top_slope_ratio']:.2f} "
                      f"prefix_len={lin['linear_prefix_len']}")
        S.evict_pop(cell["preset"], cell["M"], cell["N"], 0)
    write_json("linearity_precheck.json", out)
    return out


def phase_G(gpu_id, scratch, reps, locked_mhz):
    """GPU CV at the crossover cells (revad/ad/fusedfd), KERNEL_MAX_ITER=50."""
    print("\n=== PHASE G: GPU determinism (CV of loop_ms) ===")
    cv = {}
    for cell in CELLS:
        pop = S.get_pop(cell["preset"], cell["M"], cell["N"], 0)
        for variant in GPU_VARIANTS:
            sd = scratch / f"G_{variant}_{cell['tag']}"
            try:
                k = gpu_reps(pop, variant, gpu_id, S.KERNEL_MAX_ITER, reps, sd)
            except Exception as e:  # noqa: BLE001
                print(f"[G][FAIL] {variant} {cell['tag']}: {e}")
                append_rep(dict(phase="G", variant=variant, cell=cell["tag"],
                                status="fail", err=str(e)[:300]))
                continue
            reps_lm = k["loop_ms_reps"]
            c = compute_cv(reps_lm)
            tput_med = S.throughput(cell["M"], k["n_dropped"], c["median"])
            key = f"{variant}_{cell['tag']}"
            cv[key] = dict(preset=cell["preset"], M=cell["M"], N=cell["N"],
                           variant=variant, metric="loop_ms", reps=reps_lm,
                           throughput_at_median=tput_med,
                           med_loss_fp64=k["med_loss_fp64"],
                           n_dropped=k["n_dropped"], gpu_id=gpu_id,
                           locked_mhz=locked_mhz, **c)
            append_rep(dict(phase="G", variant=variant, cell=cell["tag"],
                            preset=cell["preset"], M=cell["M"], N=cell["N"],
                            metric="loop_ms", reps=reps_lm, cv=c["cv"],
                            median_ms=c["median"], throughput_at_median=tput_med,
                            med_loss_fp64=k["med_loss_fp64"], status="ok",
                            gpu_id=gpu_id, locked_mhz=locked_mhz, n_rep=reps))
            print(f"[G] {key}: median={c['median']:.3f}ms CV={c['cv']*100:.2f}% "
                  f"tput={tput_med:.0f} loss={k['med_loss_fp64']:.3e}")
        S.evict_pop(cell["preset"], cell["M"], cell["N"], 0)
    write_json("cv_gpu.json", cv)
    return cv


# ------------------------------------------------------------- Operon phases
def operon_reps(pop, nproc, reps, mask_cores, env_cap=None, time_budget_s=360.0):
    """R timed reps with ONE warmup, under a bound affinity mask + tenant sampler.
    Returns wall_core reps + quality + tput + the tenant/affinity evidence.
    env_cap: optional dict of env vars (e.g. OMP=1) set BEFORE the pool spawns.
    Adaptive cap: after rep 1, bound the rep count so a slow config (e.g. late-gen
    M=64k @128c) stays under time_budget_s; the actual count is recorded."""
    from cusr.benchmark.backends import OperonLM
    from cusr.benchmark import interp
    import statistics
    saved_env = {}
    if env_cap:
        for k, v in env_cap.items():
            saved_env[k] = os.environ.get(k)
            os.environ[k] = v
    got_mask = set_affinity(mask_cores)
    samp = TenantSampler(mask_cores)
    samp.start()
    capped_to = reps
    try:
        be = OperonLM(nproc=nproc, max_iter=S.OPERON_MAX_ITER)
        be.fit_pop(pop)                                  # warmup (discarded)
        cores, last = [], None
        for i in range(reps):
            res = be.fit_pop(pop)
            cores.append(res.wall_core)
            last = res
            if i == 0 and res.wall_core > 0:             # adaptive cap from rep 1
                affordable = max(5, int(time_budget_s / res.wall_core))
                capped_to = min(reps, affordable)
            if len(cores) >= capped_to:
                break
    finally:
        tenant = samp.stop_sampler()
        if env_cap:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    med_loss = S.gate_med_loss_fp64(pop, last.c_final)
    n_drop = int(np.sum(last.status == 3))
    losses = interp.loss_pop(pop, np.asarray(last.c_final, np.float64))
    return dict(wall_core_reps=cores, n_reps_used=len(cores), n_reps_requested=reps,
                med_loss_fp64=med_loss, n_dropped=n_drop,
                throughput_at_median=S.throughput(
                    pop["M"], n_drop, statistics.median(cores) * 1000.0),
                losses=losses, affinity_mask=got_mask, tenant=tenant)


def phase_O(reps):
    """Operon CV + clean baseline at the crossover cells, 64c (clean) + 128c (full)."""
    print("\n=== PHASE O: Operon determinism + clean baseline ===")
    cv = {}
    for cell in CELLS:
        pop = S.get_pop(cell["preset"], cell["M"], cell["N"], 0)
        for nc in NCORES:
            _scrap_all_operon_pools()   # fresh pool under THIS config's mask (codex fix #2)
            print(f"[O] {cell['tag']} nc={nc} (mask {nc}c) warm+{reps} reps ...")
            try:
                o = operon_reps(pop, nc, reps, MASK[nc])
            except Exception as e:  # noqa: BLE001
                print(f"[O][FAIL] {cell['tag']} nc={nc}: {e}")
                append_rep(dict(phase="O", cell=cell["tag"], ncores=nc,
                                status="fail", err=str(e)[:300]))
                continue
            c = compute_cv(o["wall_core_reps"])
            key = f"{cell['tag']}_{nc}c"
            isolated = (nc == 64 and o["tenant"]["foreign_overlap_samples"] == 0)
            cv[key] = dict(preset=cell["preset"], M=cell["M"], N=cell["N"],
                           ncores=nc, metric="wall_core", reps=o["wall_core_reps"],
                           throughput_at_median=o["throughput_at_median"],
                           med_loss_fp64=o["med_loss_fp64"], n_dropped=o["n_dropped"],
                           affinity_mask_size=len(o["affinity_mask"]),
                           isolated_verified=isolated, tenant=o["tenant"], **c)
            append_rep(dict(phase="O", cell=cell["tag"], preset=cell["preset"],
                            M=cell["M"], N=cell["N"], ncores=nc, metric="wall_core",
                            reps=o["wall_core_reps"], cv=c["cv"], median_s=c["median"],
                            throughput_at_median=o["throughput_at_median"],
                            med_loss_fp64=o["med_loss_fp64"], isolated_verified=isolated,
                            tenant=o["tenant"], status="ok", n_rep=reps))
            t = o["tenant"]
            print(f"[O] {key}: median={c['median']:.3f}s CV={c['cv']*100:.2f}% "
                  f"tput={o['throughput_at_median']:.0f} loss={o['med_loss_fp64']:.3e} "
                  f"| tenant overlap {t['foreign_overlap_samples']}/{t['n_samples']} "
                  f"samples, affinity_ok={t['affinity_inheritance_ok']}")
        S.evict_pop(cell["preset"], cell["M"], cell["N"], 0)
    write_json("cv_operon.json", cv)
    return cv


def phase_F(reps):
    """IPC-floor control (Q1): same cells/cores but homogeneous-replicated pop ->
    uniform work. Residual CV = scheduler/pool floor; excess of Phase-O CV over
    this floor = genuine load-imbalance jitter (the honestly-attributable part)."""
    print("\n=== PHASE F: IPC-floor control (homogeneous pop) ===")
    cv = {}
    for cell in CELLS:
        pop = S.get_pop(cell["preset"], cell["M"], cell["N"], 0)
        hp, rep_idx = build_homogeneous_pop(pop)
        print(f"[F] {cell['tag']}: homogeneous pop from rep_idx={rep_idx} "
              f"K={int(hp['metas'][0,3])} (all {hp['M']} trees identical)")
        for nc in NCORES:
            _scrap_all_operon_pools()   # fresh pool under THIS config's mask (codex fix #2)
            try:
                o = operon_reps(hp, nc, reps, MASK[nc])
            except Exception as e:  # noqa: BLE001
                print(f"[F][FAIL] {cell['tag']} nc={nc}: {e}")
                append_rep(dict(phase="F", cell=cell["tag"], ncores=nc,
                                status="fail", err=str(e)[:300]))
                continue
            c = compute_cv(o["wall_core_reps"])
            key = f"{cell['tag']}_{nc}c"
            cv[key] = dict(preset=cell["preset"], M=cell["M"], N=cell["N"],
                           ncores=nc, metric="wall_core_homogeneous",
                           rep_idx=int(rep_idx), reps=o["wall_core_reps"],
                           tenant=o["tenant"], **c)
            append_rep(dict(phase="F", cell=cell["tag"], preset=cell["preset"],
                            M=cell["M"], N=cell["N"], ncores=nc,
                            metric="wall_core_homogeneous", reps=o["wall_core_reps"],
                            cv=c["cv"], median_s=c["median"], rep_idx=int(rep_idx),
                            tenant=o["tenant"], status="ok", n_rep=reps))
            print(f"[F] {key}: floor median={c['median']:.3f}s "
                  f"floor_CV={c['cv']*100:.2f}%")
        S.evict_pop(cell["preset"], cell["M"], cell["N"], 0)
    write_json("cv_floor_homogeneous.json", cv)
    return cv


def phase_R(linearity, gpu_cv):
    """Intercept regression (gate-1 gated). Fit loop_ms = a + b*iters on the
    LINEAR PREFIX from Phase L. intercept_frac ~0 => no systematic short-region
    bias => the short-loop_ms concern is empirically closed."""
    print("\n=== PHASE R: intercept regression (short-timing bias) ===")
    if not linearity:
        print("[R] no linearity data; skipping")
        return {}
    out = {}
    for key, lin in linearity.items():
        iters, loops = lin["iters"], lin["loop_ms"]
        npx = lin["linear_prefix_len"]
        used_iters, used_loops = iters[:npx], loops[:npx]
        note = ("full range linear" if lin["linear"]
                else f"plateau detected; fit low-iter prefix (n={npx})")
        if len(used_iters) < 2:
            out[key] = dict(fittable=False, reason="prefix<2 pts", **lin)
            print(f"[R] {key}: NOT fittable ({note})")
            continue
        fit = fit_intercept_slope(used_iters, used_loops, ref_iter=S.KERNEL_MAX_ITER)
        out[key] = dict(fittable=True, note=note, used_iters=used_iters,
                        intercept=fit["intercept"], slope=fit["slope"],
                        r2=fit["r2"], intercept_frac=fit["intercept_frac"],
                        linear=lin["linear"])
        append_rep(dict(phase="R", key=key, status="ok", **out[key]))
        print(f"[R] {key}: intercept={fit['intercept']:.3f}ms "
              f"slope={fit['slope']:.4f}ms/iter r2={fit['r2']:.4f} "
              f"fixed_frac@50={fit['intercept_frac']*100:.1f}% [{note}]")
    write_json("timing_window.json", out)
    return out


def phase_X(reps):
    """inner-const-heavy (high-K) oversubscription spot-check: capped (OMP/BLAS=1)
    vs uncapped at 64c. The verify-first probe only covered early-gen (low K);
    high-K is the remaining caveat. If capped wall_core << uncapped, the inner-const
    Operon baseline WAS oversubscribed (and the published numbers need the caveat)."""
    print("\n=== PHASE X: inner-const oversubscription spot-check ===")
    preset, M, N, nc = "inner-const-heavy", 16000, 1000, 64
    n = max(5, reps // 4)
    pop = S.get_pop(preset, M, N, 0)
    out = {}
    for label, env in (("uncapped", None),
                       ("capped", dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                       MKL_NUM_THREADS="1"))):
        # CRITICAL: scrap ALL pools so workers respawn under env_cap (and clean
        # affinity). Reusing a prior pool would make OMP=1 a no-op => false "not
        # oversub". NB: the OMP cap only takes effect if BLAS reads the env at the
        # worker's FIRST import; corroborate with a fresh-process run if in doubt.
        _scrap_all_operon_pools()
        try:
            o = operon_reps(pop, nc, n, MASK[nc], env_cap=env)
        except Exception as e:  # noqa: BLE001
            print(f"[X][FAIL] {label}: {e}")
            append_rep(dict(phase="X", label=label, status="fail", err=str(e)[:300]))
            continue
        c = compute_cv(o["wall_core_reps"])
        out[label] = dict(preset=preset, M=M, N=N, ncores=nc, reps=o["wall_core_reps"],
                          tenant=o["tenant"], **c)
        append_rep(dict(phase="X", label=label, preset=preset, M=M, N=N, ncores=nc,
                        reps=o["wall_core_reps"], median_s=c["median"], cv=c["cv"],
                        tenant=o["tenant"], status="ok", n_rep=n))
        print(f"[X] {label}: median={c['median']:.3f}s CV={c['cv']*100:.2f}%")
    if "uncapped" in out and "capped" in out:
        ratio = out["capped"]["median"] / out["uncapped"]["median"]
        if ratio < 0.85:
            note = ("capped FASTER => uncapped was thrashing => the published "
                    "(uncapped) inner-const baseline is oversubscription-INFLATED "
                    "=> our speedup needs a caveat there")
            oversub = True
        elif ratio > 1.15:
            note = ("capped SLOWER => uncapped threads do USEFUL intra-tree work "
                    "=> the published (uncapped) baseline is Operon's BEST case "
                    "=> our speedup is FAIR/conservative (capping would undersell Operon)")
            oversub = False
        else:
            note = ("capped ~= uncapped => threads idle => baseline not "
                    "confounded by oversubscription either way")
            oversub = False
        out["verdict"] = dict(capped_over_uncapped=ratio, oversubscribed=oversub,
                              note=note)
        print(f"[X] verdict: capped/uncapped={ratio:.3f} oversubscribed={oversub}"
              f"\n     => {note}")
    S.evict_pop(preset, M, N, 0)
    write_json("oversub_innerconst.json", out)
    return out


def phase_C(gpu_cv, operon_cv):
    """crossover_clean: recompute revad-vs-{64c,128c} from THIS session's GPU +
    Operon medians, with the iso-quality gate and within-session error bands."""
    print("\n=== PHASE C: clean crossover recompute ===")
    out = {}
    missing = []
    for cell in CELLS:
        tag = cell["tag"]
        gk = f"revad_{tag}"
        if gk not in gpu_cv:                      # codex fix #3: loud, not silent
            missing.append(gk)
            print(f"[C][MISSING] GPU input {gk!r} absent -> crossover for "
                  f"cell {tag} NOT computed (a GPU phase likely failed)")
            continue
        g = gpu_cv[gk]
        for nc in NCORES:
            ok = f"{tag}_{nc}c"
            if ok not in operon_cv:
                missing.append(ok)
                print(f"[C][MISSING] Operon input {ok!r} absent -> revad-vs-{nc}c "
                      f"crossover NOT computed (an Operon phase likely failed)")
                continue
            o = operon_cv[ok]
            iso = bool(np.isfinite(g["med_loss_fp64"]) and np.isfinite(o["med_loss_fp64"])
                       and g["med_loss_fp64"] <= o["med_loss_fp64"] * S.QUALITY_TOL)
            cr = recompute_crossover(g["throughput_at_median"], o["throughput_at_median"],
                                     g["cv"], o["cv"])
            out[f"{tag}_revad_vs_{nc}c"] = dict(
                cell=tag, preset=cell["preset"], M=cell["M"], N=cell["N"], ncores=nc,
                iso_quality=iso, gpu_loss=g["med_loss_fp64"], operon_loss=o["med_loss_fp64"],
                isolated_verified=o.get("isolated_verified", False),
                tenant_overlap_frac=o["tenant"]["foreign_overlap_frac"], **cr)
            print(f"[C] {tag} revad-vs-{nc}c: {cr['multiplier']:.2f}x "
                  f"(±{cr['rel_unc']*100:.1f}% within-session) iso={iso} "
                  f"isolated={out[f'{tag}_revad_vs_{nc}c']['isolated_verified']}")
    if missing:
        out["_missing_inputs"] = missing      # recorded, never silently dropped
    write_json("crossover_clean.json", out)
    return out


def main():
    global _REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--locked-mhz", type=int, default=1410)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--phases", default="LGOFRXC",
                    help="subset of L(inearity) G(pu) O(peron) F(loor) R(egression) "
                         "X(oversub) C(rossover)")
    args = ap.parse_args()
    _REPS = args.reps
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "scratch"
    scratch.mkdir(exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[START] gpu={args.gpu} locked={args.locked_mhz}MHz reps={args.reps} "
          f"phases={args.phases}")
    print(f"[START] full affinity at launch: {sorted(os.sched_getaffinity(0))[:8]}...")

    lin = gpu_cv = op_cv = {}
    if "L" in args.phases:
        lin = phase_L(args.gpu, scratch, args.locked_mhz)
    if "G" in args.phases:
        gpu_cv = phase_G(args.gpu, scratch, args.reps, args.locked_mhz)
    if "O" in args.phases:
        op_cv = phase_O(args.reps)
    if "F" in args.phases:
        phase_F(args.reps)
    if "R" in args.phases:
        phase_R(lin, gpu_cv)
    if "X" in args.phases:
        phase_X(args.reps)
    if "C" in args.phases and gpu_cv and op_cv:
        phase_C(gpu_cv, op_cv)

    # clock readback (铁律 #2)
    rb = subprocess.run(
        ["sudo", "-n", "nvidia-smi", "-i", str(args.gpu),
         "--query-gpu=clocks.sm,clocks.applications.graphics,persistence_mode",
         "--format=csv,noheader"], capture_output=True, text=True)
    (OUT / "clock_lock_post.txt").write_text(rb.stdout or rb.stderr)
    print(f"[CLOCK post] {(rb.stdout or '').strip()}")
    print("[DONE] determinism_run complete")


if __name__ == "__main__":
    main()
