#!/usr/bin/env python
"""sweep_e6.py — corrected GPU CO kernel vs Operon throughput sweep (e6).

WHY THIS EXISTS: e5's sweep accidentally benchmarked the SLOW in-process host-FD
kernel (libcusr_co_fd.so, ~1.3-2k trees/s flat). e6 measures the REAL deployed
on-device kernels via the -DPROFILE subprocess binaries, with strict anti-fakery
discipline. Every number is real; every skip is logged.

GPU VARIANTS (both SUBPROCESS, crash-isolated):
  fusedfd -> cusr/kernel/batch_lm_fusedfd_prof   (on-device fused-FD Jacobian)
  ad      -> cusr/kernel/batch_lm_ad_prof         (on-device forward-AD, exact)
Both emit a `PROFILE_JSON {...}` line with loop_ms = pure LM-loop GPU time
(EXCLUDES CUDA init + corpus load). GPU throughput := (M - n_K0)/(loop_ms/1000).

NEVER use any *fd*/host-FD .so or libcusr_co_fd.so for a speed number (the e5 bug).

The pure helpers below are unit-tested in test_sweep_e6.py (no GPU needed). The
run machinery (subprocess invocation, phase scheduling, JSONL) is exercised by
the smoke run.

Run smoke:
  source scripts/env.sh
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/e6_kernel_sweep/sweep_e6.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS / SPEC AXES
# ---------------------------------------------------------------------------
KERNEL_DIR = Path(__file__).resolve().parents[2] / "cusr" / "kernel"
ALLOWED_BINARIES = {"batch_lm_fusedfd_prof", "batch_lm_ad_prof", "batch_lm_revad_prof"}
FORBIDDEN_SUBSTRINGS = ("libcusr_co", "co_fd", "_fd.so")
VARIANT_BINARY = {
    "fusedfd": KERNEL_DIR / "batch_lm_fusedfd_prof",
    "ad": KERNEL_DIR / "batch_lm_ad_prof",
    "revad": KERNEL_DIR / "batch_lm_revad_prof",   # reverse-mode AD (v5), drop-in
}

# THE anti-host-FD guard is STRUCTURAL (assert_device_jacobian below) + the binary
# allowlist (assert_gpu_binary), NOT an absolute throughput floor. An absolute
# trees/s floor CANNOT discriminate host-FD once N varies: a genuine small-M/large-N
# config (fusedfd inner-const M=1000 N=10000) legitimately runs at ~517 trees/s
# (each of 1000 trees evaluates 10k points), which OVERLAPS the 1.3-2k host-FD band
# (measured 2026-06-21, see _calib_floor). So the floor below is demoted to an
# anti-GARBAGE check only (catches ~0/NaN throughput from a broken run).
GPU_TPUT_FLOOR = 50.0
# rough trees/s AT N=1000 for the measured-vs-expected LOG line only (informational,
# N-naive; NOT a halt). N=100 runs faster, N=10000 far slower. Calibrated 2026-06-21.
EXPECTED_TPUT = {
    "fusedfd": {1000: 5700, 4000: 12000, 16000: 24000, 64000: 36000, 256000: 50000},
    "ad": {1000: 10600, 4000: 20000, 16000: 34000, 64000: 40000, 256000: 55000},
}

MS = [1000, 4000, 16000, 64000, 256000]
NS = [100, 1000, 10000]
ALL_PRESETS = ["early-gen", "late-gen-bloated", "inner-const-heavy"]
SEEDS = [0, 1, 2]
KERNEL_MAX_ITER = 50            # carry forward from e5 (do NOT re-litigate)
OPERON_MAX_ITER = 200           # Operon's honest point (per-tree early stop)
OPERON_NCORES = [1, 16, 64, 128]
QUALITY_TOL = 1.05
MEM_CEILING_GB = 40.0           # headroom on 80GB A100 (clocks unlocked, shared)
OPERON_TIME_CEILING_S = 15 * 60  # skip+log any projected >15 min Operon config

# kernel raw status -> unified (4=FAIL_CHOLESKY -> failed); matches backends.py
_KERNEL_STATUS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2}


# ===========================================================================
# PURE HELPERS (unit-tested in test_sweep_e6.py) — keep side-effect-free.
# ===========================================================================
def gpu_footprint_bytes(M: int, N: int, K_max: int, total_nodes: int) -> int:
    """Sum of ALL device buffers the kernel cudaMalloc's (audit blocker #3).
    Enumerated from batch_lm_fusedfd.cu lines 419-435 (ad is a subset/same):
      node arrays   : d_nt(int) + d_nv(float) + d_ci(int)  = 3 * total_nodes * 4
      d_call        : total_c * 4   (<= K_max*M, bound by K_max*M)
      d_metas       : M * sizeof(TreeMeta)=16
      d_xs          : N * n_vars * 4   (n_vars small; fold into headroom via +N*4)
      per-N tree    : d_ym,d_y,d_yp,d_r each M*N*4            = 4 * M*N*4
      d_J           : K_max * M * N * 4
      d_JtJ         : K_max*K_max * M * 4
      d_JtR,d_delta : K_max * M * 4 each                       = 2 * K_max*M*4
      d_loss,d_lam  : M * 4 each ; d_stat M*4                  = 3 * M*4
    The (4+K_max)*M*N*4 naive lower bound is the {d_ym,d_y,d_yp,d_r,d_J} term;
    everything else is added ON TOP so the gate is strictly conservative."""
    M = int(M); N = int(N); K_max = int(K_max); total_nodes = int(total_nodes)
    per_n = 4 * M * N * 4                       # d_ym,d_y,d_yp,d_r
    d_J = K_max * M * N * 4
    d_JtJ = K_max * K_max * M * 4
    d_JtR_delta = 2 * K_max * M * 4
    nodes = 3 * total_nodes * 4                 # d_nt,d_nv,d_ci
    d_call = K_max * M * 4                      # >= total_c*4
    d_metas = M * 16
    small_M = 3 * M * 4                         # d_loss,d_lam,d_stat
    d_xs = N * 4                                # n_vars folded as a small pad
    return int(per_n + d_J + d_JtJ + d_JtR_delta + nodes
               + d_call + d_metas + small_M + d_xs)


def footprint_skip(M: int, N: int, K_max: int, total_nodes: int,
                   ceiling_gb: float = MEM_CEILING_GB):
    """(skip: bool, reason: str, gb: float). Decide BEFORE launching the
    subprocess so a real OOM never crashes us into a fake-failed record."""
    nbytes = gpu_footprint_bytes(M, N, K_max, total_nodes)
    gb = nbytes / (1024.0 ** 3)
    if gb > ceiling_gb:
        reason = (f"memory_ceiling: estimated GPU footprint {gb:.2f} GB "
                  f"> {ceiling_gb:.0f} GB (M={M} N={N} K_max={K_max})")
        return True, reason, gb
    return False, "", gb


def throughput(M: int, n_dropped: int, loop_ms: float) -> float:
    """trees/sec on the work the kernel actually did (audit#4): exclude K0 drops.
    Uses loop_ms (LM-loop GPU time), NEVER total_ms (which includes CUDA init)."""
    loop_ms = float(loop_ms)
    if loop_ms <= 0:
        return float("nan")
    return (int(M) - int(n_dropped)) / (loop_ms / 1000.0)


def assert_gpu_binary(binary_path) -> None:
    """(audit blocker #2) Reject the in-process host-FD slow path; only allow the
    GPU _prof binaries. Exact-basename allowlist — a naive 'fd' not in path check
    is WRONG because 'batch_lm_fusedfd_prof' contains 'fd'."""
    binary_path = str(binary_path)
    basename = os.path.basename(os.path.realpath(binary_path))
    if basename not in ALLOWED_BINARIES:
        raise AssertionError(
            f"GPU binary basename {basename!r} not in allowlist {ALLOWED_BINARIES}. "
            f"Only batch_lm_fusedfd_prof / batch_lm_ad_prof permitted. "
            f"Path was: {binary_path}")
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden in binary_path or forbidden in basename:
            raise AssertionError(
                f"GPU binary path contains forbidden substring {forbidden!r}: "
                f"{binary_path}. This is the in-process host-FD slow path "
                f"(libcusr_co_fd.so) that caused the e5 wrong numbers.")
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"GPU binary does not exist: {binary_path}")


def assert_device_jacobian(prof: dict) -> None:
    """THE structural anti-host-FD guard (N-invariant). The on-device binaries time
    their Jacobian in a DEVICE kernel under the PROFILE 'fd_jacobian' category
    (fused-FD or AD), so a real on-device run reports fd_jacobian ms>0 AND
    launches>0. The host-FD path (batch_lm.cu / libcusr_co_fd.so) computes FD on the
    HOST and shows ZERO device fd_jacobian time. This REPLACES the absolute
    throughput floor as the host-FD discriminator: a trees/s floor cannot work once
    N varies (a genuine M=1000/N=10000 config runs ~517 trees/s, overlapping the
    1.3-2k host-FD band), but the device-Jacobian signature is the same at every N."""
    cats = prof.get("cats") or {}
    fdj = cats.get("fd_jacobian") or {}
    ms = float(fdj.get("ms", 0.0))
    launches = int(fdj.get("launches", 0))
    if not (ms > 0.0 and launches > 0):
        raise RuntimeError(
            f"PROFILE_JSON shows NO on-device Jacobian kernel (fd_jacobian "
            f"ms={ms}, launches={launches}). The Jacobian was not computed on the "
            f"device -> host-FD slow path, not batch_lm_fusedfd/ad_prof. Refusing "
            f"to record (the e5 wrong-kernel safeguard).")


def tripwire_gpu_tput(variant: str, M: int, measured_tput: float) -> None:
    """Anti-GARBAGE check + measured-vs-expected LOG. The real anti-host-FD guard is
    assert_device_jacobian (structural, N-invariant) + assert_gpu_binary (allowlist);
    an absolute trees/s floor CANNOT discriminate host-FD once N varies (genuine
    high-N corners run ~500 trees/s). So this HALTS only on ~0/NaN throughput (a
    totally broken run; floor=GPU_TPUT_FLOOR=50). The expected value is N-naive
    (calibrated at N=1000) and printed for human eyeballing only."""
    expected = EXPECTED_TPUT.get(variant, {}).get(int(M))
    if expected:
        ratio = measured_tput / expected if expected else float("nan")
        print(f"[THROUGHPUT_CHECK] {variant} M={M}: measured={measured_tput:.0f} "
              f"tr/s expected~{expected:.0f}@N=1000 ratio={ratio:.2f}x "
              f"(N-naive; device-Jacobian guard is the real check)")
    else:
        print(f"[THROUGHPUT_CHECK] {variant} M={M}: measured={measured_tput:.0f} tr/s")
    if not np.isfinite(measured_tput) or measured_tput < GPU_TPUT_FLOOR:
        raise SystemExit(
            f"\nTRIPWIRE HALT: {variant} M={M} measured {measured_tput:.0f} trees/s "
            f"is ~zero (floor={GPU_TPUT_FLOOR:.0f}) -> the run is broken (no GPU work "
            f"/ parse failure). Refusing to record. The host-FD discriminator is "
            f"assert_device_jacobian + assert_gpu_binary, not this floor.")


def quality_matched(med_kernel: float, med_operon: float,
                    tol: float = QUALITY_TOL) -> bool:
    """iso-quality gate: kernel fp64 median loss <= operon * tol. Non-finite
    either side => cannot certify => False (ported verbatim from e5)."""
    if not (np.isfinite(med_kernel) and np.isfinite(med_operon)):
        return False
    return bool(med_kernel <= med_operon * tol)


def gate_med_loss_fp64(pop: dict, c_final) -> float:
    """fp64 median loss recomputed via interp.loss_pop from the c_final the GPU
    wrote (total_c float32, SAME c_offset layout as pop['c_init']/'c_true') —
    independent of the backend (audit blocker #1; no per-tree decode needed)."""
    from cusr.benchmark import interp
    c64 = np.asarray(c_final, np.float64)
    if c64.size != pop["total_c"]:
        raise ValueError(f"c_final size {c64.size} != total_c {pop['total_c']}")
    losses = interp.loss_pop(pop, c64)
    finite = losses[np.isfinite(losses)]
    return float(np.median(finite)) if finite.size else float("inf")


def n_k0_dropped(pop: dict) -> int:
    """K0 trees (no constants) the kernel skips — the throughput denominator
    correction, computed HOST-SIDE from the pop, never trusted from subprocess."""
    return int((pop["metas"][:, 3] == 0).sum())


def config_key(rec: dict):
    """Stable key for done/skip/missing accounting."""
    return (rec.get("backend"), rec.get("variant"), rec.get("preset"),
            int(rec["M"]), int(rec.get("knob", -1)))


def is_complete_record(rec: dict) -> bool:
    """A record counts as DONE only if it carries a VALIDATED measurement:
    status=='ok', a finite throughput, and the timing field it was parsed from.
    Skip/timeout/partial/NaN records are NOT complete (audit blocker:
    resumability broken by append-before-validation)."""
    if rec.get("status") != "ok":
        return False
    backend = rec.get("backend")
    tput = rec.get("throughput")
    if tput is None or not np.isfinite(float(tput)):
        return False
    if backend == "kernel" and "loop_ms" not in rec:
        return False
    if backend == "operon" and "wall_core" not in rec:
        return False
    return True


def load_done(jsonl_path):
    """Set of COMPLETE config keys (validated). Tolerates a half-written trailing
    line, and ignores skip/partial records so a re-run re-attempts them."""
    jsonl_path = Path(jsonl_path)
    done = []
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                                  # crash mid-write
        if is_complete_record(rec):
            done.append(config_key(rec))
    return done


def load_records(jsonl_path):
    """All parseable records (complete, skipped, partial) for finalize/report."""
    jsonl_path = Path(jsonl_path)
    recs = []
    if not jsonl_path.exists():
        return recs
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


# ===========================================================================
# JSONL incremental write (validate-then-append; durable)
# ===========================================================================
def append_record(jsonl_path: Path, rec: dict) -> None:
    with jsonl_path.open("a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_losses(loss_dir: Path, key, arr: np.ndarray) -> None:
    loss_dir.mkdir(parents=True, exist_ok=True)
    backend, variant, preset, M, knob = key
    f = loss_dir / f"{backend}_{variant}_{preset}_{M}_{knob}.npy"
    np.save(f, np.asarray(arr, np.float64))


# ===========================================================================
# pop memoisation (rule 1: identical object / same bytes to both backends)
# ===========================================================================
_POP_CACHE: dict = {}


def get_pop(preset, M, N, seed):
    from cusr.benchmark.workload.gen_synth import gen_pop
    key = (preset, M, N, seed)
    pop = _POP_CACHE.get(key)
    if pop is None:
        pop = gen_pop(preset, M, N, seed)
        _POP_CACHE[key] = pop
    return pop


def evict_pop(preset, M, N, seed):
    _POP_CACHE.pop((preset, M, N, seed), None)


def pop_digest(pop_path: Path) -> str:
    """SHA256 of the serialized pop.bin — proves GPU & Operon (and resume) saw
    the IDENTICAL population (audit major: pop identity / silent mutation)."""
    import hashlib
    h = hashlib.sha256()
    with open(pop_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# GPU SUBPROCESS RUNNER (crash-isolated; validate-then-record)
# ===========================================================================
def run_gpu_kernel(pop, variant, max_iter, out_dir: Path, gpu_id, n_rep=3,
                   timeout=600):
    """Call batch_lm_{variant}_prof as a SUBPROCESS (crash-isolated). Returns a
    dict with the parsed PROFILE_JSON loop_ms, the c_final read from disk, the
    validated throughput, and the fp64 quality median — or raises on validation
    failure (the CALLER appends a status!=ok record; we never fake-complete)."""
    from cusr.benchmark import popio
    binary = VARIANT_BINARY[variant]
    assert_gpu_binary(binary)               # tripwire (b) BEFORE doing anything

    out_dir.mkdir(parents=True, exist_ok=True)
    pop_path = out_dir / "pop.bin"
    popio.save_pop_bin(pop, pop_path)
    digest = pop_digest(pop_path)
    M = pop["M"]
    n_drop = n_k0_dropped(pop)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU_BINARY] variant={variant} -> {binary} (allowlisted) "
          f"gpu={gpu_id} pop_sha256={digest[:16]}")

    walls, profs, last_status, last_cfinal = [], [], None, None
    for rep in range(n_rep):
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(binary), str(pop_path), str(out_dir),
             "--max-iter", str(max_iter), "--quiet"],
            capture_output=True, text=True, timeout=timeout, env=env)
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(
                f"{variant} M={M} N={pop['N']} subprocess exit={r.returncode}: "
                f"{(r.stderr or r.stdout)[-400:]}")
        prof_line = next((l for l in r.stdout.splitlines()
                          if l.startswith("PROFILE_JSON")), None)
        if not prof_line:
            raise RuntimeError(f"No PROFILE_JSON in {variant} M={M} output")
        prof = json.loads(prof_line[len("PROFILE_JSON"):])
        assert_device_jacobian(prof)        # structural anti-host-FD guard (per rep)
        walls.append(wall)
        profs.append(prof)
        # read outputs of THIS rep from the unique out_dir (no clobber)
        last_status = np.fromfile(out_dir / "status.bin", dtype=np.int32)
        last_cfinal = np.fromfile(out_dir / "c_final.bin",
                                  dtype=np.float32).astype(np.float64)

    # ---- validation (audit blocker: validate BEFORE record) ----
    if last_status is None or last_cfinal is None:
        raise RuntimeError(f"{variant} M={M}: no outputs read")
    if last_status.size != M:
        raise RuntimeError(f"{variant} M={M}: status.bin has {last_status.size} "
                           f"entries != M={M}")
    if last_cfinal.size != pop["total_c"]:
        raise RuntimeError(f"{variant} M={M}: c_final.bin has {last_cfinal.size} "
                           f"!= total_c={pop['total_c']}")
    # re-verify pop digest unchanged (silent mutation guard)
    if pop_digest(pop_path) != digest:
        raise RuntimeError(f"{variant} M={M}: pop.bin digest changed mid-run!")

    # median PROFILE by loop_ms (robust to clock jitter, clocks unlocked)
    profs_sorted = sorted(profs, key=lambda p: float(p["loop_ms"]))
    med_prof = profs_sorted[len(profs_sorted) // 2]
    loop_ms = float(med_prof["loop_ms"])
    tput = throughput(M, n_drop, loop_ms)
    if not np.isfinite(tput) or tput <= 0:
        raise RuntimeError(f"{variant} M={M}: non-finite throughput {tput}")

    # fp64 quality gate (independent of backend; from c_final on disk)
    med_loss = gate_med_loss_fp64(pop, last_cfinal)

    status = np.array([_KERNEL_STATUS_MAP.get(int(s), 2) for s in last_status],
                      np.int32)
    losses = None
    from cusr.benchmark import interp
    losses = interp.loss_pop(pop, last_cfinal)

    return dict(
        loop_ms=loop_ms, loop_ms_reps=[float(p["loop_ms"]) for p in profs],
        wall_median=statistics.median(walls), wall_reps=list(walls),
        throughput=tput, med_loss_fp64=med_loss, n_dropped=n_drop,
        frac_converged=float(np.mean(status == 0)) if status.size else 0.0,
        frac_failed=float(np.mean(status == 2)) if status.size else 0.0,
        pop_sha256=digest, losses=losses, prof=med_prof, n_rep=n_rep,
        setup_ms=float(med_prof.get("setup_ms", float("nan"))),
        load_ms=float(med_prof.get("load_ms", float("nan"))),
    )


# ===========================================================================
# OPERON RUNNER (Phase B; wall_core = fair match to GPU loop_ms, both
# setup-excluded; see SPEC fairness note)
# ===========================================================================
def run_operon(pop, nproc, max_iter=OPERON_MAX_ITER, n_rep=3):
    from cusr.benchmark.backends import OperonLM
    backend = OperonLM(nproc=nproc, max_iter=max_iter)
    backend.fit_pop(pop)                                 # warm on FULL pop
    cores, e2es, last = [], [], None
    for _ in range(n_rep):
        res = backend.fit_pop(pop)
        cores.append(res.wall_core)
        e2es.append(res.wall_e2e)
        last = res
    med_loss = gate_med_loss_fp64(pop, last.c_final)
    n_drop = int(np.sum(last.status == 3))
    from cusr.benchmark import interp
    losses = interp.loss_pop(pop, np.asarray(last.c_final, np.float64))
    wall_core = statistics.median(cores)
    return dict(
        wall_core=wall_core, wall_core_reps=list(cores),
        wall_e2e=statistics.median(e2es),
        throughput=throughput(pop["M"], n_drop, wall_core * 1000.0),
        med_loss_fp64=med_loss, n_dropped=n_drop,
        frac_converged=float(np.mean(last.status == 0)) if last.status.size else 0.0,
        frac_failed=float(np.mean(last.status == 2)) if last.status.size else 0.0,
        losses=losses, n_rep=n_rep,
    )


# ===========================================================================
# SMOKE RUN (small, real) — exercises the subprocess path end-to-end.
# ===========================================================================
def smoke(gpu_id: int):
    """One tiny config per variant (early-gen) + one inner-const-heavy (high K).
    Asserts each throughput is WAY above 2k (must clear the 6000 floor); if any
    is in the 2-6k gray zone, disambiguate with a larger-M run before deciding."""
    OUT = Path(__file__).resolve().parent / "out"
    scratch = OUT / "smoke_scratch"
    # DURABLE write path: prove validate-then-append + .npy sidecar + skip-record
    # end-to-end (not just via synthetic helper inputs). One JSONL per smoke run.
    jsonl = OUT / "smoke_e6.jsonl"
    jsonl.unlink(missing_ok=True)
    loss_dir = OUT / "smoke_e6_losses"
    rows = []
    tripwire_pass = True
    configs = [("early-gen", 2000, 1000, 0, 50),
               ("inner-const-heavy", 2000, 1000, 0, 50)]
    for variant in GPU_VARIANTS:   # include revad in the end-to-end smoke, not just fusedfd/ad
        for preset, M, N, seed, mi in configs:
            pop = get_pop(preset, M, N, seed)
            K_max = int(pop["K_max"])
            skip, reason, gb = footprint_skip(M, N, K_max, pop["total_nodes"])
            if skip:
                print(f"[SKIP] {variant} {preset} M={M} N={N}: {reason}")
                continue
            tag = f"{variant}_{preset}_{M}_{N}_{seed}"
            outd = scratch / f"{tag}_gpu{gpu_id}"
            try:
                k = run_gpu_kernel(pop, variant, mi, outd, gpu_id, n_rep=3)
            except SystemExit as e:
                print(str(e))
                tripwire_pass = False
                rows.append(dict(variant=variant, preset=preset, M=M, N=N,
                                 status="tripwire_halt"))
                continue
            tput = k["throughput"]
            # tripwire AFTER measuring (prints measured-vs-expected, may raise)
            try:
                tripwire_gpu_tput(variant, M, tput)
            except SystemExit as e:
                print(str(e))
                tripwire_pass = False
                rows.append(dict(variant=variant, preset=preset, M=M, N=N,
                                 throughput=tput, status="tripwire_halt"))
                continue
            in_range = tput > GPU_TPUT_FLOOR
            print(f"  {variant} {preset} M={M} N={N}: tput={tput:.0f} tr/s "
                  f"loop_ms={k['loop_ms']:.2f} K_max={K_max} "
                  f"med_loss_fp64={k['med_loss_fp64']:.3e} drop={k['n_dropped']} "
                  f"footprint={gb:.3f}GB range_ok={in_range}")
            # VALIDATE-THEN-APPEND (audit blocker): the record is only written
            # AFTER run_gpu_kernel validated outputs + throughput finite. Save
            # the per-tree fp64 losses to a durable .npy sidecar FIRST.
            key = ("kernel", variant, preset, M, mi)
            save_losses(loss_dir, key, k["losses"])
            rec = dict(backend="kernel", variant=variant, preset=preset, M=M, N=N,
                       knob=mi, max_iter=mi, K_max=K_max, status="ok",
                       throughput=tput, loop_ms=k["loop_ms"], setup_ms=k["setup_ms"],
                       med_loss_fp64=k["med_loss_fp64"], n_dropped=k["n_dropped"],
                       pop_sha256=k["pop_sha256"], n_rep=k["n_rep"])
            assert is_complete_record(rec), "smoke record failed completeness check"
            append_record(jsonl, rec)
            rows.append(dict(variant=variant, preset=preset, M=M, N=N,
                             throughput_tps=tput, loop_ms=k["loop_ms"],
                             med_loss_fp64=k["med_loss_fp64"], K_max=K_max,
                             expected_range_ok=in_range, status="ok"))

    # emit ONE ceiling-skip record (M=256k N=10k K=32 = ~344GB) so the
    # skip-record-logging contract is exercised end-to-end, NEVER silently
    # dropped. This config is geometrically possible but over the 40GB ceiling.
    cs_skip, cs_reason, cs_gb = footprint_skip(256000, 10000, 32, 256000 * 12)
    assert cs_skip
    skip_rec = dict(backend="kernel", variant="ad", preset="inner-const-heavy",
                    M=256000, N=10000, knob=50, status="skipped",
                    reason="memory_ceiling", skip_detail=cs_reason,
                    estimated_footprint_gb=cs_gb, K_max=32)
    append_record(jsonl, skip_rec)
    print(f"[SKIP-RECORD] logged ceiling skip M=256000 N=10000 K=32 "
          f"({cs_gb:.0f}GB) -> {jsonl.name}")

    # prove resumability semantics on the durable file: load_done counts ONLY
    # the validated 'ok' records (4), never the ceiling skip.
    done = load_done(jsonl)
    all_recs = load_records(jsonl)
    n_skip = sum(1 for r in all_recs if r.get("status") == "skipped")
    print(f"[RESUME-CHECK] jsonl: {len(all_recs)} records, {len(done)} complete "
          f"(done), {n_skip} logged skip(s); load_done excludes skips: "
          f"{len(done) == len(rows) and n_skip == 1}")
    return rows, tripwire_pass


# ===========================================================================
# FULL-MATRIX PLAN (deterministic enumeration + up-front ceiling evaluation)
# ===========================================================================
# AXIS ENCODING INTO config_key's 5 SLOTS (cannot extend config_key — it is
# pinned by test_load_done_only_counts_complete_records to the exact 5-tuple
# (backend, variant, preset, M, knob)). max_iter is constant per backend so it
# carries no information; we repurpose the free slots to encode N (and ncores):
#   kernel: key = (kernel, variant,      preset, M, knob=N)          [seed median]
#   operon: key = (operon, f"N{N}",      preset, M, knob=ncores)     [seed 0 only]
# This keeps every (variant,preset,M,N) / (preset,M,N,ncores) cell UNIQUE so
# load_done dedup + save_losses sidecars never collide.
GPU_VARIANTS = ["fusedfd", "ad", "revad"]


def _ref_corpus(preset):
    """Probe K_max + nodes-per-tree ONCE per preset (cached). K_max is a near-
    stable structural cap; we probe at M=16000 to capture the small 12->13 bump
    that low-M misses. total_nodes for a target M := round(nodes_per_tree * M).
    Used ONLY for up-front ceiling evaluation; the ACTUAL run re-checks
    footprint_skip with the real pop's K_max before launch (it can edge up)."""
    cache = _ref_corpus._cache
    if preset not in cache:
        pop = get_pop(preset, 16000, 1000, 0)
        cache[preset] = (int(pop["K_max"]), pop["total_nodes"] / 16000.0)
        evict_pop(preset, 16000, 1000, 0)
    return cache[preset]


_ref_corpus._cache = {}


def build_plan(presets, MS_, NS_, seeds, ncores_list):
    """Deterministic full-matrix plan. Each entry is a dict with the config_key
    fields + axis metadata + an up-front {will_run | skip:reason} verdict.
    GPU verdicts come from the memory footprint gate; Operon verdicts come from
    the always-skip rule (ncores=1 at M>=64000) plus the calibration-probe time
    projection (filled in later by mark_operon_time_skips)."""
    gpu, operon = [], []
    for variant in GPU_VARIANTS:
        for preset in presets:
            K_ref, nt_per = _ref_corpus(preset)
            for M in MS_:
                for N in NS_:
                    tn = int(round(nt_per * M))
                    skip, reason, gb = footprint_skip(M, N, K_ref, tn)
                    gpu.append(dict(
                        backend="kernel", variant=variant, preset=preset,
                        M=M, N=N, knob=N, seeds=list(seeds),
                        K_ref=K_ref, est_footprint_gb=gb,
                        will_run=not skip,
                        skip_reason=("memory_ceiling" if skip else None),
                        skip_detail=(reason if skip else None)))
    for preset in presets:
        for M in MS_:
            for N in NS_:
                for nc in ncores_list:
                    always = (nc == 1 and M >= 64000)
                    operon.append(dict(
                        backend="operon", variant=f"N{N}", preset=preset,
                        M=M, N=N, knob=nc, ncores=nc, seed=0,
                        will_run=not always,
                        skip_reason=("operon_ncores1_largeM" if always else None),
                        skip_detail=("unconditional skip: ncores=1 at "
                                     f"M={M}>=64000 (single-core would dominate "
                                     "wall-clock)" if always else None),
                        est_time_s=None))
    return gpu, operon


def operon_time_projection(probe_s, probe_M, probe_N, probe_nc, M, N, nc):
    """Extrapolate Operon wall_core by the work model time ~ M*N/ncores from one
    measured calibration point. Conservative: ignores fixed setup (warmup is
    excluded from wall_core anyway, matching the GPU loop_ms metric)."""
    if probe_s <= 0:
        return float("inf")
    scale = (M * N / nc) / (probe_M * probe_N / probe_nc)
    return probe_s * scale


def mark_operon_time_skips(operon_plan, probe_s, probe_M, probe_N, probe_nc,
                           ceiling_s=OPERON_TIME_CEILING_S):
    """Fill est_time_s for every operon entry and convert projected-over-ceiling
    configs into time skips (UNION with the already-set always-skip ncores=1
    rule — never un-skip). Mutates + returns the plan."""
    for e in operon_plan:
        est = operon_time_projection(probe_s, probe_M, probe_N, probe_nc,
                                     e["M"], e["N"], e["ncores"])
        e["est_time_s"] = est
        if e["will_run"] and est > ceiling_s:
            e["will_run"] = False
            e["skip_reason"] = "time_ceiling"
            e["skip_detail"] = (f"projected wall_core {est:.0f}s > "
                                f"{ceiling_s:.0f}s ceiling (M={e['M']} N={e['N']} "
                                f"ncores={e['ncores']}; from calib "
                                f"{probe_s:.2f}s @ M{probe_M}/N{probe_N}/"
                                f"nc{probe_nc})")
    return operon_plan


# ===========================================================================
# COST ESTIMATE (printed up front; Phase-A from smoke t/s scaled by M, Phase-B
# from the calibration probe). Transparent so a reviewer sees the budget.
# ===========================================================================
# calibration-derived points/s (M*N evaluated per second) per variant. GPU loop
# time scales with total WORK (M*N), NOT trees, so a trees/s model is N-blind and
# undercounts high-N configs ~10x. Anchored near the low-M calibration (points/s
# rises with M as occupancy saturates, so this is a conservative upper-ish budget).
_POINTS_PER_S = {"fusedfd": 8.0e6, "ad": 1.4e7, "revad": 1.4e7}  # revad≈ad speed (cost estimate only)
_SMOKE_SETUP_MS = 380.0   # observed setup_ms (CUDA init+alloc) per subprocess rep


def estimate_phase_a_min(gpu_plan, n_rep=3, n_gpus=8):
    """Wall-minutes for Phase A. Per config: n_seeds * n_rep subprocess runs, each
    ~ (M*N work / points_per_s) loop time + setup_ms. Work-based (M*N) so high-N
    configs are not undercounted. Divided by n_gpus parallelism. Rough budget."""
    total_s = 0.0
    for e in gpu_plan:
        if not e["will_run"]:
            continue
        pps = _POINTS_PER_S.get(e["variant"], 1.0e7)
        n_seeds = len(e["seeds"])
        loop_s = (e["M"] * e["N"]) / pps             # one rep loop time (work-based)
        per_rep_s = loop_s + _SMOKE_SETUP_MS / 1000.0
        total_s += per_rep_s * n_rep * n_seeds
    return (total_s / max(1, n_gpus)) / 60.0


def estimate_phase_b_min(operon_plan, n_rep=3, warmup_reps=1):
    """Wall-minutes for Phase B (SEQUENTIAL). Per will_run config:
    (n_rep + warmup) * projected wall_core. Uses the calibration projection
    already stored in est_time_s."""
    total_s = 0.0
    for e in operon_plan:
        if not e["will_run"]:
            continue
        est = e.get("est_time_s")
        if est is None or not np.isfinite(est):
            continue
        total_s += (n_rep + warmup_reps) * est
    return total_s / 60.0


def print_plan_cost(gpu_plan, operon_plan, est_a, est_b):
    n_gpu = len(gpu_plan)
    n_op = len(operon_plan)
    n_mem = sum(1 for e in gpu_plan if e["skip_reason"] == "memory_ceiling")
    n_time = sum(1 for e in operon_plan
                 if e["skip_reason"] in ("time_ceiling", "operon_ncores1_largeM"))
    gpu_run = sum(1 for e in gpu_plan if e["will_run"])
    op_run = sum(1 for e in operon_plan if e["will_run"])
    print("=" * 72)
    print("E6 SWEEP PLAN + COST ESTIMATE (full matrix)")
    print("=" * 72)
    print(f"  n_gpu_configs    = {n_gpu}   (will_run={gpu_run}, "
          f"mem-skip={n_mem})")
    print(f"  n_operon_configs = {n_op}  (will_run={op_run}, "
          f"time/ncores1-skip={n_time})")
    print(f"  expected (plan)  = {n_gpu + n_op}")
    print(f"  n_skipped (mem)  = {n_mem}")
    print(f"  n_skipped (time) = {n_time}")
    print(f"  est Phase-A wall = {est_a:.1f} min  (GPU, 8-way parallel, "
          f"3 seeds x 3 reps)")
    print(f"  est Phase-B wall = {est_b:.1f} min  (Operon, sequential, "
          f"calibration-projected)")
    print("=" * 72)
    return dict(n_gpu_configs=n_gpu, n_operon_configs=n_op,
                n_skipped_mem=n_mem, n_skipped_time=n_time,
                est_phaseA_min=round(est_a, 2), est_phaseB_min=round(est_b, 2))


# ===========================================================================
# PHASE A — GPU configs, up to 8-way parallel across GPU0..7.
# Workers do GPU work + validation and RETURN a result; the MAIN thread does
# tripwire (SystemExit must propagate from main, not a swallowed worker), then
# save_losses + append_record (single-writer; no interleaved fsync).
# ===========================================================================
def _gpu_worker(entry, gpu_id, scratch, max_iter, n_rep):
    """Runs all seeds for one GPU config on one gpu_id; returns a result dict.
    Throughput = MEDIAN across seeds; quality (med_loss_fp64) = SEED 0 (the
    iso-quality gate pairs GPU-seed0 vs Operon-seed0). Re-checks footprint with
    the REAL pop K_max before launch -> may turn into a late memory skip."""
    variant, preset, M, N = entry["variant"], entry["preset"], entry["M"], entry["N"]
    per_seed = []
    seed0 = None
    K_real = entry["K_ref"]
    for seed in entry["seeds"]:
        pop = get_pop(preset, M, N, seed)
        K_real = int(pop["K_max"])
        skip, reason, gb = footprint_skip(M, N, K_real, pop["total_nodes"])
        if skip:
            evict_pop(preset, M, N, seed)
            return dict(kind="late_mem_skip", entry=entry, reason=reason, gb=gb,
                        K_real=K_real)
        tag = f"{variant}_{preset}_{M}_{N}_{seed}"
        outd = scratch / f"{tag}_gpu{gpu_id}"
        k = run_gpu_kernel(pop, variant, max_iter, outd, gpu_id, n_rep=n_rep)
        if seed == 0:
            seed0 = k
        per_seed.append(k)
        evict_pop(preset, M, N, seed)
    tputs = [k["throughput"] for k in per_seed]
    med_idx = sorted(range(len(tputs)), key=lambda i: tputs[i])[len(tputs) // 2]
    med = per_seed[med_idx]
    q = seed0 if seed0 is not None else med
    return dict(kind="ok", entry=entry, med=med, seed0=q, gpu_id=gpu_id,
                med_tput=statistics.median(tputs), K_real=K_real, tputs=tputs)


def phase_a(gpu_plan, jsonl, loss_dir, scratch, n_gpus, max_iter, n_rep,
            done_keys, locked_mhz=None, gpu_ids=None):
    """Dispatch will_run GPU configs up to n_gpus-way parallel. As each finishes
    (main thread): tripwire (HALT on SystemExit) -> save_losses -> append_record.
    Skips analytically-skipped + already-done configs; logs a skip record for
    each up-front and late memory skip (never silently dropped).

    gpu_ids: explicit PHYSICAL GPU ids to use (e.g. [7] to pin one clean card on a
    shared box). Defaults to range(n_gpus). Each id is set as CUDA_VISIBLE_DEVICES
    on the subprocess and recorded as gpu_id, so it must be the real device index."""
    scratch.mkdir(parents=True, exist_ok=True)
    pool = list(range(n_gpus)) if gpu_ids is None else list(gpu_ids)
    gid_q = queue.Queue()
    for g in pool:
        gid_q.put(g)

    # 1) log up-front memory skips (once; skip if already a skip record exists is
    #    not needed — finalize dedups by key + status). Only log if not done.
    pending = []
    for e in gpu_plan:
        key = config_key(e)
        if not e["will_run"]:
            append_record(jsonl, dict(
                backend="kernel", variant=e["variant"], preset=e["preset"],
                M=e["M"], N=e["N"], knob=e["knob"], status="skipped",
                reason=e["skip_reason"], skip_detail=e["skip_detail"],
                estimated_footprint_gb=e["est_footprint_gb"], K_max=e["K_ref"]))
            continue
        if key in done_keys:
            print(f"[RESUME] kernel {e['variant']} {e['preset']} M={e['M']} "
                  f"N={e['N']} already complete -> skip")
            continue
        pending.append(e)

    print(f"[PHASE A] dispatching {len(pending)} GPU config(s) across "
          f"{n_gpus} GPU(s)")
    results = []

    def task(entry):
        gid = gid_q.get()
        try:
            return _gpu_worker(entry, gid, scratch, max_iter, n_rep)
        finally:
            gid_q.put(gid)

    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futs = {ex.submit(task, e): e for e in pending}
        for fut in as_completed(futs):
            entry = futs[fut]
            try:
                res = fut.result()
            except RuntimeError as err:
                # subprocess/validation failure -> goes to missing[], retried on
                # resume. NEVER write an ok record. (Tripwire is separate below.)
                print(f"[PHASE A][FAIL] {entry['variant']} {entry['preset']} "
                      f"M={entry['M']} N={entry['N']}: {err}")
                continue
            if res["kind"] == "late_mem_skip":
                e = res["entry"]
                append_record(jsonl, dict(
                    backend="kernel", variant=e["variant"], preset=e["preset"],
                    M=e["M"], N=e["N"], knob=e["knob"], status="skipped",
                    reason="memory_ceiling", skip_detail=res["reason"],
                    estimated_footprint_gb=res["gb"], K_max=res["K_real"]))
                print(f"[PHASE A][LATE-SKIP] {e['variant']} {e['preset']} "
                      f"M={e['M']} N={e['N']}: real K_max bumped over ceiling")
                continue
            # ---- MAIN-THREAD tripwire: SystemExit propagates -> halts run ----
            med = res["med"]
            tripwire_gpu_tput(entry["variant"], entry["M"], med["throughput"])
            # validate-then-record: save .npy sidecar FIRST, then append
            key = config_key(entry)
            save_losses(loss_dir, key, res["seed0"]["losses"])
            rec = dict(
                backend="kernel", variant=entry["variant"], preset=entry["preset"],
                M=entry["M"], N=entry["N"], knob=entry["knob"], seed_median=True,
                max_iter=max_iter, K_max=res["K_real"],
                status="ok",
                # PRIMARY metric: loop_ms-derived throughput (per-gen in-loop)
                throughput=med["throughput"], loop_ms=med["loop_ms"],
                # E2E metric: total_ms incl CUDA init
                total_ms=float(med["prof"].get("total_ms", float("nan"))),
                throughput_e2e=throughput(
                    entry["M"], med["n_dropped"],
                    float(med["prof"].get("total_ms", float("nan")))),
                setup_ms=med["setup_ms"], load_ms=med["load_ms"],
                # quality from SEED 0 (iso-quality gate partner)
                med_loss_fp64=res["seed0"]["med_loss_fp64"],
                med_tput_seeds=res["tputs"], n_dropped=med["n_dropped"],
                frac_converged=med["frac_converged"],
                frac_failed=med["frac_failed"],
                pop_sha256=res["seed0"]["pop_sha256"], n_rep=n_rep,
                n_seeds=len(entry["seeds"]),
                # clock-lock provenance (铁律 #3): which GPU + the locked SM clock
                # this record was measured under. locked_clock_mhz=None => DRAFT.
                gpu_id=res.get("gpu_id"), locked_clock_mhz=locked_mhz)
            assert is_complete_record(rec), "phase-a record failed completeness"
            append_record(jsonl, rec)
            results.append(rec)
            print(f"[PHASE A][OK] {entry['variant']} {entry['preset']} "
                  f"M={entry['M']} N={entry['N']}: tput={med['throughput']:.0f} "
                  f"tr/s loop_ms={med['loop_ms']:.2f} "
                  f"e2e_tput={rec['throughput_e2e']:.0f} "
                  f"med_loss={rec['med_loss_fp64']:.3e}")
    return results


# ===========================================================================
# PHASE B — Operon, SEQUENTIAL (clean timing; Phase A fully done first). Pairs
# the iso-quality gate GPU-seed0 vs Operon-seed0 on the SAME pop.
# ===========================================================================
def operon_calibration(preset, M, N, ncores):
    """Time ONE small Operon config to anchor the time-ceiling projection. Uses
    the real run_operon helper (wall_core, setup-excluded) so the calibration is
    in the SAME metric as everything else."""
    pop = get_pop(preset, M, N, 0)
    t0 = time.perf_counter()
    o = run_operon(pop, ncores, max_iter=OPERON_MAX_ITER, n_rep=1)
    wall = time.perf_counter() - t0
    evict_pop(preset, M, N, 0)
    return o["wall_core"], wall


def _kernel_seed0_loss(jsonl, variant, preset, M, N):
    """Look up the GPU seed0 fp64 median loss for the quality-gate pair, from the
    completed kernel records (preferring fusedfd then ad for the partner)."""
    for rec in load_records(jsonl):
        if (rec.get("backend") == "kernel" and rec.get("status") == "ok"
                and rec.get("variant") == variant and rec.get("preset") == preset
                and int(rec.get("M", -1)) == M and int(rec.get("knob", -2)) == N):
            return rec.get("med_loss_fp64")
    return None


def phase_b(operon_plan, jsonl, loss_dir, n_rep, done_keys):
    """Run will_run Operon configs SEQUENTIALLY. Log every time/ncores skip.
    For each ok config, evaluate the iso-quality gate vs GPU-seed0 on same pop."""
    pending = []
    for e in operon_plan:
        key = config_key(e)
        if not e["will_run"]:
            append_record(jsonl, dict(
                backend="operon", variant=e["variant"], preset=e["preset"],
                M=e["M"], N=e["N"], knob=e["knob"], ncores=e["ncores"],
                status="skipped", reason=e["skip_reason"],
                skip_detail=e["skip_detail"],
                estimated_time_s=e.get("est_time_s")))
            continue
        if key in done_keys:
            print(f"[RESUME] operon {e['preset']} M={e['M']} N={e['N']} "
                  f"nc={e['ncores']} already complete -> skip")
            continue
        pending.append(e)

    print(f"[PHASE B] running {len(pending)} Operon config(s) SEQUENTIALLY")
    results = []
    for e in pending:
        preset, M, N, nc = e["preset"], e["M"], e["N"], e["ncores"]
        pop = get_pop(preset, M, N, 0)
        try:
            o = run_operon(pop, nc, max_iter=OPERON_MAX_ITER, n_rep=n_rep)
        except Exception as err:  # noqa: BLE001 — operon worker hard-death etc.
            print(f"[PHASE B][FAIL] {preset} M={M} N={N} nc={nc}: {err}")
            evict_pop(preset, M, N, 0)
            continue
        # iso-quality gate: GPU-seed0 vs Operon-seed0 on the SAME (preset,M,N)
        gate = {}
        for gv in GPU_VARIANTS:
            kloss = _kernel_seed0_loss(jsonl, gv, preset, M, N)
            gate[gv] = (None if kloss is None
                        else quality_matched(kloss, o["med_loss_fp64"]))
        key = config_key(e)
        save_losses(loss_dir, key, o["losses"])
        rec = dict(
            backend="operon", variant=e["variant"], preset=preset, M=M, N=N,
            knob=nc, ncores=nc, seed=0, max_iter=OPERON_MAX_ITER, status="ok",
            # PRIMARY: wall_core-derived throughput (setup-excluded, fair to
            # GPU loop_ms). E2E: wall_e2e.
            throughput=o["throughput"], wall_core=o["wall_core"],
            wall_e2e=o["wall_e2e"],
            throughput_e2e=throughput(M, o["n_dropped"], o["wall_e2e"] * 1000.0),
            med_loss_fp64=o["med_loss_fp64"], n_dropped=o["n_dropped"],
            frac_converged=o["frac_converged"], frac_failed=o["frac_failed"],
            quality_gate_vs_kernel=gate, n_rep=n_rep)
        assert is_complete_record(rec), "phase-b record failed completeness"
        append_record(jsonl, rec)
        results.append(rec)
        evict_pop(preset, M, N, 0)
        gstr = ", ".join(f"{k}:{v}" for k, v in gate.items())
        print(f"[PHASE B][OK] {preset} M={M} N={N} nc={nc}: "
              f"tput={o['throughput']:.0f} tr/s wall_core={o['wall_core']:.3f}s "
              f"med_loss={o['med_loss_fp64']:.3e} gate[{gstr}]")
    return results


# ===========================================================================
# FINALIZE — report.json + DONE marker; the accounting identity must hold:
#   completed + len(skipped) + len(missing) == expected
# ===========================================================================
def finalize(gpu_plan, operon_plan, jsonl, out_dir, meta=None, done_name="DONE"):
    expected_keys = [config_key(e) for e in (gpu_plan + operon_plan)]
    expected = len(expected_keys)
    recs = load_records(jsonl)

    completed_keys = set()
    skipped = {}      # key -> skip record (last wins)
    for rec in recs:
        if is_complete_record(rec):
            completed_keys.add(config_key(rec))
        elif rec.get("status") == "skipped":
            skipped[config_key(rec)] = dict(
                key=list(config_key(rec)), reason=rec.get("reason"),
                detail=rec.get("skip_detail"),
                estimated_footprint_gb=rec.get("estimated_footprint_gb"),
                estimated_time_s=rec.get("estimated_time_s"))

    # a key that is BOTH completed and skipped counts as completed (it ran)
    skipped = {k: v for k, v in skipped.items() if k not in completed_keys}
    completed_keys &= set(expected_keys)          # only count in-plan keys
    skipped = {k: v for k, v in skipped.items() if k in set(expected_keys)}

    missing = [list(k) for k in expected_keys
               if k not in completed_keys and k not in skipped]

    completed = len(completed_keys)
    n_skip = len(skipped)
    n_miss = len(missing)
    assert completed + n_skip + n_miss == expected, (
        f"accounting identity broken: {completed}+{n_skip}+{n_miss} "
        f"!= {expected}")

    report = dict(
        experiment="e6_kernel_sweep",
        expected=expected,
        completed=completed,
        n_skipped=n_skip,
        n_missing=n_miss,
        skipped=list(skipped.values()),
        missing=missing,
        accounting_ok=(completed + n_skip + n_miss == expected),
        n_gpu_configs=len(gpu_plan),
        n_operon_configs=len(operon_plan),
        meta=meta or {},
    )
    report_path = out_dir / (meta.get("report_name", "report.json")
                             if meta else "report.json")
    report_path.write_text(json.dumps(report, indent=2, default=float))
    done_path = out_dir / done_name
    done_path.unlink(missing_ok=True)
    if not missing:
        done_path.write_text(
            f"e6 sweep complete: {completed} completed, {n_skip} skipped, "
            f"0 missing of {expected} expected.\n")
        print(f"[FINALIZE] DONE marker written: {done_path}")
    else:
        print(f"[FINALIZE] {n_miss} config(s) MISSING -> no DONE marker "
              f"(re-run to resume)")
    print(f"[FINALIZE] report -> {report_path}  "
          f"(expected={expected} completed={completed} skipped={n_skip} "
          f"missing={n_miss}; identity={'OK' if report['accounting_ok'] else 'BROKEN'})")
    return report, report_path


# ===========================================================================
# DRIVER — wires plan -> cost -> Phase A -> Phase B -> finalize. --full runs the
# whole matrix; --dryrun runs a tiny EXECUTABLE subset of the SAME pipeline.
# ===========================================================================
def run_sweep(dryrun, n_gpus, n_rep, out_dir, locked_mhz=None, gpu_ids=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if dryrun:
        jsonl = out_dir / "dryrun_e6.jsonl"
        loss_dir = out_dir / "dryrun_e6_losses"
        scratch = out_dir / "dryrun_scratch"
        report_name = "dryrun_report.json"
        done_name = "DRYRUN_DONE"
    else:
        jsonl = out_dir / "sweep_e6.jsonl"
        loss_dir = out_dir / "sweep_e6_losses"
        scratch = out_dir / "sweep_scratch"
        report_name = "report.json"
        done_name = "DONE"

    # ---- build the FULL-MATRIX plan unconditionally (plan_summary is ALWAYS
    #      the full matrix; only EXECUTION differs between full and dryrun) ----
    full_gpu, full_operon = build_plan(ALL_PRESETS, MS, NS, SEEDS, OPERON_NCORES)

    # ---- calibration probe (run even in dryrun) to set the time ceiling +
    #      Phase-B cost. Small config; same wall_core metric. ----
    print("[CALIB] timing one small Operon config to anchor time projection ...")
    cal_M, cal_N, cal_nc = 1000, 1000, 16
    cal_core, cal_wall = operon_calibration("early-gen", cal_M, cal_N, cal_nc)
    print(f"[CALIB] Operon early-gen M={cal_M} N={cal_N} nc={cal_nc}: "
          f"wall_core={cal_core:.3f}s (e2e {cal_wall:.3f}s)")
    mark_operon_time_skips(full_operon, cal_core, cal_M, cal_N, cal_nc)

    est_a = estimate_phase_a_min(full_gpu, n_rep=n_rep)
    est_b = estimate_phase_b_min(full_operon, n_rep=n_rep)
    plan_summary = print_plan_cost(full_gpu, full_operon, est_a, est_b)

    # ---- select the plan to EXECUTE ----
    if dryrun:
        # tiny EXECUTABLE plan that exercises the WHOLE pipeline:
        #   GPU: fusedfd+ad x early-gen x M=1000 x N={100,1000} x seed0  (4 cfg)
        #   Operon: early-gen x M=1000 x N=1000 x nc={16,64} x seed0     (2 cfg)
        exec_gpu, exec_op = build_plan(["early-gen"], [1000], [100, 1000], [0],
                                       [16, 64])
        # restrict operon to N=1000 only (so 2 configs, both with a GPU partner)
        exec_op = [e for e in exec_op if e["N"] == 1000]
        mark_operon_time_skips(exec_op, cal_core, cal_M, cal_N, cal_nc)
        run_gpu, run_operon_plan = exec_gpu, exec_op
        print(f"[DRYRUN] executable subset: {len(run_gpu)} GPU + "
              f"{len(run_operon_plan)} Operon configs")
    else:
        run_gpu, run_operon_plan = full_gpu, full_operon

    # ---- resume: load already-complete keys from THIS run's jsonl ----
    done_keys = set(load_done(jsonl))
    if done_keys:
        print(f"[RESUME] {len(done_keys)} complete record(s) found in "
              f"{jsonl.name}; will skip those configs")

    # ---- PHASE A (GPU) ----
    phase_a(run_gpu, jsonl, loss_dir, scratch, n_gpus,
            KERNEL_MAX_ITER, n_rep, done_keys, locked_mhz=locked_mhz,
            gpu_ids=gpu_ids)
    # ---- PHASE B (Operon, sequential; Phase A fully done) ----
    done_keys = set(load_done(jsonl))     # refresh after Phase A
    phase_b(run_operon_plan, jsonl, loss_dir, n_rep, done_keys)

    # ---- finalize against the EXECUTED plan (its own accounting identity) ----
    meta = dict(
        report_name=report_name,
        mode=("dryrun" if dryrun else "full"),
        timing_metric=dict(
            primary="GPU loop_ms vs Operon wall_core (setup-excluded, in-loop "
                    "per-generation CO cost)",
            e2e="GPU total_ms (incl CUDA init) vs Operon wall_e2e (incl marshal)",
            note="both recorded per record; primary is 'throughput', e2e is "
                 "'throughput_e2e'"),
        kernel_max_iter=KERNEL_MAX_ITER, operon_max_iter=OPERON_MAX_ITER,
        n_rep=n_rep, n_gpus=n_gpus,
        full_matrix_plan_summary=plan_summary,
        calibration=dict(preset="early-gen", M=cal_M, N=cal_N, ncores=cal_nc,
                         wall_core_s=cal_core, wall_e2e_s=cal_wall,
                         time_ceiling_s=OPERON_TIME_CEILING_S),
        seed_protocol="kernel: median throughput over seeds 0,1,2; quality "
                      "(med_loss_fp64) from seed0. operon: seed0 only.",
        locked_clock_mhz=locked_mhz,
    )
    report, report_path = finalize(run_gpu, run_operon_plan, jsonl, out_dir,
                                   meta=meta, done_name=done_name)
    return plan_summary, report, report_path


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny real GPU subprocess smoke (no full matrix)")
    ap.add_argument("--full", action="store_true",
                    help="run the FULL matrix (Phase A 8-way GPU + Phase B "
                         "sequential Operon); resumable from the JSONL")
    ap.add_argument("--dryrun", action="store_true",
                    help="run a tiny EXECUTABLE subset of the WHOLE pipeline "
                         "(4 GPU + 2 Operon configs) -> dryrun_report.json; "
                         "still prints the FULL-matrix plan + cost estimate")
    ap.add_argument("--gpu", type=int, default=int(
        os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0))
    ap.add_argument("--n-gpus", type=int, default=8,
                    help="Phase-A parallelism (GPUs 0..n-1)")
    ap.add_argument("--n-rep", type=int, default=3, help="timing reps / config")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default: <script>/out). Use a fresh dir "
                         "(e.g. e7_section2/out) when reusing Operon records.")
    ap.add_argument("--locked-mhz", type=int, default=None,
                    help="SM clock (MHz) the GPUs are locked to for this run; "
                         "recorded per kernel record. Omit => records are DRAFT.")
    ap.add_argument("--gpu-ids", type=str, default=None,
                    help="comma-sep PHYSICAL GPU ids to use instead of 0..n-1 "
                         "(e.g. '7' to pin one clean card on a shared machine).")
    args = ap.parse_args()
    gpu_ids = ([int(x) for x in args.gpu_ids.split(",")]
               if args.gpu_ids else None)

    OUT = Path(args.out).resolve() if args.out else (
        Path(__file__).resolve().parent / "out")
    if args.smoke:
        rows, tw = smoke(args.gpu)
        print(f"\nSMOKE rows={len(rows)} tripwire_pass={tw}")
        print(json.dumps(rows, indent=2, default=float))
        return
    if args.dryrun:
        # dryrun executes a tiny subset via the SAME 8-way scheduler + Phase B.
        n_gpus = max(1, min(args.n_gpus, 8))
        plan_summary, report, _ = run_sweep(
            dryrun=True, n_gpus=n_gpus, n_rep=args.n_rep, out_dir=OUT,
            locked_mhz=args.locked_mhz)
        print("\n[DRYRUN SUMMARY] full-matrix plan_summary:",
              json.dumps(plan_summary, default=float))
        print("[DRYRUN SUMMARY] dryrun accounting:",
              f"expected={report['expected']} completed={report['completed']} "
              f"skipped={report['n_skipped']} missing={report['n_missing']} "
              f"identity_ok={report['accounting_ok']}")
        return
    if args.full:
        n_gpus = len(gpu_ids) if gpu_ids else max(1, min(args.n_gpus, 8))
        run_sweep(dryrun=False, n_gpus=n_gpus,
                  n_rep=args.n_rep, out_dir=OUT, locked_mhz=args.locked_mhz,
                  gpu_ids=gpu_ids)
        return
    raise SystemExit("Choose a mode: --smoke | --dryrun | --full. "
                     "(--dryrun proves the whole pipeline on a tiny plan; "
                     "--full runs the matrix.)")


if __name__ == "__main__":
    main()
