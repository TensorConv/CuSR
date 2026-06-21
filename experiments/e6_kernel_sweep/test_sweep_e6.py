#!/usr/bin/env python
"""test_sweep_e6.py — TDD invariants for the e6 corrected GPU CO kernel sweep.

Each test encodes one anti-fakery contract the e6 harness MUST obey. They are
written against PURE helpers in sweep_e6.py so the suite needs no GPU / no .so /
no 1-hour run. The six SPEC-mandated areas:

  1. FOOTPRINT GUARD : footprint formula sums ALL device buffers (audit blocker
     #3 -- not just (4+K_max)*M*N*4); >40GB -> skip with a logged reason.
  2. THROUGHPUT      : (M - n_dropped)/(loop_ms/1000) parsed from PROFILE_JSON.
  3. VARIANT/BINARY  : exact-basename allowlist {batch_lm_fusedfd_prof,
     batch_lm_ad_prof}; reject 'fd'-alone / libcusr_co_fd host path (blocker #2).
  4. ANTI-HOST-FD    : structural device-Jacobian PROFILE guard (N-invariant) +
     binary allowlist; the absolute trees/s floor is only an anti-garbage check
     (a genuine M=1000/N=10000 corner runs ~500 tr/s, overlapping the host band).
  5. QUALITY GATE    : fp64 loss recomputed via interp.loss_pop from c_final on
     disk (independent of backend); kernel <= operon * 1.05.
  6. RESUMABLE JSONL : a partial/invalid record is NOT counted as done; a skip
     record is logged (not silently dropped).

Run:
  uv run python -m pytest experiments/e6_kernel_sweep/test_sweep_e6.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e6_sweep", _HERE / "sweep_e6.py")
e6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e6)

from cusr.benchmark import interp  # noqa: E402
from cusr.benchmark.workload.gen_synth import gen_pop  # noqa: E402


# =========================================================== 1. FOOTPRINT GUARD
def test_footprint_sums_all_device_buffers_not_just_4plusK():
    """Audit blocker #3: the naive (4+K_max)*M*N*4 formula UNDERCOUNTS — it omits
    d_JtJ (K^2*M), d_JtR/d_delta (K*M), and the node arrays. The real formula
    must be STRICTLY LARGER so the gate is conservative (never OOM-then-crash)."""
    M, N, K_max, total_nodes = 64000, 1000, 32, 64000 * 10
    naive = (4 + K_max) * M * N * 4
    real = e6.gpu_footprint_bytes(M, N, K_max, total_nodes)
    assert real > naive, "footprint must exceed the (4+K_max)*M*N*4 lower bound"
    # must at least include d_JtJ (K_max^2 * M * 4) on top of the per-N buffers
    assert real >= naive + K_max * K_max * M * 4


def test_footprint_skip_over_40gb_logs_reason():
    """N=10k, M=256k, K=32 is ~36 TB-class -> WAY over 40GB -> skip + reason."""
    skip, reason, gb = e6.footprint_skip(256000, 10000, 32, 256000 * 12,
                                         ceiling_gb=40.0)
    assert skip is True
    assert isinstance(reason, str) and reason  # non-empty logged reason
    assert "40" in reason or "ceiling" in reason.lower() or "memory" in reason.lower()
    assert gb > 40.0


def test_footprint_skip_under_40gb_does_not_skip():
    """Smoke-scale config (M=2000, N=1000, K small) is far under 40GB -> run."""
    skip, reason, gb = e6.footprint_skip(2000, 1000, 8, 2000 * 8, ceiling_gb=40.0)
    assert skip is False
    assert gb < 40.0


# =============================================================== 2. THROUGHPUT
def test_throughput_uses_loop_ms_and_excludes_dropped():
    """throughput = (M - n_dropped)/(loop_ms/1000). e.g. 60000 trees, 100 K0
    dropped, loop_ms=1000 -> 59900 trees in 1.0s -> 59900 tr/s."""
    assert e6.throughput(60000, 100, 1000.0) == pytest.approx(59900.0)


def test_throughput_parsed_from_profile_json_loop_ms():
    """The denominator is loop_ms (LM loop only), NOT total_ms (incl CUDA init).
    Parse a real-shape PROFILE_JSON line and confirm loop_ms is what is used."""
    line = ('PROFILE_JSON {"M":2000,"N":1000,"iters_run":50,"load_ms":80.0,'
            '"setup_ms":40.0,"loop_ms":250.0,"total_ms":900.0,"gpu_sum_ms":200.0,'
            '"host_residual_ms":50.0,"cats":{}}')
    prof = json.loads(line[len("PROFILE_JSON"):])
    tput = e6.throughput(2000, 0, prof["loop_ms"])
    # 2000 / 0.25s = 8000 tr/s; using total_ms (0.9s) would wrongly give ~2222
    assert tput == pytest.approx(8000.0)
    assert tput != pytest.approx(2000 / (prof["total_ms"] / 1000.0))


def test_throughput_nonpositive_loop_ms_is_nan():
    assert np.isnan(e6.throughput(2000, 0, 0.0))


# ====================================================== 3. VARIANT / BINARY PATH
def test_assert_gpu_binary_accepts_allowlisted_prof_binaries(tmp_path):
    for name in ("batch_lm_fusedfd_prof", "batch_lm_ad_prof"):
        p = tmp_path / name
        p.write_text("x")
        e6.assert_gpu_binary(str(p))  # must not raise


def test_assert_gpu_binary_rejects_host_fd_so(tmp_path):
    """libcusr_co_fd.so is the SLOW in-process host-FD path e6 exists to replace.
    Must be rejected even though it 'contains fd'."""
    p = tmp_path / "libcusr_co_fd.so"
    p.write_text("x")
    with pytest.raises((AssertionError, ValueError)):
        e6.assert_gpu_binary(str(p))


def test_assert_gpu_binary_rejects_nonprof_and_plain_fd(tmp_path):
    """'batch_lm_fusedfd' (no _prof) and any non-allowlisted name are rejected —
    but the legit 'batch_lm_fusedfd_prof' (which CONTAINS substring 'fd') is NOT
    rejected by a naive 'fd' check (the trap)."""
    for name in ("batch_lm_fusedfd", "batch_lm", "batch_lm_co_fd", "co_fd"):
        p = tmp_path / name
        p.write_text("x")
        with pytest.raises((AssertionError, ValueError)):
            e6.assert_gpu_binary(str(p))


def test_assert_gpu_binary_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, AssertionError, ValueError)):
        e6.assert_gpu_binary(str(tmp_path / "batch_lm_ad_prof"))  # not created


# ============================================ 4. ANTI-HOST-FD GUARDS (structural)
def test_tripwire_is_anti_garbage_only_not_host_fd_discriminator():
    """The absolute throughput floor is now an ANTI-GARBAGE check (floor 50). A
    genuine small-M/large-N corner legitimately runs at ~500 trees/s (overlaps the
    1.3-2k host-FD band), so the floor must NOT halt it; only ~0/NaN halts. The
    host-FD discriminator is assert_device_jacobian (below), not this floor."""
    e6.tripwire_gpu_tput("fusedfd", 1000, 517.0)     # genuine M=1000/N=10000 inner
    e6.tripwire_gpu_tput("ad", 1000, 1044.0)         # genuine, must NOT halt
    e6.tripwire_gpu_tput("fusedfd", 64000, 36000.0)  # fast config, no halt
    with pytest.raises((SystemExit, RuntimeError, AssertionError)):
        e6.tripwire_gpu_tput("fusedfd", 64000, 0.0)          # garbage -> halt
    with pytest.raises((SystemExit, RuntimeError, AssertionError)):
        e6.tripwire_gpu_tput("ad", 64000, float("nan"))      # NaN -> halt


def test_assert_device_jacobian_passes_with_device_kernel():
    """On-device fusedfd/ad time the Jacobian in a DEVICE kernel under the
    'fd_jacobian' PROFILE category (ms>0, launches>0). N-INVARIANT: this is the
    real host-FD discriminator, valid even where trees/s overlaps the host band."""
    prof = dict(loop_ms=155.0,
                cats={"fd_jacobian": {"ms": 138.6, "launches": 50},
                      "build_jtj": {"ms": 5.0, "launches": 50}})
    e6.assert_device_jacobian(prof)  # must not raise


def test_assert_device_jacobian_rejects_host_fd_profile():
    """The host-FD path computes FD on the HOST -> ZERO device fd_jacobian time.
    Must be rejected at ANY throughput (what an absolute floor cannot do)."""
    host_fd = dict(loop_ms=500.0,
                   cats={"fd_jacobian": {"ms": 0.0, "launches": 0},
                         "memcpy_H2D": {"ms": 400.0, "launches": 5000}})
    with pytest.raises((RuntimeError, AssertionError)):
        e6.assert_device_jacobian(host_fd)
    with pytest.raises((RuntimeError, AssertionError, KeyError)):
        e6.assert_device_jacobian(dict(loop_ms=500.0, cats={}))


# ============================================================== 5. QUALITY GATE
def test_gate_med_loss_fp64_from_cfinal_on_disk(tmp_path):
    """Quality gate recomputes fp64 loss via interp.loss_pop from the c_final the
    GPU subprocess wrote (total_c float32, same c_offset layout as c_true). The
    backend cannot influence this — only the c array does. c_true -> noise floor,
    c_init (unfit) -> strictly higher loss."""
    pop = gen_pop("inner-const-heavy", 200, 300, 0)
    # simulate the GPU writing c_final.bin == c_true (perfectly fit)
    cf = tmp_path / "c_final.bin"
    np.asarray(pop["c_true"], np.float32).tofile(cf)
    c_disk = np.fromfile(cf, dtype=np.float32)
    assert c_disk.size == pop["total_c"]
    med_fit = e6.gate_med_loss_fp64(pop, c_disk)
    med_init = e6.gate_med_loss_fp64(pop, np.asarray(pop["c_init"], np.float32))
    assert np.isfinite(med_fit)
    assert med_fit < med_init  # fitting the true constants lowers the loss
    # independence: must equal interp.loss_pop median directly
    direct = interp.loss_pop(pop, np.asarray(c_disk, np.float64))
    fin = direct[np.isfinite(direct)]
    assert med_fit == pytest.approx(float(np.median(fin)))


def test_quality_matched_gate_iso_quality():
    assert e6.quality_matched(1.0, 1.0) is True
    assert e6.quality_matched(1.05, 1.0) is True       # exactly at tol
    assert e6.quality_matched(1.06, 1.0) is False      # over tol
    assert e6.quality_matched(float("nan"), 1.0) is False   # cannot certify
    assert e6.quality_matched(1.0, float("inf")) is False


def test_n_k0_dropped_is_host_side_pop_property():
    """The throughput denominator drop count is a pop property (metas K==0),
    computed host-side, NOT trusted from subprocess output."""
    pop = gen_pop("late-gen-bloated", 300, 200, 0)
    expected = int((pop["metas"][:, 3] == 0).sum())
    assert e6.n_k0_dropped(pop) == expected


# ============================================================= 6. RESUMABLE JSONL
def _complete_rec(backend="kernel", variant="fusedfd", preset="early-gen",
                  M=2000, knob=50):
    return dict(backend=backend, variant=variant, preset=preset, M=M, knob=knob,
                status="ok", throughput=8000.0, loop_ms=250.0, med_loss_fp64=1e-3)


def test_partial_record_is_not_complete():
    """audit blocker (resumability): a record appended BEFORE validation (e.g.
    NaN throughput / missing loop_ms / status!=ok) must NOT be treated as done."""
    assert e6.is_complete_record(_complete_rec()) is True
    nan_rec = _complete_rec(); nan_rec["throughput"] = float("nan")
    assert e6.is_complete_record(nan_rec) is False
    no_loop = _complete_rec(); del no_loop["loop_ms"]
    assert e6.is_complete_record(no_loop) is False
    timeout_rec = _complete_rec(); timeout_rec["status"] = "timeout"
    assert e6.is_complete_record(timeout_rec) is False


def test_skip_record_is_logged_not_silently_dropped():
    """A ceiling-skip is a first-class JSONL record (status='skipped' + reason),
    so it is NEVER silently dropped and is distinguishable from done/missing."""
    skip_rec = dict(backend="kernel", variant="ad", preset="inner-const-heavy",
                    M=256000, knob=50, status="skipped",
                    reason="memory_ceiling", estimated_footprint_gb=36000.0)
    # a skip record is not "complete" (it produced no number) ...
    assert e6.is_complete_record(skip_rec) is False
    # ... but it IS a logged skip (has a reason), the anti-silent-drop contract
    assert skip_rec["status"] == "skipped" and skip_rec["reason"]


def test_load_done_only_counts_complete_records(tmp_path):
    """load_done returns the set of config keys that are COMPLETE (validated),
    tolerating a half-written trailing line and ignoring skip/partial records."""
    jsonl = tmp_path / "sweep.jsonl"
    good = _complete_rec(M=2000)
    skip = dict(backend="kernel", variant="ad", preset="early-gen", M=256000,
                knob=50, status="skipped", reason="memory_ceiling")
    partial = _complete_rec(M=4000); partial["throughput"] = float("nan")
    partial["status"] = "ok"
    lines = [json.dumps(good), json.dumps(skip), json.dumps(partial),
             '{"backend": "kernel", "M": 16000, "trunc']  # crash mid-write
    jsonl.write_text("\n".join(lines) + "\n")
    done = e6.load_done(jsonl)
    done_set = set(done)
    assert ("kernel", "fusedfd", "early-gen", 2000, 50) in done_set
    # skip + partial + truncated are NOT counted as done (would mask a re-run)
    assert ("kernel", "ad", "early-gen", 256000, 50) not in done_set
    assert ("kernel", "fusedfd", "early-gen", 4000, 50) not in done_set


def test_load_done_missing_file_is_empty(tmp_path):
    assert list(e6.load_done(tmp_path / "nope.jsonl")) == []
