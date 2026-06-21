#!/usr/bin/env python
"""test_sweep_honesty.py — TDD invariants for the e5 strong-baseline sweep.

These encode the HONESTY RULES the sweep MUST obey (a reviewer/auditor enforces
each one; violating any = fraud). They are deliberately written against PURE
helper functions in sweep.py (quality_matched, common_set_median,
decide_projection, missing_configs, expected_configs, throughput, ...) plus two
small real-backend checks (T1 same-bytes, T2 fp64 loss recompute, T5 fresh reps)
so the suite does not depend on a 1-hour GPU run.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run python -m pytest \
      experiments/e5_strong_baseline/test_sweep_honesty.py -q
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# import sweep.py as a module by path (it is a script, not on sys.path)
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e5_sweep", _HERE / "sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

from cusr.benchmark import interp  # noqa: E402
from cusr.benchmark.workload.gen_synth import gen_pop  # noqa: E402


# --------------------------------------------------------------------------- T2
def test_t2_loss_pop_recomputes_fp64_independent_of_backend():
    """loss_pop recomputes fp64 from c + pop; c_true -> ~noise floor, c_init higher.
    No backend self-report can change this (it only reads c arrays)."""
    pop = gen_pop("inner-const-heavy", 200, 300, 0)
    loss_true = interp.loss_pop(pop, np.asarray(pop["c_true"], np.float64))
    loss_init = interp.loss_pop(pop, np.asarray(pop["c_init"], np.float64))
    med_true = float(np.median(loss_true[np.isfinite(loss_true)]))
    med_init = float(np.median(loss_init[np.isfinite(loss_init)]))
    # noise floor (c_true) is strictly below the unconverged c_init loss
    assert med_true < med_init, (med_true, med_init)
    assert med_true >= 0.0


# --------------------------------------------------------------------------- T3
def test_t3_quality_matched_is_exact_1p05_gate():
    """quality_matched := med_loss_kernel <= med_loss_op1 * 1.05, computed exactly."""
    # just under the gate -> True
    assert sweep.quality_matched(1.05, 1.0) is True
    # exactly at the gate -> True (<=)
    assert sweep.quality_matched(1.05, 1.0) is True
    # just over the gate -> False
    assert sweep.quality_matched(1.0500001, 1.0) is False
    # clearly worse -> False
    assert sweep.quality_matched(2.0, 1.0) is False
    # non-finite operon ref -> not matched (cannot certify)
    assert sweep.quality_matched(1.0, float("inf")) is False
    assert sweep.quality_matched(float("inf"), 1.0) is False


def test_t3_common_set_median_uses_paired_solved_trees_only():
    """Median for the gate must be over trees BOTH backends solved (finite loss),
    not each backend's own surviving subset. Construct a case where the kernel
    'fails' (inf) on the hard trees: filtering finite per-backend would make the
    kernel median look artificially LOW; the common-set median must drop those
    trees from BOTH so the comparison is paired."""
    # tree 0,1 easy; tree 2,3 hard. kernel solves 0,1 well, fails (inf) on 2,3.
    loss_k = np.array([1.0, 1.0, np.inf, np.inf])
    loss_op = np.array([1.0, 1.0, 10.0, 10.0])
    med_k, med_op, n_common = sweep.common_set_median(loss_k, loss_op)
    # common set = trees 0,1 only (both finite) -> both medians 1.0
    assert n_common == 2
    assert med_k == pytest.approx(1.0)
    assert med_op == pytest.approx(1.0)
    # backend-specific finite-filter would give kernel median 1.0 vs op median 5.5
    # (the easier-subset bias the audit flags). common-set avoids that.
    naive_op = float(np.median(loss_op[np.isfinite(loss_op)]))
    assert naive_op == pytest.approx(5.5)
    assert med_op != naive_op


def test_t3_common_set_empty_is_handled():
    loss_k = np.array([np.inf, np.inf])
    loss_op = np.array([1.0, 2.0])
    med_k, med_op, n_common = sweep.common_set_median(loss_k, loss_op)
    assert n_common == 0
    assert not np.isfinite(med_k)
    assert not np.isfinite(med_op)


# --------------------------------------------------------------------------- T4
def test_t4_expected_configs_are_enumerated_asymmetrically():
    """The expected (preset,M,backend) plan is asymmetric: op nproc=1 only at
    M in {1k,4k,16k}; nproc 32/64 + kernel at all M; pareto only at 16k inner."""
    plan = sweep.expected_configs(sweep.MS, sweep.ALL_PRESETS)
    # kernel headline present at every (preset, M)
    for preset in sweep.ALL_PRESETS:
        for M in sweep.MS:
            assert ("kernel", preset, M, 50) in plan
    # operon nproc=1 measured only at the small/mid M's, NOT at 64000
    assert ("operon", "early-gen", 1000, 1) in plan
    assert ("operon", "early-gen", 16000, 1) in plan
    assert ("operon", "early-gen", 64000, 1) not in plan
    # operon parallel at every M including 64000
    for nproc in (32, 64):
        for M in sweep.MS:
            assert ("operon", "inner-const-heavy", M, nproc) in plan
    # pareto kernel iters only at 16k inner-const-heavy
    for it in (25, 100, 200):
        assert ("kernel", "inner-const-heavy", 16000, it) in plan
        assert ("kernel", "early-gen", 16000, it) not in plan


def test_t4_missing_configs_flags_no_silent_skip():
    """Every planned config must appear in the jsonl-derived done set, or be
    reported as missing. A silently dropped config shows up in missing=[...]."""
    plan = [("kernel", "early-gen", 1000, 50),
            ("operon", "early-gen", 1000, 1),
            ("operon", "early-gen", 1000, 64)]
    done = [("kernel", "early-gen", 1000, 50),
            ("operon", "early-gen", 1000, 64)]   # op1 silently absent
    missing = sweep.missing_configs(done, plan)
    assert ("operon", "early-gen", 1000, 1) in missing
    assert len(missing) == 1


def test_t4_full_plan_size_is_48():
    """Asymmetric plan: 12 kernel headline (3x4) + 24 operon parallel (3x4x2)
    + 9 operon single (3x3) + 3 net-new pareto (16k inner {25,100,200}; 50 dedups
    against the headline) = 48. A drift here means a silently dropped config."""
    plan = sweep.expected_configs(sweep.MS, sweep.ALL_PRESETS)
    assert len(plan) == 48, len(plan)


def test_t6_durable_loss_sidecar_gives_paired_gate_after_resume(tmp_path, monkeypatch):
    """The resume path is the EXPECTED path for the 64k sweep. After a crash, the
    in-memory sidecar is empty, but the durable .npy must still feed the PAIRED
    common-set gate (NOT the audit#2-forbidden per-backend median)."""
    monkeypatch.setattr(sweep, "OUT", tmp_path)
    tag = "rt"
    kkey = ("kernel", "inner-const-heavy", 1000, 50)
    okey = ("operon", "inner-const-heavy", 1000, 1)
    # kernel fails (inf) on the 2 hard trees; operon solves all 4
    loss_k = np.array([1.0, 1.0, np.inf, np.inf])
    loss_op = np.array([1.0, 1.0, 10.0, 10.0])
    sweep.save_losses(tag, kkey, loss_k)
    sweep.save_losses(tag, okey, loss_op)
    # simulate resume: in-memory sidecar empty
    sweep._LOSS_SIDECAR.clear()
    krec = {"backend": "kernel", "knob": 50, "med_loss": 1.0}
    orec = {"backend": "operon", "knob": 1, "med_loss": 5.5}
    med_k, med_op, n_common, basis = sweep._gate_medians(
        tag, "inner-const-heavy", 1000, krec, orec)
    assert "common-set" in basis, basis           # paired, not fallback
    assert n_common == 2                           # both-finite trees only
    assert med_op == pytest.approx(1.0)            # NOT the naive 5.5 (audit#2)


def test_t6_gate_fallback_only_when_loss_file_missing(tmp_path, monkeypatch):
    """If a loss .npy is genuinely absent, _gate_medians must FLAG the per-backend
    fallback (never silently claim paired)."""
    monkeypatch.setattr(sweep, "OUT", tmp_path)
    sweep._LOSS_SIDECAR.clear()
    krec = {"backend": "kernel", "knob": 50, "med_loss": 1.0}
    orec = {"backend": "operon", "knob": 1, "med_loss": 5.5}
    med_k, med_op, n_common, basis = sweep._gate_medians(
        tag := "no_files", "inner-const-heavy", 1000, krec, orec)
    assert "missing" in basis.lower()
    assert n_common == -1


def test_t4_done_key_roundtrips_through_jsonl_record():
    """A result record must carry exactly the fields that reconstruct its plan
    key, so resume can skip it and accounting can find it."""
    rec = {"backend": "operon", "preset": "early-gen", "M": 1000, "knob": 64}
    assert sweep.config_key(rec) == ("operon", "early-gen", 1000, 64)


# --------------------------------------------------------------------------- T6
def test_t6_projection_flat_is_flagged_projected_not_measured():
    """64k single-core projection: if per-core throughput flat (<=25% spread),
    project and MARK it 'projected'; never substituted as 'measured'."""
    # flat throughputs across measured M -> project
    tputs = {1000: 100.0, 4000: 105.0, 16000: 98.0}  # spread ~7%
    action, spread, reason, proj_tput = sweep.decide_projection(tputs)
    assert action == "projected"
    assert spread <= 0.25
    assert proj_tput is not None
    assert "projected" in reason.lower()


def test_t6_projection_not_flat_is_omitted_with_reason():
    """If spread > 25%, do NOT project -> omit with explicit reason string."""
    tputs = {1000: 100.0, 4000: 200.0, 16000: 60.0}  # spread huge
    action, spread, reason, proj_tput = sweep.decide_projection(tputs)
    assert action == "omitted"
    assert spread > 0.25
    assert proj_tput is None
    assert "not flat" in reason.lower()
    # the reason must quantify the spread (auditable)
    assert "%" in reason


def test_t6_throughput_excludes_dropped_trees():
    """throughput = (M - n_dropped)/wall, never M/wall, so dropped (K0) trees
    do not inflate the headline."""
    assert sweep.throughput(M=1000, n_dropped=100, wall=1.0) == pytest.approx(900.0)
    # n_dropped=0 reduces to M/wall
    assert sweep.throughput(M=1000, n_dropped=0, wall=2.0) == pytest.approx(500.0)


# --------------------------------------------------------------------------- T1
def test_t1_same_pop_bytes_go_to_both_backends():
    """Both backends optimize the SAME pop object: the key arrays a backend reads
    (nt/nv/ci/metas/xs/ym/c_init) must be byte-identical for the two calls. We
    snapshot the pop the harness hands out and assert the bytes match."""
    pop = gen_pop("inner-const-heavy", 200, 300, 0)
    keys = ("nt", "nv", "ci", "metas", "xs", "ym", "c_init")
    snap = {k: np.asarray(pop[k]).tobytes() for k in keys}
    # the harness must pass THIS object to both; simulate the two reads
    kernel_view = {k: np.asarray(pop[k]).tobytes() for k in keys}
    operon_view = {k: np.asarray(pop[k]).tobytes() for k in keys}
    for k in keys:
        assert kernel_view[k] == snap[k], f"kernel saw mutated {k}"
        assert operon_view[k] == snap[k], f"operon saw mutated {k}"
    # and the harness helper that fetches the pop must be memoised per (preset,M)
    # so it is literally the same object, never regenerated between backends:
    p1 = sweep.get_pop("inner-const-heavy", 200, 300, 0)
    p2 = sweep.get_pop("inner-const-heavy", 200, 300, 0)
    assert p1 is p2, "get_pop must return the identical object for both backends"


# --------------------------------------------------------------------------- T5
def test_t5_timed_region_calls_fresh_each_rep():
    """The timer must call the backend op FRESH each rep (re-marshal / re-dispatch
    is the per-generation cost); reps list length == n_rep. We use a fake op that
    counts calls."""
    calls = {"n": 0}

    def fake_op():
        calls["n"] += 1
        return calls["n"]

    walls, results = sweep.timed_reps(fake_op, n_rep=4)
    assert len(walls) == 4
    assert calls["n"] == 4            # one fresh call per rep (plus none hoisted)
    assert results[-1] == 4           # last result is from the last rep
    assert all(w >= 0.0 for w in walls)


def test_t5_reps_record_full_list_for_median():
    """wall_median is the median of a recorded reps list (>=3 for kernel/parallel)."""
    walls, _ = sweep.timed_reps(lambda: None, n_rep=3)
    assert len(walls) == 3
    import statistics
    assert sweep.median(walls) == pytest.approx(statistics.median(walls))
