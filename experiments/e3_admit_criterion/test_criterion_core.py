"""TDD — CORE admit criterion (mechanisms 1 fold + 1b identifiability + 3 LS-gap).

Written BEFORE criterion.py exists (RED), then driven to GREEN. The CORE is run
with use_structural=False; the structural mechanism (Phase C) has its own suite.

Hard gates the CORE alone must satisfy (task §1 validation set, minus EDGE rows):
  ADMIT : korns_7, korns_11, korns_12   (genuine inner constants)
  REJECT: korns_8 (absorbable / fold)   — anti-cheat
          nguyen_1 (constant-free poly)  — no inner constant
          outer-only Feynman I.11.19, I.12.1 — no inner constant

Run: .venv/bin/python -m pytest experiments/e3_admit_criterion/test_criterion_core.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.e3_admit_criterion import criterion as C
from experiments.e3_admit_criterion import prereg as P
from experiments.e3_admit_criterion import skel_problems as SP

KORNS = SP.korns_problems()


def _core(prob):
    return C.decide(prob, use_structural=False)


# ── ADMIT: genuine inner constants ──────────────────────────────────────────
@pytest.mark.parametrize("kid", ["korns_7", "korns_11", "korns_12"])
def test_core_admits_genuine_inner(kid):
    v = _core(KORNS[kid])
    assert v.verdict == "ADMIT", (kid, v.reason)
    assert v.R2_LS < P.TAU_LOW, (kid, "R2_LS", v.R2_LS)
    assert v.R2_full > P.TAU_HIGH, (kid, "R2_full", v.R2_full)
    assert v.recovery_err < P.RECOVERY_TOL, (kid, "rec", v.recovery_err)


def test_korns11_needs_seeding_not_blind():
    """The seeding policy is load-bearing: charitable blind random-start full-CO
    FAILS on korns_11 (~140 periods on [-5,5]), so admissibility must rest on the
    GT-seeded certification. R2_full_random is the MEDIAN over FULLCO_BLIND_SEEDS,
    so this is robust to a stray basin-hit seed (measured: 0/12 seeds solved)."""
    v = _core(KORNS["korns_11"])
    assert v.R2_full > P.TAU_HIGH                      # seeded certification succeeds
    assert v.R2_full_random < 0.9, v.R2_full_random    # blind genuinely fails (median)
    assert "certification" in v.seeding.lower()


# ── REJECT: anti-cheat / negative controls ──────────────────────────────────
def test_core_rejects_korns8_absorbable():
    v = _core(KORNS["korns_8"])
    assert v.verdict == "REJECT", v.reason
    assert v.reason_code in ("fold", "rank"), v.reason
    assert v.inner == (), "absorbable const must not survive folding"
    assert "c2" in v.absorbable


def test_core_rejects_nguyen1_no_inner():
    v = _core(SP.nguyen_problem("1"))
    assert v.verdict == "REJECT"
    assert v.reason_code == "no_inner", v.reason


@pytest.mark.parametrize("name", ["I.11.19", "I.12.1"])
def test_core_rejects_outer_only_feynman(name):
    v = _core(SP.feynman_problem(name))
    assert v.verdict == "REJECT"
    assert v.reason_code == "no_inner", v.reason


# ── consistency of the inner-set helper with the production positional counter ─
def test_inner_set_matches_production_counter():
    from cusr.demonstrator.taxonomy import count_inner_consts
    probs = list(KORNS.values()) + [SP.feynman_problem("I.8.14"),
                                    SP.nguyen_problem("5"), SP.nguyen_problem("6")]
    for p in probs:
        got = len(C.inner_set(p.skeleton, p.constants))
        want = count_inner_consts(p.skeleton, p.constants)
        assert got == want, (p.id, got, want)


# ── anti-cheat demo: outer absorption of a WRONG inner const must not admit ──
def test_korns8_fake_outer_absorption_is_rejected():
    """Demonstrate the gaming path: fix korns_8's inner c2 at a deliberately WRONG
    value (1.0 vs true 7.23) — the outer affine absorbs it to R²≈1 — yet the
    criterion still REJECTS (fold fires before any fit can be gamed)."""
    p = KORNS["korns_8"]
    X, y = SP.materialize(p)
    c2 = p.constants[2]
    r_wrong = C.r2_ls_at(p.skeleton, p.variables, (c2,), (1.0,), p.constants, X, y)
    assert r_wrong > 0.999, f"outer absorption should fake R²≈1, got {r_wrong}"
    # ...but the const is NOT recovered (1.0 != 7.23), so it is a fake solution:
    assert abs(1.0 - 7.23) / 7.23 > P.RECOVERY_TOL
    assert _core(p).verdict == "REJECT"


# ── identifiability: korns_8 columns are rank-deficient at the true constants ─
def test_korns8_jacobian_rank_deficient():
    p = KORNS["korns_8"]
    X, _ = SP.materialize(p)
    rank, n = C.jac_rank(p.skeleton, p.constants, p.gt, p.variables, X)
    assert rank < n, (rank, n)  # c1*sqrt(c2): d/dc2 ∝ d/dc1


@pytest.mark.parametrize("kid", ["korns_7", "korns_11", "korns_12"])
def test_genuine_inner_jacobian_full_rank(kid):
    p = KORNS[kid]
    X, _ = SP.materialize(p)
    rank, n = C.jac_rank(p.skeleton, p.constants, p.gt, p.variables, X)
    assert rank == n, (kid, rank, n)


# ── τ_low sensitivity: hard-anchor verdicts invariant across the frozen sweep ─
def test_tau_low_sensitivity_core():
    expect = {"korns_7": "ADMIT", "korns_11": "ADMIT", "korns_12": "ADMIT",
              "korns_8": "REJECT"}
    for kid, exp in expect.items():
        v = _core(KORNS[kid])
        for tau in P.TAU_LOW_SWEEP:
            if v.reason_code in ("no_inner", "fold", "rank"):
                got = "REJECT"  # threshold-independent rejections
            else:
                got = ("ADMIT" if (v.R2_LS < tau and v.R2_full > P.TAU_HIGH
                                   and v.recovery_err < P.RECOVERY_TOL) else "REJECT")
            assert got == exp, (kid, "tau_low", tau, got, exp, "R2_LS", v.R2_LS)
