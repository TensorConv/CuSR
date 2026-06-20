"""TDD — STRUCTURAL-discrete mechanism (Phase C, frozen CANON).

The structural test rejects problems whose surviving-inner constants are
GP-grammar-reachable canonical values (±1 from a subtraction node, π a grammar
constant). It is justified by reachability ALONE and CANON is frozen before the
Feynman-34 audit — never reverse-engineered from anchor counts.

The single most dangerous failure mode is OVER-rejection: a genuine inner
constant (korns_7's decay 0.547) must NOT snap to the nearby canonical 0.5 and
get wrongly rejected. That guard is the headline test here.

Run: .venv/bin/python -m pytest experiments/e3_admit_criterion/test_criterion_structural.py -q
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from experiments.e3_admit_criterion import criterion as C
from experiments.e3_admit_criterion import prereg as P
from experiments.e3_admit_criterion import skel_problems as SP

KORNS = SP.korns_problems()


# ── structural REJECTS grammar-reachable canonical constants ────────────────
def test_structural_rejects_I8_14_structural_minus1():
    """feynman I.8.14 = sqrt((c0·x0+x1)²+(c1·x2+x3)²), gt c0=c1=-1 — the Scheme-C
    literal→constant artifact: (-x0+x1) is a subtraction, reachable for free."""
    v = C.decide(SP.feynman_problem("I.8.14"), use_structural=True)
    assert v.verdict == "REJECT"
    assert v.reason_code == "structural", v.reason
    assert v.structural is True
    canon = v.details["structural"]["canon"]
    assert canon["c0"] == -1.0 and canon["c1"] == -1.0


def test_structural_rejects_nguyen5_unit_consts():
    """nguyen_5 inner c1=c2=1.0 (unit frequencies) — canonical → reject."""
    v = C.decide(SP.nguyen_problem("5"), use_structural=True)
    assert v.verdict == "REJECT" and v.reason_code == "structural"


def test_structural_rejects_pi_scale_constant():
    """π-scale demonstration: an inner frequency of exactly 2π is grammar-reachable
    (π is a terminal), so cos(2π·x) is structural even though 2π ≈ 6.28 'looks'
    like a genuine frequency. This is the π-scale tightening the task asks for."""
    x0, c0, c1 = sp.symbols("x0 c0 c1")
    prob = SP.SkelProblem(
        id="synthetic_pi_scale", source="test",
        skeleton=c0 * sp.cos(c1 * x0), variables=(x0,), constants=(c0, c1),
        gt=np.array([1.3, 2.0 * np.pi]), domain=((-1.0, 1.0),),
    )
    v = C.decide(prob, use_structural=True)
    assert v.verdict == "REJECT" and v.reason_code == "structural", v.reason
    assert abs(v.details["structural"]["canon"]["c1"] - 2 * np.pi) < 1e-9


# ── structural does NOT touch genuine inner constants ───────────────────────
def test_structural_does_not_reject_genuine_inner():
    for kid in ("korns_7", "korns_11", "korns_12"):
        v = C.decide(KORNS[kid], use_structural=True)
        assert v.verdict == "ADMIT", (kid, v.reason)
        assert v.structural is False


def test_korns7_decay_does_NOT_snap_to_canonical_half():
    """HEADLINE GUARD: korns_7's decay 0.547 is 8.6% from the canonical 0.5 — far
    outside STRUCT_TOL — so it must NOT be snapped to canonical and wrongly
    rejected. If this fails, the canonical set / STRUCT_TOL is too aggressive."""
    p = KORNS["korns_7"]
    X, y = SP.materialize(p)
    surviving = C.inner_set(p.skeleton, p.constants)  # {c1}
    struct, det = C.is_structural(p.skeleton, p.variables, surviving,
                                  p.constants, p.gt, X, y)
    assert struct is False, det
    # the true decay is genuinely far from every canonical value
    rel = min(abs(z - (-0.54723748542)) / 0.54723748542 for z in P.CANON_SET)
    assert rel > P.STRUCT_TOL, rel


def test_korns11_freq_not_canonical():
    p = KORNS["korns_11"]
    rel = min(abs(z - 7.23) / 7.23 for z in P.CANON_SET)
    assert rel > P.STRUCT_TOL, ("7.23 must not be canonical", rel)


# ── CANON is frozen & grammar-justified (not reverse-engineered) ────────────
def test_canon_set_frozen_shape():
    """Pin CANON: small integers/half + π-multiples only. A change to this set
    (e.g. adding 0.547 to force korns_7 out) would change this test, making any
    tampering visible in the diff."""
    assert len(P.CANON_SET) == 17
    for v in (0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5):
        assert v in P.CANON_SET
    assert any(abs(z - np.pi) < 1e-9 for z in P.CANON_SET)
    assert any(abs(z - 2 * np.pi) < 1e-9 for z in P.CANON_SET)
    # no "suspiciously specific" non-canonical decimals smuggled in
    suspicious = [z for z in P.CANON_SET
                  if not (abs(z) < 1e-9
                          or any(abs(z - q) < 1e-9 for q in (1, -1, 2, -2, 0.5, -0.5))
                          or any(abs(z - k * np.pi) < 1e-9 for k in (1, -1, 2, -2, 0.5, -0.5))
                          or any(abs(z - 1 / (k * np.pi)) < 1e-9 for k in (1, -1, 2, -2)))]
    assert suspicious == [], suspicious


def test_structural_requires_both_canonical_and_fit():
    """A constant near-canonical in VALUE but whose canonical fit does NOT reach
    τ_high must NOT be called structural (both conditions are required). We force
    this by a skeleton where snapping the inner const to its (canonical) GT still
    leaves a poor outer-affine fit is impossible on clean data; instead assert the
    logical contract directly: is_structural returns False when value is non-canonical."""
    # value-side gate: a clearly non-canonical inner value -> not structural
    x0, c0, c1 = sp.symbols("x0 c0 c1")
    prob = SP.SkelProblem("synthetic_noncanon", "test", c0 * sp.cos(c1 * x0),
                          (x0,), (c0, c1), np.array([1.0, 3.7]), ((-2.0, 2.0),))
    X, y = SP.materialize(prob)
    struct, det = C.is_structural(prob.skeleton, prob.variables, {c1},
                                  prob.constants, prob.gt, X, y)
    assert struct is False, det
