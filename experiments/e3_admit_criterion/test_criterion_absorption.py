"""TDD — linear-span absorption (adversarial finding: trig-phase false ADMIT).

A constant is "absorbable" not only when it factors out as a multiplicative scale
(sqrt(c·g)=sqrt(c)·sqrt(g), korns_8) but also when a grammar identity puts the
function into a FIXED finite linear span whose coefficients carry the constant:
  * trig phase:  sin(x+c) = cos(c)·sin(x) + sin(c)·cos(x)   -> span{sin x, cos x}
  * poly shift:  (x-c)²   = x² - 2c·x + c²                  -> span{1, x, x²}
Such constants are reachable by GP + OUTER linear coefficients with NO nonlinear
inner CO, so they must be REJECTED. The criterion's basis/fold pipeline must
expand these identities. Genuine FREQUENCIES (c·x inside trig) have no additive
angle and must be preserved as inner.

The trig-phase case (found by the adversarial red-team) was a real false ADMIT
before `_fold` added sp.expand_trig.

Run: .venv/bin/python -m pytest experiments/e3_admit_criterion/test_criterion_absorption.py -q
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from experiments.e3_admit_criterion import criterion as C
from experiments.e3_admit_criterion import skel_problems as SP

x0, x1, c0, c1, c2, c3 = sp.symbols("x0 x1 c0 c1 c2 c3")


def _prob(id_, skel, consts, gt, dom, variables=(x0,)):
    return SP.SkelProblem(id_, "test", skel, variables, consts,
                          np.array(gt, dtype=float), dom)


# ── trig PHASE is absorbable -> REJECT (the red-team break) ──────────────────
def test_trig_phase_sin_is_rejected():
    p = _prob("phase_sin", c0 * sp.sin(x0 + c1), (c0, c1), [1.7, 0.8], ((-3.0, 3.0),))
    v = C.decide(p, use_structural=True)
    assert v.verdict == "REJECT", v.reason
    # absorbed via angle-addition: the phase drops from the inner count -> fold
    assert v.reason_code in ("fold", "ls_reject"), v.reason


def test_trig_phase_cos_is_rejected():
    p = _prob("phase_cos", c0 * sp.cos(x0 + c1), (c0, c1), [1.3, 1.9], ((-4.0, 4.0),))
    assert C.decide(p, use_structural=True).verdict == "REJECT"


def test_trig_phase_proof_linear_in_fixed_basis():
    """Sanity: the data really IS perfectly fit by OLS on the fixed, inner-free
    basis {1, sin x0, cos x0} — proving no inner CO is needed."""
    p = _prob("phase_sin2", c0 * sp.sin(x0 + c1), (c0, c1), [1.7, 0.8], ((-3.0, 3.0),))
    X, y = SP.materialize(p)
    A = np.column_stack([np.ones(len(X)), np.sin(X[:, 0]), np.cos(X[:, 0])])
    th, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.sum((y - A @ th) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.9999, r2


# ── polynomial SHIFT is absorbable -> REJECT (same class via sp.expand) ──────
def test_quadratic_shift_is_rejected():
    """(x0-c1)² ∈ span{1,x0,x0²}; the vertex shift is recoverable from polynomial
    coefficients without inner CO. This is a frozen construction-grid candidate
    that the criterion HONESTLY rejects (a self-negative-control)."""
    p = _prob("shift_quad", c0 * (x0 - c1) ** 2, (c0, c1), [1.7, 2.7], ((-5.0, 5.0),))
    v = C.decide(p, use_structural=True)
    assert v.verdict == "REJECT", v.reason


# ── genuine FREQUENCY constants survive expand_trig -> still ADMIT ───────────
def test_frequency_not_broken_by_trig_expansion():
    K = SP.korns_problems()
    for kid in ("korns_11", "korns_12"):  # cos(c·x³), cos(c·x)·sin(c·x)
        v = C.decide(K[kid], use_structural=True)
        assert v.verdict == "ADMIT", (kid, v.reason)


def test_multivar_frequency_still_admits():
    """A genuine 2-freq product cos(c0·x0)·sin(c1·x1) at non-canonical freqs must
    still ADMIT (expand_trig does not flatten products of distinct-arg trig)."""
    p = _prob("twofreq", sp.cos(c0 * x0) * sp.sin(c1 * x1), (c0, c1),
              [3.3, 4.7], ((-2.0, 2.0), (-2.0, 2.0)), variables=(x0, x1))
    v = C.decide(p, use_structural=True)
    assert v.verdict == "ADMIT", v.reason
