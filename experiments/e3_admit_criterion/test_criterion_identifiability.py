"""TDD — identifiability must target INNER absorbability, not OUTER redundancy.

The first version of the criterion rejected whenever rank(J@gt) < n_consts. That
over-rejects: a Scheme-C-split outer scale (c0·sqrt(c1)/sqrt(c2) — three constants
for ONE effective scale) is rank-deficient yet harmless to linear scaling, and the
problem may still carry a GENUINE inner constant (a Gaussian width). The correct
test is whether the INNER constants add an identifiable direction BEYOND the outer
constants; only if they add none (the korns_8 / additive-in-exp pattern) is the
problem non-inner.

Run: .venv/bin/python -m pytest experiments/e3_admit_criterion/test_criterion_identifiability.py -q
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from experiments.e3_admit_criterion import criterion as C
from experiments.e3_admit_criterion import skel_problems as SP

x0, c0, c1, c2 = sp.symbols("x0 c0 c1 c2")


def test_admits_genuine_inner_despite_outer_redundancy():
    """Gaussian c0·c1·exp(c2·x0²): c0,c1 are a redundant OUTER scale (rank-deficient
    pair) but c2 is a GENUINE inner width at a NON-canonical value. Must ADMIT —
    the outer redundancy must not veto the genuine inner constant."""
    prob = SP.SkelProblem("synthetic_gauss_redundant_outer", "test",
                          c0 * c1 * sp.exp(c2 * x0**2), (x0,), (c0, c1, c2),
                          np.array([2.0, 1.5, -0.37]), ((-2.0, 2.0),))
    v = C.decide(prob, use_structural=True)
    assert v.verdict == "ADMIT", (v.reason_code, v.reason)
    assert v.reason_code == "admit"


def test_inner_adds_rank_true_for_genuine_inner():
    prob = SP.SkelProblem("g", "test", c0 * c1 * sp.exp(c2 * x0**2), (x0,),
                          (c0, c1, c2), np.array([2.0, 1.5, -0.37]), ((-2.0, 2.0),))
    X, _ = SP.materialize(prob)
    adds = C.inner_adds_rank(prob.skeleton, prob.constants, prob.gt, prob.variables, X)
    assert adds is True


def test_identifiability_backstop_catches_fold_missed_absorbable():
    """c0 + c1·exp(c2 + x0): c2 is additive-inside-exp, so exp(c2+x0)=e^{c2}·e^{x0}
    — absorbable, but reveal_folds does NOT split exp-of-sum, so the FOLD misses it.
    The identifiability backstop must catch it: d/dc2 ∝ d/dc1 ⇒ inner adds no rank."""
    prob = SP.SkelProblem("synthetic_additive_exp", "test",
                          c0 + c1 * sp.exp(c2 + x0), (x0,), (c0, c1, c2),
                          np.array([1.0, 2.0, 0.7]), ((-1.0, 1.0),))
    X, _ = SP.materialize(prob)
    adds = C.inner_adds_rank(prob.skeleton, prob.constants, prob.gt, prob.variables, X)
    assert adds is False, "additive-in-exp const adds no identifiable inner DOF"
    v = C.decide(prob, use_structural=True)
    assert v.verdict == "REJECT" and v.reason_code in ("rank", "fold"), v.reason


def test_feynman_I6_2a_rejects_as_structural_not_rank():
    """I.6.2a's inner width c3=-0.5 IS identifiable (a real Gaussian width), so it
    must NOT be rejected for 'rank'; -0.5 is canonical so the honest reason is
    'structural'."""
    v = C.decide(SP.feynman_problem("I.6.2a"), use_structural=True)
    assert v.verdict == "REJECT"
    assert v.reason_code == "structural", (v.reason_code, v.reason)


def test_genuine_inner_anchors_still_add_rank():
    K = SP.korns_problems()
    for kid in ("korns_7", "korns_11", "korns_12"):
        p = K[kid]
        X, _ = SP.materialize(p)
        assert C.inner_adds_rank(p.skeleton, p.constants, p.gt, p.variables, X) is True, kid
