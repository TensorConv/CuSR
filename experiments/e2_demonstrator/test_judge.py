"""TDD for 009's calibrated recovery judge (`judge.py`).

Policy (user-decided 2026-06-08, see judge_calibration.md):
  - PRIMARY = own uniform relative-tolerance judge: round each coefficient to
    `sig` significant figures (relative, no absolute floor, no 1e-4 zeroing),
    then structural equivalence. Fixes SRBench's small-constant pathologies
    (±10-25% slop below |c|=1; annihilation below ~5e-4).
  - Inner tolerance ~1e-3 relative (sig=3, ≈ SRBench's large-constant regime,
    applied uniformly across magnitudes).
  - Outer constants reported in TWO modes: strict (no outer tolerance) and
    lenient (outer add/mul tolerated, = competitor-comparable linear scaling).
  - Keep the numeric proxy (lenient fallback) for sympy-opaque rewrites.
  - `recovered_srbench` = SRBench-faithful column, for comparability.

Run:  uv run pytest experiments/009_sr_benchmark/test_judge.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from cusr.demonstrator import judge


# ---------------------------------------------------------------------------
# The headline fix: uniform relative tolerance across magnitudes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c", [0.002, 0.05, 0.5, 9.8, 1234.0])
def test_uniform_relative_tolerance_across_magnitudes(c):
    """Same expression, inner constant spanning 6 orders of magnitude: a near-
    exact match (rel 1e-4) recovers, a 5% error does not — *identically* at
    every magnitude. This is the property SRBench's ruler lacks (it goes
    ±10-25% permissive below |c|=1)."""
    true = f"sin({c}*x0)"
    near = f"sin({c * (1 + 1e-4)}*x0)"
    far = f"sin({c * 1.05}*x0)"
    assert judge.recovered(near, true) is True, f"c={c}: near-exact should recover"
    assert judge.recovered(far, true) is False, f"c={c}: 5% inner error should NOT recover"


def test_small_constant_not_annihilated():
    """A small inner constant (5e-5) off by 20% must be rejected. SRBench would
    zero both (|c|<1e-4) and call it recovered — our judge must not."""
    assert judge.recovered("sin(0.00005*x0)", "sin(0.00006*x0)") is False


def test_small_constant_near_match_recovers():
    assert judge.recovered("sin(0.00005*x0)", "sin(0.0000500001*x0)") is True


# ---------------------------------------------------------------------------
# strict vs lenient outer-constant columns
# ---------------------------------------------------------------------------

KORNS = "2.0 - 2.1*cos(9.8*x0)*sin(1.3*x1)"


def test_exact_recovers_in_both_modes():
    assert judge.recovered(KORNS, KORNS, outer="strict") is True
    assert judge.recovered(KORNS, KORNS, outer="lenient") is True


def test_outer_scale_lenient_only():
    assert judge.recovered(f"3.0*({KORNS})", KORNS, outer="lenient") is True
    assert judge.recovered(f"3.0*({KORNS})", KORNS, outer="strict") is False


def test_outer_offset_lenient_only():
    assert judge.recovered(f"({KORNS})+5.0", KORNS, outer="lenient") is True
    assert judge.recovered(f"({KORNS})+5.0", KORNS, outer="strict") is False


def test_combined_affine_rejected_both_modes():
    cand = f"3.0*({KORNS})+5.0"
    assert judge.recovered(cand, KORNS, outer="lenient") is False
    assert judge.recovered(cand, KORNS, outer="strict") is False


def test_wrong_inner_rejected_both_modes():
    cand = "2.0 - 2.1*cos(10.5*x0)*sin(1.3*x1)"
    assert judge.recovered(cand, KORNS, outer="lenient") is False
    assert judge.recovered(cand, KORNS, outer="strict") is False


# ---------------------------------------------------------------------------
# numeric proxy kept for sympy-opaque rewrites (the exp007 case)
# ---------------------------------------------------------------------------

@pytest.fixture
def X_3d():
    rng = np.random.default_rng(0)
    return rng.uniform(0.5, 2.5, size=(100, 3))


# sin(a+pi/2) = cos(a) rewrite with huge-coefficient algebra; sympy can't see it.
OPAQUE_HAT = ("4.8790879262543e-10*x0*"
              "(-4.09913e+9*sin(x1*x2 + 1.57079637050629) - 3.11678e+8)"
              " + 2.15207*x0")
OPAQUE_STAR = "2.0*x0*(1.0 - 1.0*cos(x1*x2))"


def test_numeric_proxy_catches_opaque_rewrite(X_3d):
    assert judge.recovered(OPAQUE_HAT, OPAQUE_STAR, outer="lenient", X=X_3d) is True


def test_opaque_rewrite_missed_without_proxy(X_3d):
    """Structural-only (no X / proxy off) misses it — documents why we keep the proxy."""
    assert judge.recovered(OPAQUE_HAT, OPAQUE_STAR, outer="lenient", X=None) is False
    assert judge.recovered(OPAQUE_HAT, OPAQUE_STAR, outer="lenient", X=X_3d,
                           use_numeric_proxy=False) is False


# ---------------------------------------------------------------------------
# SRBench comparability column reproduces SRBench's (known-flawed) behavior
# ---------------------------------------------------------------------------

def test_srbench_column_annihilates_small_constant():
    """Documents the comparability column's known blind spot: SRBench zeroes
    constants <1e-4, so two different small-frequency sines 'match'. Our primary
    judge rejects the same pair (test_small_constant_not_annihilated)."""
    assert judge.recovered_srbench("sin(0.00005*x0)", "sin(0.00006*x0)") is True


def test_srbench_column_recovers_exact_and_outer():
    assert judge.recovered_srbench(KORNS, KORNS) is True
    assert judge.recovered_srbench(f"({KORNS})+5.0", KORNS) is True           # outer additive
    assert judge.recovered_srbench(f"3.0*({KORNS})", KORNS) is True           # outer multiplicative


def test_srbench_column_rejects_affine_and_wrong_inner():
    assert judge.recovered_srbench(f"3.0*({KORNS})+5.0", KORNS) is False
    assert judge.recovered_srbench("2.0 - 2.1*cos(10.5*x0)*sin(1.3*x1)", KORNS) is False


def test_recovery_columns_shape(X_3d):
    cols = judge.recovery_columns(f"3.0*({KORNS})", KORNS)
    assert set(cols) == {"strict", "lenient", "srbench"}
    assert cols["strict"] is False and cols["lenient"] is True
