"""TDD test suite for `check_recovery` + `check_recovery_numeric` +
`check_recovery_composite` in `bench.sources.evogp`.

SRBench (NeurIPS 2021) recovery definition: f̂ recovers f* iff after sympy
simplification, either `f̂ = a · f*` (scaling-equivalent, a != 0) OR
`f̂ = f* + b` (offset-equivalent). Both holding (i.e. `f̂ = a · f* + b` with
both a≠1 AND b≠0) is NOT recovered — that's an affine transform, not a
structural match.

Composite (B) extends this with a numeric proxy fallback for cases where
sympy can't see the equivalence (huge coefficient algebra, transcendental
rewrites like sin(α+π/2)→cos(α)). Numeric form:
  - sample y_hat, y_star on a dense X grid via lambdify
  - test 1-param fit `a` (scaling-only, b≡0) OR 1-param fit `b` (offset-only, a≡1)
  - pass if either residual < tol_rel · max|y_hat|
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import sympy as sp

from cusr.bench.sources.evogp import check_recovery


# ---------------------------------------------------------------------------
# Positive cases — should return True
# ---------------------------------------------------------------------------


def test_recovery_identical():
    """f̂ ≡ f*: trivial, both sympy.Expr inputs."""
    x = sp.Symbol("x")
    assert check_recovery(x**2, x**2) is True


def test_recovery_scaling_equivalent():
    """f̂ = a · f* (scaling): 2·x² vs x², ratio = 2 (constant)."""
    x = sp.Symbol("x")
    assert check_recovery(2 * x**2, x**2) is True


def test_recovery_offset_equivalent():
    """f̂ = f* + b (offset): x² + 3 vs x², diff = 3 (constant)."""
    x = sp.Symbol("x")
    assert check_recovery(x**2 + 3, x**2) is True


def test_recovery_str_inputs():
    """Strings should also work — Dataset.ground_truth is str."""
    # I.14.4 case: GT = '0.5*k*x**2'; perfectly-fit hypothesis identical.
    assert check_recovery("0.5*k*x**2", "0.5*k*x**2") is True
    # Same structure, scaling: 2× the GT
    assert check_recovery("k*x**2", "0.5*k*x**2") is True


# ---------------------------------------------------------------------------
# Negative cases — should return False
# ---------------------------------------------------------------------------


def test_recovery_different_structure():
    """exp(x) vs x²: structurally different, neither ratio nor diff constant."""
    x = sp.Symbol("x")
    assert check_recovery(sp.exp(x), x**2) is False


def test_recovery_combined_scale_and_offset_rejected():
    """f̂ = a·f* + b with BOTH a≠1 and b≠0 is NOT recovered.
    SRBench definition is strictly OR — a structural match allows ONE degree
    of freedom (scale OR offset), not both. 2·x² + 3 vs x² fails on both
    individual tests: ratio = (2x²+3)/x² is not constant; diff = x² + 3 is
    not constant either.
    """
    x = sp.Symbol("x")
    assert check_recovery(2 * x**2 + 3, x**2) is False


def test_recovery_zero_ratio_rejected():
    """f̂ = 0, f* = x: ratio = 0/x = 0 is technically a constant, but this
    is a degenerate solution (a=0 means f̂ doesn't depend on f* at all)
    and must NOT count as recovered. diff = -x is not constant either.
    """
    x = sp.Symbol("x")
    assert check_recovery(sp.Integer(0), x) is False


# ---------------------------------------------------------------------------
# Robustness — failures must not raise
# ---------------------------------------------------------------------------


def test_recovery_with_typed_symbols():
    """Symbols carrying assumptions (e.g. real=True, as Tree.to_sympy_expr emits)
    must still recover against plain-sympify Symbols of same name.

    Without normalization sympy treats `Symbol('x', real=True)` and `Symbol('x')`
    as distinct objects, so `Symbol('x', real=True)*x1**2 / (x*x1**2)` does NOT
    simplify to 1. This is the regression caught by the first 007 sweep
    (2026-05-09): all 40 runs reported recovered=False even though the GP best
    tree was sympy-equal to ground truth.
    """
    x0_typed = sp.Symbol("x0", real=True, finite=True)
    x1_typed = sp.Symbol("x1", real=True, finite=True)
    e_hat = 0.5 * x0_typed * x1_typed**2  # mimics Tree.to_sympy_expr()
    e_star = "0.5*x0*x1**2"               # mimics dataset.ground_truth (str)
    assert check_recovery(e_hat, e_star) is True


def test_recovery_with_numeric_tail():
    """NLS optimization leaves tiny floating-point residual coefficients
    (e.g. `5.30e-9 * x1`) on the otherwise-correct expression. These are
    numerically negligible (well below early-stop r²>0.9999 fitness floor)
    but sympy.simplify won't drop them — `0.5*x0*x1**2 + 5.30e-9*x1` minus
    `0.5*x0*x1**2` simplifies to `5.30e-9*x1`, which is_constant() == False.

    check_recovery must fold sub-tolerance numeric tails to zero so the
    "essentially recovered" case counts as recovered. Tolerance picked to
    align with early_stop_r2=0.9999 fitness floor (~1e-6 relative).

    Three flavors of tail seen in 007 sweep 2026-05-09:
      - additive 1e-9 tail
      - multiplicative 1e-16 tail
      - mixed scale+tail
    """
    x0, x1 = sp.symbols("x0 x1")
    target = "0.5*x0*x1**2"

    # Flavor 1: additive tail (seen on memetic seed=49)
    e1 = 0.5 * x0 * x1**2 + 5.30407e-9 * x1
    assert check_recovery(e1, target) is True, "additive 1e-9 tail not absorbed"

    # Flavor 2: multiplicative inside (seen on seed=42)
    # 0.5*x1*(x0 + 1.89e-16/x1)*(x1 - 6.03e-16) ≈ 0.5*x0*x1**2 + tiny
    e2 = 0.5 * x1 * (x0 + 1.89e-16 / x1) * (x1 - 6.03e-16)
    assert check_recovery(e2, target) is True, "multiplicative 1e-16 tail not absorbed"

    # Flavor 3: tail multiplying the whole thing (seen on seed=51)
    e3 = x0 * x1 * (0.5 * x1 + 1.21e-17)
    assert check_recovery(e3, target) is True, "trailing 1e-17 tail not absorbed"


def test_recovery_does_not_absorb_meaningful_difference():
    """nsimplify must NOT collapse a meaningful (above-tolerance) difference
    to zero. `2*x**2 + 3` vs `x**2` is still NOT recovered (combined
    scale+offset rejection from earlier test), even with nsimplify.
    """
    x = sp.Symbol("x")
    assert check_recovery(2 * x**2 + 3, x**2) is False
    # And a meaningful additive term that's NOT below tolerance must reject:
    # 0.5*x0*x1**2 + 0.1*x1 — 0.1 is way above any reasonable tolerance.
    x0, x1 = sp.symbols("x0 x1")
    assert check_recovery(0.5 * x0 * x1**2 + 0.1 * x1, "0.5*x0*x1**2") is False


def test_recovery_handles_simplify_exception():
    """If sympy.simplify raises (e.g. timeout, internal error), check_recovery
    should return False rather than propagating. We don't want a sympy
    edge case to crash a 50-gen experiment run during the recovery sweep.
    """
    x = sp.Symbol("x")
    with mock.patch("cusr.bench.sources.evogp.sp.simplify",
                    side_effect=RuntimeError("simulated simplify failure")):
        assert check_recovery(x**2, x**2) is False


# ---------------------------------------------------------------------------
# Numeric proxy (B) — `check_recovery_numeric`
# ---------------------------------------------------------------------------


@pytest.fixture
def X_2d():
    """A 2D dense grid in (x0, x1). 100 random points in a non-pathological
    domain. Used for numeric proxy tests where lambdify needs an X."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.5, 2.5, size=(100, 2))


@pytest.fixture
def X_3d():
    rng = np.random.default_rng(0)
    return rng.uniform(0.5, 2.5, size=(100, 3))


def test_numeric_recovery_identical(X_2d):
    """f̂ ≡ f*: trivial. Numeric residual after fit is 0."""
    from cusr.bench.sources.evogp import check_recovery_numeric
    assert check_recovery_numeric("0.5*x0*x1**2", "0.5*x0*x1**2", X_2d) is True


def test_numeric_recovery_scaling(X_2d):
    """f̂ = 2·f*: scaling-only. 1-param fit a finds a=2, residual ~0."""
    from cusr.bench.sources.evogp import check_recovery_numeric
    assert check_recovery_numeric("x0*x1**2", "0.5*x0*x1**2", X_2d) is True


def test_numeric_recovery_offset(X_2d):
    """f̂ = f* + 3.7: offset-only. 1-param fit b finds b=3.7, residual ~0."""
    from cusr.bench.sources.evogp import check_recovery_numeric
    assert check_recovery_numeric("0.5*x0*x1**2 + 3.7", "0.5*x0*x1**2", X_2d) is True


def test_numeric_recovery_combined_affine_rejected(X_2d):
    """f̂ = 2·f* + 3 has a=2 AND b=3 — must NOT recover (SRBench OR not AND).
    Pure scaling fit: residual is std of (2f*+3) − a·f* = const+ε for any a → fails
    Pure offset fit: residual is std of (2f*+3) − f* − b = f*+3-b which scales w f* → fails
    """
    from cusr.bench.sources.evogp import check_recovery_numeric
    assert check_recovery_numeric("x0*x1**2 + 3", "0.5*x0*x1**2", X_2d) is False


def test_numeric_recovery_different_structure(X_2d):
    """exp(x0)+x1 vs 0.5·x0·x1²: structurally different, neither fit close."""
    from cusr.bench.sources.evogp import check_recovery_numeric
    assert check_recovery_numeric("exp(x0)+x1", "0.5*x0*x1**2", X_2d) is False


def test_numeric_recovery_huge_coefficient_algebra(X_3d):
    """The sympy-opaque case from gate (III.15.12 seed=42): expanded form
    with extreme coefficients that's algebraically `2·x0·(1−cos(x1·x2))`
    via `sin(α+π/2) = cos(α)`. Sympy can't simplify the coefficient
    explosion; numeric form catches it via pure-offset fit (b≈0).

    Full best_expr from `runs/panel_20260509T162646Z/feynman_III_15_12/
    results.jsonl` (seed=42, memetic, pop=1000):
        4.879e-10*x0*(-4.099e+9*sin(x1*x2 + π/2) - 3.117e+8) + 2.152*x0
    Algebraically:
        ≈ -2·x0·cos(x1*x2) - 0.152·x0 + 2.152·x0
        = -2·x0·cos(x1*x2) + 2.0·x0
        = 2·x0·(1 - cos(x1*x2))   ✓
    """
    from cusr.bench.sources.evogp import check_recovery_numeric
    f_hat = ("4.8790879262543e-10*x0*"
             "(-4.09913e+9*sin(x1*x2 + 1.57079637050629) - 3.11678e+8)"
             " + 2.15207*x0")
    f_star = "2.0*x0*(1.0 - 1.0*cos(x1*x2))"
    assert check_recovery_numeric(f_hat, f_star, X_3d) is True


def test_numeric_recovery_offset_form_from_gate(X_3d):
    """III.17.37 seed=49 case: `(x0*x1 - (0.0403 - x0)/cos(x2))*cos(x2)`
    expands to `x0*(x1*cos(x2)+1) − 0.0403`. Pure offset (b=−0.0403).
    """
    from cusr.bench.sources.evogp import check_recovery_numeric
    f_hat = "(x0*x1 - (0.0403023 - 1.0*x0)/cos(1.0*x2))*cos(x2)"
    f_star = "x0*(x1*cos(x2) + 1.0)"
    assert check_recovery_numeric(f_hat, f_star, X_3d) is True


def test_numeric_recovery_a_near_zero_rejected(X_2d):
    """f̂ ≈ 0 for varying f*: a=0 is degenerate (f̂ doesn't depend on f*).
    Pure scaling fit yields a≈0 (good residual!) but should reject.
    Pure offset fit: b = mean(0 - f*) = -mean(f*); residual = std(f*) ≠ 0.
    """
    from cusr.bench.sources.evogp import check_recovery_numeric
    # Use 1e-20 scale rather than literal 0 to avoid lambdify edge cases.
    assert check_recovery_numeric("1e-20", "0.5*x0*x1**2", X_2d) is False


def test_numeric_recovery_non_finite_rejected(X_2d):
    """y_hat with NaN/inf must return False (don't crash on
    log(neg) or 1/0 in user-supplied X domain)."""
    from cusr.bench.sources.evogp import check_recovery_numeric
    # log of x0 in [0.5, 2.5] is fine; but 1/(x0 - x0) is 0/0 = nan.
    assert check_recovery_numeric("1/(x0 - x0)", "0.5*x0*x1**2", X_2d) is False


def test_numeric_recovery_sympify_error_returns_false(X_2d):
    """Bad expression string should yield False, not raise."""
    from cusr.bench.sources.evogp import check_recovery_numeric
    assert check_recovery_numeric("***bad***", "x0*x1**2", X_2d) is False


# ---------------------------------------------------------------------------
# Composite — `check_recovery_composite` (sympy ∨ numeric)
# ---------------------------------------------------------------------------


def test_composite_sympy_passes_numeric_unused(X_2d):
    """If sympy already says yes, composite says yes — even if X is empty
    (composite shortcut on sympy-pass path)."""
    from cusr.bench.sources.evogp import check_recovery_composite
    assert check_recovery_composite("0.5*x0*x1**2", "0.5*x0*x1**2", X_2d) is True


def test_composite_sympy_misses_numeric_catches(X_3d):
    """Gate-found case: sympy couldn't see the rewrite, numeric catches it.
    This is the headline value of B."""
    from cusr.bench.sources.evogp import check_recovery_composite
    f_hat = ("4.8790879262543e-10*x0*"
             "(-4.09913e+9*sin(x1*x2 + 1.57079637050629) - 3.11678e+8)"
             " + 2.15207*x0")
    f_star = "2.0*x0*(1.0 - 1.0*cos(x1*x2))"
    # Sanity: sympy alone should miss
    assert check_recovery(f_hat, f_star) is False
    # Composite catches
    assert check_recovery_composite(f_hat, f_star, X_3d) is True


def test_composite_both_fail_returns_false(X_2d):
    """Structural mismatch: neither sympy nor numeric pass."""
    from cusr.bench.sources.evogp import check_recovery_composite
    assert check_recovery_composite("exp(x0)+x1", "0.5*x0*x1**2", X_2d) is False


def test_composite_combined_affine_rejected(X_2d):
    """SRBench OR-not-AND must hold for composite too."""
    from cusr.bench.sources.evogp import check_recovery_composite
    assert check_recovery_composite("x0*x1**2 + 3", "0.5*x0*x1**2", X_2d) is False
