"""TDD for `taxonomy.count_inner_consts` — the n_inner_consts axis.

Definition (POSITIONAL, an *upper bound* on nonlinear-fittable constants):
a constant symbol is "inner" iff it sits in a position linear least-squares
cannot fit *and* is entangled with a variable there — i.e. inside a
transcendental function's argument, an exponent, or a Pow base / denominator
that also contains a variable. "Outer" = root additive constants + pure linear
coefficients of variable monomials (c*x0, c*x0**2), and pure constant
arithmetic (c0/c1).

Positional, not absorbability: `exp(c0 + c1*x0)` counts c0 as inner even though
exp(c0) is secretly an outer multiplicative scale (LSQ-fittable). Computing true
absorbability is hard; positional is a clean upper bound. The seam is pinned by
test_seam_nested_transcendental_counts_positionally below.

Run:  uv run pytest experiments/009_sr_benchmark/test_taxonomy.py -v
"""
from __future__ import annotations

import sympy as sp

from cusr.demonstrator.taxonomy import count_inner_consts

x0, x1 = sp.symbols("x0 x1")
c0, c1, c2, c3 = sp.symbols("c0 c1 c2 c3")


def n(expr, consts):
    return count_inner_consts(expr, consts)


# --- outer-only: nothing inner -------------------------------------------------

def test_no_constants():
    assert n(x0**3 + x0**2 + x0, ()) == 0


def test_additive_and_linear_coeff():
    assert n(c0 + c1 * x0, (c0, c1)) == 0          # c0 additive, c1 linear coeff


def test_two_linear_coeffs():
    assert n(c0 * x0 + c1 * x1, (c0, c1)) == 0


def test_polynomial_coeffs_are_outer():
    assert n(c0 * x0**2 + c1 * x0, (c0, c1)) == 0  # coeffs of monomials are LSQ-fittable


def test_constant_arithmetic_not_inner():
    assert n(c0 / c1 * x0, (c0, c1)) == 0          # c0/c1 is just a constant ratio
    assert n(c0 * x0 / c1, (c0, c1)) == 0


# --- inner: nonlinear positions ------------------------------------------------

def test_frequency_inside_cos():
    assert n(c0 * sp.cos(c1 * x0), (c0, c1)) == 1  # c1 freq inner, c0 outer mult


def test_korns12_form():
    e = c0 + c1 * sp.cos(c2 * x0) * sp.sin(c3 * x1)
    assert n(e, (c0, c1, c2, c3)) == 2            # c2, c3 inner; c0 add, c1 mul


def test_decay_rate_in_exp():
    assert n(c0 * sp.exp(c1 * x0), (c0, c1)) == 1


def test_exponent_is_inner():
    assert n(c0 * x0**c1, (c0, c1)) == 1          # c1 in exponent


def test_phase_and_freq_both_inner():
    assert n(sp.sin(c0 + c1 * x0), (c0, c1)) == 2  # both inside sin


def test_denominator_with_variable():
    assert n(c0 / (c1 + x0), (c0, c1)) == 1        # c1 inner, c0 outer mult


def test_squared_binomial_shift_is_inner():
    assert n((c0 + x0)**2, (c0,)) == 1             # exp != 1 with var -> base nonlinear


def test_sqrt_inner_scale():
    # Korns-8 shape: 6.87 + 11*sqrt(7.23*x0*x1) -> only the 7.23 is inner
    e = c0 + c1 * sp.sqrt(c2 * x0 * x1)
    assert n(e, (c0, c1, c2)) == 1


def test_seam_nested_transcendental_counts_positionally():
    # exp(c0 + c1*x0): positionally BOTH inner, even though exp(c0) is an
    # absorbable outer scale. Documents the upper-bound choice.
    assert n(sp.exp(c0 + c1 * x0), (c0, c1)) == 2


# --- robustness ----------------------------------------------------------------

def test_only_listed_constants_counted():
    # x1 is a variable here, not a constant, even though it could look constant-ish
    assert n(sp.cos(c0 * x0) + x1, (c0,)) == 1


def test_constant_only_transcendental_not_inner():
    assert n(c0 * x0 + sp.sin(c1), (c0, c1)) == 0  # sin(c1) has no variable


# --- absorbable-fold detector (reveal_folds) -----------------------------------

def test_reveal_folds_flags_sqrt_scale():
    from cusr.demonstrator.taxonomy import reveal_folds
    e = c0 + c1 * sp.sqrt(c2 * x0 * x1)        # sqrt(c2*..) = sqrt(c2)*sqrt(..)
    assert n(e, (c0, c1, c2)) == 1             # positional says inner
    assert n(reveal_folds(e), (c0, c1, c2)) == 0  # but it folds out -> absorbable


def test_reveal_folds_flags_exp_additive():
    from cusr.demonstrator.taxonomy import reveal_folds
    e = sp.exp(c0 + c1 * x0)                     # exp(c0+..) = exp(c0)*exp(..)
    assert n(e, (c0, c1)) == 2
    assert n(reveal_folds(e), (c0, c1)) == 1     # c0 folds out, c1 stays


def test_reveal_folds_keeps_genuine_inner():
    from cusr.demonstrator.taxonomy import reveal_folds
    for e, cs in [
        (c0 * (1 - sp.exp(c1 * x0)), (c0, c1)),          # decay rate
        (c0 + c1 * sp.cos(c2 * x0**3), (c0, c1, c2)),     # frequency w/ cube
        (c0 + c1 * sp.cos(c2 * x0) * sp.sin(c3 * x1), (c0, c1, c2, c3)),
    ]:
        assert n(reveal_folds(e), cs) == n(e, cs)
