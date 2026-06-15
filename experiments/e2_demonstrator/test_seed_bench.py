"""First-brick TDD for the 009 SR-recovery benchmark seed set.

Proves the pipe end-to-end on a tiny cross-family seed:
  manifest entry  ->  generate (X, y)  ->  recovery judge

The judge is REUSED from `bench.sources.evogp` (already TDD'd in
`tests/bench/test_recovery.py`); we don't re-test it here. What we test is
009's own additions:
  - the manifest schema + its inner-constant taxonomy (`n_inner_consts`),
  - the data generator (true_expr -> (X, y)),
  - a thin `score()` adapter wiring a candidate against a manifest entry.

The Korns problem doubles as the stress test for the thing 009 actually
measures: a wrong *inner* (frequency) constant must NOT count as recovered,
while an outer additive/multiplicative constant must.

Run:  uv run pytest experiments/009_sr_benchmark/test_seed_bench.py -v
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from cusr.demonstrator import seed_bench as sb


REQUIRED_FIELDS = {
    "id", "source", "true_expr", "n_vars", "var_ranges",
    "n_consts", "n_inner_consts", "difficulty", "is_control",
}


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

def test_seed_has_both_families():
    ids = {p.id for p in sb.SEED}
    assert "nguyen_1" in ids, "control problem missing"
    assert "korns_12" in ids, "inner-constant problem missing"


def test_every_problem_has_required_fields():
    for p in sb.SEED:
        for f in REQUIRED_FIELDS:
            assert hasattr(p, f), f"{p.id} missing field {f!r}"


def test_inner_const_taxonomy_labeled():
    by_id = {p.id: p for p in sb.SEED}
    # Control: integer-coefficient polynomial, no free constants at all.
    assert by_id["nguyen_1"].is_control is True
    assert by_id["nguyen_1"].n_inner_consts == 0
    # Korns-12: 2.0 - 2.1*cos(9.8*x0)*sin(1.3*x1) — two inner frequencies.
    assert by_id["korns_12"].is_control is False
    assert by_id["korns_12"].n_inner_consts == 2


def test_true_expr_parses_with_declared_vars():
    for p in sb.SEED:
        syms = {f"x{i}": sp.Symbol(f"x{i}") for i in range(p.n_vars)}
        expr = sp.sympify(p.true_expr, locals=syms)
        free = {s.name for s in expr.free_symbols}
        allowed = set(syms)
        assert free <= allowed, f"{p.id}: true_expr uses symbols outside x0..x{p.n_vars-1}: {free - allowed}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def test_generate_shapes_and_ranges():
    p = {q.id: q for q in sb.SEED}["korns_12"]
    X, y = sb.generate(p, seed=0, n_samples=256)
    assert X.shape == (256, p.n_vars)
    assert y.shape == (256,)
    assert np.all(np.isfinite(y))
    for i, (lo, hi) in enumerate(p.var_ranges):
        assert X[:, i].min() >= lo and X[:, i].max() <= hi


def test_generate_is_deterministic():
    p = {q.id: q for q in sb.SEED}["korns_12"]
    X1, y1 = sb.generate(p, seed=7, n_samples=128)
    X2, y2 = sb.generate(p, seed=7, n_samples=128)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_generate_y_matches_formula_nguyen():
    """Independent re-derivation: y must equal x0^3 + x0^2 + x0."""
    p = {q.id: q for q in sb.SEED}["nguyen_1"]
    X, y = sb.generate(p, seed=1, n_samples=64)
    x0 = X[:, 0]
    expected = x0**3 + x0**2 + x0
    assert np.allclose(y, expected, rtol=0, atol=1e-9)


def test_generate_y_matches_formula_korns():
    """Independent re-derivation: y must equal 2.0 - 2.1*cos(9.8*x0)*sin(1.3*x1)."""
    p = {q.id: q for q in sb.SEED}["korns_12"]
    X, y = sb.generate(p, seed=2, n_samples=64)
    expected = 2.0 - 2.1 * np.cos(9.8 * X[:, 0]) * np.sin(1.3 * X[:, 1])
    assert np.allclose(y, expected, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# score() adapter — end-to-end through the reused judge
# ---------------------------------------------------------------------------

def test_score_recovers_exact_control():
    p = {q.id: q for q in sb.SEED}["nguyen_1"]
    X, _ = sb.generate(p, seed=0, n_samples=200)
    assert sb.score(p.true_expr, p, X) is True


def test_score_control_outer_scale_tolerated():
    p = {q.id: q for q in sb.SEED}["nguyen_1"]
    X, _ = sb.generate(p, seed=0, n_samples=200)
    assert sb.score(f"2.0*({p.true_expr})", p, X) is True


def test_score_control_wrong_structure_rejected():
    p = {q.id: q for q in sb.SEED}["nguyen_1"]
    X, _ = sb.generate(p, seed=0, n_samples=200)
    assert sb.score("exp(x0)", p, X) is False


def test_score_recovers_exact_korns():
    p = {q.id: q for q in sb.SEED}["korns_12"]
    X, _ = sb.generate(p, seed=0, n_samples=300)
    assert sb.score(p.true_expr, p, X) is True


def test_score_korns_outer_offset_tolerated():
    p = {q.id: q for q in sb.SEED}["korns_12"]
    X, _ = sb.generate(p, seed=0, n_samples=300)
    assert sb.score(f"({p.true_expr}) + 5.0", p, X) is True


def test_score_korns_wrong_inner_constant_rejected():
    """THE 009 stress test: an inner frequency off by ~7% (9.8 -> 10.5) is a
    different function and must NOT count as recovered, even though the outer
    structure is identical. This is the discrimination 009's headline rests on.
    """
    p = {q.id: q for q in sb.SEED}["korns_12"]
    X, _ = sb.generate(p, seed=0, n_samples=300)
    wrong = "2.0 - 2.1*cos(10.5*x0)*sin(1.3*x1)"
    assert sb.score(wrong, p, X) is False
