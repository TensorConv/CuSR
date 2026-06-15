from __future__ import annotations

import sympy as sp

from cusr.bench.canonicalize import hash_l1, hash_l2


def test_hash_l1_structural_sensitivity():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    a = c0 * x0 + c1
    b = c0 * x0 * x0 + c1
    assert hash_l1(a) != hash_l1(b)


def test_hash_l2_invariant_to_commutative_reorder():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    a = c0 + c1 * x0
    b = c1 * x0 + c0
    assert hash_l2(a) == hash_l2(b)


def test_hash_l2_invariant_to_constant_reindex():
    x0 = sp.Symbol("x0")
    c0, c1, c2 = sp.symbols("c0 c1 c2")
    a = c0 + c1 * x0 + c2 * x0**2
    # swap c0 <-> c2 (still "some_const + some_const*x0 + some_const*x0**2")
    b = c2 + c1 * x0 + c0 * x0**2
    assert hash_l2(a) == hash_l2(b)


def test_hash_l2_deterministic():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    e = c0 * sp.sin(x0) + c1 * x0**2
    assert hash_l2(e) == hash_l2(e)
    assert hash_l2(e) == hash_l2(e)


def test_hash_l1_and_l2_16_chars():
    x0 = sp.Symbol("x0")
    c0 = sp.Symbol("c0")
    e = c0 * x0
    assert len(hash_l1(e)) == 16
    assert len(hash_l2(e)) == 16


def test_hash_l2_distinguishes_shared_vs_independent_consts():
    # Shared constant: c0 appears in two roles (2 usages, 1 symbol).
    # Independent: c0 and c1 play the two roles (2 symbols). Different skeletons.
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    shared = c0 + c0 * x0
    independent = c0 + c1 * x0
    assert hash_l2(shared) != hash_l2(independent)
