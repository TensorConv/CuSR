from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from cusr.bench.skeleton import FitRecord, FitRequest, Skeleton


def test_skeleton_valid():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    skel = Skeleton(expr=c0 * x0 + c1, variables=(x0,), constants=(c0, c1))
    assert skel.n_vars == 1
    assert skel.n_constants == 2
    assert skel.n_outputs == 1


def test_skeleton_invalid_var_name():
    a = sp.Symbol("a")
    c0 = sp.Symbol("c0")
    with pytest.raises(ValueError):
        Skeleton(expr=c0 * a, variables=(a,), constants=(c0,))


def test_skeleton_invalid_const_name():
    x0 = sp.Symbol("x0")
    k = sp.Symbol("k")
    with pytest.raises(ValueError):
        Skeleton(expr=k * x0, variables=(x0,), constants=(k,))


def test_skeleton_extra_free_symbol():
    x0 = sp.Symbol("x0")
    c0 = sp.Symbol("c0")
    zz = sp.Symbol("zz")
    with pytest.raises(ValueError):
        Skeleton(expr=c0 * x0 + zz, variables=(x0,), constants=(c0,))


def test_evaluate_shape():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    skel = Skeleton(expr=c0 * x0 + c1, variables=(x0,), constants=(c0, c1))
    X = np.linspace(-1, 1, 7).reshape(-1, 1)
    out = skel.evaluate([2.0, 3.0], X)
    assert out.shape == (7,)
    np.testing.assert_allclose(out, 2.0 * X[:, 0] + 3.0)


def test_evaluate_no_constants():
    x0 = sp.Symbol("x0")
    skel = Skeleton(expr=x0**2, variables=(x0,), constants=())
    X = np.linspace(-1, 1, 5).reshape(-1, 1)
    out = skel.evaluate([], X)
    assert out.shape == (5,)
    np.testing.assert_allclose(out, X[:, 0] ** 2)


def test_residual_nan_guard():
    x0 = sp.Symbol("x0")
    c0 = sp.Symbol("c0")
    skel = Skeleton(expr=sp.log(c0 * x0), variables=(x0,), constants=(c0,))
    X = np.array([[1.0], [-1.0], [0.5]])
    y = np.array([0.0, 0.0, 0.0])
    # c0 = -1 => log(-x) for x>0 gives NaN in numpy
    r = skel.residual([-1.0], X, y)
    assert np.all(np.isfinite(r))
    assert np.all(np.abs(r) <= 1e10 + 1e-6)


def test_jacobian_matches_analytic():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    skel = Skeleton(expr=c0 * sp.sin(x0) + c1 * x0**2, variables=(x0,), constants=(c0, c1))
    X = np.linspace(0.1, 1.0, 10).reshape(-1, 1)
    c = np.array([0.7, 0.3])
    J = skel.jacobian(c, X)
    assert J.shape == (10, 2)
    np.testing.assert_allclose(J[:, 0], np.sin(X[:, 0]))
    np.testing.assert_allclose(J[:, 1], X[:, 0] ** 2)


def test_n_outputs():
    x0 = sp.Symbol("x0")
    c0 = sp.Symbol("c0")
    skel = Skeleton(expr=c0 * x0, variables=(x0,), constants=(c0,))
    assert skel.n_outputs == 1


def test_structural_hash_stable_under_const_rename():
    x0 = sp.Symbol("x0")
    c0, c1 = sp.symbols("c0 c1")
    a = Skeleton(expr=c0 + c1 * x0, variables=(x0,), constants=(c0, c1))
    # flip ordering: c0 <-> c1 (same structural shape c_a*x0 + c_b)
    b = Skeleton(expr=c1 * x0 + c0, variables=(x0,), constants=(c0, c1))
    assert a.structural_hash == b.structural_hash


def test_jacobian_nan_guard():
    # Skeleton that yields NaN Jacobian outside its natural domain.
    # sqrt(c0 - x0**2) is undefined for x0**2 > c0; derivative w.r.t. c0 is
    # 1 / (2*sqrt(c0 - x0**2)) which also NaNs. Without the guard, scipy LM
    # silently reports "tol converged" at step 0 (see review CRITICAL).
    x0 = sp.Symbol("x0")
    c0 = sp.Symbol("c0")
    skel = Skeleton(expr=sp.sqrt(c0 - x0**2), variables=(x0,), constants=(c0,))
    X = np.array([[1.0], [2.0], [0.5]])
    J = skel.jacobian(np.array([0.1]), X)  # 0.1 - 1 = -0.9 → NaN domain
    assert J.shape == (3, 1)
    assert np.all(np.isfinite(J))
    # Guard replaces NaN with 0 → zero gradient, LM won't step blindly.
    assert np.all(J == 0.0)
