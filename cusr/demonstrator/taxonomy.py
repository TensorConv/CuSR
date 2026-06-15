"""Problem taxonomy: count constants that sit in nonlinear positions.

`n_inner_consts` is 009's primary sorting axis — the constants that linear
least-squares cannot fit (frequencies, decay rates, exponents, phases), which is
where GPU constant optimization earns its keep. See test_taxonomy.py for the
exact definition; in short:

  INNER  = a constant entangled with a variable inside a transcendental
           argument, an exponent, or a Pow base / denominator.
  OUTER  = root additive constants, linear coefficients of variable monomials
           (c*x0, c*x0**2), and pure constant arithmetic (c0/c1).

This is a POSITIONAL definition and an UPPER BOUND: it does not detect
constants that are secretly absorbable outer scales (e.g. c0 in exp(c0+c1*x0)
= exp(c0)*exp(c1*x0)). Computing true absorbability is hard and not worth it;
the upper bound is honest and robust.
"""
from __future__ import annotations

import sympy as sp


def count_inner_consts(expr: sp.Expr, constants) -> int:
    """Number of `constants` (sympy symbols) that appear in a nonlinear position
    entangled with a variable. See module docstring for the definition."""
    consts = set(constants)

    def _has_var(node: sp.Expr) -> bool:
        return bool(node.free_symbols - consts)

    inner: set = set()

    def walk(node: sp.Expr, nonlinear: bool) -> None:
        if node.is_Symbol:
            if node in consts and nonlinear:
                inner.add(node)
            return
        if not node.args:
            return
        if isinstance(node, sp.Pow):
            base, exp_ = node.args
            power_has_var = _has_var(base) or _has_var(exp_)
            walk(exp_, nonlinear or power_has_var)
            base_nonlinear = nonlinear or (exp_ != sp.Integer(1) and power_has_var)
            walk(base, base_nonlinear)
        elif isinstance(node, (sp.Add, sp.Mul)):
            for a in node.args:
                walk(a, nonlinear)
        elif isinstance(node, sp.Function):
            arg_nonlinear = nonlinear or _has_var(node)
            for a in node.args:
                walk(a, arg_nonlinear)
        else:
            for a in node.args:
                walk(a, nonlinear)

    walk(expr, False)
    return len(inner)


def reveal_folds(expr: sp.Expr) -> sp.Expr:
    """Surface absorbable inner constants by applying the structure-preserving
    folds that linear scaling / reparametrization can exploit:
      (c*g)**p  -> c**p * g**p      (a const factor inside a Pow base, e.g. sqrt)
      log(c*g)  -> log(c) + log(g)
      exp(c0+h) -> exp(c0) * exp(h) (an additive const inside exp)

    A constant whose positional inner-count DROPS after this is a false-positive
    inner (really an outer scale/offset in disguise) — the korns_8 signature.
    This is a partial detector for the common folds, not a full absorbability
    solver; count_inner_consts itself stays the positional upper bound.
    """
    e = sp.expand_power_base(expr, force=True)
    e = sp.expand_log(e, force=True)
    return sp.expand(e)
