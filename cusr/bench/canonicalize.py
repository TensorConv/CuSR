from __future__ import annotations

import hashlib
import re

import sympy as sp


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def hash_l1(expr: sp.Expr) -> str:
    """L1: raw srepr — structurally sensitive, no canonicalization."""
    return _sha16(sp.srepr(expr))


_CONST_RE = re.compile(r"^c\d+$")
_CONST_PLACEHOLDER = sp.Symbol("_C")


def _is_const(s: sp.Symbol) -> bool:
    return s.is_Symbol and _CONST_RE.match(s.name) is not None


def _unified_key(e: sp.Expr) -> str:
    subs = {s: _CONST_PLACEHOLDER for s in e.free_symbols if _is_const(s)}
    return sp.srepr(e.subs(subs))


def _reorder(e: sp.Expr) -> sp.Expr:
    if not e.args:
        return e
    new_args = tuple(_reorder(a) for a in e.args)
    if e.func in (sp.Add, sp.Mul):
        new_args = tuple(sorted(new_args, key=_unified_key))
    return e.func(*new_args, evaluate=False)


def _rename_consts(e: sp.Expr) -> sp.Expr:
    mapping: dict[str, sp.Symbol] = {}
    counter = [0]

    def walk(x: sp.Expr) -> sp.Expr:
        if _is_const(x):
            if x.name not in mapping:
                mapping[x.name] = sp.Symbol(f"c{counter[0]}")
                counter[0] += 1
            return mapping[x.name]
        if x.args:
            return x.func(*[walk(a) for a in x.args], evaluate=False)
        return x

    return walk(e)


def hash_l2(expr: sp.Expr) -> str:
    """L2: expand + reorder commutative args ignoring const names + rename consts by first occurrence."""
    expanded = sp.expand(expr)
    reordered = _reorder(expanded)
    renamed = _rename_consts(reordered)
    payload = f"sympy={sp.__version__}|{sp.srepr(renamed)}"
    return _sha16(payload)


DEFAULT = hash_l2
