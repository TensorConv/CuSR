"""009 calibrated symbolic-recovery judge.

Why this exists (see judge_calibration.md for the full measurement):
SRBench's `round_floats` rounds coefficients to 3 *decimal* places + zeroes
anything below 1e-4. That makes its inner-constant tolerance scale-dependent —
±10-25% permissive for small constants, outright annihilation below ~5e-4 —
exactly the regime our benchmark (inner constants × realistic ranges) lives in.

The fix: round each coefficient to `sig` *significant figures* (relative, no
absolute floor, no zeroing), then do the same structural-equivalence check.
This gives a uniform ~1e-3 relative tolerance across magnitudes (sig=3).

Three reported notions of recovery:
  - recovered(..., outer="strict")  : no outer-constant tolerance.
  - recovered(..., outer="lenient") : outer add/mul tolerated (= the linear
    scaling every SR method does); falls back to a numeric proxy for
    sympy-opaque rewrites (e.g. sin(a+pi/2)=cos) when X is given.
  - recovered_srbench(...)          : SRBench-faithful, for a comparability column.

NOTE: structural checks use sympy.simplify, which can be slow/hang on
pathological expressions. Fine for the seed set; add a timeout before scaling
to the full benchmark.
"""
from __future__ import annotations

import sympy as sp
from sympy import Float, Integer, preorder_traversal, simplify, sympify

DEFAULT_SIG = 3  # ~1e-3 relative inner-constant tolerance (≈ SRBench large-c regime)


def _strip_assumptions(e: sp.Expr) -> sp.Expr:
    """Normalize Symbol('x', real=True) (as EvoGP's Tree.to_sympy_expr emits)
    to plain Symbol('x') so equality/ratio simplification sees them as equal."""
    return e.xreplace({s: sp.Symbol(s.name) for s in e.free_symbols})


def _parse(expr) -> sp.Expr:
    return _strip_assumptions(sympify(str(expr)))


def round_sig(expr: sp.Expr, sig: int = DEFAULT_SIG) -> sp.Expr:
    """Round every Float atom to `sig` significant figures (relative precision).

    Unlike SRBench's `round_floats` this has no absolute decimal floor and no
    near-zero annihilation, so small and large coefficients get the same
    relative treatment."""
    e = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            e = e.subs(a, Float(f"{float(a):.{sig}g}"))
    return e


def round_floats_srbench(expr: sp.Expr) -> sp.Expr:
    """SRBench's exact round_floats (3 decimals; |a|<1e-4 -> 0)."""
    e = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            if abs(a) < 1e-4:
                e = e.subs(a, Integer(0))
            else:
                e = e.subs(a, Float(round(float(a), 3), 3))
    return e


def _structural_recovered(candidate, true, *, sig: int, outer: str) -> bool:
    try:
        p = round_sig(_parse(candidate), sig)
        t = round_sig(_parse(true), sig)
        diff = round_sig(simplify(t - p), sig)
        if str(diff) == "0":
            return True
        if outer == "lenient":
            if diff.is_constant():            # outer additive
                return True
            frac = simplify(p / t)
            if frac.is_constant() and frac != 0:  # outer multiplicative (a != 0)
                return True
        return False
    except Exception:
        return False


def recovered(
    candidate,
    true,
    *,
    sig: int = DEFAULT_SIG,
    outer: str = "lenient",
    X=None,
    use_numeric_proxy: bool = True,
) -> bool:
    """Is `candidate` a symbolic recovery of `true`?

    `outer`: "strict" (no outer-constant tolerance) or "lenient" (outer add/mul
    tolerated). In lenient mode, if the structural check fails and `X` is given,
    fall back to the numeric proxy (`bench.sources.evogp.check_recovery_numeric`)
    to catch sympy-opaque rewrites.
    """
    if outer not in ("strict", "lenient"):
        raise ValueError(f"outer must be 'strict' or 'lenient', got {outer!r}")
    if _structural_recovered(candidate, true, sig=sig, outer=outer):
        return True
    if outer == "lenient" and use_numeric_proxy and X is not None:
        try:
            from cusr.bench.sources.evogp import check_recovery_numeric
            return bool(check_recovery_numeric(str(candidate), str(true), X))
        except Exception:
            return False
    return False


def recovered_srbench(candidate, true) -> bool:
    """SRBench-faithful recovery (comparability column). Outer add/mul tolerated,
    coefficients rounded to 3 decimals with <1e-4 zeroed (its known blind spot)."""
    try:
        p = _parse(candidate)
        t = _parse(true)
        diff = round_floats_srbench(simplify(round_floats_srbench(t - p), ratio=1))
        if str(diff) == "0" or diff.is_constant():
            return True
        frac = round_floats_srbench(simplify(p / t))
        return bool(frac.is_constant()) and frac != 0
    except Exception:
        return False


def recovery_columns(candidate, true, X=None) -> dict:
    """All reported notions at once — for the benchmark's result rows."""
    return {
        "strict": recovered(candidate, true, outer="strict", X=X),
        "lenient": recovered(candidate, true, outer="lenient", X=X),
        "srbench": recovered_srbench(candidate, true),
    }


# --- hard-timeout variant -------------------------------------------------
# sympy.simplify (line ~78) can hang in C-level / deep-recursion code that a
# SIGALRM handler CANNOT interrupt (009 lesson, measured: one pathological
# <=50-node tree spun a single core for 13h despite a SIGALRM guard). The only
# reliable fix is to run the judge in a spawned subprocess and terminate() it on
# timeout. `candidate` must be passed as a string (the EvoGP CUDA Tree is not
# picklable; convert via str(tree.to_sympy_expr()) in the parent first).
import multiprocessing as _mp  # noqa: E402

_TIMEOUT_FALSE = {"strict": False, "lenient": False, "srbench": False}


def _judge_worker(q, candidate, true, X):
    try:
        q.put(recovery_columns(candidate, true, X=X))
    except Exception as e:  # noqa: BLE001
        q.put({**_TIMEOUT_FALSE, "_err": str(e)})


def recovery_columns_timeout(candidate, true, X=None, timeout: float = 20.0) -> dict:
    """recovery_columns with a HARD wall-clock timeout via a spawned subprocess.

    'spawn' (not fork) so the child gets a fresh interpreter and never inherits
    the parent's CUDA context. On timeout the child is terminate()d and the tree
    is scored non-recovery (a tree we cannot judge in `timeout`s is not a clean
    symbolic recovery). Returns the usual {strict,lenient,srbench} dict, with a
    `_timeout`/`_err` marker on failure for honest logging."""
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_judge_worker, args=(q, str(candidate), str(true), X), daemon=True)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return {**_TIMEOUT_FALSE, "_timeout": True}
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001 — child died without producing a result
        return {**_TIMEOUT_FALSE, "_err": "no result"}
