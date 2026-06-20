"""Admit/reject criterion for inner-constant problems (experiment-1).

`decide(problem)` returns a Verdict with the linear-scaling gap, the full-CO
certification, the recovery error, and a principled reason. All ground truth is
fp64 + scipy + independent of any GPU kernel. Thresholds/grids/canonical-set are
imported from the FROZEN prereg module — nothing here tunes them.

Decision flow (each mechanism justified independently; see prereg.py rationale):
  0. no positional-inner constant            -> REJECT  (outer-only / constant-free)
  1. FOLD: all inner consts absorbable        -> REJECT  (linear scaling absorbs)
  1b. IDENTIFIABILITY: rank(J@gt) < n_consts   -> REJECT  (degenerate parameterization)
  2. STRUCTURAL (optional, Phase C): canonical -> REJECT  (GP-grammar-reachable)
  3. LINEAR-SCALING GAP:
        ADMIT iff R²_LS(worst) < τ_low  AND  R²_full > τ_high  AND  recovery < tol
        else REJECT (which gate failed is reported).

DIVISION OF LABOUR (honesty note, per adversarial review): the "admitted ⇒ genuinely
needs inner CO" guarantee is carried by the four STRUCTURAL absorbability detectors —
fold, identifiability (inner-adds-rank), structural-canonical, and the linear-span
trig/poly expansion inside R²_LS — each of which fires WITHOUT fitting at the true
constants. R²_full and recovery_err are CONSTRUCTION CERTIFICATES: GT-seeded on
noise-free data they are near-tautological (≈1.0 / ≈0 for every well-posed problem),
so they certify the admitted problem is solvable+identifiable at its true constants;
they are NOT the load-bearing anti-cheat and cannot, by themselves, reject an
absorbable problem (the structural detectors do that first). The seeding is labelled
in Verdict.seeding and the blind-solver difficulty is reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import sympy as sp
from scipy.linalg import lstsq
from scipy.optimize import least_squares

from cusr.demonstrator.taxonomy import reveal_folds
from . import prereg as P
from . import skel_problems as SP


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Verdict:
    id: str
    verdict: str                 # ADMIT | REJECT
    reason: str
    reason_code: str             # no_inner | fold | rank | structural | admit | ls_reject
    R2_LS: float                 # worst over the generic reference product grid
    R2_LS_best: float            # best over the grid (reported, diagnostic)
    R2_full: float               # seeded-certification full-CO R²
    R2_full_random: float        # blind random-start full-CO R² (difficulty, non-gating)
    recovery_err: float          # max relative inner-constant recovery error (seeded fit)
    inner: tuple                 # surviving-inner constant names
    absorbable: tuple            # folded-out (absorbable) constant names
    n_inner_positional: int      # count_inner_consts upper bound
    rank: int                    # Jacobian rank at the true constants
    n_consts: int
    structural: bool
    seeding: str
    grid_fallback: bool          # True if the product grid hit MAX_PRODUCT_GRID
    details: dict = field(default_factory=dict)


# ── helpers ─────────────────────────────────────────────────────────────────
def r2(y, yhat) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def inner_set(expr: sp.Expr, constants) -> set:
    """Positional inner-constant SET. Mirrors taxonomy.count_inner_consts (which
    returns only the count); test_inner_set_matches_production_counter asserts
    len(inner_set(...)) == count_inner_consts(...)."""
    consts = set(constants)

    def _has_var(node):
        return bool(node.free_symbols - consts)

    inner: set = set()

    def walk(node, nonlinear):
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
    return inner


def _cols(X):
    return [X[:, i] for i in range(X.shape[1])]


def _fold(expr):
    """Surface every inner constant the GP grammar + OUTER linear coefficients can
    absorb WITHOUT nonlinear CO. Beyond taxonomy.reveal_folds' multiplicative/log/
    exp folds and sp.expand's polynomial-shift expansion ((x-c)²∈span{1,x,x²}), add
    sp.expand_trig so a trig PHASE collapses into its fixed angle-addition span
    (sin(x+c)=cos(c)·sin x+sin(c)·cos x ∈ span{sin x, cos x}). Genuine FREQUENCIES
    (c·x inside trig) carry no additive angle and are left untouched, so they stay
    inner. (Red-team finding: without expand_trig, additive phases were a false ADMIT.)"""
    return sp.expand(sp.expand_trig(reveal_folds(expr)))


def _outer_basis(skel, variables, fixed_inner_map):
    """Linear-in-outer basis after fixing surviving-inner consts at numeric values
    and folding everything grammar-reachable (_fold). Any residual non-fixed constant
    left inside a term is set to 1.0 — its OLS coefficient refits the outer scale.
    Mirrors pilot.basis_from."""
    g = _fold(skel.subs(fixed_inner_map))
    basis, seen = [], set()
    for term in sp.Add.make_args(g):
        _, xpart = term.as_independent(*variables, as_Add=False)
        resid = xpart.free_symbols - set(variables)
        if resid:
            xpart = xpart.subs({c: 1.0 for c in resid})
        if xpart.free_symbols & set(variables):
            key = sp.srepr(xpart)
            if key not in seen:
                seen.add(key)
                basis.append(xpart)
    return basis


def r2_ls_at(skel, variables, inner_syms, inner_vals, constants, X, y) -> float:
    """Best OLS outer-affine R² with `inner_syms` fixed at `inner_vals`."""
    fixed = {s: float(v) for s, v in zip(inner_syms, inner_vals)}
    basis = _outer_basis(skel, variables, fixed)
    n = X.shape[0]; Xc = _cols(X)
    M = [np.ones(n)]
    with np.errstate(all="ignore"):  # bad references overflow -> non-finite -> filtered below
        for b in basis:
            fb = sp.lambdify(variables, b, "numpy")
            col = np.asarray(fb(*Xc), dtype=float)
            if col.ndim == 0:
                col = np.full(n, float(col))
            M.append(col)
    A = np.column_stack(M)
    if not np.all(np.isfinite(A)):
        return float("nan")
    theta, *_ = lstsq(A, y)
    return r2(y, A @ theta)


def r2_ls_worst(skel, variables, surviving, constants, X, y):
    """WORST (and best) outer-affine R² over the per-constant reference product
    grid. WORST is licensed by the fold/identifiability guarantee: an absorbable
    or outer constant yields R²≈1 at EVERY reference, so a low WORST proves the
    constant genuinely shapes the function."""
    syms = sorted(surviving, key=lambda s: s.name)
    grid = P.GENERIC_REF_GRID
    k = len(syms)
    fallback = len(grid) ** k > P.MAX_PRODUCT_GRID
    combos = [(v,) * k for v in grid] if fallback else list(product(grid, repeat=k))
    vals = []
    for combo in combos:
        rr = r2_ls_at(skel, variables, syms, combo, constants, X, y)
        if not np.isnan(rr):
            vals.append(rr)
    if not vals:
        return float("nan"), float("nan"), fallback
    return min(vals), max(vals), fallback


def _jac_cols(skel, diff_consts, all_constants, gt, variables, X):
    """Jacobian columns ∂model/∂c for c in `diff_consts`, evaluated at the true
    constants (the model carries all constants, so we lambdify over all of them)."""
    if not diff_consts:
        return np.zeros((X.shape[0], 0))
    allargs = list(all_constants) + list(variables)
    jf = sp.lambdify(allargs, [sp.diff(skel, c) for c in diff_consts], "numpy")
    Xc = _cols(X); n = X.shape[0]
    with np.errstate(all="ignore"):
        cols = jf(*[float(v) for v in gt], *Xc)
    out = np.empty((n, len(diff_consts)))
    for j, col in enumerate(cols):
        a = np.asarray(col, dtype=float)
        out[:, j] = a if a.ndim else np.full(n, float(a))
    return out


def _svd_rank(M, *, rcond=P.IDENTIFIABILITY_RCOND):
    """SVD numeric rank of column-normalized M. Returns -1 if M is non-finite."""
    if M.shape[1] == 0:
        return 0
    if not np.all(np.isfinite(M)):
        return -1
    norms = np.linalg.norm(M, axis=0)
    nz = norms > 0
    Mn = np.zeros_like(M)
    Mn[:, nz] = M[:, nz] / norms[nz]
    s = np.linalg.svd(Mn, compute_uv=False)
    return int(np.sum(s > rcond * s[0])) if s.size and s[0] > 0 else 0


def jac_rank(skel, constants, gt, variables, X):
    """SVD numeric rank of the full NLS Jacobian at the true constants (strategy-doc
    §4.3 certificate). Reported raw; the GATE uses inner_adds_rank, because a rank
    deficiency among purely OUTER constants (a Scheme-C-split scale) is harmless to
    linear scaling and must not veto a genuine inner constant."""
    J = _jac_cols(skel, list(constants), constants, gt, variables, X)
    return _svd_rank(J), len(constants)


def inner_adds_rank(skel, constants, gt, variables, X) -> bool:
    """Do the INNER constants add an identifiable direction BEYOND the outer
    constants? rank([J_outer | J_inner]) > rank(J_outer) ⇔ ≥1 genuine inner DOF.
    False ⇒ every inner const is absorbable into the outer/linear span (korns_8;
    additive-in-exp), so the problem does not need nonlinear inner CO."""
    inner = inner_set(skel, constants)
    outer = [c for c in constants if c not in inner]
    innerc = [c for c in constants if c in inner]
    r_out = _svd_rank(_jac_cols(skel, outer, constants, gt, variables, X))
    r_all = _svd_rank(_jac_cols(skel, outer + innerc, constants, gt, variables, X))
    if r_all < 0:  # non-finite Jacobian ⇒ ill-posed ⇒ no certified inner DOF
        return False
    return r_all > r_out


def _full_co(skel, variables, constants, gt, X, y, *, seeded, seed=P.MULTISTART_SEED):
    """LM full constant-optimization. seeded=True → GT + jitter + scale-grid starts
    (construction certification); seeded=False → random gt-magnitude-scaled starts
    (charitable blind-solver baseline). Returns (params, r2) or None."""
    allargs = list(constants) + list(variables)
    f = sp.lambdify(allargs, skel, "numpy")
    jexpr = [sp.diff(skel, c) for c in constants]
    jf = sp.lambdify(allargs, jexpr, "numpy")
    Xc = _cols(X); n = X.shape[0]; k = len(constants)
    gt = np.asarray(gt, dtype=float)

    def resid(p):
        return np.asarray(f(*p, *Xc), dtype=float) - y

    def jac(p):
        J = jf(*p, *Xc); out = np.empty((n, k))
        for j, c in enumerate(J):
            arr = np.asarray(c, dtype=float)
            out[:, j] = arr if arr.ndim else np.full(n, float(arr))
        return out

    rng = np.random.default_rng(seed)
    starts = []
    if seeded:
        starts.append(gt.copy())
        for _ in range(P.FULLCO_N_GT_JITTER):
            starts.append(gt * (1 + rng.normal(0, P.FULLCO_GT_JITTER, k)))
        for s in P.FULLCO_SCALE_GRID:
            starts.append(rng.uniform(-1, 1, k) * s)
    else:
        scale = np.maximum(np.abs(gt), 1.0)
        for _ in range(P.FULLCO_N_RANDOM):
            starts.append(rng.uniform(-1, 1, k) * scale * 3)

    best = None
    with np.errstate(all="ignore"):  # bad LM starts overflow harmlessly
        for s in starts:
            try:
                sol = least_squares(resid, s, jac=jac, method="lm", max_nfev=P.FULLCO_MAX_NFEV)
                rr = r2(y, np.asarray(f(*sol.x, *Xc), dtype=float))
                if best is None or rr > best[1]:
                    best = (sol.x, rr)
            except Exception:
                pass
    return best


def blind_difficulty(skel, variables, constants, gt, X, y) -> float:
    """NON-GATING diagnostic: MEDIAN best-R² of blind random-start multistart over
    FULLCO_BLIND_SEEDS. Median (not single-seed) because blind success is
    stochastic — a stray seed can land in the true-frequency basin."""
    rs = []
    for sd in P.FULLCO_BLIND_SEEDS:
        res = _full_co(skel, variables, constants, gt, X, y, seeded=False, seed=sd)
        rs.append(res[1] if res else float("nan"))
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.median(rs)) if rs else float("nan")


def recovery_err(params, constants, surviving, gt) -> float:
    """Max relative recovery error over surviving-inner constants. Compares
    magnitudes to tolerate sign-aliasing (cos(-w·x)=cos(w·x), decay sign)."""
    idx = {c: i for i, c in enumerate(constants)}
    gt_map = {c: float(v) for c, v in zip(constants, gt)}
    worst = 0.0
    for c in surviving:
        true_v = gt_map[c]
        est = float(params[idx[c]])
        e = abs(abs(est) - abs(true_v)) / max(abs(true_v), 1e-12)
        worst = max(worst, e)
    return worst


def is_structural(skel, variables, surviving, constants, gt, X, y, *, canon_set=None):
    """Structural-discrete test (mechanism 2): every surviving-inner const is
    within STRUCT_TOL of a CANON (GP-grammar-reachable) value AND fixing them at
    those canonical values + OLS-outer reaches R² ≥ τ_high. Justified by grammar
    reachability (±1 from subtraction, π a grammar constant), never by fitting the
    anchors. `canon_set` defaults to the FROZEN prereg.CANON_SET; it is exposed only
    so the report can run a CANON-MEMBERSHIP robustness sweep over alternative
    reachability sets without mutating the frozen default."""
    cset = P.CANON_SET if canon_set is None else canon_set
    gt_map = {c: float(v) for c, v in zip(constants, gt)}
    syms = sorted(surviving, key=lambda s: s.name)
    canon = {}
    for c in syms:
        tv = gt_map[c]
        nearest = min(cset, key=lambda z: abs(z - tv) / max(abs(tv), 1e-12))
        rel = abs(nearest - tv) / max(abs(tv), 1e-12)
        if rel > P.STRUCT_TOL:
            return False, {"reason": f"{c.name}={tv:.4g} not canonical (nearest {nearest:.4g}, "
                                     f"rel {rel:.3f} > {P.STRUCT_TOL})"}
        canon[c.name] = float(nearest)
    r = r2_ls_at(skel, variables, syms, [canon[c.name] for c in syms], constants, X, y)
    return (r >= P.TAU_HIGH), {"canon": canon, "r2_at_canon": float(r)}


# ── the criterion ───────────────────────────────────────────────────────────
def decide(prob: "SP.SkelProblem", *, use_structural: bool = True, canon_set=None) -> Verdict:
    # canon_set defaults to the frozen prereg.CANON_SET; overridden only by the
    # report's CANON-membership robustness sweep.
    skel, variables, constants, gt = prob.skeleton, prob.variables, prob.constants, prob.gt
    X, y = SP.materialize(prob)

    inner0 = inner_set(skel, constants)
    base = dict(
        id=prob.id, R2_LS=float("nan"), R2_LS_best=float("nan"),
        R2_full=float("nan"), R2_full_random=float("nan"), recovery_err=float("nan"),
        inner=tuple(sorted(s.name for s in inner0)), absorbable=(),
        n_inner_positional=len(inner0), rank=-1, n_consts=len(constants),
        structural=False, seeding="n/a", grid_fallback=False, details={},
    )

    if not inner0:
        return Verdict(verdict="REJECT", reason="no positional-inner constant (outer-only / constant-free)",
                       reason_code="no_inner", **{**base, "inner": ()})

    folded = _fold(skel)
    inner1 = inner_set(folded, constants)
    surviving = inner0 & inner1
    absorbable = inner0 - inner1
    base["inner"] = tuple(sorted(s.name for s in surviving))
    base["absorbable"] = tuple(sorted(s.name for s in absorbable))

    if not surviving:
        return Verdict(verdict="REJECT",
                       reason=f"all {len(inner0)} positional-inner const(s) absorbable via fold "
                              f"{base['absorbable']} — linear scaling absorbs them (e.g. sqrt(c·g)=sqrt(c)·sqrt(g))",
                       reason_code="fold", **base)

    rank, ncon = jac_rank(skel, constants, gt, variables, X)  # raw, for reporting
    base["rank"] = rank
    if not inner_adds_rank(skel, constants, gt, variables, X):
        return Verdict(verdict="REJECT",
                       reason=f"non-identifiable inner: the inner const(s) {base['inner']} add no "
                              f"direction beyond the outer constants (rank(J)={rank}/{ncon}); absorbable "
                              "into the outer/linear span, so no nonlinear inner CO is needed",
                       reason_code="rank", **base)

    if use_structural:
        struct, sdet = is_structural(skel, variables, surviving, constants, gt, X, y,
                                     canon_set=canon_set)
        base["details"]["structural"] = sdet
        if struct:
            base["structural"] = True
            return Verdict(verdict="REJECT",
                           reason=f"structural: surviving inner const(s) are GP-grammar-reachable "
                                  f"canonical values {sdet['canon']} (R²@canon={sdet['r2_at_canon']:.4f} "
                                  f"≥ τ_high={P.TAU_HIGH}); reachable without nonlinear CO",
                           reason_code="structural", **base)

    R2_LS, R2_LS_best, fb = r2_ls_worst(skel, variables, surviving, constants, X, y)
    seeded = _full_co(skel, variables, constants, gt, X, y, seeded=True)
    R2_full = seeded[1] if seeded else float("nan")
    R2_full_random = blind_difficulty(skel, variables, constants, gt, X, y)
    rec = recovery_err(seeded[0], constants, surviving, gt) if seeded else float("inf")
    base.update(R2_LS=R2_LS, R2_LS_best=R2_LS_best, R2_full=R2_full,
                R2_full_random=R2_full_random, recovery_err=rec, grid_fallback=fb,
                seeding="GT+jitter+scale-grid multistart (construction certification)")

    admit = (R2_LS < P.TAU_LOW) and (R2_full > P.TAU_HIGH) and (rec < P.RECOVERY_TOL)
    if admit:
        reason = (f"linear-scaling gap: R²_LS(worst)={R2_LS:.4f} < τ_low={P.TAU_LOW}, "
                  f"R²_full={R2_full:.4f} > τ_high={P.TAU_HIGH}, recovery_err={rec:.4f} < {P.RECOVERY_TOL}")
        return Verdict(verdict="ADMIT", reason=reason, reason_code="admit", **base)

    bits = []
    if not (R2_LS < P.TAU_LOW):
        bits.append(f"R²_LS(worst)={R2_LS:.4f} ≥ τ_low={P.TAU_LOW} (linear scaling already suffices)")
    if not (R2_full > P.TAU_HIGH):
        bits.append(f"R²_full={R2_full:.4f} ≤ τ_high={P.TAU_HIGH} (true skeleton can't fit even with CO)")
    if not (rec < P.RECOVERY_TOL):
        bits.append(f"recovery_err={rec:.4f} ≥ tol={P.RECOVERY_TOL} (inner const not recovered)")
    return Verdict(verdict="REJECT", reason="rejected — " + "; ".join(bits),
                   reason_code="ls_reject", **base)
