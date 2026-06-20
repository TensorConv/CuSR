"""PILOT (throwaway, pre-freeze design exploration) — NOT a deliverable test.

Settle the R²_LS operationalization by MEASUREMENT on the validation anchors,
per advisor: korns_7's decay rate discriminates max-over-grid vs worst-over-grid
vs perturbation-collapse, and reasoning can't settle it. Compute every candidate
scheme's number for each anchor, then pick the scheme that cleanly separates
{korns_7/11/12 ADMIT} from {korns_8/outer-only/structural REJECT} by first
principles. fp64, scipy, zero-noise, independent of any GPU kernel.

Run: python experiments/e3_admit_criterion/pilot.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.linalg import lstsq

from cusr.bench.sources.feynman import load_feynman
from cusr.demonstrator.taxonomy import count_inner_consts, reveal_folds

np.set_printoptions(precision=4, suppress=True)

x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")
c0, c1, c2, c3 = sp.symbols("c0 c1 c2 c3")


def cols(X):
    return [X[:, i] for i in range(X.shape[1])]


def sample_X(ranges, n, seed):
    rng = np.random.default_rng(seed)
    X = np.empty((n, len(ranges)))
    for i, (lo, hi) in enumerate(ranges):
        X[:, i] = rng.uniform(lo, hi, n)
    return X


def eval_expr(expr, variables, X):
    f = sp.lambdify(variables, expr, "numpy")
    y = np.asarray(f(*cols(X)), dtype=float)
    if y.ndim == 0:
        y = np.full(X.shape[0], float(y))
    return y


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_full(skel, variables, constants, X, y, gt, n_random=16, seed=0, gt_seed=True):
    allargs = list(constants) + list(variables)
    f = sp.lambdify(allargs, skel, "numpy")
    jac_exprs = [sp.diff(skel, c) for c in constants]
    jf = sp.lambdify(allargs, jac_exprs, "numpy")
    Xc = cols(X)
    n = X.shape[0]
    k = len(constants)
    gt = np.asarray(gt, dtype=float)

    def resid(p):
        return np.asarray(f(*p, *Xc), dtype=float) - y

    def jac(p):
        J = jf(*p, *Xc)
        out = np.empty((n, k))
        for j, c in enumerate(J):
            arr = np.asarray(c, dtype=float)
            out[:, j] = arr if arr.ndim else np.full(n, float(arr))
        return out

    rng = np.random.default_rng(seed)
    starts = []
    if gt_seed:
        for _ in range(4):
            starts.append(gt * (1 + rng.normal(0, 0.1, k)))
    scale = np.maximum(np.abs(gt), 1.0)
    for _ in range(n_random):
        starts.append(rng.uniform(-1, 1, k) * scale * 3)

    best = None
    for s in starts:
        try:
            sol = least_squares(resid, s, jac=jac, method="lm", max_nfev=4000)
            rr = r2(y, np.asarray(f(*sol.x, *Xc), dtype=float))
            if best is None or rr > best[1]:
                best = (sol.x, rr)
        except Exception:
            pass
    return best  # (params, r2)


def basis_from(skel, variables, inner_syms, inner_vals):
    """Outer-linear basis after freezing inner consts and folding absorbables."""
    subs = {s: float(v) for s, v in zip(inner_syms, inner_vals)}
    g = sp.expand(reveal_folds(skel.subs(subs)))
    basis, seen = [], set()
    for term in sp.Add.make_args(g):
        coeff, xpart = term.as_independent(*variables, as_Add=False)
        resid_consts = xpart.free_symbols - set(variables)
        if resid_consts:  # leftover outer scale inside a nonlinear -> OLS refits, set 1
            xpart = xpart.subs({c: 1.0 for c in resid_consts})
        if xpart.free_symbols & set(variables):
            key = sp.srepr(xpart)
            if key not in seen:
                seen.add(key)
                basis.append(xpart)
    return basis


def r2_ls_given_inner(skel, variables, inner_syms, inner_vals, X, y):
    basis = basis_from(skel, variables, inner_syms, inner_vals)
    Xc = cols(X)
    n = X.shape[0]
    M = [np.ones(n)]
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


# --- anchors -----------------------------------------------------------------
def make_anchor(id_, skel, variables, constants, gt, ranges, inner_syms,
                expect, n=2000, seed=0):
    return dict(id=id_, skel=skel, variables=tuple(variables),
                constants=tuple(constants), gt=np.asarray(gt, float),
                ranges=ranges, inner_syms=tuple(inner_syms), expect=expect,
                n=n, seed=seed)


def feynman_anchor(name, expect):
    p = load_feynman()[name]
    inner = [c for c in p.constants
             if count_inner_consts((c) * sp.Symbol("zzz") * 0 + p.skeleton_expr, p.constants)]
    # recompute inner set properly below; here just pass all-positional-inner
    return p


ANCHORS = [
    make_anchor("korns_7", c0 * (1 - sp.exp(c1 * x0)), [x0], [c0, c1],
                [213.80940889, -0.54723748542], [(0.1, 10.0)], [c1], "ADMIT"),
    make_anchor("korns_11", c0 + c1 * sp.cos(c2 * x0**3), [x0], [c0, c1, c2],
                [6.87, 11.0, 7.23], [(-5.0, 5.0)], [c2], "ADMIT"),
    make_anchor("korns_12", c0 + c1 * sp.cos(c2 * x0) * sp.sin(c3 * x1), [x0, x1],
                [c0, c1, c2, c3], [2.0, -2.1, 9.8, 1.3], [(-5.0, 5.0), (-5.0, 5.0)],
                [c2, c3], "ADMIT"),
    # Korns-8: absorbable sqrt scale -> fold removes c2 from inner -> REJECT.
    make_anchor("korns_8", c0 + c1 * sp.sqrt(c2 * x0 * x1 * x2), [x0, x1, x2],
                [c0, c1, c2], [6.87, 11.0, 7.23], [(0.1, 10.0)] * 3, [c2], "REJECT"),
]


def main():
    # add Feynman anchors: 2 outer-only + structural-±1 (I.8.14)
    fey = load_feynman()
    outer_only = [(n, p) for n, p in fey.items()
                  if count_inner_consts(p.skeleton_expr, p.constants) == 0]
    fey_anchors = []
    for name in ("I.8.14",):  # structural-±1: sqrt((c0 x0+x1)^2+(c1 x2+x3)^2), gt 1,1
        p = fey[name]
        inner = [c for c in p.constants
                 if c in (set(p.constants))
                 and count_inner_consts(p.skeleton_expr.subs({c: c}), (c,))]
        fey_anchors.append((f"feynman_{name}", p, "EDGE"))
    for name, p in outer_only[:2]:
        fey_anchors.append((f"feynman_{name}", p, "REJECT"))

    print("=" * 110)
    print("PILOT — R²_full + R²_LS scheme comparison on validation anchors")
    print("=" * 110)
    header = (f"{'anchor':16} {'exp':6} {'Rf_rand':>8} {'Rf_gtsd':>8} {'rec_err':>8} "
              f"{'LS+1':>7} {'LS-1':>7} {'LSmax':>7} {'LSwrst':>7} {'LSc1.5':>7} {'LSc0.5':>7} {'LScMIN':>7}")
    print(header)
    print("-" * 110)

    rows = []

    def process(id_, skel, variables, constants, gt, ranges, inner_syms, expect, n, seed):
        X = sample_X(ranges, n, seed)
        y = eval_expr(skel.subs({c: float(v) for c, v in zip(constants, gt)}), variables, X)
        rand = fit_full(skel, variables, constants, X, y, gt, gt_seed=False, seed=seed)
        gtsd = fit_full(skel, variables, constants, X, y, gt, gt_seed=True, seed=seed)
        # recovery error of inner consts from gt-seeded best
        params = gtsd[0]
        rec_err = 0.0
        for s in inner_syms:
            j = list(constants).index(s)
            true_v = float(gt[j])
            est = float(params[j])
            # frequency/decay sign aliasing: compare magnitudes
            e = abs(abs(est) - abs(true_v)) / max(abs(true_v), 1e-12)
            rec_err = max(rec_err, e)
        gt_inner = np.array([gt[list(constants).index(s)] for s in inner_syms])
        grid_vals = [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]
        ls = {}
        ls["+1"] = r2_ls_given_inner(skel, variables, inner_syms, [1.0] * len(inner_syms), X, y)
        ls["-1"] = r2_ls_given_inner(skel, variables, inner_syms, [-1.0] * len(inner_syms), X, y)
        grid_r2 = [r2_ls_given_inner(skel, variables, inner_syms, [v] * len(inner_syms), X, y)
                   for v in grid_vals]
        grid_r2 = [r for r in grid_r2 if not np.isnan(r)]
        ls["max"] = max(grid_r2) if grid_r2 else float("nan")
        ls["worst"] = min(grid_r2) if grid_r2 else float("nan")
        ls["c1.5"] = r2_ls_given_inner(skel, variables, inner_syms, list(gt_inner * 1.5), X, y)
        ls["c0.5"] = r2_ls_given_inner(skel, variables, inner_syms, list(gt_inner * 0.5), X, y)
        coll = [r2_ls_given_inner(skel, variables, inner_syms, list(gt_inner * f), X, y)
                for f in (0.5, 1.5, 2.0)]
        coll = [r for r in coll if not np.isnan(r)]
        ls["cMIN"] = min(coll) if coll else float("nan")
        print(f"{id_:16} {expect:6} {rand[1]:8.4f} {gtsd[1]:8.4f} {rec_err:8.4f} "
              f"{ls['+1']:7.3f} {ls['-1']:7.3f} {ls['max']:7.3f} {ls['worst']:7.3f} "
              f"{ls['c1.5']:7.3f} {ls['c0.5']:7.3f} {ls['cMIN']:7.3f}")
        rows.append((id_, expect, rand[1], gtsd[1], rec_err, ls))

    for a in ANCHORS:
        process(a["id"], a["skel"], a["variables"], a["constants"], a["gt"],
                a["ranges"], a["inner_syms"], a["expect"], a["n"], a["seed"])

    for id_, p, expect in fey_anchors:
        variables = p.variables
        constants = p.constants
        gt = p.ground_truth_constants
        inner_syms = tuple(c for c in constants
                           if count_inner_consts(p.skeleton_expr, constants)
                           and _is_inner(p.skeleton_expr, c, constants))
        ranges = p.sampling_ranges
        process(id_, p.skeleton_expr, variables, constants, gt, ranges,
                inner_syms, expect, 2000, 0)

    print("-" * 110)
    print("Legend: Rf_rand=R²_full random-only starts; Rf_gtsd=R²_full GT-seeded (ceiling);")
    print(" rec_err=max rel inner recovery err (|abs(est)-abs(true)|/|true|);")
    print(" LS+1/-1=fix inner@±1; LSmax/worst=best/worst over {±0.5,±1,±2}; LScX=fix inner@true×X; LScMIN=min over collapse factors.")
    print(" Want: ADMIT rows -> R²_LS LOW under chosen scheme; REJECT rows -> R²_LS HIGH.")


def _is_inner(skel, c, constants):
    """Is single const c in a nonlinear position (positional)?"""
    return count_inner_consts(skel, (c,)) > 0


if __name__ == "__main__":
    main()
