"""Pure analysis functions for the C2 determinism / clean-baseline hardening.

TDD'd in test_determinism_analysis.py (no GPU). The serial driver
determinism_run.py imports these to turn raw timing reps into CV / multipliers;
the adversarial audit re-derives every headline number from reps_raw.jsonl with
the SAME functions, so a number that does not reproduce here is a fabrication.
"""
from __future__ import annotations

import numpy as np


def compute_cv(reps):
    """median/min/max/mean/std(ddof=1)/cv=std/mean over timing reps (a fraction).
    Non-finite reps are dropped; empty -> all NaN. cv is the WITHIN-SESSION
    coefficient of variation (run-to-run spread under fixed clocks/binding)."""
    a = np.asarray(list(reps), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, median=float("nan"), min=float("nan"), max=float("nan"),
                    mean=float("nan"), std=float("nan"), cv=float("nan"))
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return dict(n=int(a.size), median=float(np.median(a)), min=float(a.min()),
                max=float(a.max()), mean=mean, std=std,
                cv=(std / mean if mean else float("nan")))


def fit_intercept_slope(iters, loop_ms, ref_iter=50):
    """OLS loop_ms = intercept + slope*iters. A nonzero intercept (relative to
    the total at ref_iter) is a FIXED short-region/launch cost that a single short
    loop over-weights => systematic short-timing bias. intercept_frac = fraction
    of loop_ms at ref_iter that is fixed; ~0 => no bias (the point-4 closure)."""
    x = np.asarray(iters, dtype=float)
    y = np.asarray(loop_ms, dtype=float)
    if x.size < 2:
        raise ValueError("need >= 2 points to fit")
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = a + b * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    total_at_ref = a + b * ref_iter
    return dict(intercept=float(a), slope=float(b), r2=float(r2),
                ref_iter=int(ref_iter), total_at_ref=float(total_at_ref),
                intercept_frac=(float(a / total_at_ref)
                                if total_at_ref else float("nan")),
                n_points=int(x.size))


def check_linearity(iters, loop_ms, plateau_ratio=0.5):
    """Guard against the early-stop saturation hole: if the GPU LM loop breaks
    once all trees converge, loop_ms(40) ~= loop_ms(80) and the regression's
    'intercept' is meaningless (the same idle-work failure that killed --repeat).
    Compare the TOP consecutive segment slope to the median segment slope; top <<
    median => it plateaued. Also returns the largest still-rising low-iter PREFIX
    length so the caller fits only the genuinely-iterating regime."""
    order = np.argsort(np.asarray(iters, dtype=float))
    x = np.asarray(iters, dtype=float)[order]
    y = np.asarray(loop_ms, dtype=float)[order]
    if x.size < 2:
        return dict(linear=False, segment_slopes=[], median_slope=float("nan"),
                    top_slope_ratio=float("nan"), linear_prefix_len=int(x.size))
    seg = np.diff(y) / np.diff(x)
    med = float(np.median(seg))
    top_ratio = float(seg[-1] / med) if med else float("nan")
    linear = bool(np.isfinite(top_ratio) and top_ratio >= plateau_ratio)
    keep = 2
    for i in range(seg.size):
        if med and seg[i] >= plateau_ratio * med:
            keep = i + 2
        else:
            break
    return dict(linear=linear, segment_slopes=[float(s) for s in seg],
                median_slope=med, top_slope_ratio=top_ratio,
                linear_prefix_len=int(min(keep, x.size)))


def recompute_crossover(gpu_tput, operon_tput, gpu_cv=0.0, operon_cv=0.0):
    """revad-vs-Operon multiplier = gpu_tput / operon_tput, with a within-session
    error band from propagated CVs: rel_unc = sqrt(gpu_cv^2 + operon_cv^2).

    HONESTY (advisor gate-3): this band is WITHIN-SESSION only. It does NOT capture
    the cross-day drift the original +-30% reflected — the controlled single
    session deliberately excludes that. State the number as a contention-free
    single-session point estimate, not as 'uncertainty is now +-cv%'."""
    if not (np.isfinite(gpu_tput) and np.isfinite(operon_tput)) or operon_tput <= 0:
        return dict(multiplier=float("nan"))
    mult = float(gpu_tput) / float(operon_tput)
    rel = float(np.hypot(gpu_cv, operon_cv))
    return dict(multiplier=mult, rel_unc=rel,
                lo=mult * (1.0 - rel), hi=mult * (1.0 + rel),
                gpu_tput=float(gpu_tput), operon_tput=float(operon_tput),
                gpu_cv=float(gpu_cv), operon_cv=float(operon_cv))


def build_homogeneous_pop(pop, rep_idx=None):
    """IPC-floor control (Q1): replicate ONE representative tree M times so every
    Operon worker does IDENTICAL work. The intent was: residual wall_core CV =
    pure scheduler/pool/IPC floor, and (heterogeneous CV - this floor), IF
    POSITIVE, would be work-induced load-imbalance jitter attributable to the CPU.

    EMPIRICAL RESULT (2026-06-24, contended shared box): the excess was <= 0 in
    all 4 cells (floor CV >= heterogeneous CV), so the decomposition CANNOT
    support a load-imbalance term — no such jitter is attributable. Two reasons:
    (a) both runs ran at tenant_overlap_frac=1.0, so the floor is itself
    co-tenant-contaminated (an UPPER BOUND on the true floor); (b) identical-work
    runs may go MORE variable (synchronized contention when all workers hit shared
    resources in lockstep). Conclusion: drop the determinism-as-load-imbalance
    contribution; it would need an idle host. Returns (homogeneous_pop, rep_idx)."""
    from cusr.benchmark import popio
    metas = pop["metas"]
    Ks = metas[:, 3]
    if rep_idx is None:
        elig = np.where(Ks > 0)[0]
        if elig.size:
            kk = Ks[elig]
            rep_idx = int(elig[np.argsort(kk, kind="stable")[len(kk) // 2]])
        else:
            rep_idx = 0
    rep_idx = int(rep_idx)
    node_off, n_nodes, c_off, K = (int(v) for v in metas[rep_idx].tolist())
    nt_j = np.asarray(pop["nt"][node_off:node_off + n_nodes]).copy()
    nv_j = np.asarray(pop["nv"][node_off:node_off + n_nodes]).copy()
    ci_j = np.asarray(pop["ci"][node_off:node_off + n_nodes]).copy()
    c_j = np.asarray(pop["c_init"][c_off:c_off + K]).copy()
    ym_j = np.asarray(pop["ym"][rep_idx]).copy()
    M = int(pop["M"])
    trees = [(nt_j.copy(), nv_j.copy(), ci_j.copy(), c_j.copy()) for _ in range(M)]
    ym_h = np.tile(ym_j, (M, 1)).astype(np.float32)
    hp = popio.build_pop(trees, np.asarray(pop["xs"]), ym_h)
    return hp, rep_idx
