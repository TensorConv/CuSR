"""analyze.py — aggregate per-cell Operon/FD/AD JSONs into report_aggregates.json.

Reads ``data/<safe>__{operon,fd,ad}.json`` for every problem with all three, joins per
tree (index m), and computes the baseline tables. ALL numbers computed here (no
hand-transcription).

Metric design (after advisor review — the honesty core):
  * EFFICACY anchor = the DISTRIBUTION of the per-tree loss ratio loss_final/loss_start
    (fp64, neutral). Immune to each optimizer's differing stop criterion. A binary
    "meaningfully improved" requires a >1% cut (ratio < 0.99), NOT "loss changed at all"
    (the latter flatters the kernel, which nudges every tree, vs Operon, which returns a
    tree UNCHANGED when it sees no progress).
  * CONVERGED-BUT-WORSENED = status==0 yet fp64 loss_final > loss_start. The honest,
    observable "the kernel's converged set contains trees CO made worse." Its genuine
    fast-math-fp32-LIE subset (kernel's OWN fp32 loss <= start, i.e. the kernel is fooled)
    is tracked within it: of FD's converged-but-worsened, ~38% (367/961) are genuine
    fp32-lies (kernel fp32 <= start); the remaining ~62% (_convworse_ownmetric) are worse
    in the kernel's OWN fp32 objective too (an accept/stop-criterion defect, not a precision
    artifact). AD: ~67% lies (785/1162). The >100x-worse subset is checked by
    probes/probe_fp32_lie.py.
  * Failures NEVER dropped from rates. Quality (neutral loss) compared ONLY where BOTH
    engines genuinely improved (excludes converged-but-worsened); the worsened RATE is
    reported separately. inf (singular) losses counted as 'diverged', never averaged.
  * AD vs FD = CONTROLLED (same fp32 fast-math LM, only Jacobian differs) -> attributable.
    kernel vs Operon = EXTERNAL YARDSTICK (fp32 simple-damping vs fp64 trust-region) ->
    bounds achievable, not attributable.

ALIGNMENT/EVAL gate: loss_start must match across all three engines (same fp32 c_init,
same fp64 evaluator) or the cell aborts loudly.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path("/home/weish/hao/CuSR/results/operon_baseline__20260617")
DATA = D / "data"
PROBLEMS = ["feynman/I.12.1", "feynman/I.18.12", "feynman/I.27.6", "feynman/I.6.2",
            "feynman/I.12.2", "feynman/I.13.12", "feynman/II.3.24",
            "nguyen/1", "nguyen/2", "nguyen/3", "nguyen/4", "nguyen/5", "nguyen/6",
            "nguyen/7", "nguyen/8", "nguyen/9", "nguyen/10"]
GENS = [0, 4, 16, 64, 100]
TIE = 1.05          # within-1.05x "tie" band for win/loss tallies
IMPROVE = 0.99      # ratio < 0.99 == meaningful (>1%) loss reduction


def _load(safe, eng):
    p = DATA / f"{safe}__{eng}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell_records(prob):
    """Yield (gen, aligned-arrays-dict). Raises on M mismatch / loss_start disagreement."""
    safe = prob.replace("/", "_")
    op_, fd_, ad_ = _load(safe, "operon"), _load(safe, "fd"), _load(safe, "ad")
    if not (op_ and fd_ and ad_):
        return
    for g in GENS:
        gs = str(g)
        og, fg, ag = op_["gens"].get(gs), fd_["gens"].get(gs), ad_["gens"].get(gs)
        if not (og and fg and ag) or "error" in fg or "error" in ag:
            continue
        M = og["M"]
        assert fg["M"] == M and ag["M"] == M, f"{prob} g{g}: M mismatch"
        assert og.get("probe_b_identical") is True, f"{prob} g{g}: Probe B not identical"
        assert og.get("aligned") and og.get("k_mismatch", 0) == 0, f"{prob} g{g}: not aligned"
        # Operon RARELY returns a worse-than-start tree (its LM diverged + didn't keep start);
        # observed 33/~340k tree-instances corpus-wide. Counted (op_worse), not silently
        # clamped. Abort only on a SYSTEMIC rate (>5% of a cell) that would signal a real bug.
        assert og.get("worse_than_start", 0) <= 0.05 * og["M"], (
            f"{prob} g{g}: Operon worse-than-start {og['worse_than_start']} > 5% of {og['M']} (systemic?)")
        ls_o = np.array(og["loss_start"]); ls_f = np.array(fg["neutral_loss_start"])
        ls_a = np.array(ag["neutral_loss_start"])
        fin = np.isfinite(ls_o) & np.isfinite(ls_f) & np.isfinite(ls_a)
        for other in (ls_f, ls_a):
            d = np.abs(ls_o[fin] - other[fin]); denom = np.maximum(np.abs(ls_o[fin]), 1.0)
            assert np.all(d / denom < 1e-4), f"{prob} g{g}: loss_start mismatch (misalignment/eval)"
        yield g, dict(
            M=M, K=np.array(og["K"]), loss_start=ls_o,
            op_loss=np.array(og["loss_operon"]), op_succ=np.array(og["success"], bool),
            op_iters=np.array(og["iters"]), op_worse=int(og.get("worse_than_start", 0)),
            fd_loss=np.array(fg["neutral_loss_final"]), fd_status=np.array(fg["status"]),
            fd_kloss=np.array(fg["kernel_fp32_loss"]),
            fd_maxit=int(fg["summary"]["maxiter"]) if fg.get("summary") else None,
            ad_loss=np.array(ag["neutral_loss_final"]), ad_status=np.array(ag["status"]),
            ad_kloss=np.array(ag["kernel_fp32_loss"]),
            ad_maxit=int(ag["summary"]["maxiter"]) if ag.get("summary") else None,
        )


def ratio_summary(final, start, mask):
    """Distribution of loss_final/loss_start on mask (start>0). diverged = inf/worse->inf."""
    m = mask & (start > 0) & np.isfinite(start)
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    f = final[m]; s = start[m]
    fin = np.isfinite(f)
    diverged = int((~fin).sum())                          # CO -> singular (inf loss)
    r = f[fin] / s[fin]
    return dict(n=n, diverged=diverged,
                median_ratio=round(float(np.median(r)), 4) if len(r) else None,
                p10_ratio=round(float(np.percentile(r, 10)), 4) if len(r) else None,
                p90_ratio=round(float(np.percentile(r, 90)), 4) if len(r) else None,
                max_finite_ratio=round(float(r.max()), 2) if len(r) else None,  # worst finite (review: source it)
                improved_1pct=int((r < IMPROVE).sum()),    # meaningful (>1%) reduction
                worsened=int((r > 1.0 + 1e-6).sum()) + diverged)


def loss_ratio_stats(a, b, mask):
    """log10(a/b) on mask where both finite & positive (quality comparison)."""
    m = mask & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() == 0:
        return dict(n=0)
    r = np.log10(a[m] / b[m])
    return dict(n=int(m.sum()), median_log10=round(float(np.median(r)), 4),
                p10=round(float(np.percentile(r, 10)), 4),
                p90=round(float(np.percentile(r, 90)), 4),
                a_better=int((a[m] < b[m] / TIE).sum()),
                b_better=int((b[m] < a[m] / TIE).sum()))


def main():
    cells = {p: list(cell_records(p)) for p in PROBLEMS}
    cells = {p: r for p, r in cells.items() if r}
    print(f"loaded {len(cells)}/{len(PROBLEMS)} problems with all 3 engines")

    by_gen = defaultdict(lambda: defaultdict(int))
    ratio_acc = defaultdict(lambda: defaultdict(list))   # gen -> eng -> [final],[start]
    migration = defaultdict(int)
    kfail = defaultdict(lambda: defaultdict(int))
    qacc = defaultdict(lambda: dict(a=[], b=[]))
    qcomp = defaultdict(lambda: defaultdict(int))   # quality-set composition (review #7)
    adfd_perprob = defaultdict(list)                # per-problem AD-vs-FD quality (F2: source "within each problem")
    per_problem = {}
    maxit_frac = []   # (prob, gen, eng, maxiter/M)

    for prob, recs in cells.items():
        pp = {}
        for g, r in recs:
            K = r["K"]; opt = K > 0; M = r["M"]; s = r["loss_start"]
            d = by_gen[g]; d["M"] += M; d["opt"] += int(opt.sum())
            # ---- per-engine efficacy ratio (anchor) + binary meaningful improvement ----
            engines = (("op", r["op_loss"], None), ("fd", r["fd_loss"], r["fd_status"]),
                       ("ad", r["ad_loss"], r["ad_status"]))
            for name, lf, st in engines:
                ratio_acc[g][name + "_f"].append(lf[opt]); ratio_acc[g][name + "_s"].append(s[opt])
                improved = opt & (s > 0) & np.isfinite(lf) & (lf < s * IMPROVE)
                d[f"{name}_impr1"] += int(improved.sum())
                if st is not None:
                    d[f"{name}_conv"] += int(((st == 0) & opt).sum())
                    d[f"{name}_chol"] += int(((st == 4) & opt).sum())
                    d[f"{name}_nan"] += int(((st == 2) & opt).sum())
                    # status==2 is "lam>1e12 -> finished=2": review fix #4 — ~99.8% of these
                    # return a FINITE best-so-far loss (a damping STALL), only a few truly NaN.
                    d[f"{name}_nan_true"] += int(((st == 2) & opt & ~np.isfinite(lf)).sum())
                    d[f"{name}_maxit"] += int(((st == 1) & opt).sum())
                    d[f"{name}_k0"] += int((st == 3).sum())
                    # converged-but-worsened (status==0 yet fp64 worse than start)
                    worse = opt & (st == 0) & (
                        (~np.isfinite(lf)) | (lf > s * (1 + 1e-6) + 1e-9))
                    d[f"{name}_convworse"] += int(worse.sum())
                    # genuine fast-math fp32-LIE subset: kernel's OWN fp32 loss <= start
                    # (kernel is FOOLED — thinks it improved/held, fp64 says worse)
                    kl = r[f"{name}_kloss"]
                    d[f"{name}_fp32lie"] += int((worse & (kl <= s * (1 + 1e-4))).sum())
                    # review fix #8: the REST of convworse are worse in the kernel's OWN fp32
                    # objective too (a genuine accept/stop-criterion defect, not a precision artifact)
                    d[f"{name}_convworse_ownmetric"] += int((worse & (kl > s * (1 + 1e-4))).sum())
                    # SERIOUS subset: converged yet >2x worse in fp64 (unambiguous, not fp32 noise)
                    d[f"{name}_convworse2x"] += int(
                        (opt & (st == 0) & ((~np.isfinite(lf)) | (lf > s * 2))).sum())
            d["op_worse"] += r["op_worse"]
            # operon iters stats (review nit b: source the env "max 498 / 0.08%" string)
            it = r["op_iters"]
            d["op_iters_max"] = max(d["op_iters_max"], int(it.max()) if it.size else 0)
            d["op_iters_near_cap"] += int((it >= 490).sum()); d["op_iters_n"] += int(it.size)
            # ---- AD vs FD migration (controlled) ----
            for fs, as_ in zip(r["fd_status"][opt], r["ad_status"][opt]):
                migration[(int(fs), int(as_))] += 1
            # ---- robustness gap: of trees Operon MEANINGFULLY improved, where kernel did NOT
            op_imp = opt & (s > 0) & np.isfinite(r["op_loss"]) & (r["op_loss"] < s * IMPROVE)
            kfail[g]["op_improved_1pct"] += int(op_imp.sum())
            for name, lf in (("fd", r["fd_loss"]), ("ad", r["ad_loss"])):
                kimp = (s > 0) & np.isfinite(lf) & (lf < s * IMPROVE)
                kfail[g][f"{name}_noimprove_here"] += int((op_imp & ~kimp).sum())
            # ---- quality on GENUINELY-converged set (both improved; exclude convworse) ----
            both_real = (r["fd_loss"] < s * IMPROVE) & (r["ad_loss"] < s * IMPROVE) & opt \
                & np.isfinite(r["fd_loss"]) & np.isfinite(r["ad_loss"])
            qacc["ad_vs_fd"]["a"].append(r["ad_loss"][both_real]); qacc["ad_vs_fd"]["b"].append(r["fd_loss"][both_real])
            _af = r["ad_loss"][both_real]; _ff = r["fd_loss"][both_real]
            _pos = (_af > 0) & (_ff > 0)
            adfd_perprob[prob].extend(np.log10(_af[_pos] / _ff[_pos]).tolist())   # for per-problem median
            for name, lf in (("fd", r["fd_loss"]), ("ad", r["ad_loss"])):
                # Operon genuinely improved AND kernel genuinely improved (both <0.99*start)
                kimp = opt & (s > 0) & np.isfinite(lf) & (lf < s * IMPROVE)
                msk = op_imp & kimp
                qacc[f"{name}_vs_operon"]["a"].append(lf[msk]); qacc[f"{name}_vs_operon"]["b"].append(r["op_loss"][msk])
                # set composition (review #7): how many trees the quality set DROPS and why
                qc = qcomp[f"{name}_vs_operon"]
                qc["op_improved"] += int(op_imp.sum()); qc["k_improved"] += int(kimp.sum())
                qc["both_kept"] += int(msk.sum())
                qc["op_only_dropped"] += int((op_imp & ~kimp).sum())     # Operon improved, kernel didn't
                qc["k_only_dropped"] += int((kimp & ~op_imp).sum())      # kernel improved, Operon didn't (kernel WINS dropped)
            for name in ("fd", "ad"):
                mx = r[f"{name}_maxit"]
                if mx is not None:
                    maxit_frac.append((prob, g, name, round(mx / M, 4)))
            # per-cell median loss-ratio (review #1: pooling hides per-problem flips) + the
            # one-sided worsening tail (counts of trees >2x / >100x worse, ANY status — review #1).
            cell = dict(M=M, opt=int(opt.sum()),
                        op_impr1=int(op_imp.sum()),
                        fd_conv=int(((r["fd_status"] == 0) & opt).sum()),
                        ad_conv=int(((r["ad_status"] == 0) & opt).sum()),
                        fd_chol=int(((r["fd_status"] == 4) & opt).sum()),
                        ad_chol=int(((r["ad_status"] == 4) & opt).sum()),
                        # robustness gap for this cell (review #2: widen vs narrow is per-problem)
                        fd_gap=int((op_imp & ~((s > 0) & np.isfinite(r["fd_loss"]) & (r["fd_loss"] < s * IMPROVE))).sum()),
                        ad_gap=int((op_imp & ~((s > 0) & np.isfinite(r["ad_loss"]) & (r["ad_loss"] < s * IMPROVE))).sum()))
            for name, lf in (("op", r["op_loss"]), ("fd", r["fd_loss"]), ("ad", r["ad_loss"])):
                m = opt & (s > 0)
                fm = m & np.isfinite(lf)
                ratio = lf[fm] / s[fm]
                cell[f"{name}_medr"] = round(float(np.median(ratio)), 5) if ratio.size else None
                if name in ("fd", "ad"):     # one-sided tail (any status), regardless of conv flag
                    d[f"{name}_worse2x_any"] += int((m & ((~np.isfinite(lf)) | (lf > s * 2))).sum())
                    d[f"{name}_worse100x_any"] += int((m & ((~np.isfinite(lf)) | (lf > s * 100))).sum())
            pp[str(g)] = cell
        per_problem[prob] = pp

    # efficacy ratio summaries per gen per engine
    eff = {}
    for g in GENS:
        if g not in ratio_acc: continue
        eff[str(g)] = {}
        for name in ("op", "fd", "ad"):
            f = np.concatenate(ratio_acc[g][name + "_f"]); s = np.concatenate(ratio_acc[g][name + "_s"])
            eff[str(g)][name] = ratio_summary(f, s, np.ones(len(f), bool))

    quality = {}
    for name, ab in qacc.items():
        a = np.concatenate(ab["a"]) if ab["a"] else np.array([])
        b = np.concatenate(ab["b"]) if ab["b"] else np.array([])
        quality[name] = loss_ratio_stats(a, b, np.ones(len(a), bool))

    # ---- review #1: per-problem efficacy at g100; the ordering FLIPS per problem, so report
    # the full breakdown (Operon strictly better / kernel strictly better / tie) not one side. ----
    glast = str(GENS[-1])
    REL = 1.02   # >2% apart counts as a strict win (else a "tie", e.g. both ~0)
    op_best_vs = {}
    for k in ("fd", "ad"):
        opb = kb = tie = 0
        for prob, pp in per_problem.items():
            if glast not in pp: continue
            c = pp[glast]
            o, m = c.get("op_medr"), c.get(f"{k}_medr")
            if o is None or m is None: continue
            if o < m / REL:    opb += 1     # Operon meaningfully lower (deeper)
            elif m < o / REL:  kb += 1       # kernel meaningfully lower
            else:              tie += 1
        op_best_vs[k] = {"operon_better": opb, "kernel_better": kb, "tie": tie}
    # ---- review #2: per-problem robustness gap widen vs narrow (g0 -> g100), FD kernel ----
    g0s = str(GENS[0]); widen = narrow = 0; gap_by_problem = {}
    for prob, pp in per_problem.items():
        if g0s in pp and glast in pp:
            def gf(c):
                den = c["op_impr1"] or 1
                return round(100 * (c["fd_gap"] + c["ad_gap"]) / (2 * den), 2)
            a, b = gf(pp[g0s]), gf(pp[glast])
            gap_by_problem[prob] = {"g0": a, "g100": b}
            widen += int(b > a + 1e-9); narrow += int(b < a - 1e-9)
    # ---- review #5: neutral net improvement (the experiment's OWN gold standard) ----
    _fd_imp = sum(by_gen[g]["fd_impr1"] for g in GENS if g in by_gen)
    _ad_imp = sum(by_gen[g]["ad_impr1"] for g in GENS if g in by_gen)
    neutral_net = {"fd_impr1_total": _fd_imp, "ad_impr1_total": _ad_imp,
                   "ad_minus_fd": _ad_imp - _fd_imp}
    # ---- F2 (source the "within each problem" claim): per-problem median log10(AD/FD) ----
    adfd_problem_medians = {p: round(float(np.median(v)), 4) for p, v in adfd_perprob.items() if v}
    adfd_max_abs_problem = round(max((abs(m) for m in adfd_problem_medians.values()), default=0.0), 4)
    # ---- F5 (source "Operon never blows up"): worst FINITE loss-ratio per engine over all gens ----
    worst_finite_ratio = {name: round(max((eff[str(g)][name].get("max_finite_ratio") or 0)
                                          for g in GENS if str(g) in eff), 2)
                          for name in ("op", "fd", "ad")}

    report = dict(
        n_problems=len(cells), problems=list(cells), gens=GENS, tie_band=TIE, improve_thresh=IMPROVE,
        by_gen={str(g): dict(by_gen[g]) for g in GENS if g in by_gen},
        efficacy_ratio=eff,
        ad_vs_fd_migration={f"{k[0]}->{k[1]}": v for k, v in sorted(migration.items())},
        kernel_fail_where_operon_improved={str(g): dict(kfail[g]) for g in kfail},
        quality_genuine_converged=quality,
        quality_set_composition={k: dict(v) for k, v in qcomp.items()},          # review #7
        per_problem_efficacy_ordering={**op_best_vs, "n_problems": len(per_problem),  # review #1
                                        "note": "median loss-ratio at gen100; >2% apart = strict win"},
        robustness_gap_widen_narrow={"widen": widen, "narrow": narrow,            # review #2
                                     "per_problem": gap_by_problem},
        neutral_net_improvement=neutral_net,                                      # review #5
        adfd_quality_per_problem={"medians_log10": adfd_problem_medians,           # F2 (sourced)
                                  "max_abs_median": adfd_max_abs_problem},
        worst_finite_loss_ratio=worst_finite_ratio,                               # F5 (sourced)
        operon_iters_max=max((by_gen[g]["op_iters_max"] for g in GENS if g in by_gen), default=0),
        operon_near_cap_frac=round(sum(by_gen[g]["op_iters_near_cap"] for g in GENS if g in by_gen)
                                   / max(sum(by_gen[g]["op_iters_n"] for g in GENS if g in by_gen), 1), 5),
        maxiter_fraction_max=max((x[3] for x in maxit_frac), default=0.0),
        maxiter_fraction_worst=sorted(maxit_frac, key=lambda x: -x[3])[:5],
        per_problem=per_problem,
    )
    (DATA / "report_aggregates.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {DATA/'report_aggregates.json'}")
    for g in GENS:
        if g not in by_gen: continue
        d = by_gen[g]; opt = d["opt"] or 1; e = eff[str(g)]
        print(f" g{g:>3} opt={d['opt']:5} | median loss-ratio Op={e['op']['median_ratio']} "
              f"FD={e['fd']['median_ratio']} AD={e['ad']['median_ratio']} | "
              f">1% impr Op={100*d['op_impr1']/opt:.0f} FD={100*d['fd_impr1']/opt:.0f} AD={100*d['ad_impr1']/opt:.0f} | "
              f"chol FD={100*d['fd_chol']/opt:.0f} AD={100*d['ad_chol']/opt:.0f} | "
              f"conv-worse FD={d['fd_convworse']}({d['fd_fp32lie']}lie) AD={d['ad_convworse']}({d['ad_fp32lie']}lie) | "
              f"Op-worse={d['op_worse']}")
    return report


if __name__ == "__main__":
    main()
