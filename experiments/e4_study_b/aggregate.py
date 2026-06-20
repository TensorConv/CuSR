"""Aggregate the parallel Study B shards into one report.

Reads every `out/<tag>.jsonl` shard (one per GPU from the parallel run), combines
the per-cell rows, and computes the full summary (reusing run_study_b.summarize)
plus the paired BINARY tests (McNemar/exact sign) for solved/recovery that the
review asked for — kept out of the continuous-R2 Wilcoxon path. Analysis is
decoupled from the (expensive) run, so this can be re-run / extended post-hoc
without re-running the experiment.

Run: .venv/bin/python -m experiments.e4_study_b.aggregate [glob]
   default glob = study_b_gpu*.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

from . import arms as ARMS
from .run_study_b import _clamp0, summarize

OUT = Path(__file__).resolve().parent / "out"
ARM_NAMES = [a.name for a in ARMS.make_arms()]


def _sign_p(nb, nc):
    n = nb + nc
    if n == 0:
        return float("nan")
    k = min(nb, nc)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def _fail_kind(r):
    """crash vs timeout for a failed row. Prefers the structured fail_kind field
    (new runs); falls back to parsing the _failed reason string (old shards)."""
    if not r.get("_failed"):
        return None
    fk = r.get("fail_kind")
    if fk:
        return fk
    return "timeout" if str(r["_failed"]).startswith("killed>") else "crash"


def fail_kinds(rows):
    """Per-arm crash/timeout split + per-seed fail vector. The 'GPU completes
    where CPU times out' claim must not lump backend CRASHES (a fixable bug) in
    with real timeouts — they are different failure modes."""
    out = {}
    for a in ARM_NAMES:
        ar = [r for r in rows if r["arm"] == a and not r["is_control"]]
        fails = [r for r in ar if r.get("_failed")]
        kinds = Counter(_fail_kind(r) for r in fails)
        by_seed = Counter(r["seed"] for r in fails)
        out[a] = {"n_admit": len(ar), "n_failed": len(fails),
                  "n_timeout": kinds.get("timeout", 0), "n_crash": kinds.get("crash", 0),
                  "fails_by_seed": dict(sorted(by_seed.items()))}
    return out


def completed_both_paired(rows, a, b):
    """Paired held-out-R2 test between arms a,b restricted to (problem,seed) cells
    where BOTH arms completed (drop either-failed). Removes the timeout/crash
    R2=0 floor that contaminates the all-cells Wilcoxon. Per-problem median over
    surviving seeds, then paired Wilcoxon across problems (matches the primary
    aggregation). Reports clamped AND raw deltas + how many cells were dropped."""
    from scipy.stats import wilcoxon
    adm = [r for r in rows if not r["is_control"]]
    cells = defaultdict(dict)
    for r in adm:
        if r["arm"] in (a, b):
            cells[(r["id"], r["seed"])][r["arm"]] = r
    kept = [(k, c) for k, c in cells.items()
            if a in c and b in c and not c[a].get("_failed") and not c[b].get("_failed")]
    dropped = sum(1 for k, c in cells.items()
                  if a in c and b in c and (c[a].get("_failed") or c[b].get("_failed")))
    pa, pb, pa_raw, pb_raw = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for (pid, _seed), c in kept:
        pa[pid].append(_clamp0(c[a]["r2_test"])); pb[pid].append(_clamp0(c[b]["r2_test"]))
        pa_raw[pid].append(c[a]["r2_test"]); pb_raw[pid].append(c[b]["r2_test"])
    pids = sorted(set(pa) & set(pb))
    xa = np.array([np.median(pa[p]) for p in pids]); xb = np.array([np.median(pb[p]) for p in pids])
    d = xb - xa
    raw_a = np.array([np.median(pa_raw[p]) for p in pids]); raw_b = np.array([np.median(pb_raw[p]) for p in pids])
    res = {"pair": f"{b}_vs_{a}", "n_pairs": len(pids), "cells_kept": len(kept),
           "cells_dropped_either_failed": dropped,
           "median_delta_r2_clamped": float(np.median(d)) if d.size else float("nan"),
           "median_delta_r2_raw": float(np.median(raw_b - raw_a)) if pids else float("nan"),
           "wins_b": int(np.sum(d > 0)), "wins_a": int(np.sum(d < 0))}
    res["wilcoxon_p"] = (float(wilcoxon(xa, xb).pvalue)
                         if d.size >= 1 and np.any(d != 0) else float("nan"))
    return res


def control_validity_per_metric(rows):
    """CO must not 'unlock' a control. The headline metric is `lenient`, but the
    reported 'validity holds' was checked on `solved` only. Report BOTH, and list
    any control where a CO arm recovers (lenient) more than no_co — a leak."""
    ctl = [r for r in rows if r["is_control"]]
    out = {"per_arm": {}, "lenient_leaks": []}
    for a in ARM_NAMES:
        ar = [r for r in ctl if r["arm"] == a]
        out["per_arm"][a] = {"n": len(ar),
                             "solved": sum(int(r["solved"]) for r in ar),
                             "lenient": sum(int(r["lenient"]) for r in ar)}
    # per-control-id no_co vs each CO arm on lenient
    byid = defaultdict(lambda: defaultdict(int))
    for r in ctl:
        if r["lenient"]:
            byid[r["id"]][r["arm"]] += 1
    for pid, d in sorted(byid.items()):
        nz = d.get("no_co", 0)
        for a in ("sparse_gpu", "cpu_every", "gpu_every"):
            if d.get(a, 0) > nz:
                out["lenient_leaks"].append({"id": pid, "arm": a,
                                             "co_lenient": d.get(a, 0), "no_co_lenient": nz})
    return out


def _poly_absorbable(true_expr, lo, hi, deg_max=4, n=400, thresh=0.999):
    """Is a univariate target well-approximated by a low-degree polynomial on its
    SAMPLED domain? If so, linear-scaling/GP can fit it WITHOUT recovering the
    inner constant, so its `solved` does not evidence inner-CO. None = can't
    classify (multivariate / eval failure)."""
    import sympy as sp
    try:
        expr = sp.sympify(true_expr)
    except Exception:  # noqa: BLE001
        return None
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(syms) != 1:
        return None
    xs = np.linspace(lo, hi, n)
    try:
        f = sp.lambdify(syms, expr, "numpy")
        ys = np.asarray(f(xs), dtype=float)
    except Exception:  # noqa: BLE001
        return None
    m = np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if len(ys) < 20 or np.ptp(ys) == 0:
        return None
    best = -np.inf
    for deg in range(1, deg_max + 1):
        try:
            yp = np.polyval(np.polyfit(xs, ys, deg), xs)
            ss_res = float(np.sum((ys - yp) ** 2)); ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            best = max(best, 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)
        except Exception:  # noqa: BLE001
            pass
    return best >= thresh


def absorbable_split(rows):
    """Split single-inner ADMIT solved/lenient into polynomial-ABSORBABLE vs NON-
    absorbable on the sampled domain. Shows whether the CO gain concentrates on
    the non-absorbable subset (where inner-CO is genuinely needed) — the honest
    scope of C2 — vs being inflated by absorbable problems any arm can fit."""
    from . import corpus
    adm_probs, _ = corpus.load()
    # classify ONLY single-inner admits (the split is about single-inner C2 scope)
    dom = {p.id: (p.true_expr, p.var_ranges, p.n_vars)
           for p in adm_probs if p.difficulty == "single_inner"}
    absb = {}
    for pid, (expr, vr, nv) in dom.items():
        absb[pid] = _poly_absorbable(expr, vr[0][0], vr[0][1]) if nv == 1 else None
    single = [r for r in rows if not r["is_control"] and r["difficulty"] == "single_inner"]
    out = {"absorbable_ids": sorted([k for k, v in absb.items() if v is True]),
           "nonabsorbable_ids": sorted([k for k, v in absb.items() if v is False]),
           "unclassified_ids": sorted([k for k, v in absb.items() if v is None]),
           "by_arm": {}}
    for a in ARM_NAMES:
        blk = {}
        for label, want in (("absorbable", True), ("nonabsorbable", False)):
            sub = [r for r in single if r["arm"] == a and absb.get(r["id"]) is want]
            blk[label] = {"n": len(sub),
                          "solved": sum(int(r["solved"]) for r in sub),
                          "lenient": sum(int(r["lenient"]) for r in sub)}
        out["by_arm"][a] = blk
    return out


def multi_inner_power(rows):
    """Multi-inner is the sub-claim the extension was built to test. Report the
    DISCORDANT-pair counts + sign-p + Wilcoxon so the multi null is read as what
    it is — UNTESTED / underpowered (near-zero discordant pairs => no power),
    NOT a refutation of 'CO helps multi-inner'."""
    from scipy.stats import wilcoxon
    multi = [r for r in rows if not r["is_control"] and r["difficulty"] == "multi_inner"]
    n_problems = len({r["id"] for r in multi})
    out = {"n_multi_problems": n_problems, "discordant": {}, "wilcoxon": {}}
    for field in ("solved", "lenient"):
        by = defaultdict(dict)
        for r in multi:
            by[(r["id"], r["seed"])][r["arm"]] = bool(r[field])
        nb = nc = 0
        for c in by.values():
            if "no_co" in c and "gpu_every" in c:
                if not c["no_co"] and c["gpu_every"]:
                    nb += 1
                elif c["no_co"] and not c["gpu_every"]:
                    nc += 1
        out["discordant"][field] = {"gpu_up": nb, "no_co_up": nc, "discordant": nb + nc,
                                    "sign_p": _sign_p(nb, nc),
                                    "note": "0 discordant => ZERO power" if nb + nc == 0 else ""}
    # per-problem clamped-R2 Wilcoxon (power floor now met: 15 >= prereg floor 8)
    pids = sorted({r["id"] for r in multi})
    pa = defaultdict(list); pb = defaultdict(list)
    for r in multi:
        if r["arm"] == "no_co":
            pa[r["id"]].append(_clamp0(r["r2_test"]))
        elif r["arm"] == "gpu_every":
            pb[r["id"]].append(_clamp0(r["r2_test"]))
    xa = np.array([np.median(pa[p]) for p in pids if pa[p] and pb[p]])
    xb = np.array([np.median(pb[p]) for p in pids if pa[p] and pb[p]])
    d = xb - xa
    out["wilcoxon"]["gpu_every_vs_no_co"] = {
        "n_pairs": int(d.size), "median_delta_r2": float(np.median(d)) if d.size else float("nan"),
        "p": float(wilcoxon(xa, xb).pvalue) if d.size and np.any(d != 0) else float("nan"),
        "verdict": "underpowered / directional — not a refutation"}
    return out


def _load(glob: str) -> list[dict]:
    rows = []
    for fn in sorted(OUT.glob(glob)):
        for line in fn.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mcnemar(rows, a, b, field):
    """Paired binary test for `field` (solved/lenient) between arm a and arm b
    over seed-matched (problem,seed) cells. Returns discordant counts + exact
    two-sided sign-test p on the discordant pairs."""
    from math import comb
    by = {}
    for r in rows:
        if r["arm"] in (a, b) and not r["is_control"]:
            by.setdefault((r["id"], r["seed"]), {})[r["arm"]] = bool(r[field])
    nb = nc = 0  # b: a=0,b=1 (b better); c: a=1,b=0 (a better)
    for cell in by.values():
        if a in cell and b in cell:
            if not cell[a] and cell[b]:
                nb += 1
            elif cell[a] and not cell[b]:
                nc += 1
    n = nb + nc
    if n == 0:
        p = float("nan")
    else:
        k = min(nb, nc)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b_better": nb, "a_better": nc, "discordant": n, "sign_p": p}


def main():
    glob = sys.argv[1] if len(sys.argv) > 1 else "study_b_gpu*.jsonl"
    rows = _load(glob)
    if not rows:
        print(f"no rows matched {glob} in {OUT}")
        return
    n_problems = len({r["id"] for r in rows})
    n_fail = sum(1 for r in rows if r.get("_failed"))
    print(f"loaded {len(rows)} cells  ({n_problems} problems, {len(ARM_NAMES)} arms)  "
          f"fails={n_fail}")

    summary = summarize(rows, ARM_NAMES)

    # paired binary tests for the recovery claim (review should-fix)
    binary = {}
    for field in ("solved", "lenient"):
        binary[field] = {}
        for a, b in (("no_co", "gpu_every"), ("cpu_every", "gpu_every"), ("no_co", "cpu_every")):
            binary[field][f"{b}_vs_{a}"] = _mcnemar(rows, a, b, field)

    # ── audit-remediation analyses (2026-06-20 adversarial audit, run wf_3916912b-830) ──
    # B3: both-completed recompute for ALL pairs (the all-cells Wilcoxon floors
    #     failed cells to 0; for cpu this is a crash+timeout artifact that fakes a
    #     'CO hurts' between-arm result. On completed cells the cpu-vs-no_co sign flips
    #     to null). B4: crash vs timeout. B5: per-metric control validity. B6:
    #     absorbable split. B7: multi-inner power disclosure.
    completed_both = {f"{b}_vs_{a}": completed_both_paired(rows, a, b)
                      for a, b in (("no_co", "gpu_every"), ("cpu_every", "gpu_every"),
                                   ("no_co", "cpu_every"))}
    fk = fail_kinds(rows)
    ctl_metric = control_validity_per_metric(rows)
    try:
        absb = absorbable_split(rows)
    except Exception as e:  # noqa: BLE001 — corpus optional; never break aggregation
        absb = {"error": repr(e)}
    mpower = multi_inner_power(rows)

    report = {"n_cells": len(rows), "n_problems": n_problems, "n_failed": n_fail,
              "arms": ARM_NAMES, "summary": summary, "paired_binary": binary,
              "paired_completed_both": completed_both, "fail_kinds": fk,
              "control_validity_per_metric": ctl_metric, "absorbable_split": absb,
              "multi_inner_power": mpower,
              "_caveats": [
                  "fixed-GENERATION protocol: NO speed/hardware claim from gen-count.",
                  "scipy (cpu_every) CO path is cell-level NONDETERMINISTIC; its fail-set "
                  "is a noisy single draw. no_co and gpu_every are deterministic. The "
                  "gpu-vs-cpu R2-parity number is robust to this ~0.006 noise.",
                  "all-cells paired_on_r2_test floors failed cells to R2=0; use "
                  "paired_completed_both for the timeout/crash-robust between-arm comparison.",
                  "multi-inner is UNDERPOWERED (near-zero discordant pairs) — the multi "
                  "null is UNTESTED, NOT a refutation of 'CO helps multi-inner'.",
              ]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    # ── console ──
    print("\n=== per-arm (ADMITTED) ===")
    print(f"{'arm':12}{'solved':>10}{'lenient':>10}{'medR2>=0':>10}{'fail':>6}{'mwall':>9}")
    for a in ARM_NAMES:
        x = summary["admitted"][a]
        print(f"{a:12}{x['solved_rec']:>4}/{x['n']:<5}{x['lenient_rec']:>4}/{x['n']:<5}"
              f"{x['median_r2_clamped']:>10.3f}{x['n_failed']:>6}{x['mean_wall_s']:>8.1f}s")
    print("\n=== paired Wilcoxon on clamped held-out R2 (per-problem) ===")
    for k, v in summary["paired_on_r2_test"].items():
        print(f"  {k:24} Δr2={v['median_delta_r2']:+.4f}  wins {v['wins_b']}/{v['n_pairs']}  p={v['wilcoxon_p']:.4g}")
    print("\n=== paired BINARY (McNemar sign-test) on recovery ===")
    for field in ("solved", "lenient"):
        for k, v in binary[field].items():
            print(f"  {field:8} {k:24} {v['b_better']}↑/{v['a_better']}↓ (disc={v['discordant']}) p={v['sign_p']:.4g}")
    print(f"\n=== by inner-count (CLAMPED med R2; n_multi={summary['n_multi_inner_problems']}) ===")
    for bucket in ("single_inner", "multi_inner"):
        blk = summary["by_inner_count"].get(bucket) or {}
        if blk:
            print(f"  {bucket:13} " + "  ".join(
                f"{a}:{blk[a]['median_r2_clamped']:.3f}(s{blk[a]['solved_rec']}/{blk[a]['n']})"
                for a in ARM_NAMES if a in blk))
    print("\n=== [B3] paired Wilcoxon on COMPLETED-BOTH cells (timeout/crash-robust) ===")
    for k, v in completed_both.items():
        print(f"  {k:24} Δr2={v['median_delta_r2_clamped']:+.4f}  wins {v['wins_b']}/{v['n_pairs']}"
              f"  p={v['wilcoxon_p']:.4g}  (dropped {v['cells_dropped_either_failed']} cells)")
    print("    ^ cpu_every_vs_no_co here is the HONEST number; the all-cells p above is a fail-floor artifact.")
    print("\n=== [B4] fail kinds (crash = backend bug, NOT a timeout) ===")
    for a in ARM_NAMES:
        x = fk[a]
        print(f"  {a:12} failed {x['n_failed']:>2}/{x['n_admit']}  (timeout {x['n_timeout']}, crash {x['n_crash']})"
              f"  by_seed={x['fails_by_seed']}")
    print("\n=== [B5] control validity PER METRIC (lenient is the headline metric) ===")
    for a in ARM_NAMES:
        x = ctl_metric["per_arm"][a]
        print(f"  {a:12} solved {x['solved']:>2}/{x['n']}   lenient {x['lenient']:>2}/{x['n']}")
    if ctl_metric["lenient_leaks"]:
        print("  lenient LEAKS (CO recovers a control no_co does not):")
        for L in ctl_metric["lenient_leaks"]:
            print(f"    {L['id']:28} {L['arm']} {L['co_lenient']} vs no_co {L['no_co_lenient']}")
    else:
        print("  no lenient leaks.")
    print("\n=== [B6] single-inner solved/lenient: ABSORBABLE vs NON-absorbable ===")
    if "by_arm" in absb:
        print(f"  absorbable={absb['absorbable_ids']}")
        print(f"  non-absorbable n_ids={len(absb['nonabsorbable_ids'])}  unclassified(multivar)={len(absb.get('unclassified_ids',[]))}")
        for a in ARM_NAMES:
            b = absb["by_arm"][a]
            print(f"  {a:12} absorbable s{b['absorbable']['solved']}/{b['absorbable']['n']} "
                  f"l{b['absorbable']['lenient']}  |  NON-abs s{b['nonabsorbable']['solved']}/{b['nonabsorbable']['n']} "
                  f"l{b['nonabsorbable']['lenient']}")
        print("    ^ C2 scope: the no_co->gpu recovery gain should concentrate on NON-absorbable.")
    else:
        print("  (skipped:", absb.get("error"), ")")
    print(f"\n=== [B7] multi-inner POWER (n_multi={mpower['n_multi_problems']}) — null is UNTESTED, not refuted ===")
    for field in ("solved", "lenient"):
        d = mpower["discordant"][field]
        print(f"  {field:8} gpu↑{d['gpu_up']}/no_co↑{d['no_co_up']}  discordant={d['discordant']}"
              f"  sign_p={d['sign_p']:.4g}  {d['note']}")
    w = mpower["wilcoxon"]["gpu_every_vs_no_co"]
    print(f"  wilcoxon gpu_vs_no_co: Δr2={w['median_delta_r2']:+.4f} p={w['p']:.4g} ({w['verdict']})")
    print("\n=== controls (CO must NOT unlock — solved should be ~equal/low) ===")
    for a in ARM_NAMES:
        x = summary["controls"][a]
        print(f"  {a:12} solved {x['solved_rec']}/{x['n']}  medR2>=0 {x['median_r2_clamped']:.3f}")
    print(f"\nwrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
