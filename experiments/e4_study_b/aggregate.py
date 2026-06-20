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
from collections import Counter
from pathlib import Path

import numpy as np

from . import arms as ARMS
from .run_study_b import summarize

OUT = Path(__file__).resolve().parent / "out"
ARM_NAMES = [a.name for a in ARMS.make_arms()]


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

    report = {"n_cells": len(rows), "n_problems": n_problems, "n_failed": n_fail,
              "arms": ARM_NAMES, "summary": summary, "paired_binary": binary}
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
    print("\n=== controls (CO must NOT unlock — solved should be ~equal/low) ===")
    for a in ARM_NAMES:
        x = summary["controls"][a]
        print(f"  {a:12} solved {x['solved_rec']}/{x['n']}  medR2>=0 {x['median_r2_clamped']:.3f}")
    print(f"\nwrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
