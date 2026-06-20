"""§2.5 application — run the admit criterion over ALL 34 positional-inner Feynman
problems and tighten the positional upper bound into a verified true set.

Reports every problem's R²_LS / R²_full / recovery / verdict / reason, and the
honest survivor (ADMIT) count — however small. No filtering, no dropping; every
one of the 34 is reported.

Run: .venv/bin/python -m experiments.e3_admit_criterion.run_feynman34
Writes: experiments/e3_admit_criterion/out/feynman34.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import criterion as C
from . import skel_problems as SP

OUT = Path(__file__).resolve().parent / "out"


def main(write: bool = True) -> dict:
    probs = SP.feynman_inner_problems()
    assert len(probs) == 34, f"expected 34 positional-inner Feynman, got {len(probs)}"

    rows = []
    t0 = time.time()
    print(f"{'id':18s} {'verdict':7s} {'code':11s} {'R2LS_w':>7s} {'R2full':>7s} "
          f"{'blind':>6s} {'rec':>7s} {'inner':>5s} {'rank':>5s}  skeleton")
    print("-" * 130)
    for p in probs:
        v = C.decide(p, use_structural=True)
        rows.append(dict(
            id=v.id, verdict=v.verdict, reason_code=v.reason_code, reason=v.reason,
            R2_LS=v.R2_LS, R2_LS_best=v.R2_LS_best, R2_full=v.R2_full,
            R2_full_random=v.R2_full_random, recovery_err=v.recovery_err,
            n_inner_positional=v.n_inner_positional, surviving_inner=list(v.inner),
            absorbable=list(v.absorbable), rank=v.rank, n_consts=v.n_consts,
            structural=v.structural, skeleton=str(p.skeleton),
            gt=[float(x) for x in p.gt],
        ))
        def f(x):
            return f"{x:7.3f}" if x == x else "    nan"  # nan-safe
        print(f"{v.id:18s} {v.verdict:7s} {v.reason_code:11s} {f(v.R2_LS)} {f(v.R2_full)} "
              f"{v.R2_full_random:6.3f} {v.recovery_err:7.4f} "
              f"{v.n_inner_positional:5d} {str(v.rank)+'/'+str(v.n_consts):>5s}  {str(p.skeleton)[:60]}")

    admits = [r for r in rows if r["verdict"] == "ADMIT"]
    by_code = {}
    for r in rows:
        by_code[r["reason_code"]] = by_code.get(r["reason_code"], 0) + 1
    summary = dict(
        n_positional_inner=len(rows),
        n_admit=len(admits),
        admit_ids=[r["id"] for r in admits],
        reject_breakdown=by_code,
        elapsed_s=round(time.time() - t0, 1),
    )
    print("-" * 130)
    print(f"POSITIONAL upper bound: {len(rows)}   ->   VERIFIED true inner (ADMIT): {len(admits)}")
    print(f"ADMIT ids: {summary['admit_ids']}")
    print(f"REJECT breakdown by reason: {by_code}")
    print(f"(elapsed {summary['elapsed_s']}s)")

    result = dict(summary=summary, rows=rows)
    if write:
        OUT.mkdir(exist_ok=True)
        (OUT / "feynman34.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {OUT / 'feynman34.json'}")
    return result


if __name__ == "__main__":
    main()
