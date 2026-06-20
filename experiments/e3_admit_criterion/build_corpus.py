"""Experiment-2 — construct a frozen inner-constant corpus using the criterion.

Runs every candidate from the FROZEN construction grid (prereg.CONSTRUCTION_GRID)
plus negative controls (Nguyen, korns_8, outer-only Feynman) through decide().
Admitted candidates form the corpus; EVERY candidate (admit and reject) is recorded
in the manifest with its R²_LS / R²_full / recovery / verdict / reason / seed, so the
corpus is reproducible from manifest+seed and is provably not cherry-picked (the
negative controls and any rejected construction candidates are logged, not dropped).

Honest by construction: a pre-registered construction candidate that the criterion
REJECTS (e.g. a quadratic shift, which is absorbable into a polynomial span) is
reported as rejected — the grid was frozen before running, so rejections are
evidence the criterion discriminates rather than rubber-stamps.

Run: .venv/bin/python -m experiments.e3_admit_criterion.build_corpus
Writes: experiments/e3_admit_criterion/out/corpus_manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

from . import criterion as C
from . import prereg as P
from . import skel_problems as SP

OUT = Path(__file__).resolve().parent / "out"


def _record(prob, v, role):
    return dict(
        id=v.id, role=role, verdict=v.verdict, reason_code=v.reason_code, reason=v.reason,
        formula=str(prob.baked), skeleton=str(prob.skeleton),
        constants=[c.name for c in prob.constants], gt=[float(x) for x in prob.gt],
        domain=[list(map(float, d)) for d in prob.domain],
        R2_LS=v.R2_LS, R2_LS_best=v.R2_LS_best, R2_full=v.R2_full,
        R2_full_random=v.R2_full_random, recovery_err=v.recovery_err,
        surviving_inner=list(v.inner), absorbable=list(v.absorbable),
        n_inner_positional=v.n_inner_positional, rank=v.rank, n_consts=v.n_consts,
        structural=v.structural, seeding=v.seeding,
        data_seed=P.DATA_SEED, multistart_seed=P.MULTISTART_SEED, n_samples=P.N_SAMPLES,
    )


def main(write: bool = True) -> dict:
    records = []

    # ── construction candidates (the frozen grid) ──────────────────────────
    print("=" * 100)
    print("CONSTRUCTION CANDIDATES (frozen grid)")
    print("=" * 100)
    for prob in SP.construction_candidates():
        v = C.decide(prob, use_structural=True)
        records.append(_record(prob, v, "construction"))
        print(f"  {v.verdict:7s} {v.reason_code:11s} {prob.id:34s} "
              f"R2_LS={v.R2_LS if v.R2_LS == v.R2_LS else float('nan'):.3f}  {str(prob.baked)[:45]}")

    # ── negative controls (must reject) ────────────────────────────────────
    print("\n" + "=" * 100)
    print("NEGATIVE CONTROLS (must REJECT — evidence the corpus is not cherry-picked)")
    print("=" * 100)
    controls = []
    for nid in ("1", "2", "3", "4", "5"):
        controls.append((SP.nguyen_problem(nid), "control_nguyen"))
    controls.append((SP.korns_problems()["korns_8"], "control_korns8_absorbable"))
    for p in SP.feynman_outer_only(limit=P.N_OUTER_ONLY_FEYNMAN_CONTROLS):
        controls.append((p, "control_feynman_outer_only"))

    n_control_leak = 0
    for prob, role in controls:
        v = C.decide(prob, use_structural=True)
        records.append(_record(prob, v, role))
        leak = ""
        if v.verdict == "ADMIT":
            leak = "  <<< CONTROL LEAKED (admitted!)"
            n_control_leak += 1
        print(f"  {v.verdict:7s} {v.reason_code:11s} {prob.id:22s} ({role}){leak}")

    # ── summary ────────────────────────────────────────────────────────────
    constr = [r for r in records if r["role"] == "construction"]
    admitted = [r for r in records if r["verdict"] == "ADMIT"]
    constr_admit = [r for r in constr if r["verdict"] == "ADMIT"]
    constr_reject = [r for r in constr if r["verdict"] == "REJECT"]
    summary = dict(
        n_construction_candidates=len(constr),
        n_construction_admitted=len(constr_admit),
        n_construction_rejected=len(constr_reject),
        construction_rejected=[(r["id"], r["reason_code"]) for r in constr_reject],
        n_controls=len(controls),
        n_controls_leaked=n_control_leak,
        corpus_size=len(admitted),
        corpus_ids=[r["id"] for r in admitted],
    )
    print("\n" + "=" * 100)
    print(f"CORPUS SIZE (admitted): {len(admitted)}  "
          f"[{len(constr_admit)}/{len(constr)} construction candidates admitted]")
    if constr_reject:
        print(f"Construction candidates REJECTED (honest, pre-registered): "
              f"{summary['construction_rejected']}")
    print(f"Negative controls: {len(controls)} total, LEAKED (wrongly admitted): {n_control_leak}")
    assert n_control_leak == 0, "a negative control was admitted — criterion is unsound"

    result = dict(summary=summary, records=records)
    if write:
        OUT.mkdir(exist_ok=True)
        (OUT / "corpus_manifest.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {OUT / 'corpus_manifest.json'}")
    return result


if __name__ == "__main__":
    main()
