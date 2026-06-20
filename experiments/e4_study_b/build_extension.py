"""Build the FROZEN multi-inner EXTENSION corpus via the UNCHANGED e3 criterion.

Runs every (family × value-pair) candidate from prereg_ext.EXTENSION_GRID, the
in-family canonical reject controls, and e3's standard negative controls
(nguyen, korns_8, outer-only Feynman) through `e3.criterion.decide()` — the SAME
function, thresholds, CANON_SET and STRUCT_TOL as the e3 corpus (imported, never
re-defined). EVERY candidate (admit and reject) is recorded, so a rejected family
is honest evidence the criterion discriminates, exactly like e3's 7 self-rejects.

Run:    .venv/bin/python -m experiments.e4_study_b.build_extension
Writes: experiments/e4_study_b/out/extension_manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

from experiments.e3_admit_criterion import criterion as C
from experiments.e3_admit_criterion import prereg as P
from experiments.e3_admit_criterion import skel_problems as SP

from . import prereg_ext as PX

OUT = Path(__file__).resolve().parent / "out"


def _build(template: str, spec: dict, values: tuple, *, outer: float = 1.7,
           prefix: str = "ext") -> SP.SkelProblem:
    """One SkelProblem from a (template, spec, inner-value-tuple) — mirrors
    e3.skel_problems.constructed_problem but reads PX's grid, not e3's."""
    skel = sp.sympify(spec["skeleton"])
    consts = tuple(sorted([s for s in skel.free_symbols if s.name.startswith("c")],
                          key=lambda s: s.name))
    variables = tuple(sorted([s for s in skel.free_symbols if s.name.startswith("x")],
                             key=lambda s: s.name))
    inner_names = [n.strip() for n in spec["inner"].split(",")]
    innermap = dict(zip(inner_names, values))
    gt = np.array([float(innermap[c.name]) if c.name in innermap else float(outer)
                   for c in consts])
    vstr = "_".join(f"{v:g}" for v in values)
    return SP.SkelProblem(f"{prefix}_{template}_{vstr}", "constructed_ext", skel,
                          variables, consts, gt, tuple(spec["domain"]))


def _candidates(grid: dict, prefix: str) -> list[SP.SkelProblem]:
    out = []
    for template, spec in grid.items():
        for vs in spec["values_2d"]:
            out.append(_build(template, spec, tuple(vs), prefix=prefix))
    return out


def _record(prob, v, role) -> dict:
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
        count_ops=int(sp.sympify(prob.skeleton).count_ops()),
        data_seed=P.DATA_SEED, multistart_seed=P.MULTISTART_SEED, n_samples=P.N_SAMPLES,
    )


def main(write: bool = True) -> dict:
    records = []

    print("=" * 100 + "\nEXTENSION FAMILIES (frozen grid — UNCHANGED e3 criterion)\n" + "=" * 100)
    for prob in _candidates(PX.EXTENSION_GRID, "ext"):
        v = C.decide(prob, use_structural=True)
        records.append(_record(prob, v, "extension"))
        r2 = v.R2_LS if (v.R2_LS == v.R2_LS) else float("nan")
        print(f"  {v.verdict:7s} {v.reason_code:11s} {prob.id:30s} "
              f"R2_LS={r2:.3f}  inner={len(v.inner)}  {str(prob.baked)[:42]}")

    print("\n" + "=" * 100 + "\nIN-FAMILY CANONICAL CONTROLS (should REJECT structural)\n" + "=" * 100)
    for prob in _candidates(PX.EXTENSION_REJECT_CONTROLS, "ctrl"):
        v = C.decide(prob, use_structural=True)
        records.append(_record(prob, v, "control_canon"))
        leak = "  <<< CONTROL LEAKED" if v.verdict == "ADMIT" else ""
        print(f"  {v.verdict:7s} {v.reason_code:11s} {prob.id:30s}{leak}")

    print("\n" + "=" * 100 + "\nGLOBAL NEGATIVE CONTROLS (e3 set; must REJECT)\n" + "=" * 100)
    controls = [(SP.nguyen_problem(n), "control_nguyen") for n in ("1", "2", "3", "4", "5")]
    controls.append((SP.korns_problems()["korns_8"], "control_korns8_absorbable"))
    for p in SP.feynman_outer_only(limit=P.N_OUTER_ONLY_FEYNMAN_CONTROLS):
        controls.append((p, "control_feynman_outer_only"))
    n_leak = 0
    for prob, role in controls:
        v = C.decide(prob, use_structural=True)
        records.append(_record(prob, v, role))
        if v.verdict == "ADMIT":
            n_leak += 1
            print(f"  {v.verdict:7s} {prob.id:22s} ({role})  <<< LEAKED")

    # ── summary ──────────────────────────────────────────────────────────────
    ext = [r for r in records if r["role"] == "extension"]
    ext_admit = [r for r in ext if r["verdict"] == "ADMIT"]
    ext_reject = [r for r in ext if r["verdict"] == "REJECT"]
    canon = [r for r in records if r["role"] == "control_canon"]
    canon_leak = [r for r in canon if r["verdict"] == "ADMIT"]
    over_cap = [r for r in ext_admit if r["count_ops"] > 40 or r["n_consts"] > 32]
    summary = dict(
        n_extension_candidates=len(ext),
        n_extension_admitted=len(ext_admit),
        n_extension_rejected=len(ext_reject),
        extension_admitted_ids=[r["id"] for r in ext_admit],
        extension_rejected=[(r["id"], r["reason_code"]) for r in ext_reject],
        n_canon_controls=len(canon), n_canon_leaked=len(canon_leak),
        n_global_controls=len(controls), n_global_leaked=n_leak,
        n_admit_over_cap=len(over_cap),
        power_floor_min_multi=PX.POWER_FLOOR_MIN_MULTI,
        total_multi_inner_after_extension=3 + len(ext_admit),  # e3 had 3
    )
    print("\n" + "=" * 100)
    print(f"EXTENSION ADMITTED: {len(ext_admit)}/{len(ext)}   "
          f"(rejected: {summary['extension_rejected']})")
    print(f"canonical controls leaked: {len(canon_leak)}   global controls leaked: {n_leak}")
    print(f"admits over CappedCO cap (count_ops>40 or K>32): {len(over_cap)}  (want 0)")
    print(f"total multi-inner after extension: {summary['total_multi_inner_after_extension']}  "
          f"(power floor: {PX.POWER_FLOOR_MIN_MULTI})")
    assert n_leak == 0, "a global negative control leaked — criterion unsound"
    assert len(canon_leak) == 0, "an in-family canonical control leaked — non-canonical claim broken"

    result = dict(summary=summary, records=records)
    if write:
        OUT.mkdir(exist_ok=True)
        (OUT / "extension_manifest.json").write_text(json.dumps(result, indent=2))
        print(f"\nwrote {OUT / 'extension_manifest.json'}")
    return result


if __name__ == "__main__":
    main()
