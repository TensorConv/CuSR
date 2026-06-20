"""Study B corpus — the e3 constructed inner-CO corpus as run-ready Problems.

Adapts the FROZEN, pre-registered admit/reject corpus
(`experiments/e3_admit_criterion/out/corpus_manifest.json`) into
`seed_bench.Problem` objects the EvoGP memetic harness consumes. We do NOT
re-derive or re-tune the corpus here — this is a read-only view of the e3
deliverable, so Study B inherits e3's pre-registration discipline verbatim.

  - ADMIT records  -> the inner-CO-necessity instrument (the arms run on these).
  - REJECT records -> the validity controls: CO must NOT "unlock" a control
                      (a no-CO vs CO gap on a control would mean the gap is not
                      attributable to inner-constant recovery). Two sub-classes
                      (per e3 REPORT §4, distinguishable via `role`): 11 PRE-
                      REGISTERED negative controls (nguyen_1-5, korns_8, 5 outer-
                      only Feynman) + 7 SELF-REJECTED construction candidates
                      (freq_sin_x2_{1,2}, freq_cos_x3_{1,2}, shift_quad_{1.3,2.7,
                      4.1}). The 7 self-rejects are the anti-cherry-pick evidence.

`difficulty` carries the inner-constant COUNT bucket ("single_inner" /
"multi_inner") so the harness can report the gain split that Claim 2's
"largest gains on problems requiring multiple internal constants" needs.
NOTE (measured 2026-06-20): only 3 of the 21 ADMIT problems are multi_inner
(all the exp*cos damped family) — that sub-claim is thinly powered until the
corpus is extended; reported honestly, not hidden.
"""
from __future__ import annotations

import json
import pathlib

from cusr.demonstrator.seed_bench import Problem

_HERE = pathlib.Path(__file__).resolve()
MANIFEST = _HERE.parents[1] / "e3_admit_criterion" / "out" / "corpus_manifest.json"
# The frozen multi-inner EXTENSION (build_extension.py), run through the SAME
# unchanged e3 criterion — its ADMITs join the instrument, its canonical siblings
# join the controls. Absent until build_extension has been run.
EXT_MANIFEST = _HERE.parent / "out" / "extension_manifest.json"


def _to_problem(rec: dict) -> Problem | None:
    """One manifest record -> Problem. Returns None if it can't be materialized
    (no formula / no domain), so non-runnable controls are skipped, not faked."""
    formula = rec.get("formula")
    domain = rec.get("domain")
    if not formula or not domain:
        return None
    n_inner = int(rec.get("n_inner_positional") or 0)
    is_control = rec.get("verdict") != "ADMIT"
    return Problem(
        id=rec["id"],
        source="constructed_e3",
        true_expr=formula,
        n_vars=len(domain),
        var_ranges=tuple(tuple(float(b) for b in d) for d in domain),
        n_consts=int(rec.get("n_consts") or 0),
        n_inner_consts=n_inner,
        difficulty=("multi_inner" if n_inner >= 2 else "single_inner"),
        is_control=is_control,
        notes=f"e3:{rec.get('role','')}:{rec.get('reason_code','')}",
    )


def load(include_controls: bool = True, include_extension: bool = True
         ) -> tuple[list[Problem], list[Problem]]:
    """Returns (admitted, controls). `admitted` = e3's 21 + (if present) the 12
    multi-inner extension ADMITs; `controls` = the runnable REJECT validity checks.
    Records are deduped by id (the extension re-runs the global controls)."""
    recs = list(json.loads(MANIFEST.read_text())["records"])
    if include_extension and EXT_MANIFEST.exists():
        for r in json.loads(EXT_MANIFEST.read_text())["records"]:
            # take the extension's own families + in-family canon controls; the
            # global nguyen/feynman controls already come from the e3 manifest.
            if r.get("role") in ("extension", "control_canon"):
                recs.append(r)
    admitted, controls, seen = [], [], set()
    for r in recs:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        p = _to_problem(r)
        if p is None:
            continue
        (controls if p.is_control else admitted).append(p)
    if not include_controls:
        controls = []
    return admitted, controls


if __name__ == "__main__":
    adm, ctl = load()
    multi = [p for p in adm if p.n_inner_consts >= 2]
    print(f"admitted={len(adm)}  (multi_inner={len(multi)})  controls={len(ctl)}")
    for p in adm:
        print(f"  [{p.difficulty:13}] {p.id:34} {p.true_expr}")
    print("controls:", [p.id for p in ctl])
