"""Assemble the reproducible report payload (out/report.json) for the admit criterion.

Gathers: validation-set verdicts, the two anti-cheat demonstrations (korns_8 outer
absorption; trig-phase linear-span absorption), the τ_low and STRUCT_TOL sensitivity
analyses, and the corpus + Feynman-34 summaries. Every number in REPORT.md comes from
here so it is reproducible.

Run: .venv/bin/python -m experiments.e3_admit_criterion.make_report
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

from . import criterion as C
from . import prereg as P
from . import skel_problems as SP

OUT = Path(__file__).resolve().parent / "out"


def _v(prob):
    v = C.decide(prob, use_structural=True)
    return dict(id=v.id, verdict=v.verdict, reason_code=v.reason_code,
                R2_LS=v.R2_LS, R2_full=v.R2_full, R2_full_random=v.R2_full_random,
                recovery_err=v.recovery_err, inner=list(v.inner), reason=v.reason)


def validation_set():
    K = SP.korns_problems()
    # (key, problem, expected, frozen_anchor?) — frozen anchors are in prereg.VALIDATION;
    # nguyen_5 is a supplementary structural demonstration (unit-const Nguyen).
    probs = [
        ("korns_7", K["korns_7"], "ADMIT", True),
        ("korns_11", K["korns_11"], "ADMIT", True),
        ("korns_12", K["korns_12"], "ADMIT", True),
        ("korns_8", K["korns_8"], "REJECT", True),
        ("nguyen_1", SP.nguyen_problem("1"), "REJECT", True),
        ("nguyen_5", SP.nguyen_problem("5"), "REJECT", False),
        ("feynman_I.8.14", SP.feynman_problem("I.8.14"), "REJECT", True),
        ("feynman_I.11.19", SP.feynman_problem("I.11.19"), "REJECT", True),
        ("feynman_I.12.1", SP.feynman_problem("I.12.1"), "REJECT", True),
    ]
    rows = []
    for key, prob, expected, frozen in probs:
        r = _v(prob)
        r["expected"] = expected
        r["frozen_anchor"] = frozen
        # cross-check the frozen anchors against the prereg checklist itself
        if frozen and key in P.VALIDATION:
            assert P.VALIDATION[key]["expect"] == expected, (key, "prereg mismatch")
        r["match"] = (r["verdict"] == expected)
        rows.append(r)
    return rows


def anticheat_demos():
    out = {}
    # (1) korns_8: outer affine absorbs a WRONG inner const to fake R²≈1, yet REJECTED.
    p = SP.korns_problems()["korns_8"]
    X, y = SP.materialize(p)
    c2 = p.constants[2]
    r_wrong = C.r2_ls_at(p.skeleton, p.variables, (c2,), (1.0,), p.constants, X, y)
    out["korns8_outer_absorption"] = dict(
        skeleton=str(p.skeleton), wrong_inner="c2=1.0 (true 7.23)",
        faked_R2_via_outer_absorption=float(r_wrong),
        verdict=C.decide(p).verdict, reason_code=C.decide(p).reason_code,
        note="OLS outer absorbs any fixed c2 to R²≈1, but the fold detects sqrt(c2·g)=sqrt(c2)·sqrt(g) -> REJECT",
    )
    # (2) trig phase: linear in fixed {1,sin,cos} basis (R²=1, no CO), yet REJECTED.
    x0, c0, c1 = sp.symbols("x0 c0 c1")
    pp = SP.SkelProblem("phase", "demo", c0 * sp.sin(x0 + c1), (x0,), (c0, c1),
                        np.array([1.7, 0.8]), ((-3.0, 3.0),))
    X, y = SP.materialize(pp)
    A = np.column_stack([np.ones(len(X)), np.sin(X[:, 0]), np.cos(X[:, 0])])
    th, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2_fixed = 1 - np.sum((y - A @ th) ** 2) / np.sum((y - y.mean()) ** 2)
    out["trig_phase_linear_span"] = dict(
        skeleton=str(pp.skeleton), R2_on_fixed_sin_cos_basis_no_CO=float(r2_fixed),
        verdict=C.decide(pp).verdict, reason_code=C.decide(pp).reason_code,
        note="angle addition makes sin(x+c) linear in {sin x, cos x} for ANY phase -> REJECT (fold via expand_trig)",
    )
    return out


def tau_low_sensitivity():
    K = SP.korns_problems()
    rows = []
    for kid, exp in (("korns_7", "ADMIT"), ("korns_11", "ADMIT"),
                     ("korns_12", "ADMIT"), ("korns_8", "REJECT")):
        v = C.decide(K[kid], use_structural=True)
        per_tau = {}
        for tau in P.TAU_LOW_SWEEP:
            if v.reason_code in ("no_inner", "fold", "rank", "structural"):
                got = "REJECT"
            else:
                got = ("ADMIT" if (v.R2_LS < tau and v.R2_full > P.TAU_HIGH
                                   and v.recovery_err < P.RECOVERY_TOL) else "REJECT")
            per_tau[tau] = got
        rows.append(dict(id=kid, expected=exp, R2_LS=v.R2_LS,
                         stable=all(g == exp for g in per_tau.values()), per_tau=per_tau))
    return rows


def struct_tol_sensitivity():
    """The 2 Feynman admits hinge on -1/3 being non-canonical. Report the flip tol."""
    third = 1.0 / 3.0
    rel_all = min(abs(z - (-third)) / third for z in P.CANON_SET)
    rel_no_pi = min(abs(z - (-third)) / third for z in P.CANON_SET
                    if abs(abs(z) - 1 / math.pi) > 1e-6 and abs(abs(z) - 1 / (2 * math.pi)) > 1e-6)
    return dict(
        admit_ids=["feynman_II.11.27", "feynman_II.11.28"],
        minus_third_nearest_canon_reldist=float(rel_all),
        nearest_is="-1/pi=-0.3183",
        struct_tol=P.STRUCT_TOL,
        admits_survive_iff_struct_tol_below=float(rel_all),
        reldist_excluding_pi_members=float(rel_no_pi),
        note=("-1/pi is NOT smuggled to catch -1/3: removing 1/pi-family from CANON leaves -1/3 "
              f"{rel_no_pi:.2f} from canonical, still >> STRUCT_TOL. The admits survive for any "
              f"STRUCT_TOL < {rel_all:.3f}; -1/3 != -1/pi are genuinely distinct constants."),
    )


def canon_membership_sensitivity():
    """The load-bearing robustness check (advisor): the 34→2 headline depends on the
    canonical SET membership, not just STRUCT_TOL. Re-run the Feynman-34 audit under
    several DEFENSIBLE alternative reachability sets (the FROZEN default is untouched;
    canon_set is passed only here) and report how the survivor count swings."""
    pi = math.pi
    pi_fam = (pi, -pi, 2 * pi, -2 * pi, pi / 2, -pi / 2,
              1 / pi, -1 / pi, 1 / (2 * pi), -1 / (2 * pi))
    base_no_pi = (0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5)
    thirds_quarters = (1 / 3, -1 / 3, 2 / 3, -2 / 3, 1 / 4, -1 / 4, 3 / 4, -3 / 4)
    sets = {
        "pm1_only          (subtraction sign only)": (0.0, 1.0, -1.0),
        "ints_half_no_pi   (±1,±2,±0.5)": base_no_pi,
        "REGISTERED (frozen default)": tuple(P.CANON_SET),
        "registered_no_pi": base_no_pi,
        "registered_plus_simple_rationals (+±1/3,±1/4,±2/3,±3/4)": tuple(P.CANON_SET) + thirds_quarters,
    }
    probs = SP.feynman_inner_problems()
    rows = []
    for name, cset in sets.items():
        admits = [C.decide(p, use_structural=True, canon_set=cset) for p in probs]
        ids = [v.id for v in admits if v.verdict == "ADMIT"]
        rows.append(dict(canon_set=name, n_canon=len(set(cset)), n_admit=len(ids), admit_ids=ids))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    corpus = json.loads((OUT / "corpus_manifest.json").read_text())["summary"]
    fey = json.loads((OUT / "feynman34.json").read_text())["summary"]
    payload = dict(
        thresholds=dict(TAU_LOW=P.TAU_LOW, TAU_HIGH=P.TAU_HIGH,
                        RECOVERY_TOL=P.RECOVERY_TOL, STRUCT_TOL=P.STRUCT_TOL,
                        GENERIC_REF_GRID=list(P.GENERIC_REF_GRID), N_SAMPLES=P.N_SAMPLES),
        validation_set=validation_set(),
        anticheat_demos=anticheat_demos(),
        tau_low_sensitivity=tau_low_sensitivity(),
        struct_tol_sensitivity=struct_tol_sensitivity(),
        canon_membership_sensitivity=canon_membership_sensitivity(),
        corpus_summary=corpus,
        feynman34_summary=fey,
    )
    (OUT / "report.json").write_text(json.dumps(payload, indent=2))

    vs = payload["validation_set"]
    print("VALIDATION SET:")
    for r in vs:
        ok = "OK " if r["match"] else "!! MISMATCH"
        print(f"  {ok} {r['id']:16s} got={r['verdict']:7s} exp={r['expected']:7s} ({r['reason_code']})")
    allok = all(r["match"] for r in vs)
    print(f"  -> all validation anchors match: {allok}")
    print(f"\nANTI-CHEAT: korns_8 faked R²={payload['anticheat_demos']['korns8_outer_absorption']['faked_R2_via_outer_absorption']:.4f} "
          f"-> {payload['anticheat_demos']['korns8_outer_absorption']['verdict']}; "
          f"trig-phase fixed-basis R²={payload['anticheat_demos']['trig_phase_linear_span']['R2_on_fixed_sin_cos_basis_no_CO']:.4f} "
          f"-> {payload['anticheat_demos']['trig_phase_linear_span']['verdict']}")
    print(f"τ_low stable across sweep: {all(r['stable'] for r in payload['tau_low_sensitivity'])}")
    print(f"STRUCT_TOL: -1/3 reldist to canon = {payload['struct_tol_sensitivity']['minus_third_nearest_canon_reldist']:.4f} (tol={P.STRUCT_TOL})")
    print(f"CORPUS: {corpus['corpus_size']} admitted ({corpus['n_construction_admitted']}/{corpus['n_construction_candidates']} candidates); controls leaked: {corpus['n_controls_leaked']}")
    print(f"FEYNMAN-34: {fey['n_positional_inner']} -> {fey['n_admit']} verified inner {fey['admit_ids']}")
    print("CANON-MEMBERSHIP sensitivity (survivor count swings with the reachability set):")
    for r in payload["canon_membership_sensitivity"]:
        print(f"    34 -> {r['n_admit']:2d}   {r['canon_set']:55s} admits={r['admit_ids']}")
    assert allok, "validation mismatch"
    assert corpus["n_controls_leaked"] == 0
    print("wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
