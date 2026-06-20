"""PRE-REGISTRATION (FROZEN) — multi-inner-constant EXTENSION corpus for Study B.

  ████  DO NOT EDIT ANY VALUE AFTER THE FIRST build_extension RUN.  ████

WHY THIS EXISTS. The e3 corpus has only 3 multi-inner ADMITs, all one near-clone
family (1.7*exp(-a*x0)*cos(b*x0)). Claim 2's "largest gains on problems requiring
MULTIPLE inner constants" is therefore underpowered AND non-diverse. This file
pre-registers a STRUCTURALLY DIVERSE multi-inner extension, to be run through the
**UNCHANGED** e3 admit criterion (same TAU_LOW/TAU_HIGH, CANON_SET, STRUCT_TOL,
RECOVERY_TOL, seeds — all imported from e3, never re-defined here). Honesty rules,
verbatim from e3:
  - Freeze the grid BEFORE running decide() even once.
  - Run the FROZEN criterion unchanged; the instant you widen CANON to admit a
    value you have broken pre-registration.
  - Keep only ADMITs; REPORT EVERY REJECT with its reason code (a reject IS the
    anti-cheat evidence, like e3's 7 self-rejected freq/shift candidates).
  - No iterate-until-admit.

DESIGN — each family puts 2 inner constants in DIFFERENT structural roles, and
each inner constant was reasoned (BEFORE freezing) to independently survive the
three structural detectors: (i) identifiability — adds a Jacobian-rank direction
beyond the outer scale; (ii) structural — value is non-canonical (> STRUCT_TOL
from {0,±1,±2,±0.5,π-mult,±1/π,±1/2π}); (iii) fold — not absorbable into a fixed
inner-free linear span. Domains avoid aliasing (freq×range ≲ 3 periods) and poles.
Whatever the criterion actually decides is reported as-is.
"""
from __future__ import annotations

# Frozen e3 thresholds/seeds — IMPORTED, never re-defined, so "unchanged criterion"
# is enforced by construction. (Referenced here only to make the dependency explicit.)
from experiments.e3_admit_criterion.prereg import (  # noqa: F401
    CANON_SET, DATA_SEED, MULTISTART_SEED, N_SAMPLES,
    RECOVERY_TOL, STRUCT_TOL, TAU_HIGH, TAU_LOW,
)

# ─────────────────────────────────────────────────────────────────────────────
# Multi-inner construction grid (FROZEN). Each entry: skeleton-template with free
# c-constants, the comma-separated INNER constant names, the (value-pair) grid,
# and the sampling domain. Outer constants default to 1.7 (as in e3).
# ─────────────────────────────────────────────────────────────────────────────
EXTENSION_GRID: dict[str, dict] = {
    # (1) Two incommensurate frequencies, both inner. Not absorbable: angle-product
    #     expands to ½[sin((c1+c2)x)+sin((c1-c2)x)] — the sum/diff frequencies are
    #     themselves non-canonical, so no fixed inner-free trig basis fits. c1≠c2.
    "freq_product": {"skeleton": "c0*sin(c1*x0)*cos(c2*x0)", "inner": "c1,c2",
                     "values_2d": ((1.3, 2.7), (0.7, 1.9), (2.3, 3.7)),
                     "domain": ((-4.0, 4.0),)},
    # (2) Gaussian WIDTH (squared decay) + FREQUENCY — distinct mechanism from the
    #     e3 damped family's linear decay exp(-c1*x). c1>0 width, c2 frequency.
    "gauss_freq": {"skeleton": "c0*exp(-c1*x0**2)*cos(c2*x0)", "inner": "c1,c2",
                   "values_2d": ((0.3, 2.7), (0.7, 1.9), (0.4, 3.3)),
                   "domain": ((-3.5, 3.5),)},
    # (3) Inner constants in DIFFERENT variables: frequency in x0, decay in x1.
    "twovar_df": {"skeleton": "c0*sin(c1*x0)*exp(-c2*x1)", "inner": "c1,c2",
                  "values_2d": ((1.3, 0.7), (2.3, 0.9)),
                  "domain": ((-3.0, 3.0), (0.1, 4.0))},
    # (4) Lorentzian: WIDTH c1 + CENTER c2, both inner, in a rational. Center sits
    #     inside (x-c2)² in a DENOMINATOR -> not the absorbable polynomial-shift
    #     case; width and center are independent DOF. Denominator c1+(x-c2)²≥c1>0.
    "lorentzian": {"skeleton": "c0/(c1 + (x0-c2)**2)", "inner": "c1,c2",
                   "values_2d": ((0.8, 1.3), (1.4, 2.3)),
                   "domain": ((-4.0, 6.0),)},
    # (5) Logistic: SLOPE c1 + CENTER c2, both inner inside the sigmoid.
    "sigmoid": {"skeleton": "c0/(1 + exp(-c1*(x0-c2)))", "inner": "c1,c2",
                "values_2d": ((1.3, 0.7), (2.3, 1.9)),
                "domain": ((-3.0, 5.0),)},
}

# In-family canonical-valued siblings that SHOULD reject (structural): same
# skeletons with inner constants set to grammar-reachable canonical values. These
# preserve the e3 validity-control structure (CO must NOT unlock a control) per
# NEW family, not only the global nguyen/feynman controls.
EXTENSION_REJECT_CONTROLS: dict[str, dict] = {
    "freq_product_canon": {"skeleton": "c0*sin(c1*x0)*cos(c2*x0)", "inner": "c1,c2",
                           "values_2d": ((1.0, 2.0),), "domain": ((-4.0, 4.0),)},
    "gauss_freq_canon": {"skeleton": "c0*exp(-c1*x0**2)*cos(c2*x0)", "inner": "c1,c2",
                         "values_2d": ((0.5, 2.0),), "domain": ((-3.5, 3.5),)},
}

# Power floor (FROZEN): the minimum number of SURVIVING multi-inner ADMIT problems
# (original e3 3 + extension admits) required before ANY inferential Claim-2
# multi-inner p-value may be reported. Below this, report per-problem deltas
# descriptively only. Set so a Wilcoxon on near-clones cannot recur silently.
POWER_FLOOR_MIN_MULTI: int = 8

# Cap-interaction note (FROZEN expectation): all extension skeletons have
# count_ops well under 40 and n_constants == 3 (< MAX_K 32), so they are NOT
# CappedCO-skipped in Study B. Verified empirically in build_extension's report.
