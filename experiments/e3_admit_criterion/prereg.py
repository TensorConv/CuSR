"""PRE-REGISTRATION (FROZEN) — admit/reject criterion for inner-constant problems.

  ████  DO NOT EDIT ANY VALUE IN THIS FILE AFTER THE FIRST `decide()` RUN.  ████

This module is the frozen contract for experiment-1 (the admit criterion) and
experiment-2 (the constructed corpus). Per the project anti-faking discipline,
every threshold, grid, canonical set, and the validation checklist are written
down HERE, BEFORE running, and are NOT to be tuned to make the validation set or
the Feynman-34 audit "pass". If the criterion does not separate the anchors, the
honest move is to FIX THE OPERATIONALIZATION (the R²_LS / fold / structural
mechanism) — never to nudge a τ. Any τ change after first-run must be accompanied
by a τ-sensitivity analysis showing the validation verdicts do not flip, and must
be recorded in the report as a deviation.

Design rationale (why three mechanisms, measured on the pilot — see pilot.py):
No single linear-scaling-reference scheme separates the anchors:
  * korns_7 (MUST ADMIT): a generic decay guess -0.5 ~ true -0.547 makes
    max-over-grid R²_LS = 0.999 -> max wrongly REJECTS it; worst-over-grid = 0.06.
  * I.8.14 (structural -1, SHOULD REJECT): the true -1 sits IN the grid, so
    worst-over-grid = 0.000 -> worst wrongly ADMITS it; max = 1.000.
These pull opposite ways, so the criterion uses three independent mechanisms,
each with a distinct, pre-registered justification:
  1. FOLD (symbolic, fires before any fitting) -> absorbable consts (korns_8:
     sqrt(c*prod) = sqrt(c)*sqrt(prod)). reveal_folds drops its inner count.
  1b. IDENTIFIABILITY (Jacobian rank at true constants == n_consts) -> catches
     non-identifiable consts (korns_8's c1*sqrt(c2) column degeneracy). This is
     also the SVD numeric-rank certificate the benchmark wants.
  2. STRUCTURAL-DISCRETE (separate, frozen CANON set; justified ONLY by
     GP-grammar reachability, never by fitting the anchors) -> grammar-reachable
     canonical constants (I.8.14's +/-1; physics pi-multiples). Staged AFTER the
     core is green, because it is the only mechanism with no must-reject anchor.
  3. GENERIC LINEAR-SCALING GAP: R²_LS = WORST over a per-constant reference
     product grid (OLS outer-affine, inner fixed) must be < TAU_LOW; AND
     R²_full > TAU_HIGH; AND inner constants recovered within RECOVERY_TOL.

Ground truth is fp64 + scipy + independent of any GPU kernel.
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (FROZEN)
# ─────────────────────────────────────────────────────────────────────────────

# SRBench accuracy-solution threshold, adopted verbatim (strategy doc §4.1, D4).
TAU_HIGH: float = 0.999

# Linear-scaling-gap threshold. Pre-registered in the strategy doc's 0.5–0.8
# band. Pilot WORST-scheme margins are huge: admits {korns_7=0.06, korns_11=0.009,
# korns_12=0.000} vs rejects {korns_8=1.000, outer-only=1.000}. 0.70 sits in the
# wide gap; τ-sensitivity (below) sweeps [0.10, 0.95] and verdicts must not flip.
TAU_LOW: float = 0.70

# Inner-constant recovery tolerance (relative). Task §1 anti-cheat: |ĉ-c|/|c|.
RECOVERY_TOL: float = 0.01

# Canonical-snap tolerance for the structural test: a true inner constant counts
# as "a canonical value" iff it is within this relative distance of a CANON entry.
# Looser than RECOVERY_TOL so near-canonical physics constants snap, but far
# tighter than korns_7's 8.6% distance from 0.5 (so korns_7 never snaps).
STRUCT_TOL: float = 0.02

# τ-sensitivity sweep: the validation verdicts must be invariant across this band.
TAU_LOW_SWEEP: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90, 0.95)

# ─────────────────────────────────────────────────────────────────────────────
# Generic linear-scaling reference grid (FROZEN)
# ─────────────────────────────────────────────────────────────────────────────
# R²_LS fixes each surviving inner constant at a GENERIC value (the kind an SR
# system proposes from an O(1) ephemeral-constant prior, WITHOUT nonlinear CO),
# fits the outer affine by OLS, and takes the WORST over the Cartesian product of
# this set across the inner constants. WORST is licensed by the fold/identifiability
# guarantee: an absorbable or outer constant yields R²=1 at EVERY reference (the
# basis shape is invariant to it), so WORST can only be low when the constant
# genuinely changes the function's shape. Magnitudes are O(1) on purpose: a true
# high-magnitude inner constant (e.g. a frequency 9.8) cannot be matched by any
# O(1) generic guess, which is exactly why such problems need nonlinear CO.
GENERIC_REF_GRID: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)

# Cap on the product-grid size (k inner consts -> len(grid)**k). Above this we
# fall back to the shared-scalar grid (same value for all inner consts) and LOG
# the fallback honestly in the per-problem record. With len=6 the cap is hit at
# k>=5 inner consts; none of the validation anchors reach it.
MAX_PRODUCT_GRID: int = 4096

# ─────────────────────────────────────────────────────────────────────────────
# Canonical (GP-grammar-reachable) constant set — STRUCTURAL test (FROZEN)
# ─────────────────────────────────────────────────────────────────────────────
# A constant whose TRUE value is one of these is reachable by the GP grammar
# WITHOUT nonlinear constant optimization, so its presence in a nonlinear
# position does not make the problem need inner CO:
#   * 0, ±1, ±2 : small-integer coefficients; ±1 in particular is what a
#     subtraction/negation node yields for free (the Scheme-C "literal→constant"
#     artifact, e.g. korns/feynman structural −1 inside (c·x + y)).
#   * ±0.5 : the reciprocal-of-2 / sqrt-exponent constant.
#   * π-multiples : π is a standard grammar/terminal constant in SR systems and
#     the only transcendental literal pervasive in the Feynman set (phases, 2π
#     frequencies, 1/(2π) prefactors).
# This set is justified by reachability ALONE and is FROZEN before the Feynman-34
# audit. It is NEVER to be edited to change a survivor count.
_PI = float(np.pi)
CANON_SET: tuple[float, ...] = (
    0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
    _PI, -_PI, 2.0 * _PI, -2.0 * _PI, _PI / 2.0, -_PI / 2.0,
    1.0 / _PI, -1.0 / _PI, 1.0 / (2.0 * _PI), -1.0 / (2.0 * _PI),
)

# ─────────────────────────────────────────────────────────────────────────────
# Full-CO certification seeding policy (FROZEN) — addresses the korns_11 blocker
# ─────────────────────────────────────────────────────────────────────────────
# For benchmark CONSTRUCTION, R²_full and recovery CERTIFY that the true
# constants are the global optimum that fits the (noise-free) data — they do NOT
# claim a blind solver finds them. cos(7.23·x³) on [−5,5] spans ~140 periods, so
# random multistart cannot find 7.23 (korns_11 is a "known ceiling"); a blind
# full-CO would wrongly reject this MUST-ADMIT anchor. We therefore seed the LM
# with the ground-truth constants (legitimate: the true skeleton is known at
# construction time) plus a scale grid, and additionally run random starts purely
# to REPORT blind-solver difficulty as a separate, non-gating field. Every
# per-problem record LABELS which seeds were used and reports both the seeded
# (certifying) and random (difficulty) R². Silent GT-seeding is forbidden.
FULLCO_GT_SEED: bool = True            # seed at GT (construction certification)
FULLCO_GT_JITTER: float = 0.10         # + small relative jitter around GT
FULLCO_N_GT_JITTER: int = 4
FULLCO_N_RANDOM: int = 16              # random starts — REPORTED, never gating
FULLCO_SCALE_GRID: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 10.0)  # |const| scales
FULLCO_MAX_NFEV: int = 4000

# Blind-difficulty diagnostic (NON-GATING; never affects a verdict). R²_full_random
# is the MEDIAN best-R² of random-start multistart over these seeds — a median (not
# a single seed) because blind success is stochastic: a lone seed can occasionally
# land in the true-frequency basin (measured: korns_11 solved 0/12 seeds, but one
# stray seed hit R²=1). The random starts are gt-MAGNITUDE-scaled on purpose — the
# CHARITABLE blind baseline (solver given the right magnitudes), so a failure
# isolates multimodal inner-constant difficulty, not a missing magnitude prior.
FULLCO_BLIND_SEEDS: tuple[int, ...] = (0, 1, 2)

# ─────────────────────────────────────────────────────────────────────────────
# Identifiability (FROZEN)
# ─────────────────────────────────────────────────────────────────────────────
# Reject if rank(J) < n_consts at the true constants, where J is the NLS Jacobian
# (∂model/∂cᵢ) column-normalized, rank by SVD with this relative cutoff on the
# largest singular value. Catches non-identifiable / absorbable constants whose
# columns are linearly dependent (korns_8: ∂/∂c2 ∝ ∂/∂c1).
IDENTIFIABILITY_RCOND: float = 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# Sampling (FROZEN)
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLES: int = 2000
DATA_SEED: int = 0          # deterministic (X, y) per problem
MULTISTART_SEED: int = 12345  # deterministic LM start jitter / random starts

# ─────────────────────────────────────────────────────────────────────────────
# Validation checklist (FROZEN) — the objective acceptance anchors (task §1)
# ─────────────────────────────────────────────────────────────────────────────
# verdict ∈ {ADMIT, REJECT}; rows tagged EDGE carry the expected post-structural
# verdict with its principled reason, but are NOT hard gates for the CORE (they
# only become gates once the structural mechanism is added in Phase C).
VALIDATION: dict[str, dict] = {
    # MUST ADMIT — genuine inner constants, cross-checked vs arXiv:2412.02126.
    "korns_7":  {"expect": "ADMIT",  "hard": True,
                 "why": "inner decay rate 0.547 (saturation curve); not canonical, not absorbable"},
    "korns_11": {"expect": "ADMIT",  "hard": True,
                 "why": "inner frequency 7.23 with cube; aliasing -> needs CO; GT-seeded full-CO certifies"},
    "korns_12": {"expect": "ADMIT",  "hard": True,
                 "why": "two inner frequencies 9.8 & 1.3"},
    # MUST REJECT — anti-cheat / negative controls.
    "korns_8":  {"expect": "REJECT", "hard": True,
                 "why": "absorbable sqrt scale: sqrt(c·prod)=sqrt(c)·sqrt(prod); fold + rank-deficient"},
    "nguyen_1": {"expect": "REJECT", "hard": True,
                 "why": "constant-free polynomial; no inner constant"},
    # Two outer-only Feynman (n_inner_consts==0) — representative negative controls.
    "feynman_I.11.19": {"expect": "REJECT", "hard": True,
                        "why": "outer-only (no positional inner constant)"},
    "feynman_I.12.1":  {"expect": "REJECT", "hard": True,
                        "why": "outer-only / trivial product"},
    # EDGE — not pre-labeled as a hard gate; expected verdict after Phase C with reason.
    "feynman_I.8.14": {"expect": "REJECT", "hard": False, "edge": True,
                       "why": "structural ±1 (gt c0=c1=-1 ∈ CANON): (c·x+y) with c=-1 is a subtraction, "
                              "grammar-reachable, not nonlinear CO. Scheme-C literal→constant inflation."},
}

# ─────────────────────────────────────────────────────────────────────────────
# Construction grid for experiment-2 (FROZEN, pre-registered before running)
# ─────────────────────────────────────────────────────────────────────────────
# Systematic difficulty axes. Domains chosen to keep y real/finite and to AVOID
# frequency aliasing (high freqs on a narrow range) and poles (shifts). Each
# entry: (family, skeleton-template, inner-constant value grid, domain). The
# constructor (build_corpus.py) materialises one problem per (template × value).
CONSTRUCTION_GRID: dict[str, dict] = {
    # Frequency: cos(c·x) and sin(c·x^k). Range kept small so c·x^k does not alias.
    "freq_cos_x":   {"skeleton": "c0*cos(c1*x0)",          "inner": "c1",
                     "values": (1.5, 3.0, 5.0, 7.0),       "domain": ((-3.0, 3.0),)},
    "freq_sin_x2":  {"skeleton": "c0*sin(c1*x0**2)",       "inner": "c1",
                     "values": (1.0, 2.0, 3.0),            "domain": ((-2.5, 2.5),)},
    "freq_cos_x3":  {"skeleton": "c0*cos(c1*x0**3)",       "inner": "c1",
                     "values": (1.0, 2.0),                 "domain": ((-2.0, 2.0),)},
    # Decay: exp(-c·x), positive domain.
    "decay_exp":    {"skeleton": "c0*exp(-c1*x0)",         "inner": "c1",
                     "values": (0.4, 0.9, 1.7, 3.0),       "domain": ((0.1, 5.0),)},
    "saturation":   {"skeleton": "c0*(1-exp(-c1*x0))",     "inner": "c1",
                     "values": (0.55, 1.3, 2.5),           "domain": ((0.1, 8.0),)},
    # Shift: (x-c)^2 and 1/(x-c) with the pole kept out of the sampling range.
    "shift_quad":   {"skeleton": "c0*(x0-c1)**2",          "inner": "c1",
                     "values": (1.3, 2.7, 4.1),            "domain": ((-5.0, 5.0),)},
    "shift_recip":  {"skeleton": "c0/(x0-c1)",             "inner": "c1",
                     "values": (6.5, 8.0, 9.5),            "domain": ((-4.0, 4.0),)},  # pole > hi
    # Power: x^c, positive domain.
    "power":        {"skeleton": "c0*x0**c1",              "inner": "c1",
                     "values": (1.3, 2.4, 3.7),            "domain": ((0.2, 4.0),)},
    # Damped pendulum: exp(-c0·x)·cos(c1·x) — exp×cos combo absent from the library.
    "damped":       {"skeleton": "c0*exp(-c1*x0)*cos(c2*x0)", "inner": "c1,c2",
                     "values_2d": ((0.4, 4.0), (0.7, 6.0), (1.2, 9.0)), "domain": ((0.1, 6.0),)},
}

# Negative controls that MUST be rejected by the criterion when mixed into the
# experiment-2 candidate stream (evidence the corpus is not cherry-picked).
CONSTRUCTION_NEGATIVE_CONTROLS: tuple[str, ...] = (
    "nguyen_1", "nguyen_2", "nguyen_3", "nguyen_4", "nguyen_5",  # constant-free
    "korns_8",                                                    # absorbable sqrt
    # plus outer-only Feynman, filled in by id at run time (n_inner_consts==0).
)
N_OUTER_ONLY_FEYNMAN_CONTROLS: int = 5

# Feynman-34 audit (§2.5): every positional-inner Feynman is run; the survivor
# count tightens the positional upper bound. The expected list is computed at run
# time from count_inner_consts (== 34) and reported in full, admit and reject.
