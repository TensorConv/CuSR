"""009 problem catalogue — the expanded problem set.

Three families, all expressed as 009 `Problem` records (see seed_bench.Problem)
with `n_inner_consts` auto-computed by taxonomy.count_inner_consts:

  - feynman : physics backbone, REUSED from bench (bench/sources/feynman.py,
              ~98 problems, AI-Feynman *original* sampling ranges). 34/98 carry
              inner constants. range_variant="ai_feynman_original".
              NOTE: n_inner_consts is relative to bench's literal->c
              parameterization (Scheme C), not an absolute property of the physics.
  - nguyen  : negative controls, REUSED from bench (bench/sources/synthetic.py,
              12 problems). Constant-free originals (n_consts=0, n_inner=0) — CO
              must not help here, else it's a false signal.
  - korns   : inner-constant magnifier, AUTHORED here. Constants buried in
              frequencies / decay rates (Korns 2011). See _KORNS below.

seed_bench.SEED (2 problems) stays as a fast smoke set for the engine; this
module is the real catalogue. Some ids overlap by construction.

SRSD *realistic* ranges and Strogatz are deferred (a later range_variant overlay
+ a new family); see README.
"""
from __future__ import annotations

import sympy as sp

from .seed_bench import Problem
from .taxonomy import count_inner_consts


# ---------------------------------------------------------------------------
# Feynman backbone (reuse bench)
# ---------------------------------------------------------------------------

def _feynman_catalogue() -> list[Problem]:
    from cusr.bench.sources.feynman import load_feynman

    out: list[Problem] = []
    for name, p in load_feynman().items():
        n_inner = count_inner_consts(p.skeleton_expr, p.constants)
        out.append(Problem(
            id=f"feynman_{name.replace('.', '_')}",
            source="feynman",
            true_expr=str(p.ground_truth_expr),
            n_vars=len(p.variables),
            var_ranges=tuple((float(lo), float(hi)) for lo, hi in p.sampling_ranges),
            n_consts=len(p.constants),
            n_inner_consts=n_inner,
            difficulty="untiered",          # SRSD difficulty tiers come with the realistic-range overlay
            is_control=False,
            range_variant="ai_feynman_original",
            notes=f"physical_vars={p.physical_vars}",
        ))
    return out


# ---------------------------------------------------------------------------
# Nguyen controls (reuse bench; constant-free originals)
# ---------------------------------------------------------------------------

def _nguyen_catalogue() -> list[Problem]:
    from cusr.bench.sources.synthetic import NGUYEN

    out: list[Problem] = []
    for name, p in NGUYEN.items():
        baked = p.ground_truth_expr  # constants substituted to their integer/unit gt
        lo, hi = p.sampling_range
        out.append(Problem(
            id=f"nguyen_{name}",
            source="nguyen",
            true_expr=str(baked),
            n_vars=len(p.variables),
            var_ranges=tuple((float(lo), float(hi)) for _ in p.variables),
            n_consts=0,                     # original Nguyen has no free constants
            n_inner_consts=0,
            difficulty="easy",
            is_control=True,
            range_variant="original",
            notes="Nguyen negative control (no free constants).",
        ))
    return out


# ---------------------------------------------------------------------------
# Korns inner-constant magnifier (authored)
# ---------------------------------------------------------------------------
# Skeleton form (constants as c-symbols) so n_inner_consts is well-defined; the
# true_expr is the baked version. We take the inner-constant subset {7,8,11,12}
# of Korns (2011) — the functions whose constants sit inside a nonlinear fn
# (decay rate, sqrt scale, frequency). Korns-1/4/5/6 are outer-only; 13/14/15
# use tan (singular) — skipped.
#
# Constants verified against the constant-optimization benchmark paper
# (arXiv:2412.02126, HTML table): Korns-7 = 213.81*(1-exp(-0.547237*x)),
# Korns-11 = 6.87 + 11*cos(7.23*x**3) (cube inside cos, confirmed). Korns-12 has
# NO cube (canonical 2 - 2.1*cos(9.8*x)*sin(1.3*x); the cube belongs to 11).
# Korns-8 constants (6.87/11/7.23) are recall + pattern-consistent (Korns reuses
# them across 8/11) — flagged in notes pending primary-source confirmation.
#
# RANGES are restricted from Korns' original U[-50,50] to the CO-benchmark
# convention ([-5,5] for trig/exp, [0.1,10] where a positive domain is needed),
# so functions stay real/finite and stay CO-testable (U[-50,50] aliases the
# frequencies into noise -> unrecoverable). Active vars are renamed x0,x1,...

_x = sp.symbols("x0 x1 x2 x3 x4")
_c = sp.symbols("c0 c1 c2 c3 c4")


def _korns_problem(id_, skel, consts, gt, ranges, *, difficulty="hard",
                   known_ceiling=False, notes=""):
    var_syms = tuple(sorted(skel.free_symbols - set(consts), key=lambda s: s.name))
    n_vars = len(var_syms)
    baked = skel.subs(dict(zip(consts, gt)))
    return Problem(
        id=id_,
        source="korns",
        true_expr=str(baked),
        n_vars=n_vars,
        var_ranges=tuple((float(lo), float(hi)) for lo, hi in ranges),
        n_consts=len(consts),
        n_inner_consts=count_inner_consts(skel, consts),
        difficulty=difficulty,
        is_control=False,
        range_variant="korns_restricted",
        known_ceiling=known_ceiling,
        notes=notes,
    )


def _korns_catalogue() -> list[Problem]:
    x0, x1, x2 = _x[0], _x[1], _x[2]
    c0, c1, c2, c3 = _c[0], _c[1], _c[2], _c[3]
    return [
        # Korns-7: 213.81*(1 - exp(-0.547237*x)) — inner decay rate.
        _korns_problem(
            "korns_7",
            c0 * (1 - sp.exp(c1 * x0)),
            (c0, c1), (213.80940889, -0.54723748542),
            ((0.1, 10.0),),
            notes="Korns-7 saturation curve; inner decay rate 0.547. x>0 domain.",
        ),
        # Korns-8 (6.87 + 11*sqrt(7.23*x0*x1*x2)) is DROPPED on purpose: its
        # "inner" 7.23 is absorbable — sqrt(7.23*prod) = sqrt(7.23)*sqrt(prod),
        # so 7.23 folds into the outer scale (the positional counter's upper-bound
        # over-count, made concrete). Not a genuine inner-constant magnifier.
        # Korns-11: 6.87 + 11*cos(7.23*x0**3) — inner freq with cube; known ceiling.
        _korns_problem(
            "korns_11",
            c0 + c1 * sp.cos(c2 * x0**3),
            (c0, c1, c2), (6.87, 11.0, 7.23),
            ((-5.0, 5.0),),
            known_ceiling=True,
            notes="Korns-11; inner freq 7.23 with cube -> severe aliasing, widely unsolved.",
        ),
        # Korns-12: 2 - 2.1*cos(9.8*x0)*sin(1.3*x1) — two inner frequencies.
        _korns_problem(
            "korns_12",
            c0 + c1 * sp.cos(c2 * x0) * sp.sin(c3 * x1),
            (c0, c1, c2, c3), (2.0, -2.1, 9.8, 1.3),
            ((-5.0, 5.0), (-5.0, 5.0)),
            notes="Korns-12 (2 active vars); inner freqs 9.8 & 1.3.",
        ),
    ]


# ---------------------------------------------------------------------------

_FAMILIES = {
    "feynman": _feynman_catalogue,
    "nguyen": _nguyen_catalogue,
    "korns": _korns_catalogue,
}


def load_catalogue(families=("feynman", "nguyen", "korns")) -> list[Problem]:
    out: list[Problem] = []
    for fam in families:
        out.extend(_FAMILIES[fam]())
    return out
