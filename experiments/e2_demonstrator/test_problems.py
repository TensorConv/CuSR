"""TDD for the 009 problem catalogue (`problems.py`).

Catalogue = Feynman backbone (reused from bench, AI-Feynman original ranges) +
Nguyen controls (reused, constant-free) + Korns inner-constant magnifier
(authored). Every problem is auto-annotated with n_inner_consts via taxonomy.

The load-bearing tests: every problem generates finite y on its declared ranges
(catches domain bugs like log/tan over a bad interval), controls have zero inner
constants, and the Feynman inner-constant distribution matches what was measured.

Run:  uv run pytest experiments/009_sr_benchmark/test_problems.py -v
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from cusr.demonstrator import problems as P
from cusr.demonstrator import seed_bench as sb


def test_catalogue_loads_and_ids_unique():
    cat = P.load_catalogue()
    assert len(cat) >= 110
    ids = [p.id for p in cat]
    assert len(ids) == len(set(ids)), "duplicate problem ids"


def test_families_present_with_expected_counts():
    cat = P.load_catalogue()
    by_source = Counter(p.source for p in cat)
    assert by_source["feynman"] == 98
    assert by_source["nguyen"] == 12
    assert by_source["korns"] >= 1


def test_controls_have_zero_inner_consts():
    for p in P.load_catalogue():
        if p.is_control:
            assert p.n_inner_consts == 0, f"{p.id}: control should have 0 inner consts"


def test_nguyen_are_controls_and_constant_free():
    nguyen = [p for p in P.load_catalogue() if p.source == "nguyen"]
    assert len(nguyen) == 12
    for p in nguyen:
        assert p.is_control is True
        assert p.n_consts == 0


def test_feynman_carries_inner_constant_signal():
    feyn = [p for p in P.load_catalogue() if p.source == "feynman"]
    n_with_inner = sum(1 for p in feyn if p.n_inner_consts >= 1)
    assert n_with_inner >= 30, f"only {n_with_inner} Feynman with inner consts — counter or corpus changed"


def test_feynman_inner_distribution_locked():
    feyn = [p for p in P.load_catalogue() if p.source == "feynman"]
    dist = Counter(p.n_inner_consts for p in feyn)
    assert dict(dist) == {0: 64, 1: 10, 2: 19, 3: 5}


def test_korns_are_inner_constant_problems():
    korns = [p for p in P.load_catalogue() if p.source == "korns"]
    assert all(not p.is_control for p in korns)
    # the magnifier's whole point: at least some Korns have inner constants
    assert any(p.n_inner_consts >= 1 for p in korns)


def test_all_true_exprs_parse():
    import sympy as sp
    for p in P.load_catalogue():
        syms = {f"x{i}": sp.Symbol(f"x{i}") for i in range(p.n_vars)}
        expr = sp.sympify(p.true_expr, locals=syms)
        free = {s.name for s in expr.free_symbols}
        assert free <= set(syms), f"{p.id}: stray symbols {free - set(syms)}"


def test_all_problems_generate_finite_y():
    """The domain guard: every problem must produce finite y on its ranges."""
    bad = []
    for p in P.load_catalogue():
        try:
            X, y = sb.generate(p, seed=0, n_samples=64)
            if X.shape != (64, p.n_vars) or y.shape != (64,) or not np.all(np.isfinite(y)):
                bad.append(p.id)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{p.id}:{type(e).__name__}")
    assert not bad, f"problems with bad generation: {bad}"


def test_var_ranges_length_matches_n_vars():
    for p in P.load_catalogue():
        assert len(p.var_ranges) == p.n_vars, f"{p.id}: var_ranges/n_vars mismatch"


def test_feynman_inner_consts_are_not_absorbable_folds():
    """Axis-correctness, not just count-stability: every Feynman constant the
    positional counter calls inner must survive fold-revealing simplification
    (no sqrt/log/exp-additive folds that are really outer scales — the korns_8
    signature). Verified 0 folds; this locks it."""
    from cusr.bench.sources.feynman import load_feynman
    from cusr.demonstrator.taxonomy import count_inner_consts, reveal_folds

    folds = []
    for name, prob in load_feynman().items():
        skel = count_inner_consts(prob.skeleton_expr, prob.constants)
        if skel < 1:
            continue
        revealed = count_inner_consts(reveal_folds(prob.skeleton_expr), prob.constants)
        if revealed < skel:
            folds.append((name, skel, revealed))
    assert not folds, f"absorbable-fold inner constants found: {folds}"
