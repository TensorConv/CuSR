from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cusr.bench.dataset import Dataset
from cusr.bench.skeleton import Skeleton
from cusr.bench.sources.feynman import GATE_EXCLUDE_IDS, FeynmanSource, load_feynman, select_problems


@pytest.fixture
def feynman_env(tmp_path: Path):
    return tmp_path, tmp_path / "registry.yaml"


def test_catalogue_shape():
    catalogue = load_feynman()
    assert len(catalogue) >= 90, f"expected ≥90 formulas after R2+R4' filter, got {len(catalogue)}"
    for name, prob in catalogue.items():
        assert prob.dataset_id == f"feynman/{name}"
        assert len(prob.constants) >= 1, f"{name}: scheme C requires ≥1 constant"
        assert len(prob.sampling_ranges) == len(prob.variables), name
        for lo, hi in prob.sampling_ranges:
            assert hi > lo, name
        assert prob.ground_truth_constants.shape == (len(prob.constants),), name


def test_default_select_excludes_gate_ids_only():
    """Default FeynmanSource drops GATE_EXCLUDE_IDS but keeps everything else —
    no dimensionality cap, since LM cost is driven by K (not n_vars) and
    statistics are segmented by n_vars in reports, not pre-filtered."""
    default_ids = select_problems(exclude_ids=GATE_EXCLUDE_IDS)
    catalogue = load_feynman()
    assert len(default_ids) == len(catalogue) - len(GATE_EXCLUDE_IDS)
    for excluded in GATE_EXCLUDE_IDS:
        assert excluded not in default_ids


def test_max_vars_filter_works_when_opted_in():
    ids = select_problems(max_vars=4)
    catalogue = load_feynman()
    for name in ids:
        assert len(catalogue[name].variables) <= 4


def test_runtime_filter_max_constants():
    few_const = select_problems(max_constants=2)
    catalogue = load_feynman()
    for name in few_const:
        assert len(catalogue[name].constants) <= 2


def test_include_ids_overrides_filter():
    ids = select_problems(include_ids=("I.14.4", "I.12.1"))
    assert set(ids) == {"I.14.4", "I.12.1"}


def test_materialize_subset_roundtrips(feynman_env):
    data_root, registry_path = feynman_env
    subset = ("I.6.2a", "I.14.4", "I.13.4")
    src = FeynmanSource(
        data_root=data_root,
        registry_path=registry_path,
        problems=subset,
        n_perturbed_per_problem=1,
        seed=0,
    )
    src._ensure()
    catalogue = load_feynman()
    for name in subset:
        assert (data_root / "feynman" / f"{name}.parquet").exists()
        ds = Dataset.from_registry(f"feynman/{name}", registry_path)
        X, y = ds.load()
        prob = catalogue[name]
        assert X.shape == (prob.n_samples, len(prob.variables))
        assert y.shape == (prob.n_samples,)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))


def test_ground_truth_matches_skeleton_eval(feynman_env):
    data_root, registry_path = feynman_env
    name = "I.14.4"  # 1/2·k·x² — one c (0.5), two vars
    FeynmanSource(
        data_root=data_root,
        registry_path=registry_path,
        problems=(name,),
    )._ensure()
    prob = load_feynman()[name]
    ds = Dataset.from_registry(prob.dataset_id, registry_path)
    X, y = ds.load()
    skel = Skeleton(expr=prob.skeleton_expr, variables=prob.variables, constants=prob.constants)
    y_pred = skel.evaluate(prob.ground_truth_constants, X)
    np.testing.assert_allclose(y_pred, y, rtol=1e-10)


def test_skeleton_sub_gt_matches_formula_original():
    """Regression test for the simultaneous=True sympy subs bug.

    build_feynman_formulas.py:remap_variables previously used
    ``expr.subs(subs)`` without ``simultaneous=True`` — a cascading dict sub
    that silently corrupts physics where physical-variable names collide
    with the target x0..xN namespace. Three parquets (I.9.18 gravity drops
    G, I.29.16 cos(θ1−θ2) → cos(x1−x3), I.50.26 cos(ω·t) → cos(x0·x2))
    fit the wrong formula; two more (I.8.14, I.11.19) were accidentally
    correct under current sympy but fragile to upgrade.

    Assertion: for every Feynman formula, evaluating skeleton.subs(GT) at
    sample points must produce the same y as evaluating the original
    physics formula at those same sample points (via physical_vars
    mapping). A bug in parameterize() or remap_variables() will break
    this, where the older tautological round-trip test would not.
    """
    import sympy as sp

    catalogue = load_feynman()
    rng = np.random.default_rng(0)
    max_rel_err = 0.0
    failures = []

    for name, prob in catalogue.items():
        physical = {physical_name: f"x{i}" for i, physical_name in enumerate(
            # parse physical_vars "x0=theta0, x1=theta1, ..." → theta0, theta1, ...
            pair.split("=")[1].strip() for pair in prob.physical_vars.split(",")
        )}
        orig_expr = sp.sympify(
            prob.formula_original.replace("ln(", "log("),
            locals={p: sp.Symbol(p) for p in physical},
        )
        orig_subs = {sp.Symbol(p): sp.Symbol(new) for p, new in physical.items()}
        orig_in_xspace = orig_expr.subs(orig_subs, simultaneous=True)

        n = 32
        X = np.empty((n, len(prob.variables)))
        for i, (lo, hi) in enumerate(prob.sampling_ranges):
            X[:, i] = rng.uniform(lo, hi, size=n)

        f_orig = sp.lambdify(prob.variables, orig_in_xspace, modules="numpy")
        y_orig = np.asarray(f_orig(*(X[:, i] for i in range(X.shape[1]))), dtype=float)

        skel = Skeleton(expr=prob.skeleton_expr, variables=prob.variables, constants=prob.constants)
        y_skel = skel.evaluate(prob.ground_truth_constants, X)

        if not np.all(np.isfinite(y_orig)) or not np.all(np.isfinite(y_skel)):
            failures.append((name, "non-finite y"))
            continue
        denom = np.maximum(np.abs(y_orig), 1e-12)
        rel_err = np.max(np.abs(y_orig - y_skel) / denom)
        max_rel_err = max(max_rel_err, float(rel_err))
        if rel_err > 1e-8:
            failures.append((name, f"rel_err={rel_err:.2e}"))

    assert not failures, f"skeleton.subs(GT) ≠ formula_original for: {failures}"


def test_iter_requests_produces_fit_requests(feynman_env):
    data_root, registry_path = feynman_env
    src = FeynmanSource(
        data_root=data_root,
        registry_path=registry_path,
        problems=("I.12.5", "I.14.3"),
        n_perturbed_per_problem=2,
        init_noise_sigma=0.1,
        seed=1,
    )
    reqs = list(src.iter_requests())
    assert len(reqs) == 4
    for r in reqs:
        assert r.skeleton.n_constants >= 1
        assert r.init_constants.shape == (r.skeleton.n_constants,)
        assert r.X.shape[0] == r.y.shape[0]
        assert r.source["origin"] == "feynman"


def test_prod_registry_has_relative_noise_variants():
    import yaml
    reg_path = Path(__file__).resolve().parents[2] / "experiments/003_nls_bench/data/registry.yaml"
    if not reg_path.exists():
        pytest.skip("prod registry not materialized")
    datasets = yaml.safe_load(reg_path.read_text())["datasets"]
    catalogue = load_feynman()
    for name in catalogue:
        for label, sigma in (("0p1pct", 0.001), ("01pct", 0.01)):
            variant_id = f"feynman/{name}-noisy-{label}"
            assert variant_id in datasets, variant_id
            t = datasets[variant_id]["transform"]
            assert t["type"] == "relative_gaussian_noise", (variant_id, t)
            assert t["sigma"] == sigma, (variant_id, t)
