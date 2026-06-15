from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from cusr.bench.restart import BestOfN, NoRestart, RandomPerturb
from cusr.bench.skeleton import FitRequest, Skeleton


class _StubDataset:
    """Minimal dataset stand-in — restart strategies don't touch dataset content."""
    def load(self):
        return np.zeros((1, 1)), np.zeros(1)


def _make_request(init: np.ndarray, k: int = 2) -> FitRequest:
    x0, x1 = sp.symbols("x0 x1")
    cs = sp.symbols(f"c0:{k}")
    expr = sum(c * x0**i for i, c in enumerate(cs))
    skel = Skeleton(expr=expr, variables=(x0, x1), constants=tuple(cs))
    return FitRequest(
        skeleton=skel,
        init_constants=np.asarray(init, dtype=float),
        dataset=_StubDataset(),
    )


# ---- NoRestart ----

def test_norestart_yields_single_init_verbatim():
    req = _make_request([3.14, -0.5])
    inits = list(NoRestart()(req, seed=0))
    assert len(inits) == 1
    np.testing.assert_array_equal(inits[0], [3.14, -0.5])


def test_norestart_name():
    assert NoRestart().name == "no_restart"


# ---- RandomPerturb ----

def test_random_perturb_first_trial_is_base():
    req = _make_request([1.0, 2.0])
    strat = RandomPerturb(n=3, scale=0.1)
    inits = list(strat(req, seed=42))
    assert len(inits) == 3
    np.testing.assert_array_equal(inits[0], [1.0, 2.0])


def test_random_perturb_trials_differ_from_base_on_average():
    req = _make_request([1.0, 2.0])
    strat = RandomPerturb(n=10, scale=0.5)
    inits = list(strat(req, seed=0))
    # Trial 0 is exact base; later trials should differ.
    diffs = [np.linalg.norm(inits[i] - np.asarray([1.0, 2.0])) for i in range(1, 10)]
    assert all(d > 0 for d in diffs), "perturbed trials should differ from base"
    assert np.mean(diffs) > 0.1, "perturbed trials should be meaningfully different"


def test_random_perturb_deterministic_with_seed():
    req = _make_request([0.0, 0.0])
    a = list(RandomPerturb(n=5, scale=1.0)(req, seed=7))
    b = list(RandomPerturb(n=5, scale=1.0)(req, seed=7))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_random_perturb_different_seeds_give_different_draws():
    req = _make_request([0.0, 0.0])
    a = list(RandomPerturb(n=5, scale=1.0)(req, seed=1))
    b = list(RandomPerturb(n=5, scale=1.0)(req, seed=2))
    # Trial 0 is always base (equal); trials 1+ must differ.
    assert not np.array_equal(a[1], b[1])


def test_random_perturb_shape_matches_constants():
    req = _make_request([1.0, 2.0, 3.0, 4.0], k=4)
    for init in RandomPerturb(n=3, scale=0.1)(req, seed=0):
        assert init.shape == (4,)


def test_random_perturb_name_encodes_config():
    assert RandomPerturb(n=5, scale=0.3).name == "random_perturb_n5_s0.3"


def test_random_perturb_validates_n():
    with pytest.raises(ValueError, match="n must be"):
        RandomPerturb(n=0, scale=0.1)


def test_random_perturb_validates_scale():
    with pytest.raises(ValueError, match="scale must be"):
        RandomPerturb(n=3, scale=-0.1)


# ---- BestOfN ----

def test_best_of_n_ignores_base_init():
    req = _make_request([999.0, -999.0])
    strat = BestOfN(n=5, scale=1.0)
    inits = list(strat(req, seed=0))
    assert len(inits) == 5
    # None should match the base (probability zero under continuous draw).
    for init in inits:
        assert not np.array_equal(init, [999.0, -999.0])


def test_best_of_n_shape_matches_n_constants():
    req = _make_request([0.0, 0.0, 0.0], k=3)
    strat = BestOfN(n=4, scale=1.0)
    inits = list(strat(req, seed=0))
    for init in inits:
        assert init.shape == (3,)


def test_best_of_n_deterministic():
    req = _make_request([0.0, 0.0])
    a = list(BestOfN(n=5, scale=1.0)(req, seed=13))
    b = list(BestOfN(n=5, scale=1.0)(req, seed=13))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_best_of_n_distribution_variance_matches_scale():
    """Gaussian draws should have std ≈ scale; larger scale → bigger spread."""
    req = _make_request([0.0, 0.0])
    small = np.stack(list(BestOfN(n=200, scale=0.1)(req, seed=1)))
    large = np.stack(list(BestOfN(n=200, scale=2.0)(req, seed=1)))
    # Empirical std should be within ~30% of nominal with n=200.
    assert 0.07 < small.std() < 0.13
    assert 1.7 < large.std() < 2.3


def test_best_of_n_uniform_distribution():
    req = _make_request([0.0, 0.0])
    strat = BestOfN(n=500, scale=1.5, distribution="uniform")
    inits = np.stack(list(strat(req, seed=0)))
    # Uniform(-scale, scale) → all in range.
    assert np.all(inits >= -1.5 - 1e-12)
    assert np.all(inits <= 1.5 + 1e-12)
    # Should populate both signs roughly evenly.
    assert (inits > 0).mean() > 0.4
    assert (inits < 0).mean() > 0.4


def test_best_of_n_name_encodes_config():
    assert BestOfN(n=5, scale=1.0).name == "best_of_5_gaussian_s1"
    assert BestOfN(n=10, scale=0.5, distribution="uniform").name == "best_of_10_uniform_s0.5"


def test_best_of_n_validates_inputs():
    with pytest.raises(ValueError, match="n must be"):
        BestOfN(n=0)
    with pytest.raises(ValueError, match="scale must be"):
        BestOfN(n=3, scale=0.0)
    with pytest.raises(ValueError, match="distribution must be"):
        BestOfN(n=3, scale=1.0, distribution="cauchy")


# ---- Runner integration ----

def test_runner_tags_restart_idx_and_strategy(tmp_path):
    """End-to-end: Runner + BestOfN should emit N records per request, each with
    a distinct restart_idx and the strategy name."""
    from cusr.bench.backends.scipy_common import ScipyLSBackend
    from cusr.bench.dataset import Dataset
    from cusr.bench.runner import Runner
    from cusr.bench.sources.synthetic import SyntheticSource

    # Use a canned Nguyen-style synthetic source.
    source = SyntheticSource(
        data_root=tmp_path,
        registry_path=tmp_path / "registry.yaml",
        problems=("1",),
        n_perturbed_per_problem=1,
        seed=0,
    )

    runner = Runner(
        source=source,
        backends=[ScipyLSBackend(method="lm")],
        seeds=[0],
        output_dir=tmp_path / "runs",
        restart_strategy=BestOfN(n=3, scale=0.5),
        name="restart_smoke",
        registry_path=tmp_path / "registry.yaml",
    )
    run_dir = runner.run()

    import json as _json
    records = []
    with open(run_dir / "records.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_json.loads(line))

    # 1 problem × 1 perturbation × 1 backend × 1 seed × 3 restarts = 3 records.
    assert len(records) == 3
    assert sorted(r["restart_idx"] for r in records) == [0, 1, 2]
    assert all(r["restart_strategy"].startswith("best_of_3_gaussian") for r in records)
    # Trials should have different init_constants (BestOfN draws fresh each time).
    inits = [tuple(r["init_constants"]) for r in records]
    assert len(set(inits)) == 3


def test_runner_norestart_default(tmp_path):
    """Without an explicit restart_strategy, Runner should default to NoRestart
    and produce exactly one record per (request, backend, seed)."""
    from cusr.bench.backends.scipy_common import ScipyLSBackend
    from cusr.bench.runner import Runner
    from cusr.bench.sources.synthetic import SyntheticSource

    source = SyntheticSource(
        data_root=tmp_path,
        registry_path=tmp_path / "registry.yaml",
        problems=("1",),
        n_perturbed_per_problem=2,
        seed=0,
    )
    runner = Runner(
        source=source,
        backends=[ScipyLSBackend(method="lm")],
        seeds=[0],
        output_dir=tmp_path / "runs",
        name="norestart_smoke",
        registry_path=tmp_path / "registry.yaml",
    )
    run_dir = runner.run()

    import json as _json
    records = []
    with open(run_dir / "records.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_json.loads(line))

    # 1 problem × 2 perturbations × 1 backend × 1 seed × 1 restart = 2 records.
    assert len(records) == 2
    assert all(r["restart_idx"] == 0 for r in records)
    assert all(r["restart_strategy"] == "no_restart" for r in records)
