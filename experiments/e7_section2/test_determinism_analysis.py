"""TDD for the C2 determinism/clean-baseline pure analysis functions.

No GPU needed. These functions are the ones the adversarial audit re-runs on
reps_raw.jsonl to falsify every headline CV / multiplier, so their correctness
is load-bearing. Run: .venv/bin/python -m pytest experiments/e7_section2/test_determinism_analysis.py -q
"""
import math

import numpy as np
import pytest

from experiments.e7_section2.determinism_analysis import (
    build_homogeneous_pop,
    check_linearity,
    compute_cv,
    fit_intercept_slope,
    recompute_crossover,
)


# ----------------------------------------------------------------- compute_cv
def test_cv_zero_when_constant():
    r = compute_cv([10.0, 10.0, 10.0, 10.0])
    assert r["cv"] == 0.0
    assert r["median"] == 10.0 and r["min"] == 10.0 and r["max"] == 10.0
    assert r["n"] == 4


def test_cv_known_value_sample_std():
    # [8,12]: mean=10, sample std (ddof=1)=sqrt(8)=2.8284..., cv=0.28284
    r = compute_cv([8.0, 12.0])
    assert math.isclose(r["mean"], 10.0)
    assert math.isclose(r["std"], math.sqrt(8.0), rel_tol=1e-9)
    assert math.isclose(r["cv"], math.sqrt(8.0) / 10.0, rel_tol=1e-9)


def test_cv_drops_nonfinite_and_handles_empty():
    r = compute_cv([5.0, float("nan"), 5.0, float("inf")])
    assert r["n"] == 2 and r["cv"] == 0.0
    assert math.isnan(compute_cv([])["cv"])


# -------------------------------------------------------- fit_intercept_slope
def test_fit_recovers_known_intercept_slope():
    iters = [5, 10, 20, 40]
    loop = [3 + 2 * k for k in iters]          # intercept 3, slope 2, exact
    f = fit_intercept_slope(iters, loop, ref_iter=50)
    assert math.isclose(f["intercept"], 3.0, abs_tol=1e-6)
    assert math.isclose(f["slope"], 2.0, abs_tol=1e-6)
    assert math.isclose(f["r2"], 1.0, abs_tol=1e-9)
    # fixed fraction at 50 iters = 3/(3+100) ≈ 0.0291 → "no short-region bias"
    assert math.isclose(f["intercept_frac"], 3.0 / 103.0, rel_tol=1e-6)


def test_fit_zero_intercept_means_no_bias():
    iters = [5, 10, 20, 40]
    loop = [2.0 * k for k in iters]            # pure proportional
    f = fit_intercept_slope(iters, loop, ref_iter=50)
    assert abs(f["intercept"]) < 1e-6
    assert abs(f["intercept_frac"]) < 1e-6


# ------------------------------------------------------------ check_linearity
def test_linearity_true_for_linear():
    iters = [5, 10, 20, 40, 80]
    loop = [3 + 2 * k for k in iters]
    c = check_linearity(iters, loop)
    assert c["linear"] is True
    assert c["linear_prefix_len"] == 5


def test_linearity_false_for_plateau():
    # early-stop signature: rises then saturates (loop(40)≈loop(80))
    iters = [5, 10, 20, 40, 80]
    loop = [10.0, 18.0, 20.0, 20.4, 20.5]      # top segments ~flat
    c = check_linearity(iters, loop)
    assert c["linear"] is False
    # only the rising low-iter prefix is kept for fitting
    assert c["linear_prefix_len"] <= 3


# --------------------------------------------------------- recompute_crossover
def test_crossover_multiplier_and_band():
    r = recompute_crossover(gpu_tput=100_000.0, operon_tput=10_000.0,
                            gpu_cv=0.01, operon_cv=0.05)
    assert math.isclose(r["multiplier"], 10.0)
    assert math.isclose(r["rel_unc"], math.hypot(0.01, 0.05), rel_tol=1e-9)
    assert r["lo"] < 10.0 < r["hi"]


def test_crossover_nan_guard():
    assert math.isnan(recompute_crossover(1.0, 0.0)["multiplier"])


# ------------------------------------------------------- build_homogeneous_pop
def test_homogeneous_pop_all_trees_identical():
    from cusr.benchmark.workload.gen_synth import gen_pop
    pop = gen_pop("early-gen", M=12, N=16, seed=0)
    hp, rep = build_homogeneous_pop(pop)
    # M and N preserved
    assert hp["M"] == 12 and hp["N"] == 16
    # every tree has the SAME K (uniform work => no structural load imbalance)
    Ks = hp["metas"][:, 3]
    assert len(set(Ks.tolist())) == 1
    # every tree's node block is byte-identical to the representative tree
    metas = hp["metas"]
    off0, n0, _, _ = metas[0].tolist()
    nt0 = hp["nt"][off0:off0 + n0]
    for m in range(hp["M"]):
        off, n, _, _ = metas[m].tolist()
        assert n == n0
        assert np.array_equal(hp["nt"][off:off + n], nt0)
    # representative index is a real K>0 tree from the source pop
    assert pop["metas"][rep, 3] > 0


def test_homogeneous_pop_respects_explicit_index():
    from cusr.benchmark.workload.gen_synth import gen_pop
    pop = gen_pop("early-gen", M=10, N=8, seed=1)
    hp, rep = build_homogeneous_pop(pop, rep_idx=3)
    assert rep == 3
    off, n, c_off, K = pop["metas"][3].tolist()
    assert np.array_equal(hp["nt"][0:n], pop["nt"][off:off + n])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
