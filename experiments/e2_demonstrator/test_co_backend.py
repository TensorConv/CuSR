"""TDD for the pluggable constant-optimization (CO) backend interface.

Contract (see project memory project_009_co_backend_interface):
  ConstantOptimizer.fit_batch(skeletons, inits, X, y, max_iter) -> [COResult]
  - common currency = bench.skeleton.Skeleton (sympy expr + c-symbols + x-vars)
  - all skeletons in one call share the SAME (X, y) (the problem's data)
  - heterogeneous n_consts in one batch must work

These tests run against EVERY backend (scipy = reference oracle, torch = new
GPU-native), so the interface is what's tested, not one implementation. Adding
the future CudaKernelLM just extends BACKENDS and it inherits this suite.

Run:  uv run pytest experiments/009_sr_benchmark/test_co_backend.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from cusr.bench.skeleton import Skeleton
from cusr.demonstrator.co_backend import COResult, ScipyLM, TorchLM

BACKENDS = [ScipyLM(), TorchLM()]
IDS = [b.name for b in BACKENDS]


def make(expr_str, var_names, const_names, true_c, *, n=200, seed=0, ranges=None):
    vars_ = [sp.Symbol(v) for v in var_names]
    consts = [sp.Symbol(c) for c in const_names]
    expr = sp.sympify(expr_str, locals={s.name: s for s in vars_ + consts})
    skel = Skeleton(expr=expr, variables=tuple(vars_), constants=tuple(consts))
    rng = np.random.default_rng(seed)
    ranges = ranges or [(-1.0, 1.0)] * len(vars_)
    X = np.column_stack([rng.uniform(lo, hi, n) for lo, hi in ranges])
    y = skel.evaluate(np.array(true_c, dtype=float), X) if const_names else skel.evaluate(np.array([]), X)
    return skel, X, y


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_recovers_linear(backend):
    skel, X, y = make("c0*x0 + c1", ["x0"], ["c0", "c1"], [2.0, -1.0])
    res = backend.fit_batch([skel], [np.array([0.0, 0.0])], X, y, max_iter=100)[0]
    assert isinstance(res, COResult)
    assert np.allclose(res.constants, [2.0, -1.0], atol=1e-3)
    assert res.final_loss < 1e-8


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_recovers_inner_frequency(backend):
    # the case the project is about: a frequency inside sin, fit from a near init
    skel, X, y = make("c0*sin(c1*x0)", ["x0"], ["c0", "c1"], [1.5, 2.0], ranges=[(0.0, 3.0)])
    res = backend.fit_batch([skel], [np.array([1.0, 2.2])], X, y, max_iter=300)[0]
    assert res.final_loss < 1e-6
    assert np.allclose(res.constants, [1.5, 2.0], atol=1e-2)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_zero_constants_is_noop(backend):
    skel, X, y = make("x0**2 + x0", ["x0"], [], [])
    res = backend.fit_batch([skel], [np.array([])], X, y, max_iter=10)[0]
    assert res.constants.shape == (0,)
    assert res.converged is True
    assert res.final_loss < 1e-8  # exact: candidate == target


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_final_loss_improves_from_bad_init(backend):
    skel, X, y = make("c0*x0 + c1", ["x0"], ["c0", "c1"], [3.0, 0.5])
    init = np.array([-5.0, 5.0])
    init_loss = float(np.mean(skel.residual(init, X, y) ** 2))
    res = backend.fit_batch([skel], [init], X, y, max_iter=100)[0]
    assert res.final_loss < init_loss


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_result_fields_typed(backend):
    skel, X, y = make("c0*x0 + c1", ["x0"], ["c0", "c1"], [2.0, -1.0])
    res = backend.fit_batch([skel], [np.array([0.0, 0.0])], X, y, max_iter=100)[0]
    assert isinstance(res.constants, np.ndarray)
    assert isinstance(res.n_iter, int)
    assert isinstance(res.converged, bool)
    assert isinstance(res.final_loss, float)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_heterogeneous_batch(backend):
    """One call, several candidates fitting the SAME (X,y), with DIFFERENT
    n_consts (2, 2, 1). The correct-structure one fits well; the wrong ones
    return valid results without crashing the batch."""
    strue, X, y = make("c0*sin(c1*x0)", ["x0"], ["c0", "c1"], [1.5, 2.0], ranges=[(0.0, 3.0)])

    x0, c0, c1 = sp.symbols("x0 c0 c1")
    s_lin = Skeleton(expr=c0 * x0 + c1, variables=(x0,), constants=(c0, c1))   # wrong, 2 consts
    s_one = Skeleton(expr=c0 * x0, variables=(x0,), constants=(c0,))            # wrong, 1 const

    res = backend.fit_batch(
        [strue, s_lin, s_one],
        [np.array([1.0, 2.2]), np.array([0.0, 0.0]), np.array([0.0])],
        X, y, max_iter=300,
    )
    assert len(res) == 3
    assert res[0].final_loss < 1e-6
    assert np.allclose(res[0].constants, [1.5, 2.0], atol=1e-2)
    assert res[1].constants.shape == (2,)
    assert res[2].constants.shape == (1,)
    # wrong structures can't fit a sine -> their loss stays well above the true one's
    assert res[1].final_loss > 1e-3 and res[2].final_loss > 1e-3


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_recovers_exp_decay(backend):
    skel, X, y = make("c0*exp(c1*x0)", ["x0"], ["c0", "c1"], [2.0, -0.5], ranges=[(0.0, 2.0)])
    res = backend.fit_batch([skel], [np.array([1.0, -0.3])], X, y, max_iter=300)[0]
    assert np.allclose(res.constants, [2.0, -0.5], atol=1e-2)


def test_scipy_and_torch_agree_on_same_skeleton():
    """The actual interchangeability claim: scipy and torch fit the SAME
    skeleton to the SAME answer (not just each vs ground truth)."""
    cases = [
        ("c0*x0 + c1", ["x0"], ["c0", "c1"], [2.0, -1.0], [(-1.0, 1.0)], [0.0, 0.0]),
        ("c0*sin(c1*x0)", ["x0"], ["c0", "c1"], [1.5, 2.0], [(0.0, 3.0)], [1.0, 2.2]),
        ("c0*exp(c1*x0)", ["x0"], ["c0", "c1"], [2.0, -0.5], [(0.0, 2.0)], [1.0, -0.3]),
    ]
    sc, tc = ScipyLM(), TorchLM(dtype="float64")
    for expr, vs, cs, true_c, rng, init in cases:
        skel, X, y = make(expr, vs, cs, true_c, ranges=rng)
        rs = sc.fit_batch([skel], [np.array(init)], X, y, max_iter=300)[0]
        rt = tc.fit_batch([skel], [np.array(init)], X, y, max_iter=300)[0]
        assert np.allclose(rs.constants, rt.constants, atol=1e-3), f"{expr}: {rs.constants} vs {rt.constants}"


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_pathological_skeleton_no_crash(backend):
    """A skeleton that goes non-finite on its domain (log of negatives) must not
    crash the batch — return a COResult, not raise. The blow-up guard."""
    skel, X, _ = make("c0*x0", ["x0"], ["c0"], [1.0], ranges=[(-2.0, 2.0)])  # X spans negatives
    x0, c0 = sp.symbols("x0 c0")
    bad = Skeleton(expr=c0 * sp.log(x0), variables=(x0,), constants=(c0,))
    y = np.ones(len(X))
    res = backend.fit_batch([bad], [np.array([1.0])], X, y, max_iter=50)[0]
    assert isinstance(res, COResult)  # finite-or-inf loss, but no exception


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs GPU for EvoGP")
def test_cross_check_on_real_evogp_skeletons():
    """The corpus check: skeletons that forest_member_to_skeleton ACTUALLY emits
    from evolved EvoGP trees (not hand-written) must (a) not break torch and
    (b) give torch ≈ scipy. This is what earns 'backends interchangeable'."""
    import torch
    from cusr.bench.sources.evogp import forest_member_to_skeleton
    from evogp.algorithm import (DefaultCrossover, DefaultMutation,
                                 DefaultSelection, GeneticProgramming)
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 1))
    y = (X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0])
    torch.manual_seed(1)
    np.random.seed(1)
    desc = GenerateDescriptor(max_tree_len=32, input_len=1, output_len=1,
                             using_funcs={"+": 1., "-": 1., "*": 1., "/": 1.,
                                          "sin": .5, "cos": .5, "tan": .5},
                             max_layer_cnt=4, const_samples=[0., 1., -1., 2., -2., 0.5])
    algo = GeneticProgramming(
        initial_forest=Forest.random_generate(pop_size=200, descriptor=desc),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.2, descriptor=desc.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01))
    problem = SymbolicRegression(
        datapoints=torch.from_numpy(X.astype(np.float32)).cuda(),
        labels=torch.from_numpy(y.astype(np.float32).reshape(-1, 1)).cuda())
    fit = problem.evaluate(algo.forest)
    for _ in range(3):
        algo.step(fit)
        fit = problem.evaluate(algo.forest)
    fit_cpu = fit.cpu()
    fit_cpu[torch.isnan(fit_cpu)] = -1e30
    top = torch.topk(fit_cpu, 30).indices.tolist()

    sc, tc = ScipyLM(), TorchLM(dtype="float64")
    n_checked = 0
    for idx in top:
        try:
            skel, init, _ = forest_member_to_skeleton(algo.forest[int(idx)], 1)
        except Exception:
            continue
        rs = sc.fit_batch([skel], [init], X, y, max_iter=100)[0]
        rt = tc.fit_batch([skel], [init], X, y, max_iter=100)[0]
        assert np.isfinite(rt.final_loss), f"torch non-finite on real skeleton {skel.expr}"
        if skel.n_constants > 0 and np.isfinite(rs.final_loss):
            n_checked += 1
            # Interchangeability bar is asymmetric: torch must never be
            # meaningfully WORSE than scipy. It may be BETTER on rank-deficient /
            # degenerate skeletons (e.g. all consts collapse to one value), where
            # the two LMs legitimately land in different places.
            assert rt.final_loss <= rs.final_loss * 1.05 + 1e-9, \
                f"torch worse than scipy on {skel.expr}: torch={rt.final_loss} scipy={rs.final_loss}"
    assert n_checked >= 3, f"only {n_checked} real skeletons with consts checked"
