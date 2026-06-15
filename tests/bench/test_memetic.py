"""TDD test suite for `_MemeticTopKPipeline` in `bench.sources.evogp`.

EvoGP fitness convention: higher = better (it's `-MSE`).  So:
- "improvement" = `fitness_after > fitness_before`
- "regression" = `fitness_after < fitness_before` (this is when we restore c0)

Tests #1, #2 are pre-flight (verify torch view + chained assignment semantics).
Tests #3-#8 are write-back invariants.
Tests #9, #10 are pipeline integration.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import sympy as sp
import torch

_evogp = pytest.importorskip("evogp")

from evogp.tree import Forest, Tree  # noqa: E402
from evogp.tree.utils import Func, NType  # noqa: E402

from cusr.bench.sources.evogp import (  # noqa: E402
    _MemeticTopKPipeline,
    _writeback_constants,
    forest_member_to_skeleton,
)


# ---------------------------------------------------------------------------
# Tree fixtures (CPU only — Tree ctor doesn't require CUDA)
# ---------------------------------------------------------------------------


def _make_linear_tree(max_len: int = 16, c0: float = 2.0, c1: float = 1.0) -> Tree:
    """Build a Tree for `c0 * x0 + c1` (default: 2*x + 1)."""
    value = torch.zeros(max_len, dtype=torch.float32)
    ntype = torch.zeros(max_len, dtype=torch.int16)
    ssize = torch.zeros(max_len, dtype=torch.int16)

    value[0] = float(Func.ADD)
    ntype[0] = NType.BFUNC
    ssize[0] = 5

    value[1] = float(Func.MUL)
    ntype[1] = NType.BFUNC
    ssize[1] = 3

    value[2] = c0
    ntype[2] = NType.CONST
    ssize[2] = 1

    value[3] = 0.0  # variable index 0
    ntype[3] = NType.VAR
    ssize[3] = 1

    value[4] = c1
    ntype[4] = NType.CONST
    ssize[4] = 1

    return Tree(input_len=1, output_len=1, node_value=value, node_type=ntype, subtree_size=ssize)


def _make_no_const_tree(max_len: int = 16) -> Tree:
    """Tree for `x0 + x0` — no CONST nodes."""
    value = torch.zeros(max_len, dtype=torch.float32)
    ntype = torch.zeros(max_len, dtype=torch.int16)
    ssize = torch.zeros(max_len, dtype=torch.int16)

    value[0] = float(Func.ADD)
    ntype[0] = NType.BFUNC
    ssize[0] = 3

    value[1] = 0.0
    ntype[1] = NType.VAR
    ssize[1] = 1

    value[2] = 0.0
    ntype[2] = NType.VAR
    ssize[2] = 1

    return Tree(input_len=1, output_len=1, node_value=value, node_type=ntype, subtree_size=ssize)


def _make_loose_div_tree(max_len: int = 16) -> Tree:
    """Tree for `LooseDiv(c0, x0) + c1` — uses degraded op."""
    value = torch.zeros(max_len, dtype=torch.float32)
    ntype = torch.zeros(max_len, dtype=torch.int16)
    ssize = torch.zeros(max_len, dtype=torch.int16)

    value[0] = float(Func.ADD)
    ntype[0] = NType.BFUNC
    ssize[0] = 5

    value[1] = float(Func.LOOSE_DIV)
    ntype[1] = NType.BFUNC
    ssize[1] = 3

    value[2] = 4.5  # CONST c0
    ntype[2] = NType.CONST
    ssize[2] = 1

    value[3] = 0.0  # VAR x0
    ntype[3] = NType.VAR
    ssize[3] = 1

    value[4] = 0.7  # CONST c1
    ntype[4] = NType.CONST
    ssize[4] = 1

    return Tree(input_len=1, output_len=1, node_value=value, node_type=ntype, subtree_size=ssize)


def _make_cpu_forest(pop_size: int, trees: list[Tree], max_len: int = 16) -> Forest:
    """Pack a list of Trees into a CPU Forest."""
    bv = torch.zeros(pop_size, max_len, dtype=torch.float32)
    bt = torch.zeros(pop_size, max_len, dtype=torch.int16)
    bs = torch.zeros(pop_size, max_len, dtype=torch.int16)
    for i, t in enumerate(trees):
        bv[i, :] = t.node_value
        bt[i, :] = t.node_type
        bs[i, :] = t.subtree_size
    return Forest(input_len=1, output_len=1,
                  batch_node_value=bv, batch_node_type=bt, batch_subtree_size=bs)


# ---------------------------------------------------------------------------
# Pre-flight tests
# ---------------------------------------------------------------------------


def test_forest_int_indexing_is_view():
    """forest[int].node_value[k] = v writes through to forest.batch_node_value[int, k]."""
    forest = _make_cpu_forest(pop_size=8, trees=[_make_linear_tree() for _ in range(8)])
    tree = forest[5]  # python int → Tree view
    tree.node_value[3] = 99.0
    assert forest.batch_node_value[5, 3].item() == 99.0


def test_absolute_index_assignment():
    """Verify absolute-index tensor assignment used by `_writeback_constants` (CPU + CUDA)."""
    # CPU: t[pos] = vals where pos is a 1-D long tensor of absolute indices.
    t = torch.zeros(10)
    pos = torch.tensor([0, 2, 4])
    t[pos] = torch.tensor([1., 2., 3.])
    expected = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert t.tolist() == expected

    if torch.cuda.is_available():
        tc = torch.zeros(4, device='cuda')
        posc = torch.tensor([0, 2], device='cuda')
        tc[posc] = torch.tensor([1.5, 2.5], device='cuda')
        assert tc.cpu().tolist() == [1.5, 0.0, 2.5, 0.0]


# ---------------------------------------------------------------------------
# Write-back invariants
# ---------------------------------------------------------------------------


def test_writeback_roundtrip():
    """Write-back constants then re-extract — init values change, expr doesn't."""
    tree = _make_linear_tree(c0=2.0, c1=1.0)
    skel_before, init_before, _ = forest_member_to_skeleton(tree, problem_n_vars=1)
    np.testing.assert_allclose(init_before, [2.0, 1.0], atol=1e-6)

    c_new = np.array([3.5, -0.5], dtype=np.float64)
    ok = _writeback_constants(tree, c_new)
    assert ok

    skel_after, init_after, _ = forest_member_to_skeleton(tree, problem_n_vars=1)
    np.testing.assert_allclose(init_after, [3.5, -0.5], atol=1e-5)
    # Expr structure unchanged
    assert sp.simplify(skel_before.expr - skel_after.expr) == 0


def test_writeback_dtype_preserved():
    """f64 c_star → f32 forest, residual unchanged within f32 tol."""
    tree = _make_linear_tree(c0=2.0, c1=1.0)

    # f64 input
    c_star = np.array([3.5, -0.5], dtype=np.float64)
    ok = _writeback_constants(tree, c_star)
    assert ok
    assert tree.node_value.dtype == torch.float32

    # Re-extract and verify residual matches in f32
    skel, init, _ = forest_member_to_skeleton(tree, problem_n_vars=1)
    X = np.array([[0.0], [1.0], [2.0], [-1.5]])
    y_target = 3.5 * X[:, 0] - 0.5
    pred = skel.evaluate(init, X)
    np.testing.assert_allclose(pred, y_target, atol=1e-5)


def test_writeback_nan_fallback():
    """If c_star contains NaN, original values must be untouched."""
    tree = _make_linear_tree(c0=2.0, c1=1.0)
    orig_val = tree.node_value.clone()
    orig_type = tree.node_type.clone()
    orig_size = tree.subtree_size.clone()

    bad = np.array([3.5, np.nan], dtype=np.float64)
    ok = _writeback_constants(tree, bad)
    assert ok is False
    assert torch.equal(tree.node_value, orig_val)
    assert torch.equal(tree.node_type, orig_type)
    assert torch.equal(tree.subtree_size, orig_size)

    # inf should also fail
    tree2 = _make_linear_tree(c0=2.0, c1=1.0)
    orig2 = tree2.node_value.clone()
    bad2 = np.array([np.inf, 1.0], dtype=np.float64)
    ok2 = _writeback_constants(tree2, bad2)
    assert ok2 is False
    assert torch.equal(tree2.node_value, orig2)


def test_writeback_with_degraded_ops():
    """Loose op tree: constants land at correct positions even though sympy expr is rewritten."""
    tree = _make_loose_div_tree()
    skel_before, init_before, deg = forest_member_to_skeleton(tree, problem_n_vars=1)
    assert len(deg) == 1 and "LooseDiv" in deg[0]
    np.testing.assert_allclose(init_before, [4.5, 0.7], atol=1e-6)

    c_new = np.array([6.6, -2.2], dtype=np.float64)
    ok = _writeback_constants(tree, c_new)
    assert ok

    skel_after, init_after, _ = forest_member_to_skeleton(tree, problem_n_vars=1)
    np.testing.assert_allclose(init_after, [6.6, -2.2], atol=1e-5)
    # Position of CONST nodes (positions 2 and 4) unchanged
    assert tree.node_value[2].item() == pytest.approx(6.6, abs=1e-5)
    assert tree.node_value[4].item() == pytest.approx(-2.2, abs=1e-5)


def test_member_id_index_integrity():
    """Multi-member forest: write-back to member 1 only; others byte-identical."""
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=1.0, c1=0.0),
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_linear_tree(c0=3.0, c1=2.0),
            _make_linear_tree(c0=4.0, c1=3.0),
        ],
    )
    snap_value = forest.batch_node_value.clone()
    snap_type = forest.batch_node_type.clone()
    snap_size = forest.batch_subtree_size.clone()

    tree1 = forest[1]
    ok = _writeback_constants(tree1, np.array([99.0, -99.0], dtype=np.float64))
    assert ok

    # member 1 changed
    assert not torch.equal(forest.batch_node_value[1], snap_value[1])
    # member 0, 2, 3 unchanged
    for i in (0, 2, 3):
        assert torch.equal(forest.batch_node_value[i], snap_value[i]), f"member {i} value altered"
        assert torch.equal(forest.batch_node_type[i], snap_type[i]), f"member {i} type altered"
        assert torch.equal(forest.batch_subtree_size[i], snap_size[i]), f"member {i} size altered"


def test_no_constants_skips():
    """Tree with zero CONST nodes — write-back should be a no-op success (or a clean skip)."""
    tree = _make_no_const_tree()
    snap = tree.node_value.clone()
    # n_consts=0; pass empty array. Helper should accept and not raise.
    ok = _writeback_constants(tree, np.array([], dtype=np.float64))
    # We treat zero-const write-back as success (nothing to do).
    assert ok is True
    assert torch.equal(tree.node_value, snap)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class _StubAlgorithm:
    """Minimal duck-typed algorithm. Records calls; exposes a forest."""
    def __init__(self, forest):
        self.forest = forest
        self.step_calls: list[torch.Tensor] = []

    def step(self, fitnesses):
        self.step_calls.append(fitnesses.detach().cpu().clone())


class _StubProblem:
    """Mockable problem.evaluate. Returns a recorded sequence of fitness tensors."""
    def __init__(self, fit_sequence: list[torch.Tensor]):
        self.fit_sequence = list(fit_sequence)
        self.calls = 0

    def evaluate(self, forest):
        out = self.fit_sequence[self.calls]
        self.calls += 1
        return out


def test_pipeline_recomputes_fitness():
    """Verify problem.evaluate call count: 2 if any NLS write-back happens, 1 if none."""
    # Case A: forest has constants → write-back succeeds → evaluate called 2x.
    forest_a = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=0.5, c1=0.0),  # bad fit
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_linear_tree(c0=1.5, c1=0.5),
            _make_linear_tree(c0=3.0, c1=2.0),
        ],
    )
    # Synthetic data: y = 3.0 * x + 1.0
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_after = torch.tensor([-0.1, -2.0, -3.0, -4.0])  # member 0 improves
    problem_a = _StubProblem([fit_before, fit_after])
    algo_a = _StubAlgorithm(forest_a)
    pipe_a = _MemeticTopKPipeline(
        algorithm=algo_a, problem=problem_a, top_k=1, problem_n_vars=1,
        X_np=X, y_np=y, generation_limit=1, is_show_details=False,
    )
    pipe_a.step()
    assert problem_a.calls == 2, f"expected 2 evaluate calls, got {problem_a.calls}"
    assert len(algo_a.step_calls) == 1
    # algorithm.step received the post-NLS fitness
    np.testing.assert_allclose(algo_a.step_calls[0].numpy(), fit_after.numpy())

    # Case B: all top-K members have no constants → no write-back → evaluate called 1x.
    forest_b = _make_cpu_forest(
        pop_size=4, trees=[_make_no_const_tree() for _ in range(4)],
    )
    fit_b = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    problem_b = _StubProblem([fit_b])
    algo_b = _StubAlgorithm(forest_b)
    pipe_b = _MemeticTopKPipeline(
        algorithm=algo_b, problem=problem_b, top_k=2, problem_n_vars=1,
        X_np=X, y_np=y, generation_limit=1, is_show_details=False,
    )
    pipe_b.step()
    assert problem_b.calls == 1, f"expected 1 evaluate call, got {problem_b.calls}"
    assert len(algo_b.step_calls) == 1
    np.testing.assert_allclose(algo_b.step_calls[0].numpy(), fit_b.numpy())


def test_pipeline_topk_fitness_strictly_improves():
    """Real EvoGP forest, ground-truth y=3.7*x0+1.2; bad initial constants → NLS lifts fitness."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for SymbolicRegression.evaluate")

    from evogp.algorithm import (
        DefaultCrossover, DefaultMutation, DefaultSelection, GeneticProgramming,
    )
    from evogp.problem import SymbolicRegression
    from evogp.tree import GenerateDescriptor

    # Synthetic ground-truth
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(64, 1)).astype(np.float32)
    y = (3.7 * X[:, 0] + 1.2).astype(np.float32)

    descriptor = GenerateDescriptor(
        max_tree_len=16, input_len=1, output_len=1,
        const_prob=0.5, out_prob=0.5,
        const_samples=[0.1, 0.5, 1.0, 2.0],
        using_funcs={"+": 1.0, "*": 1.0},
        max_layer_cnt=3, layer_leaf_prob=0.5,
    )

    pop_size = 4
    forest = Forest.random_generate(pop_size=pop_size, descriptor=descriptor)

    # Overwrite member 0 with hand-built linear tree, c0=0.1, c1=0.0 (bad)
    seed_tree = _make_linear_tree(c0=0.1, c1=0.0, max_len=descriptor.max_tree_len)
    forest.batch_node_value[0] = seed_tree.node_value.cuda()
    forest.batch_node_type[0] = seed_tree.node_type.cuda()
    forest.batch_subtree_size[0] = seed_tree.subtree_size.cuda()

    X_t = torch.from_numpy(X).cuda()
    # labels expected as (N, output_len)
    y_t = torch.from_numpy(y.reshape(-1, 1)).cuda()
    problem = SymbolicRegression(datapoints=X_t, labels=y_t)

    algorithm = GeneticProgramming(
        initial_forest=forest,
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.0, descriptor=descriptor.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.5, elite_rate=0.25),
    )

    # top_k=pop_size so the seeded member 0 is guaranteed to be NLS'd
    # (otherwise random forest could outrank our deliberately-bad seed).
    pipe = _MemeticTopKPipeline(
        algorithm=algorithm, problem=problem, top_k=pop_size, problem_n_vars=1,
        X_np=X.astype(np.float64), y_np=y.astype(np.float64),
        nls_max_nfev=200, generation_limit=1, is_show_details=False,
    )

    # Take fitness snapshot before NLS by manually calling evaluate
    fit_before_t = problem.evaluate(forest).cpu()
    fit_before_member0 = float(fit_before_t[0])

    # Run one memetic step. Note: this calls evaluate internally too, but our
    # fixture ensures member 0 = the seeded tree before any algorithm.step mutation.
    cpu_fitness_after = pipe.step()
    fit_after_member0 = float(cpu_fitness_after[0])

    # The seeded tree had only constants (c0=0.1, c1=0.0); NLS should drive them
    # toward (3.7, 1.2) and improve fitness. Higher = better in EvoGP.
    assert fit_after_member0 > fit_before_member0, (
        f"NLS should improve fitness: before={fit_before_member0}, after={fit_after_member0}"
    )


def test_pipeline_rollback_on_fitness_regression():
    """If post-NLS fitness is WORSE than pre-NLS for some top-K members,
    pipeline must (a) restore those members' constants in the live forest,
    (b) patch the device tensor handed to algorithm.step with the pre-NLS
    values, (c) reflect the same in returned cpu_fitness.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=0.5, c1=0.0),  # member 0 — top-1; will be NLS'd
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_linear_tree(c0=1.5, c1=0.5),
            _make_linear_tree(c0=3.0, c1=2.0),
        ],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    # Pre-NLS: member 0 best (-1.0). Post-NLS: member 0 WORSE (-2.0).
    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_after = torch.tensor([-2.0, -2.0, -3.0, -4.0])

    problem = _StubProblem([fit_before, fit_after])
    algo = _StubAlgorithm(forest)
    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=1, problem_n_vars=1,
        X_np=X, y_np=y, generation_limit=1, is_show_details=False,
    )

    # Snapshot member 0's row BEFORE step() — to verify byte-identity after rollback.
    snap_member0 = forest.batch_node_value[0].clone()

    cpu_final = pipe.step()

    # (a) member 0 row byte-identical to pre-NLS snapshot
    assert torch.equal(forest.batch_node_value[0], snap_member0), \
        "member 0 constants not restored after rollback"

    # (b) algorithm.step received fitness with [0] == -1.0 (pre-NLS, not -2.0)
    assert len(algo.step_calls) == 1
    np.testing.assert_allclose(algo.step_calls[0][0].item(), -1.0)

    # (c) returned cpu_fitness has [0] == -1.0
    np.testing.assert_allclose(float(cpu_final[0]), -1.0)

    # (d) nls_records last entry has status='rolled_back'
    assert pipe.nls_records[-1]['status'] == 'rolled_back', \
        f"expected status='rolled_back', got {pipe.nls_records[-1]}"


def test_pipeline_all_nan_skips_second_evaluate(monkeypatch):
    """When all top-K members' NLS returns NaN c_star, no write-back occurs,
    so problem.evaluate should be called exactly 1x (not 2x), and the fitness
    tensor handed to algorithm.step is the original pre-NLS one.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[_make_linear_tree(c0=2.0, c1=1.0) for _ in range(4)],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    problem = _StubProblem([fit_before])
    algo = _StubAlgorithm(forest)

    # Monkeypatch least_squares (bound name in bench.sources.evogp) to return NaN.
    class _FakeRes:
        def __init__(self, n):
            self.x = np.full(n, np.nan, dtype=np.float64)
            self.nfev = 1
            self.status = 1  # converged-ish; doesn't matter, NaN check fires first

    def _fake_lm(fun, x0, jac=None, method=None, max_nfev=None, **kw):
        return _FakeRes(len(x0))

    monkeypatch.setattr('cusr.bench.sources.evogp.least_squares', _fake_lm)

    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=4, problem_n_vars=1,
        X_np=X, y_np=y, generation_limit=1, is_show_details=False,
    )
    pipe.step()

    # (1) evaluate called exactly once — no second evaluation since no write-back happened
    assert problem.calls == 1, f"expected 1 evaluate call, got {problem.calls}"

    # (2) algorithm.step received the original pre-NLS fitness tensor
    assert len(algo.step_calls) == 1
    np.testing.assert_allclose(algo.step_calls[0].numpy(), fit_before.numpy())

    # (3) all nls_records entries from this step have status == 'skipped_nan'
    statuses = [r['status'] for r in pipe.nls_records]
    assert len(statuses) == 4
    assert all(s == 'skipped_nan' for s in statuses), \
        f"expected all 'skipped_nan', got {statuses}"


# ---------------------------------------------------------------------------
# nls_every — frequency control (Step 1)
# ---------------------------------------------------------------------------


def test_nls_every_5_skips_intermediate_gens():
    """nls_every=5: gen 0 triggers NLS (2 evaluate calls), gen 1-4 skip (1 each).
    NLS records should only come from gen 0; gen 1-4 are pure GP, no records.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[_make_linear_tree(c0=2.0, c1=1.0) for _ in range(4)],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    # gen 0 triggers NLS → consumes fit_before + fit_after.
    # gen 1, 2, 3, 4 skip NLS → consume one fitness each.
    fit_g0_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_g0_after = torch.tensor([-0.5, -2.0, -3.0, -4.0])
    fit_g1 = torch.tensor([-0.5, -2.0, -3.0, -4.0])
    fit_g2 = torch.tensor([-0.4, -2.0, -3.0, -4.0])
    fit_g3 = torch.tensor([-0.3, -2.0, -3.0, -4.0])
    fit_g4 = torch.tensor([-0.2, -2.0, -3.0, -4.0])
    problem = _StubProblem([fit_g0_before, fit_g0_after, fit_g1, fit_g2, fit_g3, fit_g4])
    algo = _StubAlgorithm(forest)

    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=1, problem_n_vars=1,
        X_np=X, y_np=y, nls_every=5, generation_limit=5, is_show_details=False,
    )

    for _ in range(5):
        pipe.step()

    # Total evaluate: gen 0 = 2x, gen 1-4 = 1x each → 2 + 4 = 6.
    assert problem.calls == 6, f"expected 6 evaluate calls, got {problem.calls}"
    # algorithm.step called once per generation = 5.
    assert len(algo.step_calls) == 5
    # NLS records: only from gen 0.
    nls_gens = {r["gen"] for r in pipe.nls_records}
    assert nls_gens == {0}, f"expected NLS records only from gen 0, got {nls_gens}"


def test_nls_every_1_triggers_every_gen():
    """nls_every=1: every generation triggers NLS (backward-compat with original behavior).
    3 gens × 2 evaluate calls = 6 total; nls_records has entries from gen 0, 1, 2.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[_make_linear_tree(c0=2.0, c1=1.0) for _ in range(4)],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    # 3 generations, 2 evaluate each.
    fits = [
        torch.tensor([-1.0, -2.0, -3.0, -4.0]),  # g0 before
        torch.tensor([-0.9, -2.0, -3.0, -4.0]),  # g0 after
        torch.tensor([-0.9, -2.0, -3.0, -4.0]),  # g1 before
        torch.tensor([-0.8, -2.0, -3.0, -4.0]),  # g1 after
        torch.tensor([-0.8, -2.0, -3.0, -4.0]),  # g2 before
        torch.tensor([-0.7, -2.0, -3.0, -4.0]),  # g2 after
    ]
    problem = _StubProblem(fits)
    algo = _StubAlgorithm(forest)

    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=1, problem_n_vars=1,
        X_np=X, y_np=y, nls_every=1, generation_limit=3, is_show_details=False,
    )

    for _ in range(3):
        pipe.step()

    assert problem.calls == 6, f"expected 6 evaluate calls, got {problem.calls}"
    nls_gens = {r["gen"] for r in pipe.nls_records}
    assert nls_gens == {0, 1, 2}, f"expected NLS records from gen 0,1,2; got {nls_gens}"


def test_nls_every_must_be_positive():
    """nls_every=0 or negative should raise — defensive against config typos."""
    forest = _make_cpu_forest(pop_size=4, trees=[_make_linear_tree() for _ in range(4)])
    problem = _StubProblem([torch.zeros(4)])
    algo = _StubAlgorithm(forest)

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(8, 1)).astype(np.float64)
    y = X[:, 0].astype(np.float64)

    for bad in (0, -1, -5):
        with pytest.raises(ValueError, match="nls_every"):
            _MemeticTopKPipeline(
                algorithm=algo, problem=problem, top_k=1, problem_n_vars=1,
                X_np=X, y_np=y, nls_every=bad,
                generation_limit=1, is_show_details=False,
            )


# ---------------------------------------------------------------------------
# skeleton_dedup — F mitigation (Step 4)
#
# Hash key = `str(skel.expr)`. The c-symbols are encounter-order canonical
# (forest_member_to_skeleton walks reverse prefix), so two trees with the same
# structure but different constant values share the same skeleton string. This
# is exactly what we want to dedup: NLS only needs to polish each unique
# skeleton once per batch.
#
# Iteration semantics with dedup ON:
#   - iter_pool = full forest sorted by fitness DESC (not top_k)
#   - stop when n_nls_attempted == top_k OR pool exhausted
#   - "n_nls_attempted" = trees that pass dedup AND have constants (i.e. enter
#     scipy LM). skipped_no_consts and skipped_dup_skel do NOT count.
# ---------------------------------------------------------------------------


def test_dedup_skips_duplicate_skeleton():
    """With dedup=True, duplicate-skeleton trees in the iter pool produce
    `skipped_dup_skel` records — only the highest-ranked unique skel goes to NLS.

    Forest: 4 linear trees with DIFFERENT constants but SAME skeleton
    (`c0*x0 + c1`). top_k=2. Expected:
      - rank 0: 'written' (or 'rolled_back') — first unique skeleton seen
      - rank 1, 2, 3: 'skipped_dup_skel' — same skeleton hash as rank 0
      - n_nls_attempted = 1 < top_k=2, so we exhaust the pool

    The lower-ranked dups are never tried because they'd waste an NLS slot
    on a polish-target sympy already saw.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_linear_tree(c0=3.0, c1=1.5),
            _make_linear_tree(c0=0.5, c1=0.8),
            _make_linear_tree(c0=4.0, c1=2.5),
        ],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_after = torch.tensor([-0.05, -2.0, -3.0, -4.0])
    problem = _StubProblem([fit_before, fit_after])
    algo = _StubAlgorithm(forest)

    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=2, problem_n_vars=1,
        X_np=X, y_np=y, skeleton_dedup=True,
        generation_limit=1, is_show_details=False,
    )
    pipe.step()

    statuses = [r["status"] for r in pipe.nls_records]
    assert statuses.count("skipped_dup_skel") == 3, \
        f"expected 3 skipped_dup_skel, got {statuses}"
    # Exactly 1 tree reached NLS (status in {written, rolled_back})
    nls_succeed = sum(1 for s in statuses if s in {"written", "rolled_back"})
    assert nls_succeed == 1, f"expected 1 NLS-completed, got {statuses}"


def test_dedup_disabled_default_no_skip():
    """Without `skeleton_dedup` (default False), behavior matches existing
    test_pipeline_recomputes_fitness Case A: top_k trees all NLS'd, no
    skipped_dup_skel records.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_linear_tree(c0=3.0, c1=1.5),
            _make_linear_tree(c0=0.5, c1=0.8),
            _make_linear_tree(c0=4.0, c1=2.5),
        ],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_after = torch.tensor([-0.05, -0.1, -3.0, -4.0])
    problem = _StubProblem([fit_before, fit_after])
    algo = _StubAlgorithm(forest)

    # No `skeleton_dedup` kwarg → default False
    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=2, problem_n_vars=1,
        X_np=X, y_np=y,
        generation_limit=1, is_show_details=False,
    )
    pipe.step()

    statuses = [r["status"] for r in pipe.nls_records]
    assert "skipped_dup_skel" not in statuses, \
        f"dedup off but found dup_skel records: {statuses}"
    # Both top-2 reached NLS (written or rolled_back, depending on fit_after)
    nls_succeed = sum(1 for s in statuses if s in {"written", "rolled_back"})
    assert nls_succeed == 2, f"expected 2 NLS-completed, got {statuses}"


def test_dedup_with_mixed_skeletons():
    """Forest has 2 unique skeleton classes; dedup=True picks one of each.

    Trees: [linear(c0=2,c1=1), linear(c0=3,c1=2), no_const, linear(c0=4,c1=2)]
    Skeletons: [A, A, B, A] where A = c0*x0+c1, B = 2*x0 (no-const tree).
    Fitnesses: [-1, -2, -3, -4] → ranking 0 > 1 > 2 > 3.
    top_k=2, dedup=True.

    Expected iteration:
      - rank 0 (idx 0, skel A) → NLS path (1 const-having unique seen)
      - rank 1 (idx 1, skel A — dup) → 'skipped_dup_skel'
      - rank 2 (idx 2, skel B — no consts) → 'skipped_no_consts' (does NOT
        count toward n_nls_attempted; dedup logic decides whether to add to seen)
      - rank 3 (idx 3, skel A — dup) → 'skipped_dup_skel'
      - pool exhausted, end. n_nls_attempted = 1.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_linear_tree(c0=3.0, c1=2.0),
            _make_no_const_tree(),
            _make_linear_tree(c0=4.0, c1=2.0),
        ],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_after = torch.tensor([-0.05, -2.0, -3.0, -4.0])
    problem = _StubProblem([fit_before, fit_after])
    algo = _StubAlgorithm(forest)

    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=2, problem_n_vars=1,
        X_np=X, y_np=y, skeleton_dedup=True,
        generation_limit=1, is_show_details=False,
    )
    pipe.step()

    statuses = [r["status"] for r in pipe.nls_records]
    assert statuses.count("skipped_no_consts") == 1, \
        f"expected 1 skipped_no_consts (rank 2), got {statuses}"
    assert statuses.count("skipped_dup_skel") == 2, \
        f"expected 2 skipped_dup_skel (ranks 1, 3), got {statuses}"
    nls_succeed = sum(1 for s in statuses if s in {"written", "rolled_back"})
    assert nls_succeed == 1, f"expected 1 NLS-completed, got {statuses}"


def test_dedup_with_full_unique_pool():
    """When all top-K trees have unique skeletons, dedup ON behaves exactly
    like dedup OFF — no skipped_dup_skel records, all top_k get NLS.

    Forest: [linear, no_const, loose_div, linear(other consts)]. Skeletons:
    [A=c0*x0+c1, B=2*x0, C=safe_div(c0,x0)+c1, A]. top_k=3 ⇒ pool consumes
    ranks 0, 1, 2 (skel A, B, C — all unique among first 3). Rank 3 not
    reached because n_nls_attempted hits cap when rank 2 (NLS with consts)
    increments it. Wait — rank 1 (skel B = no_const) doesn't count toward
    NLS attempts (no NLS path). So:
      - rank 0 (A, has consts) → NLS, attempted=1
      - rank 1 (B, no consts) → skipped_no_consts, attempted=1
      - rank 2 (C, has consts) → NLS, attempted=2
      - rank 3 (A — dup of 0) → would be skipped_dup_skel, but loop break
        first because attempted == top_k? Need to re-check.

    For top_k=3 with this layout, attempted only reaches 2 (B doesn't count).
    Pool exhausts → 3 records: (NLS, no_consts, NLS). No dup record.

    For top_k=2, same: NLS, no_consts, NLS — pool stops when attempted==2
    after rank 2. No dup record.
    """
    forest = _make_cpu_forest(
        pop_size=4,
        trees=[
            _make_linear_tree(c0=2.0, c1=1.0),
            _make_no_const_tree(),
            _make_loose_div_tree(),
            _make_linear_tree(c0=3.0, c1=2.0),
        ],
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(32, 1)).astype(np.float64)
    y = (3.0 * X[:, 0] + 1.0).astype(np.float64)

    # Fitness ranking 0 > 1 > 2 > 3
    fit_before = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    fit_after = torch.tensor([-0.05, -2.0, -0.4, -4.0])
    problem = _StubProblem([fit_before, fit_after])
    algo = _StubAlgorithm(forest)

    pipe = _MemeticTopKPipeline(
        algorithm=algo, problem=problem, top_k=2, problem_n_vars=1,
        X_np=X, y_np=y, skeleton_dedup=True,
        generation_limit=1, is_show_details=False,
    )
    pipe.step()

    statuses = [r["status"] for r in pipe.nls_records]
    # No dup record because all touched skeletons are unique
    assert "skipped_dup_skel" not in statuses, \
        f"unexpected dup_skel for unique pool: {statuses}"
    # No-const tree was visited (rank 1)
    assert statuses.count("skipped_no_consts") == 1, \
        f"expected 1 skipped_no_consts, got {statuses}"
    # 2 NLS attempts (target met with rank 0 + rank 2)
    nls_attempts = sum(1 for s in statuses
                       if s in {"written", "rolled_back", "skipped_nan", "nls_error"})
    assert nls_attempts == 2, f"expected 2 NLS attempts, got {statuses}"
