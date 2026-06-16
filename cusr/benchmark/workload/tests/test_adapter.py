"""test_adapter.py — Operon Tree -> pop.bin adapter round-trip (TDD).

THIS TEST FAILS UNTIL ``cusr.benchmark.workload.operon_adapter.convert`` EXISTS.
That import error is the intended red state of the TDD cycle (deliverable #5).

The killer test (spec §5): build a known Operon tree via ``InfixParser.Parse``,
get ground truth from Operon's OWN evaluator (``EvaluateTrees``), run ``convert``
to our prefix ``(nt, nv, ci, c_init)`` arrays, evaluate with the host reference
``ref_eval``, and assert ``max|truth - ours| < 1e-3`` on finite entries AND
``len(c_init) == tree.CoefficientsCount``. That single assertion covers operand
order, opcode mapping, the ``Variable -> MUL(CONST, VAR)`` expansion, and the
coefficient mapping all at once.

Operon-API facts established empirically (see spec §3/§7 and OPERON_ADAPTER_SPEC):
  * Dataset MUST be fortran-ordered AND include the target column, else
    EvaluateTrees segfaults: ``op.Dataset(np.asfortranarray(column_stack([X, y])))``.
    Default columns are auto-named X1..Xn (1-indexed); the last is the target.
  * InfixParser folds ``<number>*Var`` WRONG when ``*`` is unspaced (``3.0*X1``
    collapses to a bare constant). Use spaced ``*`` (``3.0 * X1``) — verified to
    produce the correct ``MUL(VAR, CONST)`` / ``MUL(VAR, ...)`` structure.
  * Every Operon leaf is an optimizable coefficient (default ``Optimize=True``);
    ``tree.CoefficientsCount`` == #leaves == our K.
"""
from __future__ import annotations

import numpy as np
import pyoperon as op
import pytest

# --- the not-yet-existing module under test (import failure == red TDD state) --
from cusr.benchmark.workload.operon_adapter import (  # noqa: F401  (intentional)
    UnsupportedNodeType, convert, write_operon_pop_bin,
)

from cusr.benchmark.workload.pop_io import read_pop_bin
from cusr.benchmark.workload.pop_ref import ref_eval


N = 64


def _make_dataset(n_inputs: int, seed: int = 0):
    """Fortran dataset with ``n_inputs`` input cols (X1..Xn) + a target col.

    Returns ``(ds, vmap, hash2idx, cols)`` where ``cols`` is the (N, n_inputs)
    float64 input matrix. Ranges chosen to keep DIV/SIN well-behaved.
    """
    rng = np.random.default_rng(seed)
    cols = rng.uniform(1.0, 5.0, size=(N, n_inputs)).astype(np.float64)
    y_placeholder = np.zeros(N, dtype=np.float64)
    data = np.asfortranarray(np.column_stack([cols, y_placeholder]))
    ds = op.Dataset(data)
    vmap = {v.Name: v.Hash for v in ds.Variables}
    hash2idx = {int(v.Hash): int(v.Index) for v in ds.Variables}
    return ds, vmap, hash2idx, cols


def _operon_truth(tree, ds):
    """Ground truth via Operon's own evaluator (spec-prescribed form)."""
    out = np.zeros(N, dtype=np.float32)
    op.EvaluateTrees([tree], ds, op.Range(0, N), out, 1)
    return out.astype(np.float64)


def _roundtrip_assert(tree, ds, hash2idx, cols):
    """convert -> ref_eval, compare to Operon truth; check K == CoefficientsCount."""
    truth = _operon_truth(tree, ds)
    nt, nv, ci, c_init = convert(tree, hash2idx)

    K = len(c_init)
    assert K == tree.CoefficientsCount, (
        f"K={K} != CoefficientsCount={tree.CoefficientsCount}"
    )

    X = cols.astype(np.float32)
    ours = np.asarray(ref_eval(nt, nv, ci, c_init, X), dtype=np.float64)

    finite = np.isfinite(truth) & np.isfinite(ours)
    assert finite.any(), "no finite entries to compare"
    max_abs = float(np.max(np.abs(truth[finite] - ours[finite])))
    assert max_abs < 1e-3, f"max|truth - ours|={max_abs:.3e}"
    return nt, nv, ci, c_init


# Expressions verified to parse correctly through this pyoperon wheel
# (spaced `*` only). (expr, n_inputs)
_EXPRS = [
    ("X1 - X2", 2),                       # REQUIRED golden case (non-commutative SUB)
    ("(X1 - X2) / (X1 + X3)", 3),         # REQUIRED golden case (nested, DIV)
    ("X1 / X2", 2),                       # non-commutative DIV
    ("3.0 * X1 - 2.0 * X2", 2),           # weighted vars + explicit constants
    ("sin(X1) * X2", 2),                  # unary SIN + product
    ("cos(X1) * X2", 2),                  # unary COS — pins the COS func id (vs TAN swap)
    ("tan(X1) * X2", 2),                  # unary TAN — pins the TAN func id (vs COS swap)
]


@pytest.mark.parametrize("expr,n_inputs", _EXPRS)
def test_roundtrip_parsed(expr, n_inputs):
    ds, vmap, hash2idx, cols = _make_dataset(n_inputs)
    tree = op.InfixParser.Parse(expr, vmap)
    _roundtrip_assert(tree, ds, hash2idx, cols)


def test_roundtrip_single_constant():
    """A bare Constant leaf -> one CONST node, K==1, value preserved."""
    ds, vmap, hash2idx, cols = _make_dataset(1)
    tree = op.InfixParser.Parse("3.5", vmap)
    nt, nv, ci, c_init = _roundtrip_assert(tree, ds, hash2idx, cols)
    assert len(c_init) == 1 and abs(float(c_init[0]) - 3.5) < 1e-6


def test_roundtrip_single_variable_unit_weight():
    """A bare Variable leaf -> MUL(CONST=weight, VAR=col); weight defaults to 1.0."""
    ds, vmap, hash2idx, cols = _make_dataset(2)
    tree = op.InfixParser.Parse("X2", vmap)
    nt, nv, ci, c_init = _roundtrip_assert(tree, ds, hash2idx, cols)
    assert tree.CoefficientsCount == 1


def test_roundtrip_nonunit_variable_weight():
    """Manually weight a Variable node (2.5 * X1 - X2) so the Variable->MUL(CONST,VAR)
    expansion is exercised with a NON-unit weight (the parser-built cases all carry
    weight 1.0). Postfix children-before-parent: the child nearest the parent
    (operand[0]) is the minuend."""
    ds, vmap, hash2idx, cols = _make_dataset(2)

    def var_node(hash_value, weight):
        nd = op.Node(op.NodeType.Variable)
        nd.Value = float(weight)
        nd.HashValue = int(hash_value)
        return nd  # Optimize defaults True -> counts as a coefficient

    # (2.5*X1) - (1.0*X2): postfix [X2, X1, Sub] with X1 nearest parent = minuend
    tree = op.Tree([
        var_node(vmap["X2"], 1.0),
        var_node(vmap["X1"], 2.5),
        op.Node.Sub(),
    ]).UpdateNodes()

    nt, nv, ci, c_init = _roundtrip_assert(tree, ds, hash2idx, cols)
    # two leaves -> two coefficients; the weights 2.5 and 1.0 must appear in c_init
    assert len(c_init) == 2
    assert np.isclose(sorted(map(float, c_init)), [1.0, 2.5]).all()


def test_determinism_identical_bytes():
    """Converting the same tree twice yields identical arrays (deterministic)."""
    ds, vmap, hash2idx, cols = _make_dataset(3)
    tree = op.InfixParser.Parse("(X1 - X2) / (X1 + X3)", vmap)
    a = convert(tree, hash2idx)
    b = convert(tree, hash2idx)
    for arr_a, arr_b in zip(a, b):
        assert np.array_equal(np.asarray(arr_a), np.asarray(arr_b))


def test_ci_prefix_rank_order():
    """ci must be -1 for non-CONST and 0,1,..,K-1 in prefix order for CONST nodes,
    matching c_init order (the loader/inspect ci invariant)."""
    ds, vmap, hash2idx, cols = _make_dataset(2)
    tree = op.InfixParser.Parse("3.0 * X1 - 2.0 * X2", vmap)
    nt, nv, ci, c_init = convert(tree, hash2idx)
    seen = 0
    for t, cidx in zip(nt, ci):
        if int(t) == 1:  # CONST
            assert int(cidx) == seen
            seen += 1
        else:
            assert int(cidx) == -1
    assert seen == len(c_init)


# --- write_operon_pop_bin: convert-per-tree + honesty drop-not-crash path -----
# (spec §9: count+log+filter the offending tree, never abort the whole dump.)

def test_write_operon_pop_bin_happy_path(tmp_path):
    """Raw Operon trees go in; convert runs inside the writer; pop.bin round-trips."""
    ds, vmap, hash2idx, cols = _make_dataset(2)
    X = cols.astype(np.float32)
    y = np.zeros(N, dtype=np.float32)
    trees = [op.InfixParser.Parse(e, vmap) for e in ("X1 - X2", "3.0 * X1 - 2.0 * X2")]

    stats = write_operon_pop_bin(trees, hash2idx, X, y, tmp_path / "pop.bin")

    assert stats["n_in"] == 2 and stats["n_kept"] == 2
    assert stats["dropped_unsupported"] == 0 and stats["dropped_k_over"] == 0
    assert stats["dropped_bad_type"] == 0 and stats["dropped_nonfinite"] == 0
    assert stats["M_prob"] == 2

    pop = read_pop_bin(tmp_path / "pop.bin")
    assert pop["M_prob"] == 2
    # first tree (X1 - X2) -> MUL(1,X0) - MUL(1,X1); evaluate the slice and compare.
    off, n, coff, K = pop["metas"][0]
    pred = ref_eval(pop["nt"][off:off + n], pop["nv"][off:off + n],
                    pop["ci"][off:off + n], pop["c_init"][coff:coff + K], pop["xs"])
    assert np.max(np.abs(pred.astype(np.float64) - (cols[:, 0] - cols[:, 1]))) < 1e-3


def test_write_operon_pop_bin_drops_unsupported_not_crash(tmp_path):
    """A Variable whose hash is absent from hash2idx makes convert() raise
    UnsupportedNodeType. The writer must COUNT+LOG+FILTER that one tree (spec §9
    honesty path), keep the good trees, and NOT abort the whole dump."""
    ds, vmap, hash2idx, cols = _make_dataset(2)
    X = cols.astype(np.float32)
    y = np.zeros(N, dtype=np.float32)

    good = op.InfixParser.Parse("X1 - X2", vmap)
    # Variable node with an out-of-dataset hash (the finding's own repro; also the
    # post-ChangeVariableMutation case). convert(good)==K2, convert(bad) raises.
    bad_leaf = op.Node(op.NodeType.Variable)
    bad_leaf.Value = 1.0
    bad_leaf.HashValue = 999999999
    bad = op.Tree([bad_leaf]).UpdateNodes()

    # Sanity: convert(bad) really raises UnsupportedNodeType (the path under test).
    with pytest.raises(UnsupportedNodeType):
        convert(bad, hash2idx)

    stats = write_operon_pop_bin([good, bad, good], hash2idx, X, y, tmp_path / "pop.bin")

    assert stats["n_in"] == 3
    assert stats["dropped_unsupported"] == 1, "the bad-hash tree must be counted, not crash"
    assert stats["n_kept"] == 2 and stats["M_prob"] == 2
    # the dump exists and is well-formed despite the dropped tree
    pop = read_pop_bin(tmp_path / "pop.bin")
    assert pop["M_prob"] == 2
