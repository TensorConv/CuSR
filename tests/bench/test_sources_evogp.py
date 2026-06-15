from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch

# evogp Python package must import (it lazily loads the CUDA .so on first use,
# so plain construction of a CPU Tree doesn't need a GPU).
_evogp = pytest.importorskip("evogp")

from evogp.tree import Tree  # noqa: E402
from evogp.tree.utils import Func, NType  # noqa: E402

from cusr.bench.sources.evogp import forest_member_to_skeleton  # noqa: E402


def _make_linear_tree(max_len: int = 16) -> Tree:
    """Build a Tree for `2.0 * x0 + 1.0` by hand in prefix order (CPU tensors).

    Prefix: ADD, MUL, 2.0, x0, 1.0
    - node 0: BFUNC/ADD, subtree_size 5
    - node 1: BFUNC/MUL, subtree_size 3
    - node 2: CONST 2.0, subtree_size 1
    - node 3: VAR x0 (encoded as index 0), subtree_size 1
    - node 4: CONST 1.0, subtree_size 1

    Tree ctor stores tensors verbatim; no CUDA needed for this test path.
    """
    value = torch.zeros(max_len, dtype=torch.float32)
    ntype = torch.zeros(max_len, dtype=torch.int16)
    ssize = torch.zeros(max_len, dtype=torch.int16)

    value[0] = float(Func.ADD)
    ntype[0] = NType.BFUNC
    ssize[0] = 5

    value[1] = float(Func.MUL)
    ntype[1] = NType.BFUNC
    ssize[1] = 3

    value[2] = 2.0
    ntype[2] = NType.CONST
    ssize[2] = 1

    value[3] = 0.0  # variable index 0
    ntype[3] = NType.VAR
    ssize[3] = 1

    value[4] = 1.0
    ntype[4] = NType.CONST
    ssize[4] = 1

    return Tree(input_len=1, output_len=1, node_value=value, node_type=ntype, subtree_size=ssize)


def test_linear_tree_to_skeleton():
    tree = _make_linear_tree()
    skel, init, degraded = forest_member_to_skeleton(tree, problem_n_vars=1)

    # Structural equivalence: expr should equal c0*x0 + c1 (up to commutativity).
    c0, c1 = sp.symbols("c0 c1", real=True)
    x0 = sp.symbols("x0", real=True)
    expected = c0 * x0 + c1
    assert sp.simplify(skel.expr - expected) == 0

    # Constant values are captured in encounter order (prefix walk).
    assert init.shape == (2,)
    np.testing.assert_allclose(init, [2.0, 1.0], atol=1e-6)

    # No loose ops in this tree.
    assert degraded == []
    assert skel.metadata["degraded_ops"] == []

    # Sanity: skeleton variable / constant symbols.
    assert skel.n_vars == 1
    assert skel.n_constants == 2
    assert [v.name for v in skel.variables] == ["x0"]
    assert [c.name for c in skel.constants] == ["c0", "c1"]

    # Skeleton can evaluate: plug in the init constants and recover 2*x+1.
    X = np.array([[0.0], [1.0], [-1.0], [2.5]])
    pred = skel.evaluate(init, X)
    np.testing.assert_allclose(pred, 2.0 * X[:, 0] + 1.0, atol=1e-6)
