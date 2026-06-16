"""test_pop_io.py — pop.bin writer/reader + C-inspector + ref_eval round-trip.

Builds a hand-made tiny population (a multi-coefficient tree + a single-CONST
tree), writes it with ``write_pop_bin``, and asserts:
  * the bytes pass ``cusr/kernel/inspect`` (subprocess, exit 0 + "PASS"),
  * ``read_pop_bin`` mirrors ``loader.c`` (header + arrays + invariants),
  * the writer round-trips byte-exactly (re-read arrays equal the inputs),
  * ``ym`` is the shared ``y`` tiled M_prob times,
  * ``ref_eval`` reproduces the analytic ``y`` for the hand-built tree,
  * ``compute_max_stack`` matches the simulated depth.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from cusr.benchmark.workload.pop_io import (
    MAX_K, Func, NType, compute_max_stack, read_pop_bin, write_pop_bin,
)
from cusr.benchmark.workload.pop_ref import ref_eval

from .conftest import INSPECT_BIN


# --- hand-built population -------------------------------------------------
# tree A (prefix):  c0*x0 + c1*x1
#   ADD( MUL(CONST c0, VAR x0), MUL(CONST c1, VAR x1) )
_NT_A = np.array([NType.BFUNC, NType.BFUNC, NType.CONST, NType.VAR,
                  NType.BFUNC, NType.CONST, NType.VAR], np.int32)
_NV_A = np.array([Func.ADD, Func.MUL, 0.0, 0.0, Func.MUL, 0.0, 1.0], np.float32)
_CI_A = np.array([-1, -1, 0, -1, -1, 1, -1], np.int32)
_C_A = np.array([2.0, -3.0], np.float32)

# tree B: a single Constant leaf (= c, K=1)
_NT_B = np.array([NType.CONST], np.int32)
_NV_B = np.array([0.0], np.float32)
_CI_B = np.array([0], np.int32)
_C_B = np.array([7.0], np.float32)


def _make_pop(N=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(1.0, 5.0, size=(N, 2)).astype(np.float32)
    y = (2.0 * X[:, 0] + (-3.0) * X[:, 1]).astype(np.float32)  # = tree A
    trees = [(_NT_A, _NV_A, _CI_A, _C_A), (_NT_B, _NV_B, _CI_B, _C_B)]
    return trees, X, y


def test_compute_max_stack():
    # tree A: deepest pending = ADD waiting on left MUL while right operand on stack
    assert compute_max_stack(_NT_A) == 3
    assert compute_max_stack(_NT_B) == 1


def test_write_read_roundtrip(tmp_path):
    trees, X, y = _make_pop()
    out = tmp_path / "pop.bin"
    stats = write_pop_bin(out, trees, X, y)

    assert stats == dict(M_prob=2, total_nodes=8, total_c=3, N=16, n_vars=2,
                         K_max=2, max_stack=3)

    pop = read_pop_bin(out)
    assert pop["magic"] == 0x4D4C344D and pop["version"] == 1
    assert (pop["M_prob"], pop["total_nodes"], pop["total_c"]) == (2, 8, 3)
    assert (pop["N"], pop["n_vars"], pop["K_max"], pop["max_stack"]) == (16, 2, 2, 3)

    # arrays round-trip byte-exactly
    assert np.array_equal(pop["nt"], np.concatenate([_NT_A, _NT_B]))
    assert np.array_equal(pop["nv"], np.concatenate([_NV_A, _NV_B]))
    assert np.array_equal(pop["ci"], np.concatenate([_CI_A, _CI_B]))
    assert np.allclose(pop["c_init"], [2.0, -3.0, 7.0])
    assert np.array_equal(pop["xs"], X)

    # metas: (node_offset, n_nodes, c_offset, K)
    assert pop["metas"][0].tolist() == [0, 7, 0, 2]
    assert pop["metas"][1].tolist() == [7, 1, 2, 1]

    # ym = shared y tiled M_prob times
    assert pop["ym"].shape == (2, 16)
    assert np.array_equal(pop["ym"][0], y)
    assert np.array_equal(pop["ym"][0], pop["ym"][1])


def test_ref_eval_matches_analytic(tmp_path):
    trees, X, y = _make_pop()
    out = tmp_path / "pop.bin"
    write_pop_bin(out, trees, X, y)
    pop = read_pop_bin(out)

    off, n, coff, K = pop["metas"][0]
    pred = ref_eval(pop["nt"][off:off + n], pop["nv"][off:off + n],
                    pop["ci"][off:off + n], pop["c_init"][coff:coff + K], pop["xs"])
    assert np.max(np.abs(pred.astype(np.float64) - y.astype(np.float64))) < 1e-4

    # tree B is a bare constant -> 7.0 everywhere
    off, n, coff, K = pop["metas"][1]
    predB = ref_eval(pop["nt"][off:off + n], pop["nv"][off:off + n],
                     pop["ci"][off:off + n], pop["c_init"][coff:coff + K], pop["xs"])
    assert np.allclose(predB, 7.0)


def test_inspect_passes(tmp_path):
    if not INSPECT_BIN.exists():
        pytest.skip(f"inspector not built ({INSPECT_BIN}); run: make -C cusr/kernel inspect")
    trees, X, y = _make_pop()
    out = tmp_path / "pop.bin"
    write_pop_bin(out, trees, X, y)

    r = subprocess.run([str(INSPECT_BIN), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, f"inspect failed rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "PASS" in r.stdout
    # no warnings about ci inconsistency / K-over / tfunc
    assert "WARNING" not in r.stdout
    assert "不一致" not in r.stdout


def test_empty_population(tmp_path):
    """Zero trees -> header all-zero counts, still a valid (degenerate) file."""
    X = np.zeros((4, 2), np.float32)
    y = np.zeros(4, np.float32)
    out = tmp_path / "pop_empty.bin"
    stats = write_pop_bin(out, [], X, y)
    assert stats["M_prob"] == 0 and stats["total_nodes"] == 0 and stats["K_max"] == 0
    pop = read_pop_bin(out)
    assert pop["M_prob"] == 0 and pop["c_init"].size == 0 and pop["ym"].size == 0


def test_max_k_constant():
    assert MAX_K == 32
