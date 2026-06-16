"""test_sr_problems.py — parity of the shared problem library + sampler.

Two guarantees:
1. ``sample_xy`` reproduces ``y = skeleton(c_true, X)`` for every problem
   (sanity that the sampler math matches the declared skeleton; for noise=0 it
   must be exact in fp32 up to lambdify rounding).
2. ``sr_problems.PROBLEMS`` is identical to ``dump_evogp.py``'s ``PROBLEMS`` dict,
   so the Operon path and the evogp path share one definition. We extract
   dump_evogp's dict by parsing its **source** with ``ast`` — never importing it,
   since that pulls in evogp/torch (CUDA init) which is absent in operon-venv.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from cusr.benchmark.workload import sr_problems
from cusr.benchmark.workload.sr_problems import PROBLEMS, sample_xy

REPO_ROOT = Path(__file__).resolve().parents[4]
DUMP_EVOGP_SRC = REPO_ROOT / "cusr" / "kernel" / "dump_evogp.py"


def _extract_assign_from_source(src_path: Path, name: str):
    """Return the literal value of a module-level ``name = <literal>`` assignment
    by parsing ``src_path`` with ast — without importing the module."""
    tree = ast.parse(src_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found as a module-level assignment in {src_path}")


def _extract_func_from_source(src_path: Path, name: str):
    """Return a callable for the module-level ``def name(...)`` in ``src_path`` by
    parsing+compiling just that FunctionDef — without importing the module (which
    would pull in torch/evogp + CUDA init, absent from operon-venv).

    The function body may ``import`` what it needs (``dump_evogp._sample_xy`` does
    ``import sympy`` internally), so only ``np`` must be supplied in globals."""
    tree = ast.parse(src_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"np": np}
            exec(compile(mod, f"<{src_path.name}:{name}>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"def {name} not found as a module-level function in {src_path}")


def test_problems_match_dump_evogp_source():
    """sr_problems.PROBLEMS must equal dump_evogp.py::PROBLEMS (by source AST)."""
    assert DUMP_EVOGP_SRC.exists(), f"missing {DUMP_EVOGP_SRC}"
    dump_problems = _extract_assign_from_source(DUMP_EVOGP_SRC, "PROBLEMS")
    assert set(dump_problems) == set(PROBLEMS), (
        f"problem id mismatch: only-in-dump={set(dump_problems) - set(PROBLEMS)} "
        f"only-in-sr={set(PROBLEMS) - set(dump_problems)}"
    )
    assert dump_problems == PROBLEMS, "PROBLEMS dicts differ between sr_problems and dump_evogp"


def test_funcset_and_const_samples_match_dump_evogp_source():
    """USING_FUNCS / CONST_SAMPLES must also match (parity of the GP funcset)."""
    assert _extract_assign_from_source(DUMP_EVOGP_SRC, "USING_FUNCS") == sr_problems.USING_FUNCS
    assert _extract_assign_from_source(DUMP_EVOGP_SRC, "CONST_SAMPLES") == sr_problems.CONST_SAMPLES


@pytest.mark.parametrize("problem_id", sorted(PROBLEMS))
def test_sample_xy_reproduces_skeleton(problem_id):
    """y returned by sample_xy must equal skeleton(c_true, X) recomputed in fp64."""
    prob = PROBLEMS[problem_id]
    N = 256
    seed = 0
    X, y = sample_xy(problem_id, N, seed, noise=0.0)

    assert X.dtype == np.float32 and y.dtype == np.float32
    assert X.shape == (N, len(prob["variables"]))
    assert y.shape == (N,)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))

    # Independently recompute y from X via the skeleton (fp64) and compare.
    var_syms = tuple(sp.Symbol(v) for v in prob["variables"])
    const_syms = tuple(sp.Symbol(c) for c in prob["constants"])
    locals_ = {str(s): s for s in (*var_syms, *const_syms)}
    expr = sp.sympify(prob["skeleton_expr"], locals=locals_)
    expr = expr.subs({c: float(v) for c, v in
                      zip(const_syms, prob["ground_truth_constants"])})
    f = sp.lambdify(var_syms, expr, modules="numpy")
    y_ref = np.asarray(f(*X.astype(np.float64).T), dtype=np.float64).reshape(-1)

    # y was cast to fp32; compare with an fp32-scale relative tolerance.
    scale = max(1.0, float(np.max(np.abs(y_ref))))
    max_abs = float(np.max(np.abs(y.astype(np.float64) - y_ref)))
    assert max_abs < 1e-4 * scale, f"{problem_id}: max|y - skeleton(X)|={max_abs:.3e} (scale={scale:.3g})"


@pytest.mark.parametrize("problem_id,N,seed,noise", [
    ("feynman/I.18.12", 128, 0, 0.0),   # default problem (repo pop.bin source)
    ("feynman/I.12.1", 256, 2, 0.05),   # noise>0: pins the rng draw order (X then noise)
    ("nguyen/5", 64, 1, 0.0),           # sin/cos skeleton, constant-free
])
def test_sample_xy_byte_parity_with_dump_evogp(problem_id, N, seed, noise):
    """sr_problems.sample_xy must reproduce dump_evogp._sample_xy bit-for-bit for the
    same (problem, N, seed, noise) — the corpus-comparability guarantee (spec §8).

    test_sample_xy_reproduces_skeleton only checks sample_xy against its OWN re-derived
    sympy skeleton (self-consistency); the PROBLEMS/USING_FUNCS/CONST_SAMPLES *dicts* are
    AST-compared, but the verbatim-copied _sample_xy numerics (rng draw order, dtype-cast
    timing, noise model) were unpinned. This drives dump_evogp's actual sampler (extracted
    from source via ast+exec — no torch/evogp import) and asserts np.array_equal."""
    dump_sample_xy = _extract_func_from_source(DUMP_EVOGP_SRC, "_sample_xy")
    prob = sr_problems.PROBLEMS[problem_id]

    X_sr, y_sr = sample_xy(problem_id, N, seed, noise=noise)
    X_dump, y_dump = dump_sample_xy(prob, N, seed, noise=noise)

    assert np.array_equal(X_sr, X_dump), f"{problem_id}: X differs from dump_evogp._sample_xy"
    assert np.array_equal(y_sr, y_dump), f"{problem_id}: y differs from dump_evogp._sample_xy"
    assert X_sr.dtype == X_dump.dtype == np.float32
    assert y_sr.dtype == y_dump.dtype == np.float32


def test_sample_xy_deterministic_and_seed_sensitive():
    """Same (id, N, seed) -> identical bytes; different seed -> different X."""
    X0a, y0a = sample_xy("feynman/I.18.12", 128, 0)
    X0b, y0b = sample_xy("feynman/I.18.12", 128, 0)
    assert np.array_equal(X0a, X0b) and np.array_equal(y0a, y0b)
    X1, _ = sample_xy("feynman/I.18.12", 128, 1)
    assert not np.array_equal(X0a, X1)


def test_sample_xy_noise_changes_y_only():
    """noise>0 perturbs y (RMS-relative) but leaves X identical (same rng draw order:
    X is drawn first, then noise)."""
    X0, y0 = sample_xy("feynman/I.12.1", 256, 3, noise=0.0)
    Xn, yn = sample_xy("feynman/I.12.1", 256, 3, noise=0.05)
    assert np.array_equal(X0, Xn), "X must be identical regardless of noise"
    assert not np.array_equal(y0, yn), "noise>0 must perturb y"
