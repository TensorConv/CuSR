"""test_shared_header.py — P1 step "shared-header" regression guard.

See docs/kernel/INPROCESS_CO_PLAN.md, implementation step 1 (EXTRACT CORE):

    "Factor the __global__ kernels + the LM-loop body into a shared header
     lm_core.cuh #include'd by BOTH the standalone and co_lib.cu so device
     codegen cannot drift (byte-parity is FMA-contraction sensitive)."

This step is a PURE refactor: the shared device kernels (eval_tree_d,
eval_kernel_batched, residual_kernel, loss_kernel, build_jtj_jtr_kernel) +
the shared scalar helpers (STATUS_* codes, CUDA_CHECK, write_blob) move out of
the two standalone .cu files into cusr/kernel/lm_core.cuh, which BOTH
batch_lm.cu (FD) and batch_lm_ad.cu (AD) then #include. No behavior change.

GATES (all must hold AFTER the extraction):
  1. test_shared_header_exists — lm_core.cuh exists.
  2. test_both_standalones_include_shared_header — batch_lm.cu AND batch_lm_ad.cu
     literally `#include "lm_core.cuh"`. (Without this the "single source of
     truth" claim is vacuous — the kernels could still be duplicated.)
  3. test_shared_kernels_not_duplicated — the shared __global__ kernel
     DEFINITIONS no longer live in either standalone .cu (they were extracted,
     not copied). A duplicate definition is both a codegen-drift hazard and a
     compile error; assert the bodies are gone from the .cu files.
  4. test_rebuilt_standalones_match_golden — REGRESSION GUARD. Rebuild both
     standalones from the refactored sources with the repo's NVCC_FLAGS and
     assert their output is STILL byte-identical to the step-1 goldens. If any
     byte differs, the extraction altered an expression — this is the load-
     bearing gate the step's `green` rides on.

Build/run (GPU pinned, device idle):
    source scripts/env.sh
    export CUDA_VISIBLE_DEVICES=2
    uv run pytest cusr/kernel/tests/test_shared_header.py -v
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# Reuse the single source of truth for variants / pops / MAX_ITER / golden dir.
import sys
HERE = Path(__file__).resolve().parent          # cusr/kernel/tests
sys.path.insert(0, str(HERE))
from test_inproc_parity import (  # noqa: E402
    GOLDEN, MAX_ITER, POPS, PROJ, VARIANTS,
)

SHARED_HEADER = PROJ / "lm_core.cuh"

# The __global__ / __device__ entities that the refactor extracts into the
# shared header. These are exactly the functions that were textually identical
# in batch_lm.cu and batch_lm_ad.cu before the refactor.
SHARED_KERNELS = (
    "eval_tree_d",
    "eval_kernel_batched",
    "residual_kernel",
    "loss_kernel",
    "build_jtj_jtr_kernel",
)

# The two standalone .cu files that must now share the header.
STANDALONE_SRC = {
    "fd": PROJ / "batch_lm.cu",
    "ad": PROJ / "batch_lm_ad.cu",
}

NVCC_FLAGS = "-O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math"


# ----------------------------------------------------------------------------
# GATE 1: the shared header exists.
# ----------------------------------------------------------------------------
def test_shared_header_exists():
    assert SHARED_HEADER.exists(), (
        f"{SHARED_HEADER} not created yet — step 'shared-header' extracts the "
        f"shared __global__ kernels + scalar helpers here."
    )


# ----------------------------------------------------------------------------
# GATE 2: both standalones #include the shared header.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("variant", list(STANDALONE_SRC))
def test_both_standalones_include_shared_header(variant):
    src = STANDALONE_SRC[variant].read_text()
    assert '#include "lm_core.cuh"' in src, (
        f"{STANDALONE_SRC[variant].name} does not #include \"lm_core.cuh\" — "
        f"the shared-header single-source-of-truth is not wired in."
    )


# ----------------------------------------------------------------------------
# GATE 3: the shared kernels are DEFINED in the header, not duplicated in .cu.
# A definition in a .cu while also in the (included) header is a redefinition
# compile error; this gate makes the "extracted, not copied" invariant explicit
# and catches a partial refactor that left a stale copy behind.
# ----------------------------------------------------------------------------
def test_shared_kernels_defined_in_header():
    hdr = SHARED_HEADER.read_text()
    for k in SHARED_KERNELS:
        # a definition opens a brace `(... ) {` — accept the function signature.
        assert k in hdr, f"shared kernel {k} not present in {SHARED_HEADER.name}"


@pytest.mark.parametrize("variant", list(STANDALONE_SRC))
def test_shared_kernels_not_duplicated_in_standalone(variant):
    src = STANDALONE_SRC[variant].read_text()
    dupes = []
    for k in SHARED_KERNELS:
        # A DEFINITION looks like `<qualifiers> <type> <name>(...) {`. We detect
        # a leftover definition by the kernel's launch-config-free signature line
        # ending in `{` on the same or next lines. Simpler robust heuristic: the
        # opening `__global__ void <name>(` or `__device__ float <name>(` token.
        for marker in (f"__global__ void {k}(", f"__device__ float {k}("):
            if marker in src:
                dupes.append((k, marker))
    assert not dupes, (
        f"{STANDALONE_SRC[variant].name} still DEFINES shared kernels "
        f"{[d[0] for d in dupes]} — they must be extracted to lm_core.cuh, not "
        f"duplicated (codegen-drift + redefinition hazard)."
    )


# ----------------------------------------------------------------------------
# GATE 4 (load-bearing): rebuild both standalones from the refactored sources
# and assert byte-for-byte parity with the step-1 goldens.
# ----------------------------------------------------------------------------
def _build(target: str) -> None:
    r = subprocess.run(
        ["make", target],
        cwd=str(PROJ), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"make {target} exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )


def _run_standalone(binary: Path, pop: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(binary), str(pop), str(out_dir), "--quiet", "--max-iter", str(MAX_ITER)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{binary.name} exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )


@pytest.mark.parametrize(
    "variant,pop",
    [(v, p) for v in VARIANTS for p in POPS],
    ids=[f"{v}-{p}" for v in VARIANTS for p in POPS],
)
def test_rebuilt_standalones_match_golden(variant, pop, tmp_path):
    spec = VARIANTS[variant]
    binary = spec["binary"]
    # Force a rebuild from the (refactored) sources so the byte-parity check is
    # against freshly-compiled-from-the-header codegen, not a stale binary.
    target = binary.name
    if binary.exists():
        binary.unlink()
    _build(target)
    assert binary.exists(), f"{target} did not build"

    pop_path = POPS[pop]
    gdir = GOLDEN / variant / pop
    missing = [n for n in spec["outputs"] if not (gdir / n).exists()]
    assert not missing, f"golden missing for {variant}/{pop}: {missing}"

    out = tmp_path / "fresh"
    _run_standalone(binary, pop_path, out)
    for name in spec["outputs"]:
        fresh = (out / name).read_bytes()
        golden = (gdir / name).read_bytes()
        assert fresh == golden, (
            f"REGRESSION: {variant}/{pop}/{name} differs from golden after the "
            f"shared-header extraction — the refactor altered an expression. "
            f"(fresh {len(fresh)} B vs golden {len(golden)} B)"
        )
