"""test_inproc_parity.py — in-process CO drop-in (P1) parity suite.

See docs/kernel/INPROCESS_CO_PLAN.md.

STEP baseline-golden (this commit): there is NO .so yet. The two gates here are
the durable BYTE-DETERMINISM baseline the audit demanded (MUST-FIX #2) plus the
saved golden references that every later parity step (single-call,
parity-under-reuse, status-branch span, end-to-end) compares against.

  1. test_standalone_determinism — run EACH standalone variant TWICE on the same
     pop and assert c_final.bin + status.bin (+ AD loss_init/loss_final) are
     byte-for-byte identical. If the kernel were nondeterministic run-to-run,
     byte-parity vs a golden would be meaningless, so this gate must pass first.
  2. test_golden_matches_standalone — assert the committed golden files exist and
     that a FRESH standalone run reproduces them byte-for-byte. This proves the
     golden is a faithful, reproducible reference (not a stale/hand-edited blob).

MUST-FIX #1: parity is always pinned to the SAME variant on both sides (AD .so vs
AD standalone; FD .so vs FD standalone). Goldens are therefore per-variant.

Build prerequisite (run once):
    source scripts/env.sh
    cd cusr/kernel && make batch_lm batch_lm_ad
Run (GPU pinned, device must be idle):
    source scripts/env.sh
    export CUDA_VISIBLE_DEVICES=2
    uv run pytest cusr/kernel/tests/test_inproc_parity.py -v
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent          # cusr/kernel/tests
PROJ = HERE.parent                              # cusr/kernel
ROOT = HERE.parents[2]                          # repo root
GOLDEN = HERE / "golden"

# Fixed across golden-gen and every later parity compare. Goldens are only valid
# for THIS max_iter; later steps must call the .so with the same value.
MAX_ITER = 50

# ---- variants -------------------------------------------------------------
# Each variant: standalone binary + the exact set of output blobs it writes
# (FD writes c_final/status; AD additionally writes loss_init/loss_final).
VARIANTS = {
    "fd": {
        "binary": PROJ / "batch_lm",
        "outputs": ("c_final.bin", "status.bin"),
    },
    "ad": {
        "binary": PROJ / "batch_lm_ad",
        "outputs": ("c_final.bin", "status.bin", "loss_init.bin", "loss_final.bin"),
    },
}

# ---- representative pops ---------------------------------------------------
# pop.bin: real EvoGP dump, M=1000, K_max=8 (the parity-gate workhorse).
# operon snapshot: M=4000, K_max=28 — high-K, stresses Cholesky/fail paths.
OPERON_SNAP = (ROOT / "data" / "workload" / "snapshots"
               / "operon_feynman_I.12.1_pop4000_noise0_len64_seed0" / "pop_gen0100.bin")

POPS = {
    "pop1000": ROOT / "data" / "fixtures" / "pop.bin",
}
if OPERON_SNAP.exists():
    POPS["operon4000_g100"] = OPERON_SNAP

CASES = [(v, p) for v in VARIANTS for p in POPS]


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_binary(binary: Path) -> None:
    if not binary.exists():
        pytest.skip(f"{binary.name} not built — run `make batch_lm batch_lm_ad` in {PROJ}")


# --------------------------------------------------------------------------
# GATE 1 (FIRST): run-to-run byte-determinism of each standalone variant.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("variant,pop", CASES, ids=[f"{v}-{p}" for v, p in CASES])
def test_standalone_determinism(variant, pop, tmp_path):
    spec = VARIANTS[variant]
    binary = spec["binary"]
    _require_binary(binary)
    pop_path = POPS[pop]
    assert pop_path.exists(), f"pop not found: {pop_path}"

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    _run_standalone(binary, pop_path, out_a)
    _run_standalone(binary, pop_path, out_b)

    for name in spec["outputs"]:
        fa, fb = out_a / name, out_b / name
        assert fa.exists(), f"{binary.name} did not write {name}"
        assert fb.exists(), f"{binary.name} did not write {name}"
        ba, bb = fa.read_bytes(), fb.read_bytes()
        # byte-for-byte; surface the sha if it ever diverges
        assert ba == bb, (
            f"{variant} {pop} {name} NOT byte-identical run-to-run "
            f"(sha A={_sha256(fa)} B={_sha256(fb)}, len {len(ba)} vs {len(bb)})"
        )


# --------------------------------------------------------------------------
# GATE 2: committed goldens exist AND a fresh standalone run reproduces them
# byte-for-byte (proves the golden is a faithful, regenerable reference).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("variant,pop", CASES, ids=[f"{v}-{p}" for v, p in CASES])
def test_golden_matches_standalone(variant, pop, tmp_path):
    spec = VARIANTS[variant]
    binary = spec["binary"]
    _require_binary(binary)
    pop_path = POPS[pop]
    assert pop_path.exists(), f"pop not found: {pop_path}"

    gdir = GOLDEN / variant / pop
    missing = [n for n in spec["outputs"] if not (gdir / n).exists()]
    assert not missing, (
        f"golden missing for {variant}/{pop}: {missing} — regenerate with "
        f"`python {HERE/'gen_golden.py'}` (GPU pinned, idle)"
    )

    out = tmp_path / "fresh"
    _run_standalone(binary, pop_path, out)
    for name in spec["outputs"]:
        fresh = (out / name).read_bytes()
        golden = (gdir / name).read_bytes()
        assert fresh == golden, (
            f"golden DRIFT for {variant}/{pop}/{name}: fresh standalone run does "
            f"not match committed golden (sha fresh={_sha256(out/name)} "
            f"golden={_sha256(gdir/name)}). Rebuild may have changed codegen, or "
            f"golden is stale — regenerate."
        )
