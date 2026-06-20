"""test_inproc_reuse.py — in-process CO drop-in (P1) parity-under-reuse + FD-.so parity.

STEP parity-reuse-and-fd (this commit). Builds on the per-variant byte-parity
goldens established by `gen_golden.py` / `test_inproc_parity.py` (step baseline-golden)
and the shared device header `lm_core.cuh` (step shared-header).

What this suite proves (docs/kernel/INPROCESS_CO_PLAN.md "Parity test design"):

  (b) SINGLE-CALL parity — for EACH variant, one `co_optimize` on a pop is
      byte-identical to that variant's own standalone golden. MUST-FIX #1: both
      sides are pinned to the SAME variant (AD .so vs AD golden, FD .so vs FD
      golden — NEVER AD-vs-FD). The FD `.so` is added this step.

  (a) PARITY-UNDER-REUSE (MUST-FIX #4 — THE discriminating gate) — in ONE
      persistent process, after ONE `co_init`, run a sequence of optimizes with
      VARYING M / total_c (popA -> popB larger -> popC smaller -> popA again, plus
      a 65536-then-1000 large-alloc-then-shrink stress) and assert EVERY call is
      byte-identical to its OWN single-call golden. A persistent context that does
      not re-seed `h_c` / reset `d_stat` / slice D2H to the current M (the bugs
      MUST-FIX #4 calls out) corrupts a later, smaller pop with a prior, larger
      pop's optimized constants — FINITE values that the `isfinite` writeback guard
      misses. This gate is the only detector, so it must actually catch that.

  (c) STATUS-BRANCH + MAGNITUDE-SPAN fixtures — synthesise small pops that drive
      each status branch (CONVERGED / MAXITER / FAIL_NAN / K0_SKIP) and assert .so
      == standalone per branch; use the operon high-K snapshot for FAIL_CHOLESKY
      and the scale-1e± fixtures for the magnitude span (if present).

The .so itself re-adds `load_pop_bin`'s sum-consistency checks (MUST-FIX #6) inside
`co_optimize` before the H2D copies — exercised here by every valid pop passing and
asserted negatively by `test_co_optimize_rejects_inconsistent_sums`.

Build prerequisite (run once):
    source scripts/env.sh
    cd cusr/kernel && make batch_lm batch_lm_ad libcusr_co_fd.so libcusr_co_ad.so
Run (GPU pinned, device must be idle):
    source scripts/env.sh
    export CUDA_VISIBLE_DEVICES=2
    uv run pytest cusr/kernel/tests/test_inproc_reuse.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent          # cusr/kernel/tests
PROJ = HERE.parent                              # cusr/kernel
ROOT = HERE.parents[2]                          # repo root
GOLDEN = HERE / "golden"

sys.path.insert(0, str(PROJ))                   # for co_inproc
sys.path.insert(0, str(ROOT))                   # for cusr.benchmark.popio

# Pinned identical to the goldens (test_inproc_parity.MAX_ITER); the .so MUST be
# called with this value or byte-parity fails legitimately.
MAX_ITER = 50

# ---- node-type / op enums (pop_format.h / ad_interp.cuh) -------------------
N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_SIN, F_LOG, F_EXP = 14, 20, 22

# ---- variants --------------------------------------------------------------
# Each variant: standalone binary + .so + the EXACT output blobs it defines.
# Goldens are PER-VARIANT (MUST-FIX #1): FD .so vs FD golden, AD .so vs AD golden.
VARIANTS = {
    "fd": {
        "binary": PROJ / "batch_lm",
        "so": PROJ / "libcusr_co_fd.so",
        "outputs": ("c_final", "status"),
    },
    "ad": {
        "binary": PROJ / "batch_lm_ad",
        "so": PROJ / "libcusr_co_ad.so",
        "outputs": ("c_final", "status", "loss_init", "loss_final"),
    },
}


# ---- representative on-disk pops (same as the goldens) ---------------------
OPERON_SNAP = (ROOT / "data" / "workload" / "snapshots"
               / "operon_feynman_I.12.1_pop4000_noise0_len64_seed0" / "pop_gen0100.bin")
POPS = {
    "pop1000": ROOT / "data" / "fixtures" / "pop.bin",
}
if OPERON_SNAP.exists():
    POPS["operon4000_g100"] = OPERON_SNAP


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _require(variant: str):
    """Load co_inproc + assert this variant's .so is built. FAIL (not skip) if the
    .so/module is missing on a GPU box — the step is not done until they exist."""
    spec = VARIANTS[variant]
    if not spec["so"].exists():
        pytest.fail(
            f"{spec['so'].name} not built — run "
            f"`make libcusr_co_fd.so libcusr_co_ad.so` in {PROJ}"
        )
    try:
        import co_inproc  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"import co_inproc failed: {e!r}")
    return spec


def _load_pop_dict(pop_path: Path) -> dict:
    from cusr.benchmark import popio
    return popio.load_pop_bin(pop_path)


def _run_standalone_blobs(binary: Path, pop_path: Path, tmp: Path) -> dict:
    """Run the standalone on a pop.bin file; return {blob_name: np.ndarray}."""
    tmp.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [str(binary), str(pop_path), str(tmp), "--quiet", "--max-iter", str(MAX_ITER)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{binary.name} exit={r.returncode}\n{r.stdout}\n{r.stderr}")
    out = {}
    out["c_final"] = np.fromfile(tmp / "c_final.bin", dtype=np.float32)
    out["status"] = np.fromfile(tmp / "status.bin", dtype=np.int32)
    if (tmp / "loss_init.bin").exists():
        out["loss_init"] = np.fromfile(tmp / "loss_init.bin", dtype=np.float32)
    if (tmp / "loss_final.bin").exists():
        out["loss_final"] = np.fromfile(tmp / "loss_final.bin", dtype=np.float32)
    return out


def _golden_blobs(variant: str, pop: str) -> dict:
    gdir = GOLDEN / variant / pop
    spec = VARIANTS[variant]
    dtypes = {"c_final": np.float32, "status": np.int32,
              "loss_init": np.float32, "loss_final": np.float32}
    out = {}
    for n in spec["outputs"]:
        f = gdir / f"{n}.bin"
        if not f.exists():
            return {}
        out[n] = np.fromfile(f, dtype=dtypes[n])
    return out


def _assert_bit_equal(got: dict, ref: dict, ctx: str, outputs) -> None:
    for n in outputs:
        g = np.ascontiguousarray(got[n])
        r = np.ascontiguousarray(ref[n])
        assert g.dtype == r.dtype, f"{ctx}/{n}: dtype {g.dtype} != {r.dtype}"
        assert g.shape == r.shape, f"{ctx}/{n}: shape {g.shape} != {r.shape}"
        assert g.tobytes() == r.tobytes(), (
            f"{ctx}/{n}: NOT byte-identical "
            f"(first diff at {np.argmax(g.view(np.uint8) != r.view(np.uint8))} of "
            f"{g.nbytes} B; got[:4]={g.ravel()[:4]} ref[:4]={r.ravel()[:4]})"
        )


# ===========================================================================
# (b) SINGLE-CALL parity: .so == standalone golden, SAME variant.
# ===========================================================================
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("pop", list(POPS))
def test_single_call_matches_golden(variant, pop):
    spec = _require(variant)
    import co_inproc
    pd = _load_pop_dict(POPS[pop])
    ref = _golden_blobs(variant, pop)
    if not ref:
        pytest.skip(f"golden absent for {variant}/{pop}")

    co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]),
                               variant=variant)
    try:
        got = co.optimize(pd, max_iter=MAX_ITER)
    finally:
        co.teardown()
    _assert_bit_equal(got, ref, f"{variant}/{pop}", spec["outputs"])


# ===========================================================================
# (a) PARITY-UNDER-REUSE: one co_init, varying M / total_c, every call byte-equal
#     to its OWN single-call standalone result. MUST-FIX #4.
# ===========================================================================
def _subset_pop(pd: dict, n: int) -> dict:
    """First n trees of pd as a fresh, self-consistent pop dict (offsets rebuilt)."""
    from cusr.benchmark import popio
    return popio.slice_pop(pd, n)


def _tile_pop(pd: dict, reps: int) -> dict:
    """Replicate pd's trees `reps` times -> larger M (offsets rebuilt by build_pop)."""
    from cusr.benchmark import popio
    M = pd["M"]
    trees = []
    for _ in range(reps):
        for m in range(M):
            no, nn, co_, K = pd["metas"][m].tolist()
            trees.append((pd["nt"][no:no + nn], pd["nv"][no:no + nn],
                          pd["ci"][no:no + nn], pd["c_init"][co_:co_ + K]))
    ym = np.tile(pd["ym"], (reps, 1))
    return popio.build_pop(trees, pd["xs"], ym)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parity_under_reuse(variant):
    """popA -> popB(larger) -> popC(smaller) -> popA again, ALL byte-identical to
    their own single-call standalone result, in ONE persistent process."""
    spec = _require(variant)
    import co_inproc

    base = _load_pop_dict(POPS["pop1000"])         # M=1000
    popA = _subset_pop(base, 600)                  # M=600
    popB = _subset_pop(base, 1000)                 # M=1000 (LARGER than A)
    popC = _subset_pop(base, 200)                  # M=200  (SMALLER than B)

    seq = [("A0", popA), ("B", popB), ("C", popC), ("A3", popA)]

    # standalone reference per distinct pop (run once on a tmp .bin each)
    import tempfile
    from cusr.benchmark import popio
    refs = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for tag, p in seq:
            key = p["M"]
            if key in refs:
                continue
            binp = td / f"pop_{key}.bin"
            popio.save_pop_bin(p, binp)
            refs[key] = _run_standalone_blobs(spec["binary"], binp, td / f"out_{key}")

    co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]),
                               variant=variant)
    try:
        for tag, p in seq:
            got = co.optimize(p, max_iter=MAX_ITER)
            _assert_bit_equal(got, refs[p["M"]],
                              f"{variant}/reuse[{tag}] M={p['M']}", spec["outputs"])
    finally:
        co.teardown()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parity_under_reuse_mixed_k(variant):
    """K_max-MIXING under reuse: every call byte-equal to its own reference (MUST-FIX #6).

    Buffers d_J = M*K_max*N and d_JtJ = M*K_max*K_max are PRODUCTS, so a realloc
    keyed on a single dim under-allocates when M shrinks but the product grows.
    This sequence VARIES K_max within ONE persistent context:
        pop1000(M=1000,K=8) -> operon(M=4000,K=28) -> low-K slice(M=50,K=8)
                            -> operon(M=4000,K=28) again
    and checks each call against its own reference, so an under-allocated d_J
    (stale/garbage J columns) corrupts JtJ -> different bytes. NOTE: it does NOT
    by itself discriminate byte-capacity from an M-keyed cap — here the big-product
    pop (operon, M=4000) is also the LARGEST-M pop, so an M-keyed cap would (re)grow
    on it too and not under-allocate. The dedicated under-allocation discriminator is
    test_parity_under_reuse_shrinkM_growK, which runs a LARGER-M/low-K pop BEFORE the
    smaller-M/high-K operon so M shrinks while the d_J product grows.
    Needs the operon snapshot (high K); skips otherwise."""
    if "operon4000_g100" not in POPS:
        pytest.skip("operon snapshot absent — mixed-K realloc test needs high K")
    spec = _require(variant)
    import co_inproc
    import tempfile
    from cusr.benchmark import popio

    base = _load_pop_dict(POPS["pop1000"])          # K_max=8
    operon = _load_pop_dict(POPS["operon4000_g100"])  # K_max=28, M=4000
    small = _subset_pop(base, 50)                    # K_max<=8, M=50 (shrink)

    # operon ref comes from the committed golden (huge to re-run twice); the small
    # pops get a one-shot standalone reference.
    operon_ref = _golden_blobs(variant, "operon4000_g100")
    if not operon_ref:
        pytest.skip("operon golden absent")

    seq = [("pop1000", base), ("operon", operon),
           ("small50", small), ("operon2", operon)]

    refs = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for tag, p in seq:
            if tag.startswith("operon"):
                refs[tag] = operon_ref
                continue
            binp = td / f"{tag}.bin"
            popio.save_pop_bin(p, binp)
            refs[tag] = _run_standalone_blobs(spec["binary"], binp, td / f"o_{tag}")

        # sanity: the sequence genuinely mixes K_max (else it's not discriminating)
        kmaxes = {tag: int(p["K_max"]) for tag, p in seq}
        assert max(kmaxes.values()) >= 28 and min(kmaxes.values()) <= 8, kmaxes

        co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]), variant=variant)
        try:
            for tag, p in seq:
                got = co.optimize(p, max_iter=MAX_ITER)
                _assert_bit_equal(got, refs[tag],
                                  f"{variant}/mixedK[{tag}] M={p['M']} K={p['K_max']}",
                                  spec["outputs"])
        finally:
            co.teardown()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_parity_under_reuse_shrinkM_growK(variant):
    """THE under-allocation discriminator for byte-capacity realloc (MUST-FIX #6).

    Constructs the exact hazard the spec names: in ONE persistent context, run a
    LARGER-M/low-K pop FIRST, then a SMALLER-M/high-K pop whose d_J / d_JtJ PRODUCT
    is larger:
        bigM_lowK = pop1000 tiled x8  (M=8000, K_max=8)
        operon    = snapshot           (M=4000, K_max=28)
    M strictly DECREASES (8000 -> 4000) while the products
        d_J   = M*K_max*N   and   d_JtJ = M*K_max*K_max
    strictly INCREASE (operon's K=28 >> 8 dominates the M shrink). A realloc keyed
    on a single dim (M alone, "M=4000 <= 8000 -> no realloc") would keep the K=8-
    sized d_J / d_JtJ and under-allocate on operon -> OOB writes in build_jtj_jtr /
    the AD Jacobian (garbage-or-crash). Only a BYTE-CAPACITY realloc (the products
    in bytes) grows the buffers and stays correct.

    The reference is a FRESH standalone run on the operon snapshot (same variant,
    same MAX_ITER) -- the in-process == standalone parity property, NOT the frozen
    committed golden (those goldens predate later shared-kernel edits and are stale
    vs the current source; this test is immune to that). Needs the operon snapshot
    (high K); skips otherwise.

    Non-vacuity is proven out-of-test by sabotaging co_lib.cu to key the d_J/d_JtJ
    realloc TRIGGER on M (skip regrow when M_prob <= prior M): that sabotage fails
    this test (under-alloc on operon-after-bigM), while byte-capacity passes.
    """
    if "operon4000_g100" not in POPS:
        pytest.skip("operon snapshot absent — shrinkM/growK realloc test needs high K")
    spec = _require(variant)
    import co_inproc
    import tempfile
    from cusr.benchmark import popio

    base = _load_pop_dict(POPS["pop1000"])          # M=1000, K_max=8
    bigM_lowK = _tile_pop(base, 8)                   # M=8000, K_max=8
    assert bigM_lowK["M"] == 8000, bigM_lowK["M"]
    operon = _load_pop_dict(POPS["operon4000_g100"])  # M=4000, K_max=28

    def _products(p):
        M, K, N = p["M"], int(p["K_max"]), int(p["N"])
        return M * K * N, M * K * K       # d_J, d_JtJ

    dj_big, djtj_big = _products(bigM_lowK)
    dj_op, djtj_op = _products(operon)
    # The hazard must genuinely hold or the test is vacuous: M shrinks AND BOTH
    # products grow on the bigM -> operon transition (an M-keyed cap under-allocs).
    assert operon["M"] < bigM_lowK["M"], (operon["M"], bigM_lowK["M"])
    assert dj_op > dj_big, (dj_op, dj_big)            # d_J product GROWS on M-shrink
    assert djtj_op > djtj_big, (djtj_op, djtj_big)    # d_JtJ product GROWS too

    # Reference = fresh standalone on the operon snapshot (parity oracle, not the
    # frozen golden). pop1000 (data/fixtures/pop.bin) is the snapshot for bigM_lowK
    # tiling; the standalone runs on a file, so save the tiled pop first.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        operon_bin = POPS["operon4000_g100"]         # already on disk
        operon_ref = _run_standalone_blobs(spec["binary"], operon_bin, td / "operon_ref")

        co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]),
                                   variant=variant)
        try:
            # FIRST the larger-M/low-K pop (sets the byte capacity at K=8 sizes if a
            # per-dim-M cap were used), THEN the smaller-M/high-K pop (product grows).
            co.optimize(bigM_lowK, max_iter=MAX_ITER)
            got = co.optimize(operon, max_iter=MAX_ITER)
        finally:
            co.teardown()

    # operon-AFTER-bigM must be byte-identical to its standalone single-call result.
    # Under an M-keyed cap d_J/d_JtJ are under-allocated -> OOB -> different bytes
    # (or a CUDA error); byte-capacity reallocs and stays correct.
    _assert_bit_equal(got, operon_ref,
                      f"{variant}/shrinkM_growK[operon after M=8000] "
                      f"M={operon['M']} K={int(operon['K_max'])}",
                      spec["outputs"])


def test_parity_under_reuse_large_then_shrink():
    """65536 -> 1000 large-alloc-then-shrink stress (FD variant; the reuse logic
    is variant-independent). The huge pop forces a large realloc; the immediate
    shrink to 1000 must NOT read past current M (capacity-vs-M slicing bug)."""
    spec = _require("fd")
    import co_inproc
    from cusr.benchmark import popio

    base = _load_pop_dict(POPS["pop1000"])
    big = _tile_pop(base, 66)                       # 66*1000 = 66000 >= 65536
    assert big["M"] >= 65536, big["M"]
    small = _subset_pop(base, 1000)                 # M=1000

    import tempfile
    refs = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for p in (big, small):
            binp = td / f"pop_{p['M']}.bin"
            popio.save_pop_bin(p, binp)
            refs[p["M"]] = _run_standalone_blobs(spec["binary"], binp, td / f"o_{p['M']}")

        co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]), variant="fd")
        try:
            for p in (big, small, big, small):
                got = co.optimize(p, max_iter=MAX_ITER)
                _assert_bit_equal(got, refs[p["M"]],
                                  f"fd/largeshrink M={p['M']}", spec["outputs"])
        finally:
            co.teardown()


# ===========================================================================
# (c) STATUS-BRANCH fixtures: drive each status code and check .so == standalone.
# ===========================================================================
def _build_status_branch_pop():
    """Synthesise a small pop that exercises CONVERGED, MAXITER, FAIL_NAN, K0_SKIP.

    Returns (pop_dict, {branch_name: tree_index}). Branches are validated by the
    standalone status output in test_status_branch_coverage (we don't hardcode
    which index lands in which bucket beyond asserting the set is hit)."""
    from cusr.benchmark import popio
    xs = np.linspace(0.5, 3.0, 64).astype(np.float32).reshape(-1, 1)
    x = xs[:, 0].astype(np.float64)

    trees = []          # (nt, nv, ci, c_init)
    ys = []             # per-tree target

    # 1) CONVERGED: linear y = c0 + c1*x, c_init near truth -> LM converges fast.
    nt = [N_BFUNC, N_CONST, N_BFUNC, N_CONST, N_VAR]
    nv = [F_ADD, 0.0, F_MUL, 0.0, 0.0]
    ci = [-1, 0, -1, 1, -1]
    c_true = [2.0, -0.5]
    trees.append((nt, nv, ci, [1.9, -0.45]))
    ys.append(c_true[0] + c_true[1] * x)

    # 2) K0_SKIP: no constants at all (y = x). K=0 -> STATUS_K0_SKIP.
    trees.append(([N_VAR], [0.0], [-1], []))
    ys.append(x.copy())

    # 3) FAIL_NAN: target is +inf-producing; c_init huge into exp -> initial loss
    #    non-finite OR step drives eval to Inf. y = exp(c0 * x), c_init large.
    nt = [N_UFUNC, N_BFUNC, N_CONST, N_VAR]
    nv = [F_EXP, F_MUL, 0.0, 0.0]
    ci = [-1, -1, 0, -1]
    trees.append((nt, nv, ci, [90.0]))     # exp(90*x) overflows fp32 -> Inf loss
    ys.append(np.exp(0.3 * x))             # finite target, but model blows up

    # 4) MAXITER: hard nonlinear y = sin(c0*x)*c1, c_init far -> slow, hits maxiter.
    nt = [N_BFUNC, N_UFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST]
    nv = [F_MUL, F_SIN, F_MUL, 0.0, 0.0, 0.0]
    ci = [-1, -1, -1, 0, -1, 1]
    trees.append((nt, nv, ci, [0.3, 0.2]))
    ys.append(np.sin(4.7 * x) * 2.3)

    M = len(trees)
    ym = np.stack([np.asarray(y, np.float32) for y in ys], axis=0)
    return popio.build_pop(trees, xs, ym), {
        "CONVERGED": 0, "K0_SKIP": 1, "FAIL_NAN": 2, "MAXITER": 3}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_status_branch_coverage(variant):
    """The synthetic pop hits multiple status branches AND the .so reproduces the
    standalone byte-for-byte on it (so branch handling is parity-correct)."""
    spec = _require(variant)
    import co_inproc
    import tempfile
    from cusr.benchmark import popio

    pop, _ = _build_status_branch_pop()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        binp = td / "branch.bin"
        popio.save_pop_bin(pop, binp)
        ref = _run_standalone_blobs(spec["binary"], binp, td / "out")

    # Must actually exercise each intended branch (else the test is vacuous).
    # Verified actuals (both variants): CONVERGED(0), MAXITER(1), FAIL_NAN(2),
    # K0_SKIP(3). FAIL_CHOLESKY(4) is covered by test_fail_cholesky_high_k.
    distinct = set(int(s) for s in ref["status"])
    for code, nm in [(0, "CONVERGED"), (1, "MAXITER"), (2, "FAIL_NAN"), (3, "K0_SKIP")]:
        assert code in distinct, (
            f"status-branch fixture did NOT exercise {nm}({code}); got {sorted(distinct)}"
        )

    co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]), variant=variant)
    try:
        got = co.optimize(pop, max_iter=MAX_ITER)
    finally:
        co.teardown()
    _assert_bit_equal(got, ref, f"{variant}/status-branch", spec["outputs"])


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_fail_cholesky_high_k(variant):
    """Operon high-K snapshot (K_max=28) drives the Cholesky-breakdown branch;
    the .so must reproduce the standalone byte-for-byte there too. Auto-skips if
    the snapshot is absent on this machine."""
    if "operon4000_g100" not in POPS:
        pytest.skip("operon snapshot absent")
    spec = _require(variant)
    import co_inproc
    pd = _load_pop_dict(POPS["operon4000_g100"])
    ref = _golden_blobs(variant, "operon4000_g100")
    if not ref:
        pytest.skip("operon golden absent")
    # FAIL_CHOLESKY (4) should appear in this high-K snapshot.
    assert 4 in set(int(s) for s in ref["status"]), \
        "operon golden has no FAIL_CHOLESKY (4) — fixture no longer stresses it"

    co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]), variant=variant)
    try:
        got = co.optimize(pd, max_iter=MAX_ITER)
    finally:
        co.teardown()
    _assert_bit_equal(got, ref, f"{variant}/operon-cholesky", spec["outputs"])


# ---- magnitude span: scale-1e± fixtures (if generated) --------------------
_SCALE_BINS = sorted(HERE.glob("pop_fixture_scale_*.bin"))


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("scale_bin", _SCALE_BINS, ids=[p.stem for p in _SCALE_BINS])
def test_magnitude_span(variant, scale_bin):
    """scale-1e±6 fixtures: .so == standalone across constant magnitudes."""
    spec = _require(variant)
    import co_inproc
    pd = _load_pop_dict(scale_bin)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref = _run_standalone_blobs(spec["binary"], scale_bin, td / "out")
    co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]), variant=variant)
    try:
        got = co.optimize(pd, max_iter=MAX_ITER)
    finally:
        co.teardown()
    _assert_bit_equal(got, ref, f"{variant}/{scale_bin.stem}", spec["outputs"])


# ===========================================================================
# (d) MUST-FIX #6: sum-consistency checks inside co_optimize before H2D.
# ===========================================================================
def test_co_optimize_rejects_inconsistent_sums():
    """A pop whose header total_c disagrees with sum(K) must be REJECTED by
    co_optimize (positive status code), not silently under-allocate / OOB-read."""
    spec = _require("fd")
    import co_inproc
    pd = _load_pop_dict(POPS["pop1000"])
    # corrupt total_c so sum(metas.K) != total_c
    bad = dict(pd)
    bad["total_c"] = pd["total_c"] + 7
    co = co_inproc.InProcessCO(device_id=0, lib_path=str(spec["so"]), variant="fd")
    try:
        with pytest.raises(co_inproc.COError):
            co.optimize(bad, max_iter=MAX_ITER)
    finally:
        co.teardown()
