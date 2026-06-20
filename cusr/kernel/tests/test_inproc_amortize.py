"""test_inproc_amortize.py — in-process CO drop-in (P1) AMORTIZATION suite.

STEP amortization. Builds on the in-process .so + ctypes wrapper from step
parity-reuse-and-fd (co_lib.cu / co_inproc.py) and the per-variant goldens.

This is the suite that PROVES the headline of P1: paying CUDA primary-context
init ONCE per process (not per generation) and running the per-gen call as just
the LM loop. See docs/kernel/INPROCESS_CO_PLAN.md "Amortization test design" and
MUST-FIX #3 (the draft's `time(co_init) ∈ 350-450ms` gate is WRONG — torch creates
the CUDA context lazily at fitness eval BEFORE any CO call, so that band fails
exactly when amortization works). Corrected gates:

  (1) test_primary_context_created_once — the CUDA primary context bound as
      current is ONE STABLE OBJECT across co_init + every co_optimize. Probed via
      the driver API (cuCtxGetCurrent over libcuda.so.1): the CUcontext handle is
      non-null and byte-identical at every probe. A per-call destroy+recreate (the
      failure amortization prevents) would change the handle. "Whoever pays it"
      (torch or co_init) is irrelevant — only that it is one stable context. Paired
      with: co_init invoked exactly once. (We deliberately DO NOT assert active==0
      at process start — order-dependent / flaky under pytest + lazy torch ctx.)

  (2) test_co_optimize_wall_index_independent — repeated co_optimize on the SAME
      pop, in ONE persistent process, is INDEX-INDEPENDENT: no per-call re-init,
      no upward drift. One untimed WARM-UP call absorbs one-time lazy module/kernel
      load; calls 2..N must be flat (max ≤ 1.5× median, and the back half is not
      systematically slower than the front half). PLUS an ABSOLUTE bound
      (MUST-FIX #3): the median co_optimize wall must be ≤ 1.3× the STANDALONE
      `batch_lm_ad` `loop_ms` (the pure LM-loop wall, parsed from the -DPROFILE
      standalone's PROFILE_JSON on the same pop). The flatness checks above are
      intra-run RELATIVE — they are blind to a CONSTANT per-call overhead
      regression (an injected sleep / a redundant H2D added to every call leaves
      max/median and back/front near 1.0 yet inflates the absolute wall). The
      absolute bound catches exactly that.

  (3) test_end_to_end_gen_speedup — THE per-gen headline. A TIMED end-to-end
      generation (forest extraction + CO + writeback) inproc vs the subprocess
      baseline (kernel_bridge._run_kernel). Extraction (per-tree GPU→host syncs)
      and writeback (per-tree GPU scatter) are reported as SEPARATE line items
      (MUST-FIX #7: they are in the shipped hot path and were unmeasured; folding
      them invisibly would hide exactly what the audit flagged). median-of-3.

  (4) test_setup_loop_split_prof — the setup/loop SPLIT from libcusr_co_ad_prof.so
      (-DPROFILE), read via the NEW co_last_split symbol. Reported, not gated on a
      hardcoded ms (the split is corpus-dependent). Asserts only that setup << loop
      for a real corpus (the loop dominates per-gen once setup is amortized).

All numbers are MEASURED actuals (never a hardcoded 180ms). GPU pinned to 2 (set
CUDA_VISIBLE_DEVICES=2), nvidia-smi idle-checked before timing.

Build prerequisite (run once):
    source scripts/env.sh
    cd cusr/kernel && make batch_lm batch_lm_ad batch_lm_ad_prof \
        libcusr_co_fd.so libcusr_co_ad.so \
        libcusr_co_fd_prof.so libcusr_co_ad_prof.so
    # batch_lm_ad_prof is the -DPROFILE standalone gate-2's absolute bound
    # (MUST-FIX #3) reads PROFILE_JSON.loop_ms from.
Run (GPU pinned, device must be idle):
    source scripts/env.sh
    export CUDA_VISIBLE_DEVICES=2
    uv run pytest cusr/kernel/tests/test_inproc_amortize.py -v -s
"""
from __future__ import annotations

import ctypes
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent          # cusr/kernel/tests
PROJ = HERE.parent                              # cusr/kernel
ROOT = HERE.parents[2]                          # repo root

sys.path.insert(0, str(PROJ))                   # for co_inproc
sys.path.insert(0, str(ROOT))                   # for cusr.*

# Pinned identical to the goldens / step-3 (the .so must run max_iter=50).
MAX_ITER = 50
DEVICE_ID = 0                                    # driver ordinal under CVD=2

# AD is the end-to-end variant (plan); the amortization headline is reported for AD.
SO_AD        = PROJ / "libcusr_co_ad.so"
SO_AD_PROF   = PROJ / "libcusr_co_ad_prof.so"
BIN_AD       = PROJ / "batch_lm_ad"
BIN_AD_PROF  = PROJ / "batch_lm_ad_prof"   # -DPROFILE standalone -> PROFILE_JSON.loop_ms
POP1000      = ROOT / "data" / "fixtures" / "pop.bin"

# Absolute-bound headroom (MUST-FIX #3): the Python-timed co_optimize WALL
# (marshalling + H2D + initial eval + LM loop + D2H) is compared against the
# STANDALONE's pure LM-loop wall (PROFILE_JSON.loop_ms — loop only). The bound
# absorbs the legitimate per-call non-loop overhead (~1MB H2D + initial eval +
# D2H). Measured on this corpus: med wall ~56ms vs standalone loop_ms ~69ms ->
# ratio ~0.81 (the warm in-process context has no cold-process JIT/module load,
# so the wall is actually BELOW the cold-standalone loop). 1.3x is comfortable
# headroom that still FAILS on a constant per-call overhead regression (injected
# sleep / redundant H2D), which the intra-run flatness checks are blind to.
WALL_OVER_LOOP_MAX = 1.3


# ---------------------------------------------------------------------------
# fixtures / guards
# ---------------------------------------------------------------------------
def _require(so: Path):
    if not so.exists():
        pytest.fail(f"{so.name} not built — run `make {so.name}` in {PROJ} "
                    f"(see test docstring)")
    try:
        import co_inproc  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"import co_inproc failed: {e!r}")


def _nvidia_smi_print():
    """Print the nvidia-smi table FOR THE RECORD (NOT a gate). The real idle-check
    is the manual pre-run inspection the operator does before timing (the step
    protocol). We deliberately do not assert here: mapping the CUDA-visible ordinal
    back to a physical index without NVML is unreliable, and the timing gates
    self-protect (warm-up + ratio tolerance). This is purely diagnostic output."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        print("\n[nvidia-smi, diagnostic only]\n" + out.strip())
    except Exception as e:  # noqa: BLE001
        print(f"\n[nvidia-smi unavailable: {e!r}]")


def _load_pop():
    from cusr.benchmark import popio
    return popio.load_pop_bin(POP1000)


def _standalone_loop_ms(reps: int = 3) -> float:
    """Median standalone LM-loop wall (ms) from the -DPROFILE standalone
    `batch_lm_ad_prof` on POP1000, max_iter=MAX_ITER. Parses PROFILE_JSON.loop_ms
    (the pure loop wall — load + setup/context-init excluded; printed regardless
    of --quiet). This is the ABSOLUTE baseline MUST-FIX #3's gate compares the
    in-process co_optimize wall against: a per-call overhead regression that the
    intra-run flatness checks can't see shows up as wall/loop_ms > 1.3x. Each rep
    is a fresh cold process (median-of-3 so one outlier can't set the baseline).

    We use the standalone's pure loop_ms, NOT gate-3's `_run_kernel` subprocess
    TOTAL wall: that total (~500ms) folds in the per-process CUDA context init
    (~400ms `setup_ms`) the in-process path amortizes away, so `~56ms <= 1.3*500ms`
    would be vacuously true and catch no regression. The pure loop_ms is the only
    apples-to-apples anchor for the in-process co_optimize wall."""
    import json
    import tempfile

    if not BIN_AD_PROF.exists():
        pytest.fail(
            f"{BIN_AD_PROF.name} not built — run `make {BIN_AD_PROF.name}` in "
            f"{PROJ} (the -DPROFILE standalone; emits PROFILE_JSON.loop_ms)"
        )
    loops = []
    with tempfile.TemporaryDirectory(prefix="co_loopms_") as td:
        for _ in range(reps):
            r = subprocess.run(
                [str(BIN_AD_PROF), str(POP1000), td,
                 "--max-iter", str(MAX_ITER), "--quiet"],
                capture_output=True, text=True)
            if r.returncode != 0:
                pytest.fail(f"{BIN_AD_PROF.name} exit={r.returncode}: "
                            f"{(r.stderr or r.stdout)[-500:]}")
            line = next((ln for ln in r.stdout.splitlines()
                         if ln.startswith("PROFILE_JSON ")), None)
            if line is None:
                pytest.fail(f"no PROFILE_JSON line from {BIN_AD_PROF.name} "
                            f"(built without -DPROFILE?): {r.stdout[-300:]}")
            j = json.loads(line[len("PROFILE_JSON "):])
            loops.append(float(j["loop_ms"]))
    print(f"\n[standalone {BIN_AD_PROF.name} loop_ms] reps={reps} "
          f"loops(ms)={[round(x, 2) for x in loops]} "
          f"median={statistics.median(loops):.2f}")
    return statistics.median(loops)


# ---------------------------------------------------------------------------
# driver-API primary-context probe (cuCtxGetCurrent over libcuda.so.1)
# ---------------------------------------------------------------------------
class _Cu:
    """Thin ctypes binding of the few CUDA driver-API calls we need to probe the
    current context handle. cuInit(0) is safe — it does NOT create a context."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cuCtxGetCurrent.restype = ctypes.c_int

    def init(self):
        rc = self.lib.cuInit(0)
        assert rc == 0, f"cuInit failed rc={rc}"

    def current_ctx(self):
        h = ctypes.c_void_p()
        rc = self.lib.cuCtxGetCurrent(ctypes.byref(h))
        assert rc == 0, f"cuCtxGetCurrent failed rc={rc}"
        return h.value          # int address or None


# ===========================================================================
# GATE 1: primary context created exactly once (handle-identity, not timing).
# ===========================================================================
def test_primary_context_created_once():
    _require(SO_AD)
    import co_inproc

    cu = _Cu()
    cu.init()                                  # safe; creates no context

    pd = _load_pop()
    co = co_inproc.InProcessCO(device_id=DEVICE_ID, lib_path=str(SO_AD), variant="ad")

    # GENUINELY count co_init invocations: wrap the loaded library's co_init so any
    # extra call (e.g. an accidental re-init inside optimize()) increments the
    # counter and fails the assertion. This is a REAL check, not a constant — if a
    # bug called co_init per optimize, n_co_init would be 5, not 1.
    n_co_init = {"n": 0}
    _orig_co_init = co._lib.co_init
    def _counting_co_init(*a, **k):            # noqa: ANN001
        n_co_init["n"] += 1
        return _orig_co_init(*a, **k)
    co._lib.co_init = _counting_co_init

    try:
        handles = []
        handles.append(("post-init", cu.current_ctx()))
        for i in range(4):
            co.optimize(pd, max_iter=MAX_ITER)   # must NOT call co_init
            handles.append((f"post-opt{i}", cu.current_ctx()))
    finally:
        co.teardown()

    print("\n[ctx handles]")
    for tag, h in handles:
        print(f"  {tag}: {hex(h) if h else None}")
    print(f"  co_init calls during 4 optimizes: {n_co_init['n']}")

    # The bound primary context must be a single stable, non-null object.
    addrs = [h for _, h in handles]
    assert all(a is not None for a in addrs), \
        f"current context became NULL at some probe: {handles}"
    assert len(set(addrs)) == 1, (
        f"CUDA primary context handle CHANGED across calls — a per-call "
        f"destroy/recreate (amortization broken): {handles}"
    )
    # co_optimize must NEVER call co_init (the wrap was installed AFTER the one
    # legitimate co_init in __init__, so any count > 0 here is a per-call re-init).
    assert n_co_init["n"] == 0, (
        f"co_optimize triggered co_init {n_co_init['n']} times — context is being "
        f"re-created per call (amortization broken)"
    )


# ===========================================================================
# GATE 2: co_optimize wall is index-independent (no per-call re-init / drift).
# ===========================================================================
def test_co_optimize_wall_index_independent():
    _require(SO_AD)
    import co_inproc
    _nvidia_smi_print()

    pd = _load_pop()
    co = co_inproc.InProcessCO(device_id=DEVICE_ID, lib_path=str(SO_AD), variant="ad")
    try:
        # One untimed WARM-UP: absorbs one-time lazy module/kernel load (that is
        # NOT per-call re-init; amortization is about the steady state).
        co.optimize(pd, max_iter=MAX_ITER)

        N = 10
        walls = []
        for _ in range(N):
            t0 = time.perf_counter()
            co.optimize(pd, max_iter=MAX_ITER)
            walls.append((time.perf_counter() - t0) * 1e3)   # ms
    finally:
        co.teardown()

    med = statistics.median(walls)
    mx = max(walls)
    half = N // 2
    front = statistics.median(walls[:half])
    back = statistics.median(walls[half:])
    print(f"\n[index-independence] N={N} walls(ms)="
          f"{[round(w, 2) for w in walls]}")
    print(f"  median={med:.2f}  max={mx:.2f}  front_med={front:.2f}  back_med={back:.2f}")

    # No single call spikes (a re-init would show as an outlier).
    assert mx <= 1.5 * med, (
        f"co_optimize wall has a per-call spike: max={mx:.2f}ms > 1.5*median "
        f"({1.5*med:.2f}ms) — possible per-call re-init: {walls}"
    )
    # No systematic upward DRIFT (back half not materially slower than front).
    assert back <= 1.3 * front, (
        f"co_optimize wall DRIFTS upward: back_med={back:.2f}ms > 1.3*front_med "
        f"({1.3*front:.2f}ms) — leak / re-alloc churn: {walls}"
    )

    # ABSOLUTE bound (MUST-FIX #3): the flatness checks above are intra-run
    # RELATIVE — a CONSTANT per-call overhead (injected sleep / redundant H2D)
    # shifts every wall equally and leaves max/median and back/front near 1.0, so
    # it passes both flatness checks while silently regressing the per-gen cost.
    # Anchor to the STANDALONE LM-loop wall (loop_ms from the -DPROFILE
    # standalone) and require the in-process median wall to stay within 1.3x of
    # it. The in-process wall adds marshalling + H2D + initial eval + D2H over the
    # pure loop but pays NO context init (amortized in co_init), so on a warm
    # context it sits at/below loop_ms; a constant regression pushes it past 1.3x.
    loop_ms = _standalone_loop_ms(reps=3)
    ratio = med / loop_ms
    print(f"  [absolute bound] median co_optimize wall={med:.2f}ms / standalone "
          f"loop_ms={loop_ms:.2f}ms = {ratio:.3f}x  (must be <= {WALL_OVER_LOOP_MAX})")
    assert med <= WALL_OVER_LOOP_MAX * loop_ms, (
        f"co_optimize median wall {med:.2f}ms > {WALL_OVER_LOOP_MAX}x standalone "
        f"loop_ms ({WALL_OVER_LOOP_MAX * loop_ms:.2f}ms) — a CONSTANT per-call "
        f"overhead regression the intra-run flatness checks are blind to "
        f"(ratio={ratio:.3f}x): walls={walls}"
    )


# ===========================================================================
# GATE 3: timed END-TO-END generation (extraction + CO + writeback), inproc vs
#         subprocess. THE per-gen headline. MUST-FIX #7: extraction + writeback
#         reported as SEPARATE line items, not folded into the loop.
# ===========================================================================
def _build_native_forest(pop_size: int, n_vars: int, seed: int):
    """A real EvoGP forest (live GPU trees) — the genuine extraction source
    MUST-FIX #7 wants timed (per-tree GPU->host syncs), not a pre-extracted
    pop.bin. Mirrors test_kernel_bridge.test_extract_vs_skeleton_function_agreement.
    Returns (forest, list_of_native_trees)."""
    import torch
    from evogp.tree import Forest, GenerateDescriptor

    FUNCS = {"+": 1., "-": 1., "*": 1., "/": 1., "sin": .5, "cos": .5,
             "exp": .3, "log": .3, "sqrt": .3, "tanh": .2}
    desc = GenerateDescriptor(max_tree_len=64, input_len=n_vars, output_len=1,
                              using_funcs=FUNCS, max_layer_cnt=6,
                              const_range=[-5., 5.], sample_cnt=10000,
                              layer_leaf_prob=0.3)
    torch.manual_seed(seed)
    forest = Forest.random_generate(pop_size=pop_size, descriptor=desc)
    return forest, [forest[m] for m in range(len(forest))]


def _extract_gen(natives, n_vars, X, y):
    """Extraction phase: live trees -> kernel-eligible pop dict + slot map.
    Times the per-tree GPU->host extraction (kb.extract_native -> _extract_tree)."""
    from cusr.demonstrator import kernel_bridge as kb
    from cusr.benchmark import popio

    kernel_slots = []          # (orig_idx, ext)
    for j, nat in enumerate(natives):
        ext = kb.extract_native(nat, n_vars)
        if ext is None:
            continue
        nt, nv, ci, c_init = ext
        if not kb._kernel_eligible(nt):
            continue
        kernel_slots.append((j, ext))
    if not kernel_slots:
        return None, []
    trees = [ext for _, ext in kernel_slots]
    ym = np.tile(y, (len(trees), 1))
    pop = popio.build_pop(trees, X, ym)
    return pop, kernel_slots


def _writeback_gen(forest, kernel_slots, pop, c_final):
    """Writeback phase: per-tree GPU scatter of optimized constants back into the
    live forest (MUST-FIX #7's `_writeback_constants` host->GPU path)."""
    from cusr.bench.sources.evogp import _writeback_constants
    n_written = 0
    for slot, (j, _ext) in enumerate(kernel_slots):
        _, _, c_off, K = pop["metas"][slot].tolist()
        c = np.asarray(c_final[c_off:c_off + K], dtype=float)
        if _writeback_constants(forest[j], c):
            n_written += 1
    return n_written


def test_end_to_end_gen_speedup():
    _require(SO_AD)
    import co_inproc
    from cusr.demonstrator import kernel_bridge as kb
    _nvidia_smi_print()

    n_vars = 2
    pop_size = 1000
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (256, n_vars)).astype(np.float64)
    y = rng.uniform(-2, 2, 256).astype(np.float64)

    forest, natives = _build_native_forest(pop_size, n_vars, seed=0)

    # CRITICAL (honesty): _writeback_constants mutates the live forest's
    # node_value IN-PLACE (view aliasing), and natives[j] IS forest[j]. Without a
    # reset, each rep would re-extract the PREVIOUS rep's LM-converged constants,
    # so a later rep's CO loop converges in ~2 iters instead of ~50 — understating
    # the inproc CO phase and inflating the speedup. We snapshot the fresh
    # (un-optimized) constants once and RESTORE them at the start of every gen,
    # BEFORE t0 (off-clock), so every rep optimizes IDENTICAL fresh constants.
    # This is also more representative: a real memetic generation faces freshly
    # mutated trees, i.e. fresh (un-converged) constants each gen.
    import torch
    # Forest stores constants in batch_node_value[(pop_size, max_tree_len)];
    # forest[m].node_value is a view into it, so _writeback_constants mutates
    # this buffer. Snapshot it once (fresh, un-optimized) and restore each gen.
    _fresh_node_value = forest.batch_node_value.clone()

    def _reset_forest():
        forest.batch_node_value.copy_(_fresh_node_value)
        torch.cuda.synchronize()

    # ---- subprocess baseline: extraction + _run_kernel(subprocess) + writeback ----
    def run_subprocess_gen():
        _reset_forest()                          # off-clock: fresh constants
        t0 = time.perf_counter()
        pop, slots = _extract_gen(natives, n_vars, X, y)
        t1 = time.perf_counter()
        if pop is None:
            pytest.skip("no kernel-eligible trees in forest")
        c_final, _status = kb._run_kernel(pop, BIN_AD, MAX_ITER)
        t2 = time.perf_counter()
        _writeback_gen(forest, slots, pop, c_final)
        t3 = time.perf_counter()
        return dict(extract=(t1 - t0) * 1e3, co=(t2 - t1) * 1e3,
                    writeback=(t3 - t2) * 1e3, total=(t3 - t0) * 1e3,
                    n_kernel=len(slots))

    # ---- inproc: extraction + InProcessCO.optimize + writeback (ONE co_init) ----
    co = co_inproc.InProcessCO(device_id=DEVICE_ID, lib_path=str(SO_AD), variant="ad")

    def run_inproc_gen():
        _reset_forest()                          # off-clock: fresh constants
        t0 = time.perf_counter()
        pop, slots = _extract_gen(natives, n_vars, X, y)
        t1 = time.perf_counter()
        if pop is None:
            pytest.skip("no kernel-eligible trees in forest")
        out = co.optimize(pop, max_iter=MAX_ITER)
        c_final = out["c_final"]
        t2 = time.perf_counter()
        _writeback_gen(forest, slots, pop, c_final)
        t3 = time.perf_counter()
        return dict(extract=(t1 - t0) * 1e3, co=(t2 - t1) * 1e3,
                    writeback=(t3 - t2) * 1e3, total=(t3 - t0) * 1e3,
                    n_kernel=len(slots))

    try:
        # warm up the inproc context (pay co_init's one-time cost off the clock)
        run_inproc_gen()

        REPS = 3
        sub = [run_subprocess_gen() for _ in range(REPS)]
        inp = [run_inproc_gen() for _ in range(REPS)]
    finally:
        co.teardown()

    def med(rows, k):
        return statistics.median(r[k] for r in rows)

    print(f"\n[end-to-end gen] forest pop_size={pop_size}, "
          f"n_kernel={sub[0]['n_kernel']}, REPS={REPS}, median ms:")
    print(f"  {'phase':<12}{'subprocess':>14}{'inproc':>14}")
    for k in ("extract", "co", "writeback", "total"):
        print(f"  {k:<12}{med(sub, k):>14.2f}{med(inp, k):>14.2f}")
    speedup = med(sub, "total") / med(inp, "total")
    co_speedup = med(sub, "co") / med(inp, "co")
    print(f"  total speedup (sub/inproc): {speedup:.2f}x   "
          f"(CO-phase alone: {co_speedup:.2f}x)")

    # Headline assertion: the inproc CO PHASE is strictly faster than the
    # subprocess CO phase (subprocess pays process-spawn + disk + context init
    # every gen; inproc pays none of it). This is the amortization win.
    assert med(inp, "co") < med(sub, "co"), (
        f"inproc CO phase ({med(inp,'co'):.2f}ms) not faster than subprocess "
        f"({med(sub,'co'):.2f}ms) — amortization not realized"
    )
    # And the inproc end-to-end gen is faster overall.
    assert med(inp, "total") < med(sub, "total"), (
        f"inproc end-to-end ({med(inp,'total'):.2f}ms) not faster than "
        f"subprocess ({med(sub,'total'):.2f}ms)"
    )
    # Sanity: extraction + writeback are NON-trivial line items (MUST-FIX #7 —
    # they must be COUNTED, not assumed ~0). We assert they're measured (>0), and
    # surface them in the printout so the A-over-C tie-break can be revisited if
    # they rival the loop.
    assert med(inp, "extract") > 0 and med(inp, "writeback") > 0


# ===========================================================================
# GATE 4: setup/loop split from the -DPROFILE .so (reported, lightly gated).
# ===========================================================================
def test_setup_loop_split_prof():
    _require(SO_AD_PROF)
    import co_inproc

    pd = _load_pop()
    co = co_inproc.InProcessCO(device_id=DEVICE_ID, lib_path=str(SO_AD_PROF),
                               variant="ad")
    try:
        # warm up (lazy module load lands in the first call's setup), then measure
        co.optimize(pd, max_iter=MAX_ITER)
        splits = []
        for _ in range(3):
            co.optimize(pd, max_iter=MAX_ITER)
            s, l = co.last_split()
            splits.append((s, l))
    finally:
        co.teardown()

    setups = [s for s, _ in splits]
    loops = [l for _, l in splits]
    # The prof build MUST expose real data (this is the symbol the step adds).
    assert all(s is not None and s >= 0 for s in setups), \
        f"co_last_split returned no data from the -DPROFILE .so: {splits}"
    setup_med = statistics.median(setups)
    loop_med = statistics.median(loops)
    print(f"\n[setup/loop split, libcusr_co_ad_prof.so, M={pd['M']}]")
    print(f"  setup_ms (median) = {setup_med:.2f}")
    print(f"  loop_ms  (median) = {loop_med:.2f}")
    print(f"  setup fraction    = {setup_med/(setup_med+loop_med)*100:.1f}%")

    # On a real corpus the LM loop dominates the per-call cost once the CUDA
    # context is amortized away (setup here is only H2D + initial eval, NOT
    # context init — that is paid once in co_init, outside co_optimize).
    assert loop_med > setup_med, (
        f"setup ({setup_med:.2f}ms) >= loop ({loop_med:.2f}ms) — unexpected; "
        f"the per-call setup should be small vs the LM loop"
    )
