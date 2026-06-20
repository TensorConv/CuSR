# In-process CO drop-in (P1) — plan of record

## IMPLEMENTATION STATUS (2026-06-17, workflow `wf_32afd8bc-900`)
**Adversarial-review verdict (12 auditors): PASS_WITH_FIXES. No faked green.** The review
re-ran every test, confirmed goldens are byte-exact vs the STANDALONE (non-circular,
12/12 sha256 match), and found no skip/xfail/tolerance-loosening. Caught localized
overclaims-of-rigor (not faked results) + disclosed deferrals.

**WORKS (verified):** the in-process library is built and functionally correct.
- `co_lib.cu` + `co_lib.h` (extern "C" `co_init`/`co_optimize`/`co_teardown`, persistent ctx,
  exit→return, realloc-on-grow by byte-capacity, per-call h_c re-seed + current-M slicing),
  `lm_core.cuh` (5 FMA-sensitive device kernels single-sourced; standalone byte-identical
  after refactor), `co_inproc.py` (ctypes; PopHeader 64B/TreeMeta 16B; M→M_prob+magic).
- 4 `.so` built: `libcusr_co_{fd,ad}{,_prof}.so`. Goldens under `cusr/kernel/tests/golden/`.
- Parity GREEN: single-call + parity-under-reuse, FD & AD each vs its own standalone golden,
  byte-exact (`test_inproc_parity.py` 8, `test_shared_header.py` 10, `test_inproc_reuse.py` 24).
- **Amortization MEASURED (the P1 payoff):** CO phase **504ms→38ms = 13.2×**; end-to-end gen
  (extract+CO+writeback) **817ms→353ms = 2.3×**; setup is 2.3% of a call; `co_init` called
  **0×** across 4 optimizes (context paid once); per-call wall index-independent (1.005×).

**BRIDGE WIRED (2026-06-17, workflow `wf_12795200-b06`, verdict PASS_WITH_FIXES, no faked green):**
`kernel_bridge.fit_natives` gained `inproc`/`device_id`; when `inproc=True` it routes to
`co_inproc.get_inproc_co(device, variant).optimize(pop, max_iter)` (variant tracks the `binary`
arg → AD/FD .so), keeping `_run_kernel` as the subprocess baseline; `CudaKernelLM` gained
`inproc`/`device_id`; `pipeline.py` UNCHANGED. Tests (`experiments/e2_demonstrator/test_inproc_bridge.py`):
end-to-end inproc==subprocess (routing **sabotage-proven** — breaking it fails the test) +
**MUST-FIX #7 mixed-batch dropped-individual** (no `None` in `results[M]`, scipy-fallback slots
byte-correct). Regression 31 passed. TR work git-proven untouched (only 2 Python files changed).

**STILL OPEN (the `Fixups` agent died on API idle-timeout — these were NOT applied):**
- **MUST-FIX #2:** `audit_shrinkM_growK.py` still not collected by pytest (M-shrink/d_J-grow
  realloc-guard); `mixed_k` docstring still overclaims. (Realloc code is correct; guard missing.)
- **MUST-FIX #3:** amortization gate-2 still has no absolute bound (only intra-run flatness;
  blind to a constant per-call overhead regression). Add: `co_optimize` median ≤ 1.3× standalone
  `loop_ms`.
- **NEW (medium, review-found):** mixed-batch test value-checks only the scipy-fallback slots;
  eligible-slot value correctness on the COMPACTED writeback path (equal-K swap) is unverified
  by either test — add a per-slot eligible-value check on a mixed batch.
- **LOW:** `co_lib.cu:21-22` comment falsely claims `d_stat/h_solve_stat` are per-call reset —
  fix the comment OR add the memset (output-neutral today, write-before-read). `make clean`
  omits the `.so`. LM-loop copied not shared (byte-gate covers drift). int32 index >~2M trees.
- **Pre-existing uncommitted TRUST_REGION edits** (NOT this work): `batch_lm_fusedfd.cu`, parts of
  `batch_lm_ad.cu`/`Makefile`/`.gitignore`, untracked `batch_lm_ad_tr`. **User manages these** —
  keep them; the P1 commit must touch only P1 files (`batch_lm_ad.cu` mixes TR + shared-header).

**Remaining to finish P1:** the 3 small test/comment items above, then commit P1 apart from TR.
P1's core (library + bridge + parity + 13.2×/2.3× amortization + dropped-individuals) is DONE.

---

**Status:** design plan, **audited verdict = SOUND-WITH-FIXES**, NOT yet implemented.
Produced by the `cusr-inprocess-co-plan` workflow (10-agent understand→design→synthesize)
+ a 12-reviewer adversarial audit (run `wf_c3baab86-74b`). This file is the durable
record (the workflow output lives only in the run transcript).

## Goal
Make the CO kernel a Python-callable, **in-process** drop-in so CUDA-context init
(~360–410 ms, one-time) is paid ONCE per process instead of every generation, and the
per-gen call costs only the LM loop. This is the long-pole prerequisite for the
end-to-end experiment (CuSR as EvoGP's per-gen constant-optimization step). See
memory `cusr-korns-corpus`, `cusr-lm-v4-ad`, `cusr-direction`.

## Chosen approach — Option A: ctypes-loaded `libcusr_co.so`
Refactor the kernel's monolithic `main()` into three `extern "C"` functions over a
persistent context, loaded from Python via ctypes. The EvoGP loop's public contract is
unchanged (it keeps calling `co.fit_batch`); only the per-gen **subprocess + disk** inside
`kernel_bridge.fit_natives` is replaced by a persistent handle.

Why A over the others (judge panel): **C (torch CUDA extension, zero-copy GPU tensors)**
is the true end-state but much larger scope (on-device repack + GPU-scatter writeback) and
ABI-coupled to the torch build; **B (pybind11)** adds a build dep for no win over ctypes
here; **D (persistent subprocess)** still pays IPC/serialization and doesn't cleanly share
the CUDA context with torch. A is the minimal change that pays the one-time-setup prize and
reuses the existing `fit_natives` host marshalling verbatim. **A is an explicitly-billed
interim toward C, not zero-copy** (see must-fix #7 and open question on zero-copy scope).

### C ABI (`cusr/kernel/co_lib.cu` + `co_lib.h`)
```
int  co_init(int device_id, void **handle_out);   // 0 OK; forces CUDA primary context once
int  co_optimize(void *handle, const PopHeader *header,
                 const int *nt, const float *nv, const int *ci,   // [total_nodes]
                 const TreeMeta *metas,                            // [M]
                 const float *c_init, const float *xs, const float *ym,
                 int max_iter,
                 float *c_final_out, int *status_out,
                 float *loss_init_out, float *loss_final_out);     // caller-allocated
                 // 0 OK; 2 = K_max>MAX_K(32) or max_stack>MAX_STACK(64) (caller pre-filters); <0 = CUDA error
void co_teardown(void *handle);
```
### Python (`cusr/kernel/co_inproc.py`)
`InProcessCO(device_id, lib_path)` → `dlopen` + `co_init` once + `atexit` teardown;
`.optimize(pop_dict, max_iter) -> (c_final f32[total_c], raw_status i32[M])` (same dict
`_run_kernel` takes today). `get_inproc_co(device)` = process-wide singleton so setup is
paid once across all gens. Bridge selects it via `CudaKernelLM(inproc=True)`.

## Files
- change: `cusr/kernel/Makefile` (add `libcusr_co.so`: `NVCC_FLAGS -Xcompiler -fPIC -shared`),
  `cusr/demonstrator/kernel_bridge.py` (route `fit_natives` to `get_inproc_co().optimize`,
  keep `_run_kernel` as subprocess baseline), `cusr/demonstrator/co_backend.py`
  (`CudaKernelLM` gains `inproc`, `device_id`).
- new: `cusr/kernel/co_lib.{cu,h}`, `cusr/kernel/co_inproc.py`,
  `cusr/kernel/tests/test_inproc_parity.py`, `cusr/kernel/tests/test_inproc_amortize.py`.

## Implementation steps (condensed)
0. PREP: `source scripts/env.sh`; confirm the chosen standalone variant builds and is
   bit-exact deterministic run-to-run (cmp two runs of `c_final.bin`+`status.bin`) — this
   is the golden baseline (**see must-fix #2**).
1. EXTRACT CORE: copy the chosen `batch_lm*.cu`; delete arg-parse/`dirname`/`load_pop_bin`/
   `write_blob`/`main()`. **Factor the `__global__` kernels + the LM-loop body into a shared
   header `#include`d by BOTH the standalone and `co_lib.cu`** so device codegen cannot drift
   (byte-parity is FMA-contraction sensitive).
2. `exit()` → RETURN: convert every `CUDA_CHECK` (×44) and the `write_blob` exits into stored
   error + negative return. A `.so` must never `exit()` the interpreter mid-evolution.
3. PERSISTENT CONTEXT + ALLOC: `CoCtx` holds all ~17 device handles + host scratch + a
   **per-buffer BYTE capacity**. `co_init`: `cudaSetDevice` + `cudaFree(0)`. `co_optimize`:
   realloc-on-grow by byte-capacity (**not per-dim** — `d_J = M·K_max·N`, `d_JtJ = M·K_max²`
   are products; M-shrink/K-grow can under-allocate → OOB; **must-fix #6**). Every launch grid
   and every D2H slice uses the CURRENT pop's `M`/`total_c`, never capacity (**must-fix #4**).
4. PER-CALL RE-INIT (parity-load-bearing): every call, re-seed `h_c` from THIS pop's
   `c_init`, set `h_lam=1e-3`, zero `h_finished/h_iter/h_rejected`, re-apply K=0 pre-mark,
   initial eval/residual/loss, snapshot `h_loss_init`. **Include `d_stat`/`h_solve_stat`**
   (omitted in the draft — must-fix #4). Slice ALL outputs (`c_final/status/solve-status/loss`)
   to current `M`/`total_c`.
5. BUILD: add the `.so` target; verify ctypes load + `nm` shows the three symbols unmangled.
6. ctypes WRAPPER: `ctypes.Structure` for **PopHeader (16×int32 = 64 B, incl. `reserved[7]`,
   magic `0x4D4C344D`, version 1) and TreeMeta (4×int32 = 16 B)**; `assert sizeof==64/16` at
   import; remap `pop['M']→header.M_prob` (dict has key `M`, no magic/version — **must-fix #5**);
   C-contiguous + exact-dtype + GC-pinned buffers.
7. WIRE BRIDGE: in `fit_natives`, `inproc` path calls `get_inproc_co(device).optimize(...)`;
   keep `_run_kernel` as baseline; `pipeline.py` unchanged.
8. PARITY tests (see below). 9. AMORTIZATION tests (see below). 10. Re-point
   `test_parity_gate.py` + `test_convergence_honesty.py` at the in-process path; doc the API.

## MUST-FIX before/while implementing (from the adversarial audit)
1. **Variant: RESOLVED — support BOTH (user decision 2026-06-17).** Factor the `__global__`
   kernels + LM-loop body into a shared header; build the `.so` parameterized to FD
   (`batch_lm`) and AD (`batch_lm_ad`) — variant-independent refactor, low marginal cost.
   End-to-end experiment uses **AD**; FD stays buildable. **Pin BOTH sides of every parity
   comparison to the SAME variant** (AD .so vs AD standalone; FD .so vs FD standalone) — never
   AD-vs-FD. Parity + amortization suites run per variant. (A win for AD here also feeds the
   deferred v4 AD-adoption decision; see memory `cusr-lm-v4-ad`.)
2. **Parity rests on an uncommitted self-asserted determinism claim + an undefined fp
   metric.** Add a step-0 that runs the standalone twice and commits the cmp as the durable
   determinism baseline; define an explicit inproc-vs-standalone fp tolerance + NaN policy
   (`verify.py`'s `c_rel` is vs scipy, NOT vs the standalone — not reusable as-is).
3. **Amortization test can't prove its headline and its gate is wrong.** The
   `time(co_init) ∈ 350–450ms` gate FAILS exactly when amortization works, because torch
   creates the CUDA context at fitness eval BEFORE any CO call (no `cudaSetDevice/cuInit`
   exists; context is lazy on first `cudaMalloc`). Replace with "CUDA primary context created
   exactly once per process (whoever pays it)" + `co_optimize` wall index-independence, and
   add a **timed END-TO-END generation** (extraction + co_optimize + writeback), since the
   `540→180ms` headline is per-GENERATION but the draft times only the LM loop.
4. **Stale-buffer forest corruption under reuse.** Re-seed `h_c` from current `c_init` every
   call and slice all D2H to current `M`/`total_c` (never capacity); add `d_stat/h_solve_stat`
   to the reset list. Non-accepting trees return `c_final==c_init` ONLY because the standalone
   re-`memcpy`s it each run — a persistent buffer that isn't re-seeded writes a prior, larger
   pop's optimized constants onto current nodes (FINITE → the `isfinite` writeback guard
   misses it). **The parity-under-reuse gate (larger→smaller→repeat) is the only detector.**
5. **PopHeader ABI marshalling.** `build_pop` returns `{'M':..}` (no `M_prob`, no magic/
   version); a naive pass-through KeyErrors or under-sizes the 64-B struct → OOB device reads.
   Synthesize the full header in the wrapper.
6. **Re-add `load_pop_bin`'s sum-consistency checks** (`sum(K)==total_c`,
   `sum(n_nodes)==total_nodes`) inside `co_optimize` before H2D; track per-buffer byte capacity.
7. **Correct false claims + verify dropped-individual routing.** The draft rationale says
   writeback is "a Tree-level CPU path, NOT a GPU scatter" — FALSE: `evogp.py:254` DOES scatter
   into the GPU forest view (per-tree H2D + 2 syncs), and that residual cost must be COUNTED,
   not deferred. Also: the per-tree GPU→host extraction (~5 syncs/tree × ~4000 = ~20k tiny
   syncs/gen) is in the shipped hot path and unmeasured — time it; if comparable to the
   ~180 ms loop, the A-over-C tie-break must be revisited. Add a mixed-batch `fit_natives`
   test (kernel-eligible + TFUNC/over-K trees) asserting no `None` remains in `results[M]`.

## Parity test design (corrected)
Golden = byte-for-byte vs the standalone (chosen variant, identical flags, A100 sm_80).
Gates: (1) single-call `np.array_equal(c_final/status)`; (2) **PARITY-UNDER-REUSE** — one
`co_init`, then `popA → popB(larger) → popC(smaller) → popA` all byte-identical to their own
goldens (incl. 65536→1000 to stress large-alloc-then-shrink); (3) status-branch + magnitude
span across fixtures (incl. operon high-K for `fail_cholesky`, scale-1e±6 fixtures);
(4) end-to-end `fit_natives` inproc-vs-subprocess identical. Keep `test_parity_gate.py` +
`test_convergence_honesty.py` as secondary "still-passes-quality".

## Amortization test design (corrected)
Build `libcusr_co_prof.so` (`-DPROFILE`) for the setup/loop split ONLY; use the non-prof
build for headline wall. Assert: context created exactly once per process; `co_optimize`
median wall ≤ 1.3× standalone `loop_ms` AND index-independent (no per-call re-init); a timed
end-to-end gen (extraction + optimize + writeback) inproc vs subprocess for the speedup.
median-of-3, `CUDA_VISIBLE_DEVICES` pinned, `nvidia-smi` idle-check first. State the split as
corpus-dependent; report measured actuals, don't assert a 180 ms number on an unprofiled corpus.

## Open questions for the human
1. ~~Variant: FD or AD?~~ **RESOLVED 2026-06-17: support BOTH (shared header); end-to-end uses AD.**
2. Alloc strategy: realloc-on-grow (memory-lean, small per-gen cost during bloat — fires
   ~6/9 gens on the Operon corpus) vs allocate-to-max-bound (wastes VRAM, OOM risk alongside
   torch). Pick a max M (e.g. pop_size 4000) + policy.
3. Zero-copy scope: is literal "minimize host↔device copies for the GPU-resident forest" a
   hard requirement for THIS milestone (→ that's Option C), or is "no per-gen disk + no
   subprocess" sufficient for P1 (→ A)?
4. Dataset residency / GPU-scatter writeback: keep `xs`/`ym` device-resident across gens and
   move writeback to a GPU scatter — both are v2 optimizations orthogonal to P1; confirm
   out-of-scope.
