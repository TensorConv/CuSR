# W5 — kernel CO bridge (tree → pop.bin → batch_lm) parity record

CudaKernelLM wired as a 009 CO backend. Path = **native EvoGP tree → pop.bin →
008 batch_lm kernel** (not sympy Skeleton → bytecode): reuses the already-
parity-gated `dump_evogp._extract_tree` + `popio.build_pop`, so E2 runs the
identical kernel path E1 benchmarks. Only the subprocess + per-tree slice/align
is new (`kernel_bridge.py`).

* bridge `kernel_bridge.fit_natives` — extract → build_pop → batch_lm → COResult
* contract: `fit_batch(..., native=trees)`; scipy/torch ignore `native`, pipeline passes `forest[mid]`
* fallback: K>32 / stack>64 / TFUNC trees → scipy (counted in `last_stats`), no candidate dropped
* loss currency: kernel c* (fp32) re-scored fp64 via `skel.residual` → mean r², same as scipy/torch

## Parity gate (laptop RTX 5070 Ti, PTX JIT; sanity only — A100 is paper-grade)

GPU-free oracle (encoding correctness, no kernel):
* arrays↔sympy: median rel-err 4e-7 over 40 synth trees (structural-bug gate; bulk = float32 input precision)
* extract round-trip: nt/ci + injected c_init recovered exactly

GPU kernel vs scipy, synth `inner-const-heavy` first 24 trees, mean r²:
* **direction-correct (kernel loss ≤ init): 24/24** — the "not garbage" gate (LM never accepts an uphill step; a wrong-function encoding would push loss measured via the correct skeleton UP)
* tier-B (≤1.05× scipy-fp64): 22/24
* the 2 misses are fp32 optimizer plateaus, NOT encoding: both reduced loss from init (0.81→0.49, 0.45→0.095) and function-identity at init = 2.4e-7 / 3.4e-8 (kernel `interp` semantics == scipy `skel` semantics). Consistent with the documented fp32 quality gap (W0/E1); fp64 build is W8.

Realistic-range gate (advisor must-do): `sin(c0·x0)`, c0_true=1e-5, init 20% off
→ kernel tracks scipy within tier-B. Exercises the W0 relative `eps_fd` on a tiny
inner constant (absolute FD would give a degenerate Jacobian here).

## Cross-representation invariant (the false-null gate)

The kernel input (`_extract_tree`) and the function scipy fits / writeback assumes /
the judge sees (`forest_member_to_skeleton`) are two separately-maintained walks of
the same tree. If they computed different functions, the kernel would optimize fn A,
writeback would stuff A-optimal constants into fn B, EvoGP would re-eval worse →
rollback → **p=1.0 collapses toward p=0 and we'd misread a wiring bug as the claim-3
null**. The earlier parity tests do NOT cover this — they compare the kernel to
`arrays_to_skeleton` (a mirror of the kernel's own semantics), never to the pipeline's
actual skeleton.

* checked 377 live EvoGP trees (random forest, full 009 FUNCS, SIGNED inputs to expose any `Abs` divergence): **0% function divergence, 0 const-count mismatch** → `_extract_tree` ≡ `forest_member_to_skeleton` on every tree that reaches the kernel
* the one real divergence point: **`LOOSE_SQRT`/`LOOSE_POW`** — `_extract_tree` sends them to plain `sqrt`/`pow`, but EvoGP's `SYMPY_MAP` (hence the skeleton/judge) uses `sqrt(|x|)`/`|x|^y`. These differ on negative args. EvoGP does NOT emit loose ops with the 009 FUNCS set (0/377), but the bridge **routes any such tree to the scipy fallback** (`_has_divergent_loose`, stat `n_loose_div`) so a config/mutation change can't silently reintroduce the false-null.
* codified as a regression test (`test_extract_vs_skeleton_function_agreement`).
* live in-loop smoke (korns_7, tiny pop): p=1.0 → 54/54 fits to kernel, 0 fallback, 0 kmismatch; p=0 → 0 CO calls (stock).

## Not done here (commit 2 / pilot)
* `co_probability` knob (p∈{0,0.14,0.5,1.0}); p=0 = honest stock EvoGP
* `run_pilot.py` scaffold (p∈{0,1.0} × ~5 problems × 10 seeds, ≤6/16 checkpoint) — NOT run
* A100 single-run cost (laptop cost is a rough proxy only)
