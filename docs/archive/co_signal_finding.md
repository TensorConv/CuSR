# CO-signal gate (2026-06-08): the LM↔EvoGP seam carries signal

Before spending a full sweep on the integration, two checks that the constant
optimization (CO) actually changes outcomes — i.e. `_writeback_constants` isn't
silently inert (view-vs-copy aliasing, or the f32-truncation→rollback trap).

## 1. Direct: one writeback changes a member's fitness

Built a 300-tree EvoGP forest on `2·sin(2.5·x0)`, took the top-30 members with
constants, and for each: extract skeleton → `TorchLM.fit_batch` (f32) →
`_writeback_constants` → re-evaluate.

- **26/28 members changed fitness; all 26 improved**, median Δfitness **+0.041**.
- Well above f32 eval noise → writeback lands (aliasing works) and the
  f32 truncation does NOT null it.

## 2. End-to-end: CO-on beats CO-off where CO is required

Same seed, same everything, `co_every=1` (on) vs `co_every=∞` (off), on
`2·sin(2.5·x0)` — frequency 2.5 is **not** in EvoGP's `const_samples`
{0,1,-1,2,-2,0.5}, so GP can't reach it by terminals; only CO can.

| arm | best_fit | recovered | best expr |
|---|---|---|---|
| CO-off | -1.23e-2 | **False** | `-sin(0.068·x0) + 2·sin(2.546·x0)` (freq missed) |
| CO-on  | -1.5e-13 | **True**  | `2.0·cos(2.5·x0 − π/2)` ≡ `2·sin(2.5·x0)` |

CO-on nailed the frequency and recovered; CO-off didn't. (Also exercises the
judge's transcendental-rewrite handling: `cos(θ−π/2)=sin(θ)`.) Backend = TorchLM
f32 — so the in-loop f32 dtype works.

## Caveats / what this is NOT

- This is a **constructed, constant-bottlenecked** problem chosen to isolate CO,
  not a benchmark result. It proves the mechanism, not the magnitude.
- It does not yet test the **recipe** (adaptive λ + selection) vs naive CO — that
  (and whether naive aggressive CO causes lock-in) is the sweep's job (exp010).
- Backends compared by outcome here, not compute. The sweep must fix the compute
  axis (evals / LM-nfev), not wall-clock.
