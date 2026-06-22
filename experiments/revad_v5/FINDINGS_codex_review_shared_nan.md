# Codex review — shared-NaN residual contamination (REPRODUCED, CONFIRMED, then FIXED, 2026-06-22)

An external Codex review flagged that our reverse-AD correctness story has a blind spot.
I reproduced its central claim on the A100 + scipy fp64 oracle (Codex itself had no CUDA
device). **The finding was correct, and is now FIXED + verified.** This file is the honest record.

> **RESOLUTION (2026-06-22):** A 0-annihilating multiply (`revad_safe_mul`) at the 3 backward-pass
> push sites eliminates the residual. Verified: host test 45/45 (TDD RED→GREEN on 2 singular
> fixtures), GPU re-dump scipy classify **404/404 rev-correct, 0 residual** (was 368/404, 36),
> GPU parity both-non-finite **9052 → 52**, rev-worse **0**, mutation **10/10**. Perf cost +1.86%
> on the jac kernel (speedup vs forward 1.79× → 1.75×). Gate gaps #4 (MAX_NODES fail-fast) and
> #5 (dup-ci + poisoned-buffer fixtures, closing survivors A+B) also fixed. See the "RESOLUTION"
> section at the bottom for the full evidence trail.

## What was claimed
The parity gate (`test_revad_parity.cu:134`) auto-accepts **both-non-finite** (shared-NaN)
elements as "equal/OK", and `compare_scipy.py` only ever classified **disputed** elements
(`isfinite(fwd) != isfinite(rev)`) — so the shared-NaN bucket was validated by *neither*.
Codex extended the scipy check and found shared-NaN elements whose TRUE derivative is finite,
i.e. reverse-AD is *also* wrong there (not just forward).

## Reproduction (`classify_all_scipy.py` over the 13-tree sample, 404 elements)
```
both finite          = 300   (all 300 match scipy)
disputed (1 finite)  = 68    (all 68 rev-correct, fwd-wrong — the original finding, still true)
shared non-finite    = 36    <-- auto-accepted by parity, unexamined by compare_scipy
  -> tree value non-finite (undefined; NaN defensible)        = 0
  -> scipy also non-finite (true singularity; NaN defensible) = 0
  -> value finite & scipy FINITE -> BOTH AD MODES WRONG       = 36
        of which scipy == 0 exactly (0*inf signature)         = 36
        of which scipy != 0                                   = 0
reverse-AD vs scipy (finite-truth elements): 368 correct / 36 WRONG
```
Forward scores 300/404 (74.3%); reverse scores 368/404 (91.1%). Reverse is **strictly better
but NOT NaN-safe.**

Cited example confirmed verbatim: `m=1247`, pt0 — `fwd=[nan×6]`, `rev=[0,0,0,nan,nan,0]`,
tree value `= -0`. Columns k=0,1,2,5 reverse isolates correctly to 0; k=3,4 are shared-NaN
while scipy=0. Same singular point, reverse fixes 4/6 columns, poisons 2.

## Mechanism (revad_interp.cuh:130-134)
The backward pass does `sa[asp++] = adj * d1[i]` (and `adj * d2[i]`) unconditionally.
When a node's adjoint `adj == 0` but its local partial `d1/d2` is `inf`/`nan`, the product is
`0*inf = NaN` / `0*nan = NaN`. All 36 residuals are `adj==0` cases (true derivative is exactly 0):
  - **global-zero value** (m=1247, m=1568): some factor zeroes the whole output; every const's
    true deriv is 0, but the singular sub-path yields 0*inf.
  - **inactive branch** (m=2134, value=3.9168 ≠ 0): a MAX/MIN/comparison selects the other branch,
    so the dead branch's selector partial is exactly 0, but its inner partial is nan → 0*nan.

This is a **well-known limitation of naïve AD** (forward has it too, on *more* elements). It is
NOT a forward-vs-reverse difference in kind; reverse just contaminates fewer entries (column-
isolated instead of all-K-poison).

## Full-corpus scale (from self_test_parity_corrected.txt, 4000 trees / 27.43M elements)
```
both-non-finite          = 9052    (reverse NaN count)
fwd-nan/rev-finite       = 17048   (reverse FIXES these)
rel-fail / rev-worse     = 0 / 0   (reverse never worse than forward — Pareto)
```
forward NaN total = 26100 (0.095%); reverse NaN total = 9052 (0.033%) → reverse cuts NaN by ~65%.
The 9052 are NOT all "OK": the sample shows the bucket is dominated by finite-true-derivative
(residual contamination). The exact full-corpus split (true-singularity vs both-wrong) was not
measured (would need scipy over all 4000); the sample had 36/36 both-wrong, 0 true-singularities.

## Verdict on the prior claims
- "reverse-AD is MORE correct than forward-AD" — **TRUE** (368 vs 300 / 404; fixes 17048; never worse).
- "scipy validates reverse correctness (68/68)" — **OVERCLAIM**: scoped to the disputed set only.
  On the shared-NaN set reverse is 0/36. Correct statement: reverse matches scipy on the recovered
  (disputed) set and on all both-finite elements, but the gate never checked shared-NaN, where
  reverse is still wrong.
- "rev/fwd diverge ONLY as fwd=nan/rev=finite" — **FALSE**: they also agree-by-both-being-NaN on
  9052 elements, many/most of which are residual reverse contamination.

## Impact on the LM optimization
A NaN J entry poisons `(JᵀJ)[k][k] = Σ_i J[k][i]²` for that constant → its Gauss-Newton update is
NaN → trial rejected (NaN < cost is false), λ raised. Benign (no crash; self-heals by step
rejection), and strictly better than forward (which poisoned all K columns whenever any was
singular). But a single singular data point still poisons that constant's whole update — a 0 would
just not contribute. So the residual is a real (small, 0.033%) correctness gap at degenerate/
non-identifiable points, not merely cosmetic.

## RESOLUTION — APPLIED + VERIFIED (2026-06-22)

### The fix: `revad_safe_mul` (0-annihilating multiply) at the 3 backward push sites
```c
__host__ __device__ inline float revad_safe_mul(float a, float b) {
    return (a == 0.0f || b == 0.0f) ? 0.0f : a * b;
}
```
**Refinement note (honest):** I first proposed masking only `adj==0`. Prototyping (reverse_vjp_py.py)
+ fixture B showed that is INSUFFICIENT: the `inf-adjoint × 0-partial` shape (e.g. `sqrt(c*(x-x))`,
where `sqrt'(0)=inf` rides a nonzero adjoint down to a `×0` partial) needs the FULL form that
annihilates when *either* factor is exactly 0. `revad_safe_mul` is sound either way: a true-zero
factor (zero adjoint ⟹ node doesn't affect output; zero partial ⟹ operand doesn't affect node)
mathematically annihilates the product regardless of the other (possibly non-finite) factor. It only
kills spurious NaN; a genuine inf gradient (inf partial × nonzero adjoint, true singularity) still
returns inf.

### Evidence trail (all artifacts in experiments/revad_v5/ + mask/)
- **Python prototype** (reverse_vjp_py.py, fp32 mirror, faithful: SAFE_MUL OFF reproduces 36/36,
  fixed=0/broke=0): SAFE_MUL ON → **residual 0, fixed 36, broke 0**.
- **TDD host** (test_revad_host.cu): added `NaNsafe_A_zeroXinf` (adj==0 × inf) + `NaNsafe_B_infXzero`
  (inf × 0-partial). RED pre-fix (mask/red_host.txt: 43 PASS 2 FAIL), GREEN post-fix
  (mask/green_host.txt: **45 PASS 0 FAIL**).
- **GPU re-dump + scipy** (jac_sample_fixed.jsonl, mask/classify_fixed.txt): rev matches scipy
  **404/404**, residual **0** (was 368/404, 36). "Codex finding NOT reproduced" = fixed.
- **GPU parity** (mask/parity_green.txt, `--use_fast_math`): both-non-finite **9052 → 52**,
  fwd-nan/rev-finite 17048 → 26048, **rev-worse 0**, RESULT PASS.
- **Corpus-wide host probe** (_check_residual.cu, mask/check_*.txt — ALL 27.43M inner-const + 45.41M
  early-gen elements, precise host math): reverse non-finite **= 0**, contamination (NaN & finite value)
  **= 0**. So the precise algorithm leaves ZERO finite-value NaN.
- **The 52 are `--use_fast_math` artifacts, NOT algorithm contamination** (corrected — I first wrongly
  called them "genuine value-NaN singularities"). Confirmed: a no-fast-math GPU parity build
  (mask/parity_nofast.txt) gives both-non-finite **= 0** (matching the precise host). The 52 are FTZ/
  approx-transcendental edge effects at finite-value near-singular points under fast-math; they hit
  forward and reverse equally (rev-worse=0), ~0.0002% of elements, benign (LM rejects a NaN step).
- **Perf A/B** (median-5; mine + independent re-run on a 2nd GPU): safe_mul jac-kernel cost
  **+2.3% inner-const** (57→58 ms) and **+3.6% early-gen** (70→72 ms) — both well under 5%. Speedup
  vs forward AD stays **1.76× inner-const**; early-gen is **1.43×** but that regime is intrinsically
  <1.5× reverse-vs-forward BEFORE any patch (pre-fix 1.48×), so the patch is not the cause. (Honesty
  correction: my first "+1.86%" was inner-const-only and slightly optimistic vs the re-measured +2.34%.)
- **Math soundness** (adversarial property-test, _fuzz_safe_mul.cu, mask/fuzz_safe_mul.txt): **12 trees,
  0 counterexamples**. 5 singular shapes (0*inf, inf*0, 0*nan, MAX dead-branch) → finite correct 0;
  7 normal controls → AD matches central-difference (max rel ~1e-4), proving safe_mul never zeroes a
  genuinely-nonzero gradient. Sound: a factor is annihilated only when EXACTLY 0.0f, which mathematically
  kills the edge contribution regardless of the other factor.
- **Mutation** (mutation_test.py, mask/mutation_extended.txt): **10/10 caught**, incl. disable-safe_mul,
  +=→= (survivor B closed), delete-zero-init (survivor A closed).

### Independent multi-agent review (复核) — 5 reviewers + my fuzzer
- TDD-integrity: **HONEST** (pass-criteria strict, RED→GREEN causal, /tmp mutation re-break, 45/0 rebuild).
- Honesty/git audit: **CLEAN** (all numeric claims match artifacts; old result files untouched; work
  uncommitted; RESULTS.md dirtiness is unrelated pre-existing editorial work, no revad numbers touched).
- Independent GPU re-run (2nd GPU): **MATCH** (parity PASS, rev-worse 0, both-non-finite 52, no-fast-math 0,
  scipy 404/404).
- Corpus-wide residual (3rd GPU): **CONFIRMED** (contamination 0/0 over 27.4M+45.4M, fast-math 52 → no-fast 0).
- Perf (2nd GPU): patch cost small (+2.3–3.6%); flagged the +1.86% as inner-const-only (corrected above).
- Math soundness: my _fuzz_safe_mul.cu (the pure-reasoning agent kept hitting the 90s stream-idle watchdog).

### Gate fixes (#4, #5) — DONE
1. `test_revad_parity.cu`: **FAIL-FAST** (exit 2, RESULT FAIL) when `max_nn > MAX_NODES`, before launch.
   Verified: normal build PASS; `-DMAX_NODES=8` build exits 2 without launching (mask/parity_tiny.txt).
2. host test: **duplicated-ci** fixture `dupci_c0x_plus_c0` (discriminates `+=` vs `=`) and
   **poisoned-buffer** fixture `poisonbuf_zeroinit` (forces internal zero-init) — both proven by mutation.
3. `classify_all_scipy.py` classifies the shared-non-finite bucket against the scipy oracle (the gate
   that the original compare_scipy.py / parity test lacked).

### Not changed (scope) — forward AD
Forward-AD (ad_interp.cuh / batch_lm_ad.cu, the deployed v4) has the SAME NaN-contamination class,
worse (all-K poison). Left untouched: ad_interp.cuh is a shared invariant ("绝不修改"), and reverse
giving finite where forward NaNs is the ACCEPTED parity bucket (rev-worse stays 0). A forward fix is
a separate effort.

## Codex P2 follow-up — removable singularities (safe_mul can silently give 0 instead of nonzero)

Codex flagged that the global 0-annihilating multiply can erase a GENUINE nonzero derivative at a
removable singularity. **Verified correct (mathematically):** `pow(sqrt(c0), x0)` at `c0=0, x0=2` is
`(√c0)²=c0` on the valid domain, so the true right-derivative is **1**, but the chain rule gives
`[2√c0]·[1/(2√c0)] = 0·∞` and safe_mul annihilates it to **0**. Probe `tests/_probe_removable.cu`
(mask/fuzz... no — direct run) confirms: x0=2 → rev=0 (wrong, true=1); x0=1 → inf (genuine sing., correct);
x0=3,4 → 0 (correct, true=0). Forward AD and pre-fix reverse give **NaN** at x0=2 (also wrong);
central-FD gives NaN (boundary). Only one-sided FD / symbolic gives 1. So it is a **fundamental
first-order-AD limitation**, not a safe_mul bug; safe_mul trades loud-wrong (NaN) for silent-wrong (0).

**Does it occur in the real workload? NO.** `tests/_probe_corpus_silent.cu` compares plain-VJP vs
safe-VJP vs a robust in-domain FD oracle over EVERY element:
```
inner-const (27.43M):  affected (plain-nan -> safe-finite) = 9000, oracle-finite = 9000, SILENT-WRONG = 0
early-gen   (45.41M):  affected                            = 3000, oracle-finite = 3000, SILENT-WRONG = 0
```
All 9000+3000 safe_mul-affected entries AGREE with the FD oracle (all are genuine zeros). The probe
would flag a Codex-type case (safe=0 vs FD=1: rel 0.999 > 0.1) — it found none. Codex's case needs an
exact structural coincidence (exponent exactly 2 over a √, constant exactly at the singularity),
measure-zero and absent here.

**Decision: keep safe_mul, document the limitation.** Rationale: (1) correct for 100% of the actual
workload; (2) the pre-fix alternative (NaN) is ALSO wrong at Codex's point AND worse for LM (one NaN
poisons a constant's whole Gauss-Newton column; a conservative 0 only under-weights one point); (3) a
purely-LOCAL "narrower mask" cannot separate true-zero from removable-nonzero — both are locally `0·∞`
(separating them needs per-column dependency tracking, i.e. higher-order info); (4) no naive AD mode
handles removable singularities. `_probe_corpus_silent.cu` is kept as a regression guard: re-run it if
the workload changes, and if SILENT-WRONG > 0 we revisit (value-epsilon regularization or fail-loud).
