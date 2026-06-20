# Study B — end-to-end EvoGP + constant-optimization: honest findings

**Run:** `run_parallel.sh` (5 seeds × 4 arms × 53 problems = 1060 cells; 8-GPU, sharded
by seed×group), 2026-06-21, AFTER the adversarial audit (workflow `wf_3916912b-830`) and the
ComplexInfinity-crash fix. Data: `out/study_b_sh*.jsonl`; analysis: `out/report.json`
(`aggregate.py`). This file is the human-readable record; every number below is reproducible
from the shards.

## Arms
`no_co` (CO never invoked) · `sparse_gpu` (GPU fp32 kernel CO every 5 gen) · `cpu_every`
(scipy fp64 CO every gen) · `gpu_every` (GPU fp32 kernel CO + fp64-honesty-guard, every gen).
Identical init population per (problem,seed); CappedCO (count_ops≤40, K≤32) symmetric.
**Fixed-GENERATION protocol — NO speed/hardware claim from generation count.**

## Per-arm (165 admit cells/arm)
| arm | solved (R²>.999) | lenient (symbolic) | med R²≥0 | fails | mean wall |
|---|---|---|---|---|---|
| no_co | 32 | 0 | 0.921 | 0 | 0.5s |
| sparse_gpu | 45 | 14 | 0.925 | 2 (timeout) | 3.9s |
| cpu_every | 39 | 12 | 0.905 | 5 (timeout) | 25.2s |
| gpu_every | **47** | **16** | **0.943** | 3 (timeout) | 8.9s |

## What is SUPPORTED

**C1 — the kernel is a sound systems artifact** (fp32 + fp64-honesty guard: delivered ≤ init
per tree). 0 crashes across 1060 cells after the ComplexInfinity fix.

**C2 — integrated EvoGP+CO improves end-to-end RECOVERY vs no-CO, SCOPED TO SINGLE-INNER.**
- Symbolic recovery (lenient), the judge-based headline: **gpu_every 16 vs no_co 0**,
  McNemar **p=3.1e-5**; cpu_every 12 vs no_co 0, p=4.9e-4. All 16 gpu gains are genuine
  non-foldable inner recoveries (9 freq_cos, 5 decay_exp, 2 shift_recip) — verified, none are
  control-type artifacts.
- Solved (R²>0.999, judge-independent): gpu_every 24↑/9↓ vs no_co, p=0.014.
- **The gain concentrates on the polynomial-NON-absorbable single-inner subset** (where inner-CO
  is genuinely needed): non-absorbable no_co **6 solved / 0 lenient → gpu 20 / 12**. On the
  absorbable subset (any arm can fit a low-degree poly on-domain) the arms tie (~23 solved).
- Stock EvoGP (no_co) reaches high R² via BLOATED overfit forms; CO finds the compact true
  skeleton — this is why no_co has 32 solved but 0 lenient.

**GPU value = quality-parity-with-CPU + completes CO where scipy fails.** NOT "better quality"
and NOT "every-generation CO".
- Quality parity: on cells where both arms completed, gpu-vs-cpu median ΔR²=+0.0005, p=0.22
  (null). The fp32-guard kernel matches fp64 scipy quality.
- Systems advantage: scipy (cpu_every) hits 5 wallclock timeouts (150s SIGKILL) + previously
  15 hard crashes; the kernel completes those cells in budget.

## What is NOT supported / corrected by the audit

- **cpu_every-vs-no_co is NOT "CO hurts".** The previously-reported significant Wilcoxon
  (Δ=-0.0012, p=0.03) was a **fail-floor artifact** — 15 ComplexInfinity crashes (all seed 4) +
  6 timeouts floored to R²=0. After the crash fix the all-cells number is +0.0001, p=0.44, and
  completed-both is +0.0001, p=0.48 (null). cpu CO is R²-parity + helps recovery, like gpu.
- **Held-out R² is SATURATED (~0.92–0.99) on these single-inner problems** (linear scaling
  already fits the outer constant); the between-arm R² Wilcoxons are all null. **RECOVERY
  (solved + symbolic), not R², is the signal.**
- **"CO every generation" buys nothing over "every 5 generations".** gpu_every-vs-sparse_gpu is
  null (solved p=0.84, lenient p=0.73). Do not claim every-gen value or "GPU lets you do CO
  every gen where CPU can't".
- **Multi-inner is UNTESTED, not refuted.** On the 15 multi-inner problems the gpu-vs-no_co
  comparison has near-ZERO power: lenient discordant pairs = **0**, solved discordant = 7
  (p=1.0), per-problem Wilcoxon Δ=-0.0036 p=0.45. GP rarely finds a representable multi-inner
  skeleton at all (4/15 unsolvable by ANY arm), so CO has nothing to optimize. Do NOT claim "CO
  does / does not help multi-inner" — the study cannot test it.

## Validity (controls — CO must not "unlock" a non-inner-CO problem)
- On `solved`/R²: holds — arms ~equal (no_co 39 vs gpu 42 of 100).
- On `lenient`: the fold-absorbable `shift_quad` control subclass IS "recovered" by ALL CO arms
  (no_co 0; sparse/cpu/gpu ~4 each) — a judge-compression effect on an already-R²≈1 class,
  SYMMETRIC across CO arms (so it does not bias gpu-vs-cpu). Excluded/footnoted from the lenient
  validity statement; the 16 gpu admit-side recoveries are not of this type.

## Reproducibility caveat
`no_co` and `gpu_every` are deterministic per seed; `cpu_every` (scipy LM) is cell-level
NONDETERMINISTIC and its fail-set is a noisy single draw (~±0.006 on R²). The gpu-vs-cpu parity
result is robust to this noise.

## Provenance
Adversarial audit (12 dimensions, 3-skeptic rebuttal, run `wf_3916912b-830`): 8 findings upheld,
10 rejected as false positives. Crash fix: `cusr/bench/skeleton.py` `_is_pathological` guard +
`cusr/demonstrator/co_backend.py` ScipyLM no-double-fault, TDD in `test_co_backend.py`. Analyses
[B3]–[B7] in `aggregate.py`. Pre-audit data archived in `out/_pre_rerun_20260620/`.
