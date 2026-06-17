# fp32-honesty regression

Regression cases for the **fp32-honesty gap**: pre-CO trees on which the kernel reports
`status==0` (converged) yet its returned `c_final`, scored in **exact fp64**, is *worse*
than the starting `c_init`. The kernel's fast-math fp32 arithmetic is fooled near
singularities (fast reciprocal stays finite where exact fp64 blows up), so the LM accepts
a bad step and declares convergence. In a real GP loop the fp32 fitness would then *select*
these "looks-good / actually-terrible" trees — a silent quality leak. See the writeup:
`results/operon_baseline__20260617/report.html` finding **F5**, and the source check
`results/operon_baseline__20260617/probes/probe_fp32_lie.py`.

## The property under test
> A tree reported converged (`status==0`) must NOT be worse in exact fp64 than its start.

## Baseline (the bug, as committed) — established kernels FAIL
257 offender trees (the SERIOUS, >2×-worse subset across 17 problems × gens{0,4,16,64,100},
cap32/seed0/noise0), one fixture `pop.bin` per problem.

| kernel | serious (>2× worse) converged trees | verdict |
|---|---|---|
| `batch_lm_fusedfd` (FD) | 57 | FAIL |
| `batch_lm_ad` (AD) | 208 (worst 7.6×10⁵×) | FAIL |

AD is worse — its exact derivatives drive harder into the fp32-singular regions.

## Use
```bash
# baseline (expected FAIL — documents the bug):
python cusr/kernel/tests/fp32_honesty_regression/check_fp32_honesty.py cusr/kernel/batch_lm_fusedfd
python cusr/kernel/tests/fp32_honesty_regression/check_fp32_honesty.py cusr/kernel/batch_lm_ad
# after optimizing the kernel, point it at the new binary — PASS == serious offenders cleared:
python cusr/kernel/tests/fp32_honesty_regression/check_fp32_honesty.py path/to/kernel [--max-iter 1000] [--gpu 0]
```
The checker runs CO from each tree's original `c_init`, recomputes the neutral fp64 0.5·SSE
at `c_final`, and **exits non-zero if any serious (>2×) converged-but-worse tree remains**.
It also reports the softer counts (all converged-but-worse, catastrophic >100×) and how many
baseline-serious offenders the binary still fails. A legitimate fix may either converge to a
genuinely better point OR honestly report non-convergence (`status!=0`) — both PASS; only
`status==0` + fp64-worse FAILs.

## Files
- `fixture__<problem>.bin` — committed pop.bin fixtures (the offender trees + their `(X,y)`).
- `manifest.json` — per tree: origin `(problem, gen, orig_index)`, `K`, fp64 start loss, and
  the baseline FD/AD behaviour (status, fp64-final, kernel-fp32, serious flag).
- `check_fp32_honesty.py` — the regression test. `_eval.py` — self-contained fp64 evaluator.
- `build_fixture.py` — how the fixtures were extracted (one-time; needs the corpus + the
  committed baseline run on disk). The fixtures + manifest are the durable artifact.
