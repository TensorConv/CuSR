# CuSR HPEC claims after reset

更新 2026-06-22. 本文件记录新的三条贡献。所有 e1-e6 结果在当前论文语境中都是 pilot lessons,
不是最终 paper evidence。最终证据必须按 [`experiment_plan.md`](experiment_plan.md) 重跑。

## Contribution 1: Workload and Benchmark Framework

> We define a benchmark framework for symbolic-regression problems that genuinely require nonlinear
> optimization of internal constants.

What it includes:
- Deterministic admissibility criterion.
- Detection of no-inner, outer-only, foldable, canonical/reachable, non-identifiable, and outer-linear-scaling
  sufficient cases.
- Sensitivity over canonical reachability sets.
- Constructed admitted corpus and matched controls.
- Neutral scoring and blind baseline validation.

Pilot basis:
- e3 criterion and constructed corpus.
- AI-Feynman audit showing the survivor count depends on reachability assumptions, while the robust statement is
  that most nominal inner constants are simple/reachable/foldable.

Paper evidence still needed:
- Audit table over standard suites: AI-Feynman/Feynman, Nguyen, Korns, Livermore or SRBench-available suites.
- Blind baseline checks: no-inner/linear-scaling/full nonlinear CO from random and canonical initializations.

Boundaries:
- Not a new statistical test.
- Not a universal claim that existing benchmarks ignore constants.
- Safe claim: widely used suites often contain few problems requiring nonlinear optimization of non-canonical
  internal constants.

## Contribution 2: GPU Batched Heterogeneous SR-CO Primitive

> We implement a CUDA primitive for batched second-order LM constant optimization over heterogeneous SR trees.

What it includes:
- Structurally heterogeneous trees in one batch.
- Heterogeneous constant count K.
- A per-tree Jacobian built by reverse-mode AD: one backward pass gets all K partials, independent of K.
  This is the main speed lever, because the Jacobian is 60-66% of the loop.
- LM solve and accept/reject.
- fp32 iteration with an fp64 honesty guard: the reported result is never worse than the starting point.

Pilot basis:
- e2 bridge/parity.
- kernel tests and honesty guard.
- e6 synthetic pilot and PROFILE measurements.

Paper evidence still needed:
- Real-dump fixed-tree apples-to-apples CO replay.
- Locked synthetic scaling stress test.
- Strong CPU baselines: Operon LM, PySR BFGS, scipy where appropriate.

Boundaries:
- Not first GPU CO.
- Not a novel optimizer.
- Difference from adjacent work should be narrow: custom CUDA, second-order LM, explicit per-tree Jacobian,
  structurally heterogeneous SR trees, heterogeneous K.

## Contribution 3: Deploying the Primitive Inside a GPU GP Engine

> We integrate the kernel into a GPU-native GP engine (EvoGP) and show how to run constant
> optimization cheaply inside the search loop.

What it includes:
- In-process integration: the kernel is called directly, so the population stays on the GPU and
  is not copied back and forth to the CPU every generation.
- A tree-format bridge with a correctness check: pull the heterogeneous trees out of the engine
  and feed them to the kernel (checked on 377 trees, no extraction error).
- Warm-start: offspring start from the parent's constants, so they converge in a few iterations
  instead of a full cold solve.
- Sparse scheduling: running CO every few generations works as well as every generation.
- Equal-wall-clock comparison of the deployment options (no CO / cold full solve / warm few-step).

(The "when is it fast / when not" performance picture is part of Contribution 2's evaluation, not here.)

Pilot basis:
- e2 bridge and extraction parity.
- e4 loop pilot: sparse CO works as well as every-generation; fp32 kernel quality matches fp64 scipy;
  the kernel finishes cases where scipy crashes.
- in-process .so drop-in (measured in-process speedup; exact number to confirm).

Paper evidence still needed:
- How many iterations warm-start saves.
- Equal-wall-clock EvoGP experiment (no CO / cold-50 / warm-5).
- Per-generation CO cost breakdown.

Boundaries:
- We do NOT claim "CO improves SR" — that is prior work. We only claim it can be deployed cheaply.
- No conclusion that needs the multi-inner case (the study cannot test it).
- No hardware speed claim from fixed-generation loop wall time.

## Prior-Art Boundaries

| Axis | Prior ownership / adjacent work | Our safe position |
|---|---|---|
| CO is useful in GP | memetic GP, Kommenda et al. | Motivation only |
| LM/NLS algorithms | standard numerical optimization | system implementation, not optimizer novelty |
| GPU CO over SR trees | Kozax / de Vries et al. | we differ by custom CUDA + second-order LM + explicit per-tree Jacobian |
| GPU batched NLS | Gpufit, JAXFit | they are fixed-model homogeneous fits; ours is heterogeneous SR trees |
| Conditioning/honesty | Kronberger et al. | honesty guard is a system feature |
| Linear scaling / benchmark filtering | Keijzer scaling, SRBench-style filters | our contribution is workload operationalization and controls |

## Safe Intro Bullets

1. A benchmark framework that identifies SR problems genuinely requiring nonlinear internal-constant optimization,
   with audit tables, sensitivity reporting, and matched controls.
2. A CUDA primitive for batched second-order constant optimization over heterogeneous SR trees, with a
   reverse-mode AD Jacobian and an fp64 honesty guard.
3. A deployment of the primitive inside a GPU GP engine (EvoGP): in-process integration and warm-started
   few-step refinement, showing constant optimization can be run cheaply in the search loop.
