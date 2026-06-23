标题: CuSR HPEC 2026 direction reset
更新 2026-06-22.

本文件记录 2026-06-22 的方向重置。硬规则:

- 不被 e1-e6 的旧叙事和术语牵引。
- 现有实验全部降级为 **pilot / infrastructure / failure-mode discovery**。
- 论文数字和主 claim 的证据要按下面的新设计重新跑。
- `gen_synth` / e6 只能作为 synthetic scaling stress test, 不能替代真实 workload。

相关文件:
- 贡献和红线: [`research/contributions.md`](research/contributions.md)
- 重跑实验计划: [`research/experiment_plan.md`](research/experiment_plan.md)
- 旧实验台账: [`../experiments/RESULTS.md`](../experiments/RESULTS.md)

## New Core Story

Symbolic regression needs constant optimization, but not every "constant" benchmark actually tests the
same capability. Many standard equations have outer-only constants, canonical constants, foldable constants,
or constants absorbable by outer linear scaling. We first define the workload that genuinely requires
nonlinear internal-constant optimization. Then we build a GPU batched heterogeneous LM primitive for that
workload. Finally, we evaluate it under fixed-tree apples-to-apples protocols, and deploy it inside an EvoGP loop.

The paper is not:
- "CO is useful" as a broad claim. That is prior memetic GP knowledge.
- "GPU is always faster" as a broad claim.
- "e6 already proves the paper." e6 is a pilot synthetic scaling sweep.

The paper is:

> A benchmark/workload definition for nonlinear internal constants, a GPU batched heterogeneous LM primitive
> serving that workload (per-tree Jacobian by reverse-mode AD), and a way to deploy it cheaply inside a GPU GP loop.

## Three Contributions

### Contribution 1: Workload and Benchmark Framework

Claim:
We define and operationalize a benchmark framework for SR problems that genuinely require nonlinear
optimization of internal constants. The framework includes a deterministic admissibility criterion,
canonical/reachability sensitivity reporting, a constructed admitted corpus, and matched controls.

Why this comes first:
If existing benchmarks rarely stress nonlinear internal constants, then a GPU CO system has no clean target
workload unless we define one. This is the motivation and the evaluation foundation.

Pilot assets:
- e3 criterion prototype.
- AI-Feynman 34 audit showing most positional-inner constants are simple/reachable/foldable under reasonable
  canonical sets.
- Constructed admitted/control corpus.
- e4 hint that admitted single-inner cases are where CO changes symbolic recovery.

Paper evidence to rerun/build:
- Benchmark audit table over AI-Feynman/Feynman, Nguyen, Korns, Livermore or SRBench-accessible suites,
  plus constructed admitted/control corpus.
- Blind baseline validation: no-inner/linear-scaling/full nonlinear CO from random and canonical initializations.
- Sensitivity over canonical reachability sets, reported as a range rather than one brittle count.

Safe wording:
"Widely used suites often contain few problems that require nonlinear optimization of non-canonical internal
constants." Do not write "all existing benchmarks ignore internal constants."

### Contribution 2: GPU Batched Heterogeneous SR-CO Primitive

Claim:
We implement a CUDA primitive for batched second-order LM constant optimization over structurally heterogeneous
SR trees with heterogeneous K, a per-tree Jacobian built by reverse-mode AD, and fp64 honesty guarding.

Why this is HPEC:
This is a high-throughput primitive for a workload with irregular control flow, heterogeneous model structure,
and many small nonlinear least-squares problems. The contribution is system design and performance
characterization, not a new optimizer.

Pilot assets:
- e2 bridge/parity infrastructure.
- kernel tests and fp64 honesty guard.
- e6 pilot sweep showing possible large clean wins in many-cheap-tree/large-M regimes and quality boundaries
  in high-K rank-deficient workloads.
- PROFILE pilot: host round-trip is small, Jacobian dominates.

Paper evidence to rerun/build:
- Fixed-tree apples-to-apples CO benchmark on real EvoGP no-CO dumps.
- Locked synthetic scaling stress test for M/N/K/nodes axes.
- Baselines: Operon LM, PySR BFGS, scipy where appropriate.
- Neutral fp64 scoring, same trees, same data, same initial constants.

Safe wording:
- We do not claim first GPU CO.
- We do not claim a novel LM algorithm.
- We claim a custom CUDA second-order batched heterogeneous SR-tree CO primitive and its measured envelope.

### Contribution 3: Deploying the Primitive Inside a GPU GP Engine

Claim:
We integrate the kernel into a GPU-native GP engine (EvoGP) and show how to run constant optimization cheaply
inside the search loop: call it in-process so the population stays on the GPU, and use warm-started few-step
refinement instead of a full cold 50-iteration solve every generation. (The "when is it fast / when not"
performance picture is part of Contribution 2's evaluation, not here.)

Pilot assets:
- e2: the tree-format bridge and extraction parity (377 trees, no extraction error).
- e4 loop pilot: running CO every few generations works as well as every generation; the fp32 kernel matches
  fp64 scipy quality; the kernel finishes cases where scipy crashes.
- in-process .so drop-in (measured in-process speedup; exact number to confirm).

Paper evidence to rerun/build:
- How many iterations warm-start saves.
- Equal-wall-clock EvoGP experiment (no CO / cold-50 / warm-5), not only fixed-generation.
- Per-generation CO cost; iteration diagnostics (kernel `iter.bin`, active-tree histogram).

Safe wording:
- We do NOT claim "GPU CO improves SR" — that is prior work. We claim it can be deployed cheaply.
- No multi-inner conclusion unless the new experiment has power.
- No hardware speed claim from fixed-generation EvoGP wall time.

## What Existing Experiments Mean After Reset

- e1: infrastructure/pilot harness only. Old laptop numbers are not paper evidence.
- e2: bridge infrastructure and extraction parity evidence.
- e3: criterion prototype and pilot audit. Needs broader audit + blind baselines.
- e4: deployment pilot. Needs cleaner deployment protocol and possibly equal-wall-clock rerun.
- e5: failure-mode lesson only; do not use speed numbers.
- e6: synthetic scaling pilot. Useful for choosing axes and spotting envelopes; must be rerun locked and
  supplemented by real-dump fixed-tree replay.

## New Experiment Stack

### Parameter Axes Are Not Frozen

The paper should not treat `M`, `N`, `K`, tree size/depth, or LM iteration budget as fixed constants chosen
once. They are workload and budget axes. At least one paper table or figure should report how changing these
axes changes throughput, quality, and backend ranking.

Important axes:
- `M`: population/batch size. GPU efficiency may only appear after the batch is large enough.
- `N`: samples per tree. This changes arithmetic intensity, memory traffic, and CPU/GPU relative advantage; do
  not generalize from one `N`.
- `K` plus nodes/depth: controls Jacobian size, rank/conditioning, stack pressure, and solve cost.
- `max_iter` / warm-start budget: controls latency-quality tradeoff and decides whether full cold solves are
  necessary inside EvoGP.
- CPU thread count / Operon core count: needed to separate algorithmic envelope from CPU parallel-scaling effects.

Minimum deliverable:
- A parameter sensitivity table/figure covering at least `N`, `M`, `K`/tree-size presets, and `max_iter`.
- Explicitly mark which parameter settings are used for headline results and why.

### A. Benchmark Audit and Corpus Validation

Goal: prove the workload gap and validate the admitted/control split.

Outputs:
- Suite audit table.
- Canonical sensitivity table.
- Blind baseline behavior table.

This supports Contribution 1.

### B. Fixed-Tree Apples-to-Apples CO Benchmark

Goal: compare CO backends without search confounds.

Protocol:
- Same frozen trees.
- Same `X/y`.
- Same `c_init`.
- Same constant slot mapping.
- Backends may only optimize constants.
- Final loss scored by neutral fp64 evaluator.

Sources:
- Real EvoGP no-CO dumps.
- Benchmark ground-truth skeletons from admitted/control corpus.
- Synthetic calibrated workloads only as scaling stress tests.

This supports Contributions 2 and 3.

### C. Locked Synthetic Scaling Stress Test

Goal: map M/N/K/nodes envelope under controlled scaling.

This is a scaling stress test, not real workload evidence. It replaces e6 as a clean rerun, not as a copy
of e6's old story.

This experiment is also the main place to choose and justify paper parameter settings. If the final paper uses
only a small set of headline points, the sensitivity table/figure should still show how nearby choices change
the conclusion, especially along the `N` axis.

### D. EvoGP Loop Deployment

Goal: show the primitive matters inside search.

Minimum arms:
- no CO.
- CPU CO reference if affordable.
- GPU cold small-budget.
- GPU warm-start small-budget.

Metrics:
- symbolic recovery.
- best loss / time-to-threshold.
- equal-wall-clock comparison.
- CO overhead per generation.
- ranking/selection agreement with full CO reference if possible.

### E. Diagnostics

Goal: decide which algorithmic tweak is worth claiming.

Must add:
- kernel `iter.bin`.
- active-tree histogram.
- Operon iteration distribution.
- budget sweep `max_iter={5,10,20,50}`.

Optional:
- shape-frequency statistics for future interpreter specialization.

## Algorithmic Changes Allowed Before Claims Freeze

Reverse-mode AD (v5) is DONE and is now Contribution 2's main technical piece — it is the #1 speed lever
because the Jacobian is 60-66% of the loop. (This overrides the earlier "do not start reverse-AD" note.)
Rerun C2's performance numbers with v5, not the old forward-mode v4.

Allowed because they directly support the new experiments:
- write `iter.bin` and active histogram.
- expose Operon iteration statistics in result files.
- implement warm-start constant inheritance if small and local.
- run small budget sweeps.

Do not start now:
- full interpreter specialization.
- on-device accept/reject as a headline optimization.
- new damping/rank-rescue attempts for inner-heavy quality.

## Paper Skeleton

1. Introduction: SR CO workload ambiguity; need nonlinear internal-constant benchmark; need GPU primitive.
2. Benchmark framework: definitions, filters, canonical sensitivity, corpus and controls.
3. GPU primitive: tree encoding, heterogeneous batching, LM/Jacobian, honesty guard.
4. Experimental protocols: fixed-tree CO, synthetic scaling, EvoGP deployment.
5. Results A: benchmark audit and blind baseline validation.
6. Results B: fixed-tree backend comparison and scaling envelope.
7. Results C: EvoGP deployment and warm-start/few-step characterization.
8. Limits: high-K rank-deficient quality boundary, high-N collapse, multi-inner open frontier.
9. Conclusion.
