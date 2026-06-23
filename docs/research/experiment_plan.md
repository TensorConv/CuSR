# Experiment plan after reset

更新 2026-06-22. All paper-supporting experiments are to be rerun. Existing e1-e6 results are pilot
lessons only.

## Global Parameter-Axis Policy

Do not freeze `M`, `N`, `K`, tree size/depth, or LM iteration budget before the evidence is in. These are part
of the workload definition and deployment budget, not incidental constants.

Required sensitivity axes:
- `M`: population/batch size.
- `N`: samples per tree; this is especially important because it changes both throughput and CPU/GPU ranking.
- `K` and nodes/depth presets.
- `max_iter` and warm-start budget.
- CPU thread count / Operon core count.

Paper deliverable:
- Include a table or figure showing the effect of these choices on throughput, fp64 quality, and backend ranking.
- State which values are used for headline experiments and justify them from the sensitivity sweep.

## P0: Define and Validate the Workload

### 1. Benchmark Audit Table

Question:
Which standard SR benchmarks actually require nonlinear optimization of non-canonical internal constants?

Suites:
- AI-Feynman / Feynman subset.
- Nguyen.
- Korns.
- Livermore or SRBench-accessible suites if parsable.
- Constructed admitted corpus and matched controls.

Output columns:
- total problems.
- positional-inner count.
- no-inner / outer-only.
- foldable.
- non-identifiable.
- canonical/reachable.
- linear-scaling sufficient.
- admitted nonlinear internal constants.
- sensitivity range under canonical-set variants.

Acceptance:
The table must support a bounded claim, not a universal one: common suites often contain few genuine
nonlinear internal-constant problems.

### 2. Blind Baseline Validation

Question:
Does the constructed admitted/control split change solver behavior without using ground-truth seeding?

Arms:
- no-inner or outer linear scaling.
- nonlinear CO from random initialization.
- nonlinear CO from canonical/reachable initialization.
- optional: Operon LM / scipy / PySR as separate optimizers.

Report:
- admitted corpus: linear/no-inner fails, nonlinear CO succeeds.
- controls: linear/no-inner already succeeds or nonlinear CO gives no special unlock.
- neutral fp64 scoring.

## P1: Fixed-Tree CO System Evaluation

### 3. Real-Dump Fixed-Tree CO Replay

Question:
On real SR populations, how do CO backends compare when tree search is frozen?

Protocol:
- Same frozen trees.
- Same `X/y`.
- Same `c_init`.
- Same constant slot mapping.
- Backend may only change constants.
- Final score by neutral fp64 evaluator.

Tree sources:
- EvoGP no-CO snapshots selected to cover early, late/bloated, and high-K/internal-constant regimes.
- If useful, Operon dumps as separate workload corpus, not mixed with EvoGP in the same claim.

Backends:
- GPU AD.
- GPU fusedfd.
- Operon LM.
- PySR BFGS.
- scipy for smaller reference subsets.

Outputs:
- throughput vs fp64 loss Pareto.
- [ISO]/[SPD] tags.
- status and iteration distributions.
- nodes/K/depth summaries.

### 4. Locked Synthetic Scaling Stress Test

Question:
How does the primitive scale across M/N/K/nodes when workload axes are controlled?

Role:
This replaces e6 as a clean rerun, but remains synthetic. It supports scaling envelope, not real workload
claims.

Axes:
- M.
- N.
- K/nodes/depth presets.
- max_iter for a small subset if it changes quality or backend ranking.
- AD vs fusedfd.
- Operon core counts.

Outputs:
- locked-clock throughput.
- fp64 quality gate.
- N-axis collapse and large-M behavior.
- sensitivity table/figure for paper parameter choices.
- clear labeling as synthetic calibrated stress test.

## P2: Deployment and Algorithmic Diagnostics

### 5. Iteration Diagnostics

Question:
Are 50 LM iterations a conservative cold-start budget, and how many iterations do competing backends actually use?

Implementation:
- kernel writes `iter.bin`.
- kernel records active-tree histogram if cheap.
- Operon result files record `summary.Iterations` median/p90/max.

Sweep:
- GPU AD `max_iter={5,10,20,50}` on representative real-dump and synthetic points.

Outputs:
- loop time vs budget.
- fp64 quality vs budget.
- active-tree decay.

### 6. Warm-Start / Few-Step Deployment

Question:
Can in-loop CO be deployed as warm-started few-step refinement rather than cold full solves?

Arms:
- no CO.
- CPU CO reference if affordable.
- GPU `cold-50`.
- GPU `cold-10`.
- GPU `warm-10`.
- GPU `warm-5`.

Metrics:
- symbolic recovery.
- best loss.
- time-to-threshold under equal wall-clock.
- selection ranking agreement with full CO reference.
- CO wall time per generation.

## P3: Do Not Block the Paper

Reverse-mode AD (v5) is DONE and IS paper-critical now — it is Contribution 2's main technical piece
(the Jacobian is 60-66% of the loop). Rerun C2 numbers with v5. (This overrides the earlier note that
listed reverse-AD as "do not make paper-critical".)

Do not make these paper-critical:
- full interpreter specialization/JIT.
- on-device accept/reject as a headline optimization.
- new damping/rank-rescue attempts for high-K quality.

Small supporting diagnostics are allowed:
- shape-frequency statistics.
- profile summaries.
- simple active-tree stats.
