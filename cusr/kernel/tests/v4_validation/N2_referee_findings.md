# N2 Novelty Claim — Hostile Referee (Angle C) Findings

CLAIM N2: No surveyed GPU-SR system does gradient/nonlinear-least-squares CO ON the GPU for
heterogeneous per-individual SR trees; Gpufit fits one fixed model batched over datasets, not
heterogeneous per-tree Jacobians; so a GPU LM-CO kernel for heterogeneous SR trees is an open gap.

VERDICT: SURVIVES (as a narrow, carefully-scoped claim). Minor OVERSTATEMENT risk only in breadth
of phrasing, not in substance. Recommend the maximal-defensible revision below to be audit-proof.

## Systems checked and where their CO runs

- EvoGP (arXiv 2501.17168) — FULL-TEXT (HTML). Constants handled ONLY by genetic operators
  (mutation/random gen); NO numerical optimizer. GPU = tensorized parallel eval + genetic ops.
  https://arxiv.org/html/2501.17168v1
- Beagle (arXiv 2603.12292) — FULL-TEXT-via-summarizer. No numeric constant optimizer; GPU = genetic
  operators + fitness eval. https://arxiv.org/pdf/2603.12292
- DISCOVER (arXiv 2602.06986) — FULL-TEXT-via-summarizer. Sparse LINEAR feature selection
  (OMP/MIQP/SA); no nonlinear per-tree CO. GPU = feature gen + model eval.
  https://arxiv.org/html/2602.06986
- PSRN / PSE (Nature Comp Sci 2026; arXiv 2407.04405) — STRONGEST THREAT. Source code FULL-TEXT:
  model/regressor.py fits constants via scipy.optimize.minimize(method="Powell") on CPU with NumPy
  loss; GPU only enumerates/evaluates millions of expressions. CO is CPU, not GPU. NOT genetic
  programming (enumeration). https://raw.githubusercontent.com/intell-sci-comput/PSE/main/model/regressor.py
- SyMANTIC (arXiv 2502.03367) — FULL-TEXT. GPU torch.linalg.lstsq = LINEAR least squares for outer
  coefficients of pre-built features; explicitly CANNOT fit nonlinear inner constants (sin(2*pi*x)
  needs 2pi in operator set). Near-neighbor; does NOT preempt nonlinear heterogeneous-tree CO.
  https://arxiv.org/html/2502.03367v1
- Operon (ACM 2020; operongp.readthedocs.io) — ABSTRACT/REVIEW. Nonlinear-least-squares CO via
  Ceres+Eigen = CPU library; AD via dual numbers on CPU. GPU/SIMD = evaluation backend, not CO.
- Gpufit (PMC5691161) — FULL-TEXT. Confirms claim verbatim: SAME model function (compiled in) across
  all fits, batched over many independent datasets w/ per-fit start params; LMA entirely on GPU but
  "users cannot supply per-fit custom expressions without rebuilding." => single fixed model, not
  heterogeneous per-tree. https://pmc.ncbi.nlm.nih.gov/articles/PMC5691161/

## Prior art that exists (do NOT claim as novel; claim is narrower than these)
- Gradient/LM/Adam CO INSIDE GP trees is well established but CPU/conceptual: Topchy & Punch 2001
  (local gradient search of numeric leaf values); "Parametrizing GP Trees through Gradient Descent"
  GECCO 2023 (dl.acm.org/doi/10.1145/3583133.3590574); constant optimization in multiobjective GP
  (s10710-021-09410-y). These establish "gradient CO in GP matters" — NOT a GPU kernel.

## Null results (weak negative evidence)
- Repeated targeted searches for "GPU + per-individual/per-tree gradient/LM constant optimization +
  CUDA kernel" returned no system; one search engine summary itself concluded the combination "may
  not have substantial published work yet."

## Residual risk (honesty)
- ABSENCE-based + null-search claim; an obscure/industrial/unpublished GPU-CO system cannot be fully
  excluded. SRBench/PMLB issue search returned nothing usable (search engine limitation, not a clear
  null). PSRN had a use_constant flag that LOOKS like GPU CO but is CPU Powell on inspection — shows
  how easy it is to mistake eval-with-constants for GPU CO; keep the claim's wording precise.

## Maximal defensible (revised) claim
"Among surveyed and recent GPU symbolic-regression systems (EvoGP, TensorGP/KarooGP-class,
SymbolicRegressionGPU/PSRN, Beagle, DISCOVER, SyMANTIC, Operon's GPU eval backend), GPU acceleration
covers expression/fitness evaluation, genetic operators, and at most LINEAR coefficient solving;
numeric NONLINEAR constant optimization, where present, runs on the CPU (scipy Powell in PSRN,
Ceres/Eigen in Operon). General-purpose GPU Levenberg-Marquardt (Gpufit) fits a single compiled
model batched over datasets, not heterogeneous per-individual SR trees with per-tree Jacobians.
We found no system performing GPU-side nonlinear/gradient constant optimization of heterogeneous
per-tree SR expressions, so this remains an open gap (subject to the caveat that an
obscure/unpublished system cannot be fully excluded)."
