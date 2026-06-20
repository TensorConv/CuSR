# N3 adversarial audit (fp64-honesty guard novelty) — VERDICT: OVERSTATED

ANGLE C (2023-2025 preprints / benchmark repos / SRBench-PMLB). Hostile referee searched for a preemptor.

## What survives
- Narrow positioning holds: NO surveyed GPU SR system (EvoGP, TensorGP, SymbolicRegressionGPU/RayZhhh,
  CUDA-GP papers) discusses or measures numerical honesty of CO near singularities. SRBench/PMLB has no
  such issue. The honest-fp64-revert guard for GPU SR CO is not found in the surveyed literature.

## Why OVERSTATED (the words "no documented precedent" overreach)
1. Gpufit (Scientific Reports 2017, PMC5691161) = GPU fp32 Levenberg-Marquardt curve fitting. It DOES
   acknowledge fp32-vs-fp64 divergence: "At very high SNR, differences appear due to the limited numerical
   precision of floating point operations in CUDA (single precision) vs. MINPACK (double precision)."
   -> fp32-GPU-LM precision loss is DOCUMENTED. (But: not SR; framed as minor high-SNR artifact; NO
   fake-improvement pathology, NO recompute-revert honesty guard.) FULL-TEXT (PMC + nature).
2. Mechanism = folklore: "validate accept/reject in higher precision than the optimization runs in" is a
   cousin of (a) trust-region/LM actual-vs-predicted reduction (rho) step accept/reject [Wikipedia Trust
   region / LM], and (b) mixed-precision iterative refinement [Wikipedia]. Cross-precision specifically
   (fp32 compute / fp64 validate-and-revert) for SR CO is not found, but the building block is standard.
3. Interval/affine-arithmetic SR (Keijzer 2003; Dick 2017 arXiv:1704.04998; Pennachin affine) rejects
   invalid/singular models — but at MODEL-STRUCTURE level (trees with asymptotes/undefined regions), NOT
   fp32-vs-fp64 CO precision. Dick 2017 abstract explicitly does not touch precision/LM. Different
   mechanism, not a preemptor. ABSTRACT (Dick) / REVIEW (others).

## Maximal defensible (revised) claim
Replace "no documented precedent" with: the specific pathology (fp32 fast-math LM accepting catastrophic
fake improvements near singularities, ratios 854x / 1.5e16x) and the fp64 recompute-and-revert guard are
UNDOCUMENTED IN THE GPU SR LITERATURE; fp32-GPU-LM precision loss is known in general GPU curve fitting
(Gpufit) and the validate-in-higher-precision principle is numerical-optimization folklore, but neither
targets SR CO honesty near singularities, and no GPU SR system claims/measures it.

## Counter-sources
- Gpufit paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC5691161/ , https://www.nature.com/articles/s41598-017-15313-9
- Gpufit issue #63 (double vs single precision): https://github.com/gpufit/Gpufit/issues/63
- Dick 2017 interval-aware GP: https://arxiv.org/abs/1704.04998
- EvoGP (no precision/honesty/CO discussion): https://arxiv.org/html/2501.17168v1
- Trust region / LM rho-ratio folklore: https://en.wikipedia.org/wiki/Trust_region , https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
