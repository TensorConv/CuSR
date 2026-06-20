# N3 adversarial referee — ANGLE B (broader names in stats/optimization/ML)

VERDICT: OVERSTATED

## Claim N3 (novelty, load-bearing)
fp64-honesty guard = correctness asset with "no documented precedent": near singularities,
use_fast_math fp32 LM accepts catastrophically-wrong steps as fake improvements (worst
final/init loss ratios 854x, 1.5e16x); guard recomputes honest fp64 loss at c_final & c_init
post-loop, reverts any tree non-finite or worse-than-init; guarantees delivered<=init in
honest fp64 for EVERY tree at ~0% overhead. "No surveyed GPU SR system claims or measures
numerical honesty of CO near singularities."

## Why OVERSTATED: the MECHANISM is a recombination of 3 textbook ideas

1. "Reject any point worse than the initial / never return worse-than-start" = Marquardt's
   ORIGINAL (1963) LM safeguard.
   - Wikipedia LM (FULL-TEXT, verbatim): "If both of these are worse than the initial point,
     then the damping is increased by successive multiplication by nu until a better point is
     found." https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
   - Generalized by trust-region gain-ratio: "When rho < 0 ... the step is rejected"; "only
     steps reducing the objective function value are accepted" (Ceres / GSL / Cornell TR wiki,
     ABSTRACT/REVIEW). And by Grippo-Lampariello-Lucidi non-monotone safeguarding (WebSearch).

2. "Compute in low precision, evaluate the key residual/loss in HIGHER precision" =
   mixed-precision ITERATIVE REFINEMENT (Wilkinson).
   - Wikipedia Iterative refinement (FULL-TEXT, verbatim): "iterative refinement ... produces a
     solution correct to working precision if double the working precision is used in the
     computation of r"; "mixed-precision evaluation of r_m where intermediate results are
     computed with unit round-off eps2 before the final result is rounded ... with eps1."
     https://en.wikipedia.org/wiki/Iterative_refinement
   - Four-precision LS refinement: residuals computed at the HIGHEST precision u_r
     (arxiv 2406.16499, ABSTRACT/search-snippet; NOTE: a too-perfect verbatim quote returned by
     the PDF summarizer looked FABRICATED and is NOT relied on — the u_r framing is corroborated
     independently by the royalsocietypublishing tensor-core IR result).

3. "fast-math can silently produce finite-but-WRONG values; VALIDATE final results in strict
   precision" = explicitly documented fast-math best practice / folklore.
   - Simon Byrne, "Beware of fast-math" (FULL-TEXT): recommended mitigation steps include
     "Validate the final numeric results" and compare against non-fast-math benchmarks; notes
     -freciprocal-math reduces accuracy and -ffinite-math-only lets the compiler delete isnan
     checks ("your compiler has just removed all those checks").
     https://simonbyrne.github.io/notes/fastmath/
   - Corroborated: GCC -fno-honor-nans/-fno-honor-infinities + isnan->false
     (kristerw.github.io/2021/10/26/fast-math-ub/, FULL-TEXT-via-summarizer); real breakage
     (Cantera issue #1155; Nutrient iOS PDF SDK). The exact hazard the guard defends against
     (approx reciprocal/sqrt + assume-no-NaN/Inf -> finite-but-wrong) is textbook.

=> Each ingredient is standard; combining them is engineering, not an undocumented mechanism.
   "No documented precedent" for the mechanism is FALSE/OVERSTATED.

## What SURVIVES (the maximal defensible / revised claim)
No SURVEYED GPU symbolic-regression constant optimizer is documented to apply a post-hoc
higher-precision (fp64) acceptance gate guaranteeing PER-TREE non-worsening under fast-math
fp32 LM, with MEASURED fake-improvement ratios — and most surveyed GPU SR systems do not
perform GPU constant optimization at all, so they cannot have addressed it.
- EvoGP (arxiv 2501.17168, FULL-TEXT): NO constant optimization, NO precision/fast-math/NaN
  honesty discussion (only NaN as array padding). https://arxiv.org/html/2501.17168v5
- TensorGP (arxiv 2103.07512, ABSTRACT/snippet): RMSE fitness; no CO-honesty discussion.
- SR "protected operators / safe log-exp / clip exp / replace NaN+/-Inf" literature
  (arxiv 1704.04998 interval arithmetic; arxiv 2603.21836; 2605.03841 — ABSTRACT/snippet)
  guards EVALUATION domain errors, NOT the distinct optimizer-fooled-by-fast-math problem.
  This distinction actually supports the NARROW novelty.

This is a "first to do/measure X in this niche" engineering-novelty + first-to-quantify claim,
NOT "no documented precedent." Keep the honest framing ("UNCLAIMED feature, not proof a
competitor is dishonest"). Drop/soften "no documented precedent"; cite Marquardt + iterative
refinement + fast-math best-practice as the lineage, and position the contribution as the
ENGINEERED + MEASURED guarantee in a GPU SR CO kernel.

## Could NOT find (gaps)
- No paper doing this EXACT post-hoc fp64 acceptance gate inside an SR/GP constant optimizer
  (so not REFUTED). Springer ch. 978-3-031-70055-2_20 (alt FP primitives in GP) was paywalled
  (303 redirect) — unread; could conceivably touch GP fast-math accuracy (low risk to verdict).
