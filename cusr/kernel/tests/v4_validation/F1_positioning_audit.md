# Adversarial audit of claim F1 (positioning / narrative) — overnight lit study

VERDICT: OVERSTATED (narrative is mostly correct + useful, but "only" and the
inclusion-criterion novelty are inflated relative to prior art a reviewer will cite).

## What SURVIVES (verified, reviewer-creditable)
- "Prior benchmarks lack inner constants" is FALSE → correct to NEVER claim it.
  Source: 2412.02126 Table 1 (probe, prior) has inner constants; Kommenda 2020 uses
  trig frequency detection. access: FULL-TEXT-via-summarizer (prior probe).
- Benchmark = headline, GPU kernel = enabling infra ("runnable at scale, and honestly").
  Supported by SR community's own roadmap, which wants exactly this kind of artifact.
  Call-for-Action / SRBench-2025 (arXiv:2505.03977, FULL-TEXT, PDF read directly):
  "An effective benchmark should contain diverse subclasses of problems, identifying
  which algorithms perform best in each scenario, mapping problems to algorithms rather
  than providing a one-size-fits-all solution. Additionally, the benchmark must challenge
  contemporary SR algorithms while remaining computationally feasible..."
  Also names linear scaling as the standard constant control → confirms Keijzer = control,
  not contribution: "This was alleviated with the use of linear scaling [32] by scaling
  and translating the solution before measuring the loss function."
- "Pure 'we made CO faster' is the weakest framing." Supported:
  - EvoGP (arXiv:2501.17168, FULL-TEXT abstract): headline = GPU architecture + speed
    (140.89x vs SOTA GPU TGP; 304x reported elsewhere); accuracy only stated as
    maintained/comparable (NOT a claimed quality breakthrough). Cautionary contrast holds.
    [low-confidence detail: two fetches gave "comparable accuracy" vs "maintaining or
    exceeding accuracy" — exact qualifier uncertain; the *headline-is-speed* fact is stable.]
  - Fast SR Benchmarking (arXiv:2508.14481, FULL-TEXT): frames compute savings (-41%/-63%)
    as a contribution, but ONLY secondary to a benchmarking-methodology contribution
    (equivalence-aware early stopping). Compute cost is never the headline in SR.

## What is OVERSTATED (why the verdict is OVERSTATED, not SURVIVES)
1. "ONLY defensible differentiators are (a) and (b)" — absolute "only" undercounts.
   The fp64-honesty guard (delivered<=init in honest fp64 for every tree; worst pre-guard
   fake gain 854x/1.5e16x) is a THIRD independent CORRECTNESS differentiator, and the real
   contribution is a bundle (criterion + honest kernel + isolated corpus artifact +
   end-to-end evolutionary R2/recovery scoring). Drop the word "only".
2. Linear-scaling-gap inclusion criterion is NOT a novel methodology — it is a domain
   instantiation of DISCRIMINATIVE / ADVERSARIAL-FILTERING benchmark construction, which is
   well-known: SWAG/HellaSwag (arXiv:1808.05326; ACL D18-1009) "select items a baseline
   fails." access: REVIEW-ONLY (search snippet + listing). A hostile reviewer WILL cite this.
   BUT the SR-specific application is genuinely first-of-kind: 2412.02126 applies no selection
   rule; LLM-SRBench (arXiv:2504.10415, FULL-TEXT) constructs by REFORMULATION not baseline-gap
   filtering; SRBench/Call-for-Action group by difficulty but never isolate CO-necessity via a
   linear-scaling gap. So position it as "first principled CO-necessity isolation in SR,
   instantiating discriminative filtering," NOT as inventing the filter.

## Revised (minimal-defensible) claim
The principal differentiators are (a) the linear-scaling-gap inclusion criterion — the FIRST
application of discriminative/adversarial-filtering benchmark construction to isolate the
inner/nonlinear-CO-necessary regime in SR (cite Keijzer as the control and SWAG/HellaSwag as
the general construction principle; do NOT claim the filter idea as new) — and (b) the honest,
fast GPU CO kernel (fp64-honesty guard + speed) that makes the benchmark runnable at scale and
trustworthy near singularities. Lead with the benchmark; position the kernel as enabling
infrastructure; never claim "prior benchmarks lack inner constants" (false); never lead with
raw kernel speed (EvoGP already owns the speed story and disclaims quality — CuSR's edge is
correctness + end-to-end SR value, which a pure-speed framing throws away).

## Gaps / caveats
- Could not read SWAG/HellaSwag full text (ACL/arXiv listings only) → REVIEW-ONLY on the exact
  adversarial-filtering mechanism; the *existence* of baseline-gap filtering as prior art is solid.
- EvoGP exact accuracy qualifier uncertain between two fetches (see above) → confidence=low on
  that one phrase; does not affect the cautionary-contrast logic.
- Did not find a paper that already isolates CO-necessity via a linear-scaling gap (searched);
  absence is supportive but is "not found," not "proven absent."
