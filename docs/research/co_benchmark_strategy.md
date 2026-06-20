# CuSR — FINAL de-inflated deliverable (experiment design + paper narrative)

Read by the project lead, who wants inflation removed. Every recommendation is formatted **Claim → Evidence (source + access_level) → Strongest surviving objection → Net recommendation + confidence**. REFUTED/OVERSTATED content appears only in the "must NOT claim / cut" lists. Reviewer kill_list items of severity fatal/major have been removed or downgraded; where I cut something the prior drafts asserted, I say so.

Evidence-discipline note carried throughout: `access_level` is reported verbatim and never upgraded. Three classes of evidence are kept visibly distinct: **(i) committed + reproducible** (in `results/`, with `report.json`); **(ii) project-internal single-run** (real but uncommitted, e.g. the in-process speed numbers); **(iii) absence-of-evidence** (novelty first-ness, R13, NO-SOURCE). Mixing these was the central inflation both reviewers caught.

---

## 1. Bottom line up front

### 1a. The one defensible headline

**Headline (state exactly one):** *CuSR contributes a fast, in-process, **fp64-honest** GPU constant-optimization kernel for heterogeneous symbolic-regression trees, together with a **designed** selection criterion (the instrument, to be built) for measuring where nonlinear (inner) constant optimization is necessary. On byte-identical pre-CO trees the kernel is competitive with a mature CPU optimizer (Operon) in honest fp64 (near-tie median, per-problem mixed), with a longer degradation tail, while delivering a guarantee Operon does not: the delivered fit is never worse than the starting fit, for every tree.*

This is the **systems + honesty** framing, with the benchmark/criterion as the instrument to be built and run. It is deliberately *not* "we made CO faster" (the weakest, most-rejectable framing) and *not* "the first inner-constant benchmark" (false / promissory).

### 1b. The linear-scaling-gap NOVELTY verdict — BLUNT

**Verdict: NOT NOVEL as a statistical principle; UNCERTAIN (bounded) as an SR operationalization; and — verified in the repo — NOT YET IMPLEMENTED as a selection rule.**

- The decision principle (fit a reduced model, fit a full model, use the SSE/R² gap to decide whether nonlinear params are needed) is a re-skin of the textbook extra-sum-of-squares / general-linear F-test for nested models (R1, `online.stat.psu.edu/stat462/node/135`, FULL-TEXT). Not novel — decisive.
- Filtering benchmark problems by a linear baseline is prior art (SRBench 2.0's one-sided exclude-trivial filter, R2, `arxiv.org/html/2505.03977v1`, FULL-TEXT) and so is linear-fit quality as a descriptive difficulty meta-feature (Oliveira et al. 2018, R3, `arxiv.org/pdf/1805.10365`, FULL-TEXT). The bounded-novel residual is only the *prescriptive two-sided Keijzer-scaled inclusion rule*.
- **Verified:** the rule is not built. `cusr/demonstrator/taxonomy.py` is, by its own docstring, "a POSITIONAL definition and an UPPER BOUND" that "does not detect constants that are secretly absorbable outer scales"; `cusr/demonstrator/judge.py` uses linear scaling only as a recovery-*scoring* leniency (`outer="lenient"`). There is no `tau_low`/`tau_high`, no `R²_LS`/`R²_full`, no admit/reject computation anywhere in `cusr/` or `experiments/`.

**Therefore lead with the pivot.** The criterion is not novel-enough *and* not built-enough to headline a benchmark today. Headline the systems + honesty contribution; treat the criterion as the instrument. The build-back path is explicit: *building and running* the criterion (§4) is how a criterion-led framing could be earned later — this declines to headline an unbuilt, principle-non-novel thing, it does not abandon it.

### 1c. Contribution-status ledger (the organizing spine)

| Status | Asset | Evidence class |
|---|---|---|
| **DONE** | Study A: Operon LMOptimizer (fp64) vs GPU kernel (FD+AD), CO vs no-CO, on byte-identical pre-CO trees | Committed + adversarially reviewed (`results/operon_baseline__20260617/report.json`) |
| **DONE** | fp64-honesty guard + its un-gameable regression gate (`delivered ≤ init` for every tree) | Committed (`check_no_worse_than_init.py`); guarantee + Operon-33 contrast in `report.json` finding[4] |
| **DONE** | AD ≈ FD *quality* (cost differs ~1.25–1.3×) | Committed (`report.json` finding[1]: 312,382 trees, median log10(AD/FD)=−0.0) |
| **DONE** | Corpus characterization (current corpus is outer/linear-heavy) | Committed (`sr_problems.py`: 10 Nguyen `constants:[]`; only `feynman/I.6.2` inner) |
| **PROJECT-INTERNAL (uncommitted)** | In-process .so: CO 504→38 ms (13.2×); end-to-end 817→353 ms (2.3×) | Single-run, GPU-internal, not in `results/` |
| **PLANNED** | Criterion as an admit/reject rule (pre-registered τ + anti-gaming) | Not built |
| **PLANNED** | Frozen constructed inner-constant corpus | Not built (only korns/11, korns/12, feynman I.6.2 in-repo) |
| **PLANNED** | 4-arm end-to-end study with a *live* CPU-CO arm | Not built (pipeline is 2-arm p=0/p=1 + palette; no Operon/PySR/scipy in the evolutionary loop) |
| **PLANNED** | SVD numeric-rank check on the constructed corpus; criterion-fires-on-wild / rejects-controls validity check; real-world arm | Not built |

This ledger is the spine of both the design (§4–5) and the risk register (§8). Every "we show / we make runnable" below is conditional design language, not a present-tense result.

---

## 2. Defensible novelty — what we CAN claim vs what we must NOT

### 2a. The two defensible differentiators (CAN claim, hedged)

**Claim.** The contribution is a *bundle*: (i) a fast, in-process, **fp64-honest** GPU LM/NLS kernel with per-tree Jacobians over heterogeneous SR trees, and (ii) a linear-scaling-gap problem-selection criterion as a clean SR-specific instrument.

**Evidence.**
- Honest kernel, narrow existential: among surveyed GPU SR systems (EvoGP, TensorGP, Beagle, GSGP-CUDA, KAN-SR), none documents/claims/measures CO numerical honesty near singularities (N3 survivor). The committed un-gameable gate proves the guarantee on the kernel (`check_no_worse_than_init.py`, FULL-TEXT codebase fact).
- GPU LM-with-per-tree-Jacobians, restricted existential: GPU NLS solvers (Gpufit; JAXFit, arXiv:2208.12187) fit one fixed model batched over datasets; GPU gradient CO for evolved trees *exists* (Kozax, GECCO 2025), so only the batched-LM-with-per-tree-Jacobian-over-heterogeneous-trees slice is open (N2 survivor).
- Criterion: the two-sided Keijzer-scaled inclusion rule is unoccupied to the best of an extensive search (R2/R3 bound it; R13 ceilings it).

**Strongest surviving objection.** After every concession the residual is "first to combine three known pieces (Keijzer reduced model + deterministic two-sided gap as an *inclusion* rule + a fast honest kernel)" — the canonical incremental-benchmark template — and a single un-searched reference can dent the "first" (reviewer #1 major; R13 NO-SOURCE).

**Net recommendation + confidence.** Frame as a bundle; never say "the *only* two differentiators" (drops the honesty guard, which is a third). State first-ness only as "unoccupied to the best of an extensive search, bounded by §3." **Confidence: HIGH** the differentiators are real; **MEDIUM** that they are sufficient to headline as a benchmark (which is exactly why §1b pivots to systems + honesty).

### 2b. What we must NOT claim / cut (each is a FULL-TEXT disproof or a refuted/uncommitted statement)

1. **"Prior benchmarks lack inner constants."** FALSE. arXiv 2412.02126 Table 1 contains them (Korns F7, F11=6.87+11·cos(7.23x³), decay 10e⁻⁰·⁵ˣ, damped pendulum); BRACIS 2015 (de Melo/Fowler/Banzhaf) reports f11/f12 solved by **none** of 6 optimizers (R4, FULL-TEXT). The single most dangerous sentence — never write it.
2. **"We prove / are first to show nonlinear CO matters."** Kommenda 2020 already shows it (Poly-10 R² 0.537→>0.8); prior-art *motivation only* (P3, FULL-TEXT).
3. **"No prior benchmark filters by a linear baseline."** FALSE (SRBench 2.0, R2; Oliveira 2018, R3).
4. **"Our criterion is a novel statistical test."** FALSE — extra-sum-of-squares F-test (R1).
5. **"No GPU SR does CO" / "GPU gradient CO for evolved trees is an open gap."** FALSE (Kozax, N2). Claim only the LM-with-per-tree-Jacobian slice.
6. **"The fp64-honesty guard is a novel mechanism."** FALSE as a mechanism (textbook LM return-best + mixed-precision iterative refinement, N3). Claim only "first GPU-SR system to report/measure this."
7. **"Operon goes ~3× deeper on bloated/high-K trees."** ← **CUT ENTIRELY.** This was asserted in both prior drafts and is **refuted by CuSR's own committed report**: `report.json` finding[2] states the per-problem ordering FLIPS (g100: Operon better on 7 of 17, kernel on 5, tied on 5; for AD it is Operon 6 / AD 7 / tie 4) and the table caption says the pooled headline "HIDES these flips." finding[3]: of 17 problems, only 6 widen g0→g100 while 11 NARROW. The ~3× is a pooled-median artifact dominated by a few high-K problems. (Reviewer #2 FATAL.)
8. **"fp32 LM accepts fake improvements near singularities" as the SOLE characterization of the honesty win.** Incomplete. `report.json` finding[4]: of FD trees that claimed convergence yet were worse in fp64 (961), only 367 (38.2%) were genuinely fast-math-fooled; the other 594 (62%) were worse in the kernel's *own* fp32 objective — a stop-criterion defect since fixed (commit 8b9234b). Do not pin the headline on fast-math.
9. **The 13.2×/2.3× in-process numbers presented with the same weight as committed evidence, and "~0% overhead" as a measured fact.** Both are uncommitted / unbacked (verified: not in `results/`; no `noguard` A/B timing artifact). Downgrade per §6.
10. **"Runnable at scale" / "makes full-model fitting cheap at corpus scale" as a present-tense capability and as the kernel's justification.** Refuted by the project's own scale analysis (§8 R-c). Cut the wording *and* re-scope the kernel's value to per-generation CO inside the evolutionary host.
11. **"The SR community recognizes this gap."** Overstated — one PhD thesis (Aldeia 2025, arXiv 2512.01682, a *thesis* not a survey) in the SRBench lineage (P1). Say "noted by one recent thesis as an adjacent open direction."
12. **Tree-Edit-Distance** as a recovery metric (R12; §5).
13. **The word "first" in headline position** (R13). Carry first-ness only in the hedged "unoccupied to the best of an extensive search."

---

## 3. Prior-art citation + differentiation table

The genus "construct/select benchmark instances a weak baseline fails and a strong method solves, to discriminate" is **established GP/EC methodology** (N1: McDermott 2012 "Tunably Difficult / Varied", FULL-TEXT; Instance Space Analysis, Muñoz/Smith-Miles 2021) — concede the genus, claim only the SR operationalization. Do not import SWAG/HellaSwag as load-bearing (only in a REVIEW-ONLY repo note; N1's McDermott/ISA grounding already carries the point).

| Prior work | Source + access_level | Establishes (do NOT re-claim) | Surviving differentiator | Objection / hazard | Rec + confidence |
|---|---|---|---|---|---|
| Kommenda 2020 (GPEM) | via `arxiv.org/pdf/2505.03977`, FULL-TEXT (GPEM body paywalled, R13) | Nonlinear CO matters (Poly-10 0.537→>0.8); trig frequency = canonical hard CO | None vs the finding — it's our motivation | Cite via 2505.03977, not the paywalled body | Motivation only. **HIGH** |
| Keijzer 2003 | via `arxiv.org/pdf/2505.03977`, FULL-TEXT (body confirmed paywalled, R13) | Linear scaling handles outer/affine almost for free | Our **required CONTROL** + the reduced model in the criterion | Body un-retrieved | Control, not contribution. **HIGH** |
| **arXiv 2412.02126** (closest competitor) | `arxiv.org/abs` + `/html`, FULL-TEXT-via-summarizer, cross-checked | *Method* comparison: 8 CO optimizers × 10 fixed problems; ~half have genuine inner constants; metrics MSE/R²/TED; "no single best optimizer" | Applies **no selection rule, no inner/outer isolation**; "linear scaling" never appears | Disproves "they lack inner constants" — never claim it | Differentiate on the criterion only; avoid its TED. **HIGH** |
| **de Melo, Fowler & Banzhaf 2015** (BRACIS, DOI 10.1109/BRACIS.2015.55) — **NOT "Kommenda 2015"** | `cs.mun.ca/~banzhaf/.../BRACIS2015_Symbolic.pdf`, FULL-TEXT | 6 optimizers on Korns; f11/f12 solved by none | Same: method benchmark, no selection rule | **Citation trap (R6):** "Kommenda 2015 BRACIS" is wrong; Kommenda 2013 GECCO is only its ref [13] | Cite as de Melo 2015; never "Kommenda 2015". **HIGH** |
| SRBench (La Cava 2021) + SRBench 2.0 (2505.03977) | `arxiv.org/pdf/2107.14351` FULL-TEXT; `arxiv.org/html/2505.03977v1` FULL-TEXT | 252 problems (130 ground-truth + 122 PMLB black-box); 14 SR + 7 ML baselines; Def 4.1 + R²>0.999; one-sided linear-baseline exclude-trivial filter | Adopt Def 4.1 + R²>0.999 **verbatim**; differentiate: two-sided **scaled** include vs one-sided raw-feature exclude (R2) | **R7:** reviewers will demand a black-box/real-world arm + Operon CPU LM run **end-to-end** (the CO ceiling) | Adopt metrics; state the partition; treat ground-truth-only as a scoped limitation. **HIGH** (facts), **MEDIUM** (reviewer acceptance) |
| SRSD (Matsubara 2022/2024) | `arxiv.org/abs/2206.10540` FULL-TEXT; `arxiv.org/html/2401.00282` **REVIEW-ONLY** | Curating to isolate one capability is accepted practice; R²/solution-rate are brittle binary metrics → graded NED-style recovery | Our analog isolates inner-CO-necessity — same accepted move (P2) | Attribute "extrapolation R²" to the general generalization literature, NOT SRSD | Cite as isolation precedent + graded-recovery motivation. **MEDIUM** (REVIEW-ONLY) |
| VarPro / separable NLS in SR (Kammerer/Kronberger/Kommenda 2022) | `ar5iv.labs.arxiv.org/html/2209.09675` FULL-TEXT | Linear/nonlinear separation + inner-constant optimization are known method concepts | Readable SR-VarPro uses monotonic {id,log,exp,sqrt,inverse}, **no trig**, no multimodal/frequency regime, no local-optima/init discussion | Downgrade "Operon folds LS into LM" to REVIEW-ONLY (primary texts paywalled; joint-LM-vs-post-hoc tension) | Cite for differentiation; scope the gap to the readable record. **MEDIUM** |
| Kozax (de Wolff 2025, GECCO) | arXiv 2502.03047 (per N2) | GPU first-order gradient CO per-individual over evolved trees | Ours is **LM/NLS with explicit per-tree Jacobians**, not first-order GD | Refutes "no GPU SR does CO" — pre-rebut "why not Gpufit?" (heterogeneous trees, not one fixed model) | Claim only the LM-Jacobian slice. **MEDIUM** (restricted existential) |

---

## 4. Benchmark + experiment design

> **Structural decision — two complementary studies (kept verbatim from the design spine, which both reviewer keep_lists endorse):**
> - **Study A — fixed-tree CO ablation (DONE, carries the ISOLATION claim).** Operon LMOptimizer (fp64) vs the GPU kernel, CO vs no-CO, on **byte-identical pre-CO trees**. No search confound → the clean place inner-constant value is isolated. *This is the committed, adversarially-reviewed asset and it carries the "isolates the inner-CO regime" claim that matches the title.*
> - **Study B — evolutionary 4-arm run (PLANNED, carries END-TO-END value).** Answers "does CO help SR end-to-end" and *necessarily bundles search dynamics* (CO changes fitness → selection → which trees are discovered). Linear scaling removes the outer-coefficient confound but **not** the search-dynamics confound — so its gap is "value of *running* CO," not clean inner-constant isolation.

The reviewer-corrected figure priority (§6) follows directly: the **committed, clean** Study A leads on isolation; the **planned, confounded** Study B is the end-to-end demonstration, explicitly labeled.

### 4.1 The inclusion criterion — PLANNED build (precise recipe)

**Claim.** Build a selection rule (it does not exist today): include a candidate problem iff a Keijzer-scaled reduced model fits poorly and full nonlinear CO fits well.
- **Reduced model** (Keijzer 2003): OLS fit `y ≈ a + b·f(x)` over a fixed reference structure → `R²_LS`.
- **Full model**: nonlinear CO over all constants of the *known true minimal skeleton* → `R²_full`.
- **Rule**: include iff `R²_LS < τ_low` **AND** `R²_full > τ_high`. Set `τ_high = 0.999` to match SRBench's accuracy-solution threshold verbatim (D4, FULL-TEXT). **Pre-register `τ_low`** (e.g. 0.5–0.8) before seeing the corpus.
- **Anti-gaming (load-bearing):** compute the high-R² side against the **known true minimal skeleton** (not a flexible fit) AND **require recovery of the inner constant** (`|ĉ−c|/|c|` small), so included problems provably need nonlinear CO and the rule cannot be gamed by free-parameter overfit.
- **Validity check (must run):** show the rule FIRES on in-the-wild inner-constant problems (Korns F7/F11/F12, decay, damped pendulum) and is REJECTED by negative controls (Nguyen constant-free, outer-only Feynman) — evidence it is not circular or hand-tuned.

**Evidence.** `τ_high=0.999` is SRBench canon (D4, `PMC11074949`, FULL-TEXT); two-sided framing mandated by R2/R3 (FULL-TEXT); Keijzer reduced model is P3 (FULL-TEXT). Anti-gaming attribution: graded-recovery / brittle-binary critique from SRSD (R8, REVIEW-ONLY); SRBench already pairs accuracy R² with a separate symbolic-recovery check (R8, precedent).

**Strongest surviving objection.** The principle is the textbook extra-sum-of-squares F-test (R1); noise-free SR drives `RSS_full→0`, sending F→∞ and degenerating the test, so this is a *deterministic R²-threshold analogue*, not a significance test. Defensible novelty is only the SR-construction application.

**Net recommendation + confidence.** Build it; pre-register τ; frame as a clean operationalization, never a discovery. **Confidence: HIGH** the recipe is correct and SRBench-aligned; **the build is PLANNED, not done.**

### 4.2 Curate-vs-construct

**Claim.** You **must construct** the inner-CO-necessary regime; you cannot curate it. Keep a small curated anchor as the criterion's validity check.

**Evidence.** Verbatim DSO `benchmarks.csv` audit (R5, `arxiv.org/pdf/2211.10873`, FULL-TEXT): Nguyen **0**, Korns **3** {7,11,12}, Keijzer **0** free, Vladislavleva ~1 (up to ~3–4 with shift constants), Livermore **0** (L22 is a degree-8 polynomial — corrects the original "L22"), Jin **0** free, Feynman/SRSD **~0** (physical input variables, SI-fixed constants). Total: **~5 strict, ~10 with quadratic-shift constants** out of 150+. Cross-confirmed in `sr_problems.py`: 10 Nguyen `constants:[]`, six of seven Feynman a single outer `c0`, only `feynman/I.6.2` a genuine inner constant.

**Strongest surviving objection.** A thin or hand-picked corpus is a direct reviewer-exposure point ("you constructed problems your method is good at"). Mitigations: mechanical, pre-registered construction (canonical skeletons × difficulty grid) + the curated-anchor validity check.

**Net recommendation + confidence.** Two-part corpus: curated anchor (validity) + larger constructed body (the instrument). Report the rubric-dependent count honestly (~5 strict / ~10 with shifts). **Confidence: HIGH** (audit FULL-TEXT + repo-confirmed).

### 4.3 Difficulty axes + the SVD rank check

**Claim.** Grade constructed problems on three axes; run an SVD numeric-rank check.
- **Conditioning** — well-conditioned operator set incl. exp/trig/log. Kronberger 2023 (`arxiv.org/html/2209.00942`, FULL-TEXT): {+,*,/} alone is far more rank-deficient. Scope (R11): for the *constructed targets* the inner constants are genuine DOF by construction; the operator-set benefit is about keeping *evolved search trees* better-conditioned.
- **Frequency** — the canonical multimodal axis. Sweep low (near-monotonic) → aliasing; restrict ranges to avoid aliasing frequency into noise (e.g. `cos(c·x³)` on [−5,5]; `sr_problems.py` already documents the [−5,5] restriction for Korns).
- **Sensitivity** — `∂loss/∂c` near the true minimum (flat valley ⇒ many near-equivalent constants).

**SVD rank check.** On selected constructed problems, SVD the NLS Jacobian and check numeric rank == parameter count. This is **standalone evidence** that the constructed inner constants are genuine degrees of freedom.

**Strongest surviving objection.** SVD rank==param-count is strong *evidence*, not proof — it is data/sampling/threshold dependent. Report the sampling and singular-value threshold; say "evidence," not "PROVE."

**Net recommendation + confidence.** Run all three axes; run the SVD check (PLANNED — verified not yet run; no rank-vs-param artifact in `results/`). **Critically: ground the SVD check on its own merits, NOT as a rebuttal to an "Operon-3×-deeper" gap — the project's own committed data shows no robust depth gap (§2b item 7).** The "genuine DOF, not bloat" point stands alone. **Confidence: HIGH** on the method (FULL-TEXT); the run is PLANNED.

### 4.4 The 4-arm end-to-end study (PLANNED — the largest gap)

**Claim.** Build the four arms — **no-CO < sparse/periodic-CO < CPU-CO-every-gen < GPU-CO** — under **both** fixed-generation and fixed-wallclock, treated as complementary protocols.

**Verified gap.** The current pipeline is **2-arm** (`co_probability` p=0 vs p=1) plus a palette-isolation arm (`closed_loop_constructed.py`); **there is no CPU-CO arm (Operon/PySR/scipy) in the evolutionary loop** (grep of non-test `e2_demonstrator` runners finds none), and `run_pilot.py` is documented "NOT auto-run." The 4-arm ladder must be built before any end-to-end claim.

**Evidence for the design.**
- *Fixed-generation/fixed-evaluations* answers "is CO algorithmically valuable?" hardware-neutrally: CuSR-CPU-LM ≡ CuSR-GPU-LM by construction (same LM steps → same fitted constants), so a positive result cannot be an A100 artifact.
- *Fixed-wallclock* is the only place GPU value can surface: at equal time GPU-CO affords CO every generation while CPU-CO cannot.
- Anchor the two-protocol structure to SRBench's disjunctive "500k evals OR 48h" practice and issue #67 (Cranmer: run both); the field trend toward runtime budgets (2505.03977; 2512.01682) *supports* the wallclock arm (D1 surviving core).

**Two CPU-CO arms, distinctly labeled (R11):**
- **(a) CuSR-CPU-LM** — same LM (Jacobian type, tolerance, iteration cap, **precision policy**) as the GPU kernel → the clean GPU-vs-CPU hardware claim.
- **(b) Operon** — the field's empirical CO ceiling (SRBench top black-box method, p ≤ 6.5e-05; R7, `arxiv.org/pdf/2107.14351`, FULL-TEXT) → an explicitly-labeled **system** comparison.

**Strongest surviving objection.** "Identical by construction" holds ONLY for CuSR-CPU-LM vs CuSR-GPU-LM. The committed Operon comparison is a **3-axis-confounded** system comparison — `report.json` env: Operon 500 iters vs kernel 1000; fp32-fast-math+guard vs fp64; damping-λ vs trust-region (R11, repo-confirmed). The same confound must be controlled on the planned arms or they inherit it. Also "A100 vs one core" is a trivial dismissal unless CPU thread/core count is stated.

**Net recommendation + confidence.** Build the 4 arms with the two labeled CPU-CO roles; match numerics for any hardware claim or report fp32-guard-vs-fp64 as a separately named factor; state cores vs one A100. Also wire PySR (BFGS default / Nelder-Mead via Optim.jl) and add a real-world/black-box arm or a defended scope statement (R7). **Confidence: HIGH** on the design; **the entire study is PLANNED.**

### 4.5 Confound controls

**Claim.** Hold population size, generation count, operator set, and a **seed-matched initial population** identical across all arms; keep linear scaling **ON in every arm**; match numerics for any hardware claim.

**Evidence.** D3 (`arxiv.org/html/2209.00942`, FULL-TEXT). Seed-matching is both the init control and what licenses *paired within-problem statistics* (+ common-random-numbers variance reduction). **Linear scaling on everywhere is load-bearing, not hygiene:** it makes the no-CO arm the criterion's linear-scaling-only floor and the GPU-CO arm the LS+inner-CO ceiling, so the measured gap is attributable to inner CO specifically.

**Strongest surviving objection.** In the *evolutionary* setting, linear-scaling-on removes the outer-coefficient confound but the no-CO→CO gap still bundles search-trajectory effects (D2 OVERSTATED). This is exactly why Study A (fixed-tree) carries clean isolation and Study B carries end-to-end value.

**Net recommendation + confidence.** Apply all controls; label Study B's gain "end-to-end CO value (search + constants combined)" and point to Study A for the decomposed inner-constant contribution. **Confidence: HIGH** (D3, FULL-TEXT).

### 4.6 Seeds and statistics

**Claim.** Make the per-problem, seed-matched **paired** comparison the *primary* statistical object; treat the across-problem aggregate as supporting (underpowered by construction).

**Evidence (D4, FULL-TEXT, SRBench-aligned):**
- Within a problem, across seed-matched arms → PAIRED → **Wilcoxon signed-rank**, NOT Mann-Whitney (Mann-Whitney is for *independent* runs).
- Across problems → pairwise Wilcoxon signed-rank + **Holm** (aeon/Benavoli-2016 modern CD-diagram default, avoids Nemenyi mean-ranks composition-dependence) over Friedman+Nemenyi. SRBench's norm is Wilcoxon + Bonferroni (Demsar); Holm dominates Bonferroni-Dunn — a defensible upgrade. Cite **Benavoli et al. 2016**, not just Demsar 2006.
- Adopt SRBench **Definition 4.1** (symbolic solution) + **R²>0.999** verbatim.
- For the honesty guard: per-tree **paired** comparison (guarded vs unguarded honest-fp64 loss) via paired Wilcoxon / win-loss sign test — the correct object for a "delivered ≤ init for every tree" guarantee.
- Sizing: **problems are the power lever** — expand the problem set wherever the criterion admits more instances. Derive the effect size from the project's own pilots, not the literature. SRBench's 10→30 trials/dataset is a convention floor, justified by "statistical significance," not a power analysis.

**Strongest surviving objection.** The constructed corpus is **small by construction**, so the across-problem omnibus is **underpowered with a power ceiling**: with few qualifying problems, irreducible across-instance variance caps class-level power and *adding seeds cannot lift it* (seeds only shrink within-instance variance). This is why citing Campelo & Takahashi to size *seeds* by power is wrong — Campelo sizes *problems* by power, *seeds* by accuracy (D6 OVERSTATED).

**Net recommendation + confidence.** Lead with per-problem paired Wilcoxon + fixed-generation R²/recovery + the fixed-wallclock efficiency curve; report the aggregate as supporting and *explicitly acknowledge the power ceiling*. Do NOT cite Campelo for seed-power sizing. Note the pilot effect sizes themselves are PLANNED (`run_pilot.py` not auto-run). **Confidence: HIGH** on the machinery and the caveat.

### 4.7 Ablations

**Claim.** Run ablations **one-factor-at-a-time (OFAT)** from a fixed baseline, not a full-factorial grid.

**Evidence (D5 surviving core).** Keep: CO frequency (every-gen / periodic / none), CO budget per call (1/5/25 LM steps), Lamarckian-vs-Baldwinian writeback, with/without linear scaling. **Drop FD-vs-AD as an SR-value axis** — CuSR's own committed measurement says AD ≈ FD *quality* (`report.json` finding[1]: 312,382 trees, median log10(AD/FD)=−0.0; differs only ~1.25–1.3× in cost). FD/AD belongs in the kernel micro-benchmark.

**Strongest surviving objection.** A 72-cell full factorial carrying a null axis is the opposite of "airtight" — it dilutes signal; credible SR work uses OFAT (D5 OVERSTATED). The eval-budget-charging convention (charge CO inner iterations against a fixed-eval budget) is an honest *control*, not an established norm — do not present it as one.

**Net recommendation + confidence.** OFAT from a fixed baseline; FD/AD only in the micro-benchmark. Use the eval-budget charge as the honest control for "does CO help at all," remembering the GPU is invisible under it (same LM steps tie) and shows value only under wallclock. **Confidence: HIGH** (grounded in CuSR's own AD≈FD measurement).

---

## 5. Metrics

**Claim.** Report a triad — **(a)** held-out/extrapolation **R²** + SRBench accuracy-solution (R²>0.999); **(b)** SRBench **symbolic-solution rate** (form); **(c)** graded per-inner-constant **`|ĉ−c|/|c|`** (value) — plus **R²-after-linear-scaling** as the no-CO floor. **Do NOT adopt Tree-Edit-Distance.**

**Evidence per metric.**
- Held-out R² + symbolic recovery; report training error only as a diagnostic train-test gap, never as the metric of record (D2 surviving core — "never report training MSE" was too absolute and is corrected to "not the metric of record"). The held-out/extrapolation remedy is from the general SR generalization literature, NOT SRSD (R8).
- Symbolic-solution rate aligns with the field (R9, `assess_symbolic_model.py`, FULL-TEXT, byte-verified): SRBench's check (round to 3 decimals, then diff/ratio `is_constant`, gated R²>0.5) forgives a wrong *outer* additive-or-multiplicative constant but **breaks on a wrong inner constant** (sin(7.2x) vs sin(7.23x) leaves x in both difference and ratio). It quotients out essentially the regime the criterion discards and penalizes the regime it isolates — alignment, at the model-scoring level (criterion operates at problem-selection).
- R²-after-linear-scaling is the no-CO floor and the left half of the criterion (§4.1).
- Anti-gaming (R8, `arxiv.org/html/2401.00282`, **REVIEW-ONLY**): raw in-domain R²/MSE is distrusted (SRSD Table 7: R² PCC=0.00466 vs NED PCC=−0.416). For *inclusion*, compute the high-R² side against the known true minimal skeleton AND require inner-constant recovery; keep inclusion-use distinct from scoring-use.

**Strongest surviving objection + the one UNCERTAIN metric.** The graded per-constant `|ĉ−c|/|c|` has precedent **only in PDE-discovery** (E₂/E∞, Weak-PDE-Net), not SR-native — R12 is the only UNCERTAIN claim in the ledger (REVIEW-ONLY, `arxiv.org/pdf/2408.11515`); cite it as an *adaptation*, not existing SR practice. And the SymPy recovery metric is brittle, *under-reporting* via false negatives on equivalent forms (R9: SymbolicRegression.jl 26.7%→44.7% after curated acceptable-form lists). Pre-registering acceptable-form lists for the constructed forms is a plausible *design choice/aspiration*, **not a demonstrated result** — present it as such.

**Net recommendation + confidence.** Adopt the triad + R²-after-linear-scaling; **reject TED** (R12: the competitor's distrusted, syntactic, size-sensitive metric). Importantly, the *positive* recommendation (SRBench symbolic-rate + graded per-constant error) rests on R8 and R9 **independently of R12** — so even if R12 (UNCERTAIN) falls, the metric choice survives; R12 only carries the "don't adopt TED" negative. **Confidence: HIGH** on held-out R² + symbolic-rate (FULL-TEXT) and on rejecting TED's positive replacement; **MEDIUM/hedged** on the graded per-constant metric (R12 UNCERTAIN, REVIEW-ONLY) and on the pre-registration benefit (aspiration).

---

## 6. Paper narrative

### 6a. Headline + story arc

**Headline:** the systems + honesty sentence in §1a. **Never** lead with raw speed (EvoGP owns that and claims only quality parity — F1 survivor). **Never** put "first" in the headline; carry first-ness only as "unoccupied to the best of an extensive search."

**Arc.** (1) Motivation: nonlinear CO matters (Kommenda 2020, *prior art*) — but where, and how much beyond free outer coefficients (Keijzer 2003)? (2) Gap: existing benchmarks *contain* inner constants (2412.02126) but none *isolates* the regime where inner CO is *necessary*. (3) Method: the honest fast kernel (DONE) + the linear-scaling-gap selection criterion (the instrument, to be built). (4) Result, DONE-first: Study A shows, on byte-identical trees, the kernel matches Operon in honest fp64 with a guarantee Operon lacks; Study B (planned) measures end-to-end value. (5) Honesty + alignment: the `delivered ≤ init` guarantee, and that SRBench's own recovery metric already quotients out the outer regime the criterion discards.

### 6b. Figure plan (figure priority CORRECTED per reviewer #1 major)

**Figure 1 — the ISOLATION figure = Study A (DONE, committed, clean).** Per-tree CO vs no-CO on byte-identical pre-CO trees, kernel vs Operon, neutral fp64. This is the committed, adversarially-reviewed, no-search-confound asset, so it — not an unrun evolutionary panel — carries the "isolate the inner-CO regime" claim that matches the title. (The prior narrative draft inverted this, making the confounded evolutionary panel the headline; corrected.)

**Figure 2 — the HONESTY figure (DONE, committed, currently UNDER-claimed).** The un-gameable guarantee: kernel **0 trees worse-than-start (guaranteed)** vs **Operon's 33** (`report.json` finding[4]) — apples-to-apples, committed, reproducible. Demote the dramatic 854× / 1.5e16× ratios to "worst observed *pre-guard*" with the §2b-item-8 caveat (partly fast-math near singularities, partly — the majority of FD cases — a since-fixed stop-criterion defect). This swap fixes the inflation both reviewers flagged: over-claiming an uncommitted ratio while under-claiming a committed guarantee.

**Figure 3 — the END-TO-END figure = Study B (PLANNED).** Per-problem paired no-CO-with-linear-scaling floor vs full-CO ceiling, fixed-generation, test R² + recovery; each problem one paired point (Wilcoxon). **Label its gap honestly as "end-to-end CO value (includes search-trajectory effects)," NOT "attributable to inner CO specifically"** (D2). Defuse R8 in construction: high-R² side against the known true minimal skeleton + require inner-constant recovery.

**Figure 4 — fixed-wallclock efficiency (PLANNED, second-tier, NAME the baseline).** R²/recovery vs wallclock; GPU-CO reaches the CPU-every-generation ceiling at sparse-CO cost. Never the headline (philosophy: not perf-chasing). Name the CPU baseline; match numerics or label a system comparison; state cores vs one A100.

**Alignment paragraph (not a figure).** R9: "we align with SRBench at a different level (problem-selection vs model-scoring)."

### 6c. The IF-novel / IF-not pivot (treat the second branch as load-bearing)

Because R1/R2/R3 all *survive at FULL-TEXT* attacking criterion-novelty, a reviewer collapsing the criterion into prior art is the **expected** case. Build the paper so the pivot costs at most a re-titling, not a re-experiment — the same Study A + the same planned 4-arm study serve both framings.
- **IF the criterion is accepted as a novel operationalization:** benchmark = headline, kernel = enabling infrastructure. (Venue language rewards task/selection design — P4 survivor.)
- **IF judged not novel (the framing §1b already adopts):** systems + honesty is the headline; the corpus is the demonstration vehicle. The honesty guard is the most pivot-portable asset (converts "faster" → "more correct"), but it is *complementary*, not the most venue-portable novelty — the venue rewards the *selection design*, i.e. the criterion.

### 6d. What to CUT

| Cut | Why | Rec + confidence |
|---|---|---|
| **"Operon ~3× deeper on bloated trees"** | Refuted by CuSR's own `report.json` finding[2] (ordering flips; pooled headline "HIDES these flips") | Cut entirely; re-ground SVD check as standalone. **HIGH** |
| **"fast-math near singularities" as the sole honesty story** | finding[4]: 62% of FD dishonesty was a since-fixed stop-criterion bug | Split the two causes. **HIGH** |
| **13.2×/2.3× presented as committed; "~0% overhead" as measured** | Not in `results/`; no `noguard` A/B timing artifact | Tag project-internal single-run; soften overhead to architectural expectation. **HIGH** |
| **"runnable at scale" / "cheap at corpus scale"** | A ~15–30-problem corpus runs fine on CPU (see §8 R-c) | Re-scope to per-generation CO in the host. **HIGH** |
| **"first" in headline; present-tense "we show"** | R13; no e2e results exist | Hedged "unoccupied to the best of an extensive search"; conditional tense. **HIGH** |
| **"the only two differentiators"** | Drops the honesty guard (a third); F1 OVERSTATED | Frame as a bundle. **HIGH** |
| **"the SR community recognizes this gap"** | One thesis (Aldeia 2025), not a survey | "one recent thesis, adjacent open direction." **HIGH** |
| **"prior benchmarks lack inner constants"** | FALSE (R4) | Delete on sight. **HIGH** |
| **"novel statistical test" / "novel honesty mechanism" / "no GPU SR does CO"** | R1 / N3 / N2 | Replace with the narrow survivors. **HIGH** (refutations) |
| **Full-factorial "airtight" ablation; FD-vs-AD as SR-value axis** | D5; AD≈FD quality | OFAT; FD/AD → micro-benchmark. **HIGH** |
| **Tree-Edit-Distance** | R12 (competitor's distrusted metric) | SRBench symbolic-rate + graded per-constant error. **MEDIUM** (anti-TED is R12-UNCERTAIN; replacement is R8/R9-HIGH) |
| **SWAG/HellaSwag as load-bearing** | Only in a REVIEW-ONLY repo note | Use McDermott/ISA (FULL-TEXT) for the "known genus" point. **MEDIUM** |

---

## 7. Top reviewer objections + how to preempt each

1. **"The headline benchmark and its result don't exist yet."** (Reviewer #1 FATAL — verified: no constructed gap-selected corpus, no committed no-CO-vs-CO recovery table; `run_pilot.py` not auto-run.) *Preempt:* don't claim them. The §1c ledger makes DONE-vs-PLANNED explicit; the paper leads with the *committed* Study A + honesty guarantee, and presents the criterion + 4-arm study as the designed follow-up. A criterion-led benchmark framing is earned only after §4.1/§4.4 are built and run.
2. **"The 4-arm ladder isn't implemented; no CPU-CO arm in the loop."** (Reviewer #1 FATAL — verified.) *Preempt:* state it as the primary planned build (§4.4); do not assert any fixed-wallclock conclusion as achieved.
3. **"The one defensible novelty (the criterion) isn't built as a selection rule."** (Reviewer #1 FATAL — verified: `taxonomy.py` is a positional upper bound; `judge.py` is scoring leniency.) *Preempt:* §1b states this bluntly and pivots to systems + honesty; §4.1 specifies the rule to build with pre-registered τ + anti-gaming + validity check.
4. **"Incremental: first to combine three known pieces."** (Reviewer #1 major.) *Preempt:* concede the genus (N1: McDermott/ISA); claim only the SR operationalization; lead with the honesty/correctness contribution, which is not a recombination.
5. **"Why need a GPU CO kernel for a corpus this small?"** (Reviewer #1 major.) *Preempt:* the kernel's value is **per-generation CO inside the evolutionary host** (many generations × many individuals), not corpus construction; corpus-scale is explicitly *not* the bottleneck it solves (§8 R-c). Drop "runnable at scale."
6. **"Your headline figure can't attribute its gain to inner CO."** (Reviewer #1 major.) *Preempt:* the figure-priority correction (§6b) — Study A (clean) leads on isolation; Study B (confounded) is labeled end-to-end value.
7. **"Ground-truth-only; no real-world arm, no PySR, only fixed-tree Operon."** (Reviewer #1 major; R7 FULL-TEXT.) *Preempt:* add a real-world/black-box arm and PySR, or give a defended scope statement; run Operon end-to-end (not only fixed-tree).
8. **"The recovery metric is brittle/unprecedented."** (Reviewer #1 major.) *Preempt:* SRBench symbolic-rate (byte-verified, aligns with us) as primary form metric; graded per-constant error hedged as a PDE-discovery adaptation; pre-registered acceptable-form lists presented as a design choice, not a demonstrated result.
9. **"Operon-3×-deeper claim contradicts your own report."** (Reviewer #2 FATAL.) *Preempt:* cut the claim (§2b item 7); the SVD check stands alone.
10. **"The honesty win is over-attributed to fast-math; the speed numbers are uncommitted."** (Reviewer #2 major.) *Preempt:* split the fp32 causes (§2b item 8); tag the in-process numbers project-internal; for "context paid once," cite the *committed* setup cost ~359–488 ms across batch size (committed in `results/profile_fusedfd_a100__20260616`, `setup_ms`), which is reproducible.

---

## 8. Honest risk register (what is weak / could get us rejected)

- **R-a (FATAL today).** The paper's headline empirical object — the constructed gap-selected corpus and the no-CO-vs-CO end-to-end recovery table — **does not exist**. As of today the work is closer to the systems + correctness framing than the novel-criterion-benchmark framing. *Mitigation:* the §1c ledger; lead with committed Study A + the honesty guarantee; earn the criterion framing only by building+running §4.1/§4.4. **Verified, not estimated.**
- **R-b (FATAL today).** The 4-arm study and the fixed-wallclock GPU-vs-CPU figure have **no implementation** (2-arm pipeline, no CPU-CO arm in the loop). *Mitigation:* §4.4 is the primary build; no wallclock conclusion is asserted as achieved.
- **R-c (major, internal tension).** "The kernel makes the benchmark runnable at scale" vs "the benchmark is small by construction." A ~15–30-problem corpus runs fine on CPU; under a fixed-eval budget the GPU is invisible. *Mitigation:* re-scope — the kernel earns its keep on the **per-generation CO workload inside the host**, not on corpus construction; say so plainly and drop "runnable at scale."
- **R-d (major, incrementality).** After every concession (R1/R2/R3/R4), the criterion is a recombination of known pieces — a fair weak-reject. *Mitigation:* systems + honesty headline; the honesty guarantee is the least-contestable, fully-committed asset.
- **R-e (major, baseline completeness).** Ground-truth-only, no real-world arm, no PySR, fixed-tree-only Operon, vs SRBench's 130+122 split and Operon-end-to-end norm (R7). *Mitigation:* add the arms or defend the scope explicitly.
- **R-f (major, novelty ceiling).** First-ness rests on absence-of-evidence from a finite, US-biased search with paywalled/failed-extraction sources (R13, NO-SOURCE). *Mitigation:* never write "first"; "unoccupied to the best of an extensive search," bounded by §3.
- **R-g (major, metric trust).** The discriminating signal is inner-constant recovery, but the SymPy form-metric is brittle (under-reports), the value-metric is a cross-domain adaptation (R12 UNCERTAIN, REVIEW-ONLY), and the acceptable-form-list mitigation is unbuilt. *Mitigation:* SRBench symbolic-rate as primary; graded metric hedged; lists presented as aspiration.
- **R-h (minor → resolved by cutting).** The "Operon-3×-deeper" claim is refuted by the project's own committed report; the SVD defense was built on it. *Mitigation:* cut the claim; SVD stands alone (§4.3).
- **R-i (minor, evidence status).** "~0% overhead" has no committed A/B timing artifact (the `noguard` build exists; the measurement does not). *Mitigation:* state as architectural expectation or run+commit the A/B.

**Net.** The *design* is sound and SRBench-aligned (both reviewer keep_lists endorse: two complementary studies, Study A as clean isolation, paired Wilcoxon / Holm / Def 4.1 / reject TED / OFAT / FD-AD-out-of-SR-value). The *contribution as of today is promissory*: the path to acceptance is to build and run the criterion + the constructed corpus + a genuine 4-arm study (with a live CPU-CO arm and a real-world arm), promote the committed Study A to carry the inner-CO isolation claim, and lead the honesty story with the committed `delivered ≤ init` guarantee (0 vs Operon's 33) rather than the uncommitted fake-improvement ratios. Confidence ceiling for the whole narrative is **MEDIUM**, correctly bounded by the absence-of-evidence novelty (R13) and the promissory status of the headline experiment.

Key paths (absolute): `/home/weish/hao/CuSR/results/operon_baseline__20260617/report.json` (Study A, committed; findings 1/2/4 are load-bearing); `/home/weish/hao/CuSR/cusr/kernel/tests/fp32_honesty_regression/check_no_worse_than_init.py` (committed un-gameable guarantee); `/home/weish/hao/CuSR/cusr/demonstrator/taxonomy.py` (positional upper bound — NOT the selection rule); `/home/weish/hao/CuSR/cusr/demonstrator/judge.py` (linear scaling as scoring leniency only); `/home/weish/hao/CuSR/experiments/e2_demonstrator/closed_loop_constructed.py` + `run_pilot.py` (2-arm, "NOT auto-run" — the 4-arm gap); `/home/weish/hao/CuSR/cusr/benchmark/workload/sr_problems.py` (corpus is outer-heavy; korns/11, korns/12, feynman/I.6.2 the only inner constants).
