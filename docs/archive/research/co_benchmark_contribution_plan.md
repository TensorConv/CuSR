# CuSR — 基准贡献的叙事恢复方案 (Benchmark-Contribution Recovery Plan)

**目的 (Purpose).** Answer one question: *can the BENCHMARK/CORPUS be recovered from "instrument"
to a defensible, headline-worthy contribution — and if so, exactly how?* This builds **on**
`co_benchmark_strategy.md` (which pivoted to systems+honesty and, at the time, marked the criterion
"PLANNED/not built"). The decisive update since that doc: **the criterion is now BUILT** —
`experiments/e3_admit_criterion/` (34 tests pass, frozen pre-registration, 21-problem corpus,
Feynman-34→2 audit). This changes the *promissory* objections, not the *novelty ceiling*.

Honesty discipline (carried verbatim): `access_level` reported and never upgraded; never bare
"first"; absence-of-evidence stays absence-of-evidence; every elevation is bounded by named prior art.

> **对齐说明 (2026-06-21).** 贡献陈述以 [`contributions.md`](contributions.md) 为准。本文的"两支柱(kernel + benchmark)+ 由 FORCEFUL EMPIRICAL FINDING 驱动的升格"结论与之**一致**;contributions.md 只是把它**显式化为三条**(系统 kernel / 选题判据 / 实证发现),并把诚实性收为贡献①的特性(本文已记 Kronberger 2022 占轴)。Kozax = **de Vries 2025**(非 de Wolff)。

---

## 0. 一句话结论 (Bottom line up front)

**The benchmark recovers from "instrument" to a *co-equal, headline-eligible secondary* contribution —
earned by an EMPIRICAL FINDING (the Feynman-34 audit now; the 4-arm which-methods-fail study next), NOT
by the criterion's statistical novelty. It cannot be the sole headline.** The fp64-honest in-process GPU
CO kernel stays the co-equal systems pillar. This is the verdict to write, and it does not block the paper.

Why not the sole headline (three FULL-TEXT concessions that all still stand):
1. the underlying statistic is the textbook extra-sum-of-squares / nested-model F-test (R1, FULL-TEXT);
2. one-sided linear-baseline *filtering* of SR benchmarks is already published (SRBench 2.0, FULL-TEXT);
3. capability-isolation-by-construction is an established genus (SRSD dummy-variables; CLEVR; McDermott 2012).

What the "built" status *does* buy: it demolishes the three reviewer-#1 FATAL objections the strategy doc
flagged ("the criterion isn't built as a selection rule"; "the corpus doesn't exist"; "no admit/reject
computation"). Those are now false — `criterion.py::decide()` is the rule, `out/corpus_manifest.json` is
the corpus. So option (a) rises from *promissory secondary* to *real, co-equal-eligible secondary*. It does
**not** rise to *sole headline* — the novelty ceiling is untouched.

---

## 1. 张力的裁决 (Resolving the four-way tension — decided by the verified verdicts)

The task names four candidate positions. The verified niche verdicts already decide each; map directly.

| Candidate | Verified verdict | Disposition |
|---|---|---|
| **(a)** criterion/corpus as a novel SR *operationalization* | niche #1/#2 **NOT refuted**, MEDIUM, bounded narrow-existential | **Secondary, co-equal-eligible, now BUILT.** Real & citable; never sole headline. |
| **(b)** DIAGNOSTIC "which-SR-methods-fail-where" empirical benchmark on the corpus | niche #3 **refuted the inference** "isolation ⇒ headline" (isolation = table-stakes), but its *residual* is the highest-ceiling path: headline status comes from the **force of the empirical finding** | **TOP route to elevation.** Lead here. |
| **(c)** honesty/correctness stress-test corpus | niche #4 **REFUTED.** Kronberger 2022 (2209.00942, FULL-TEXT) owns the CO-conditioning/honesty-near-singularities axis | Corpus stays **instrument** here; not a novel axis. |
| **(d)** keep it a pure instrument; double down on systems+honesty | the honest **floor** | Always-available fallback; the pivot target if (a)/(b) are judged too incremental. |

**The discriminator between "headline" and "instrument" is whether a FORCEFUL EMPIRICAL FINDING exists.**
It does, and the prior strategy doc underweighted it because, at the time, it was unbuilt:

- **The Feynman-34→2 audit IS that finding, and it is DONE (not promissory).** Applying the criterion to
  the field's canonical SR benchmark reveals that benchmark *essentially lacks the inner-CO-necessity
  regime*: the inner constants are overwhelmingly grammar-reachable (±1 subtraction signs, ±0.5 Gaussian
  widths, π phases); the only non-canonical motif in all 34 is −1/3 (Lorentz/Clausius-Mossotti
  polarization, 2 problems). This is a **failure-mode analysis of an existing benchmark** — an explicitly
  welcomed NeurIPS Evaluations & Datasets category (niche #3, evidence point 2, FULL-TEXT).
- **The 4-arm which-methods-fail study is the higher-ceiling finding still to BUILD** (no-CO / sparse-CO /
  CPU-CO / GPU-CO). It earns the "which SR methods fail in the isolated regime" headline if it lands.

**Load-bearing caveat on the Feynman finding (must frame set-independent).** Report the robust statement
that survives *any* defensible reachability boundary — "the Feynman set contains essentially zero inner
constants needing nonlinear CO beyond simple reachable values; at most the single −1/3 motif (2 problems),
and 0 if even small rationals {±1/3,±1/4,…} are deemed reachable" — **never a bare "exactly 2."** The count
swings 0–11 across defensible canonical sets (`run_feynman34.py` table; REPORT.md §5). The set-independent
statement is the headline; the "2" is one point on a transparently-reported sweep.

---

## 2. 叙事选项 (Narrative options, best-first)

### Option 1 — DIAGNOSTIC empirical finding (TOP)
**Framing (one sentence).** *We build a deterministic inner-CO-necessity selection rule, apply it as a
diagnostic lens, and report two empirical findings — (i) the field's canonical SR benchmark (Feynman)
essentially lacks the inner-CO-necessity regime, and (ii) on a corpus constructed to isolate that regime,
[which SR/CO methods fail and by how much] — with the criterion as the instrument that makes the finding
legible, not as the novelty.*

**Evidence.**
- Genre is accepted AND isolation-alone is table-stakes: CLEVR (Johnson 2017, CVPR, REVIEW-ONLY — 403 on PDF,
  framing confirmed across two snippets); SRSD-Feynman (2206.10540, REVIEW-ONLY); bAbI/DERAIL (2012.01365,
  FULL-TEXT). NeurIPS 2026 E&D track *explicitly welcomes* "analyze strengths, limitations, or failure modes
  of existing benchmarks" (FULL-TEXT). → headline must rest on the **finding**, not the lens.
- The Feynman finding is DONE: `experiments/e3_admit_criterion/run_feynman34.py` → `out/feynman34.json`
  (34→2 under the frozen set; 0–11 sweep reported), 34 tests pass, fp64+scipy ground truth, GPU-independent.
- The criterion is BUILT: `criterion.py::decide()` (5 mechanisms, 4 structural anti-cheat detectors that
  fire *without* fitting at the true constants); pre-registration frozen in `prereg.py`.

**Strongest surviving objection.** The diagnostic-isolation genre is crowded (CLEVR/bAbI/SRSD) so the lens
is not the novelty (niche #3, refuted "isolation⇒headline"); and the *which-methods-fail* half is still
PLANNED — only the Feynman-audit half is done. A reviewer can say "you have a failure-mode note on Feynman
+ a built rule, but the marquee method-failure result isn't here yet."

**What to BUILD to earn it (map to assets).**
- *Already done:* Feynman-34 audit (e3) → the "existing benchmark lacks the regime" finding; the 21-problem
  corpus (`out/corpus_manifest.json`); the criterion (`criterion.py`).
- *Build:* the **4-arm end-to-end study** (no-CO / sparse-CO / CPU-CO / GPU-CO) on the constructed corpus —
  the current pipeline is 2-arm with no live CPU-CO arm in the loop (strategy doc §4.4 / R-b). This is the
  marquee finding. Add at least one external SR method (Operon end-to-end, PySR) so "which methods fail"
  is not self-referential. Reuse Study A (fixed-tree, committed) as the clean-isolation supporting panel.

**Confidence.** MEDIUM. The Feynman half is HIGH-confidence DONE; the headline-grade method-failure half is
PLANNED, and the genre's table-stakes nature caps how much the lens alone earns.

---

### Option 2 — Bounded-novel SR operationalization (secondary, now BUILT)
**Framing.** *First SR benchmark CONSTRUCTED by a deterministic two-sided, Keijzer-scaled include-iff rule
(admit iff scaled `y≈a+b·f(x)` fits poorly AND full nonlinear inner CO fits well AND the inner constant is
recovered) to isolate the inner-CO-necessity regime — unoccupied to the best of an extensive search,
bounded by SRBench++ / SRBench 2.0 / 2412.02126 / Oliveira 2018.*

**Evidence.** niche #1/#2 NOT refuted after targeted search across CO-method benchmarks, in-algorithm linear
scaling, instance-space analysis, and difficulty-stratified benchmarks. The rule is built and pre-registered
(e3); the anti-gaming the operationalization needs is real and demonstrated (korns_8 outer-absorption REJECT;
trig-phase linear-span REJECT, found by red-team and fixed at root cause — REPORT.md §6).

**Strongest surviving objection.** Three FULL-TEXT concessions stand: F-test (R1), one-sided linear filtering
(SRBench 2.0), isolation genus (SRSD). The residual is "first to *combine* known pieces into this SR-specific
inclusion rule" — the canonical incremental-benchmark template; one un-indexed GECCO/EuroGP reference can
dent the existential (absence-of-evidence, finite US-indexed search).

**What to BUILD to earn it.** Mostly DONE: the rule, the pre-registration, the corpus, the τ/STRUCT_TOL
sensitivity sweeps, the SVD numeric-rank certificate (`inner_adds_rank` — the identifiability gate is exactly
the SVD check the strategy doc §4.3 asked for). *To add:* the explicit head-to-head **SRBench++ confrontation**
paragraph (§3) and the bounded-language related-work section; a validity table showing the rule fires on
in-the-wild inner-constant anchors and rejects negative controls (already in e3: 9 anchors match, 0/11
controls leak).

**Confidence.** MEDIUM. Real, citable, built — but novelty ceiling fixed by R1/SRBench 2.0/SRSD; sole-headline
is not available.

---

### Option 3 — Honesty/correctness stress-test corpus (demoted; axis is occupied)
**Framing.** *The corpus as a stress-test vehicle for the fp64-honesty guard in the inner-CO regime —
"first GPU in-process CO kernel that runs a Kronberger-style honesty diagnostic as an enforced runtime
guarantee (delivered ≤ init for every tree) at GP scale."*

**Evidence.** The honesty guard is committed (`check_no_worse_than_init.py`; kernel 0 trees worse-than-start
vs Operon 33, `report.json` finding[4]).

**Strongest surviving objection — fatal to "novel axis."** niche #4 **REFUTED**: Kronberger et al. 2022
(2209.00942, FULL-TEXT, IEEE SYNASC) already measures CO conditioning/rank/honesty near singularities via
SVD of the LM Jacobian. The "delivered ≤ init" guarantee is standard trust-region/return-best LM (JAXFit on
GPU is already monotone). So the corpus here is an **instrument/demonstration vehicle**, not a benchmark axis.

**What to BUILD.** Nothing new for novelty — it is the systems pillar's demo. Keep the corpus as the regime
that *exercises* the guard; do not claim the honesty axis as the benchmark's contribution.

**Confidence.** HIGH that this is instrument-only (the refutation rests on a direct primary source).

---

### Option 4 — Pure instrument + systems/honesty headline (FLOOR / fallback)
**Framing.** Exactly the strategy doc's current pivot: systems + fp64-honesty is the headline; the corpus is
the demonstration vehicle. **Evidence/objection/build:** as in `co_benchmark_strategy.md` §1a, §6c-IF-not.
**Confidence.** HIGH it is defensible; it forgoes the recoverable secondary contribution that Options 1–2
now make available, so use it only if a venue/reviewer collapses (a)+(b) as too incremental.

---

## 3. 先前工作 (Prior-art table — adds SRBench++ and CLEVR, which the strategy doc §3 omits)

The strategy doc §3 table predates and omits **SRBench++**, called by niche #1 "the most dangerous new
candidate and the nearest miss." It must be confronted head-on. CLEVR is added as the genre precedent for
Option 1. Rows below are *incremental to* strategy-doc §3 (which already covers Kommenda 2020, Keijzer 2003,
2412.02126, de Melo 2015 BRACIS, SRBench/2.0, SRSD, Kozax, VarPro).

| Paper | access_level | Establishes (do NOT re-claim) | Surviving differentiator | Objection / hazard |
|---|---|---|---|---|
| **SRBench++** (de Franca et al., IEEE TEVC 2024, PMC12321164) | **FULL-TEXT** | (i) linear baseline as a **method**-qualification gate ("disqualified approaches ranked lower than plain linear regression"); (ii) **property-isolation construction tracks**, one being "sensitivity to / avoiding (noisy) local optima" | Their linear gate ranks **methods, not problems**; their "local optima" track = **deceptive-shortcut** resistance via noisy meta-features of a polynomial, **not** inner-constant (frequency/decay) recovery, no `sin(c·x)`, no two-sided scaled linear-gap, no inner-recovery gate | The nearest miss. A hostile reviewer presses "isn't your local-optima isolation just SRBench++?" — answer: method-gate≠problem-selection; deceptive-shortcut≠inner-CO-recovery. Confront in §related-work, do not bury. |
| **Kronberger et al. 2022** "Local Optimization Often is Ill-conditioned…" (2209.00942, IEEE SYNASC) | **FULL-TEXT** | The CO-conditioning / honesty-near-singularities **axis** (SVD rank + condition number of LM Jacobian; rank-deficient/ill-conditioned ⇒ inaccurate NLS steps) | None for the *axis* — it is occupied. Our survivor is only the **GPU in-process enforced-runtime-guarantee instantiation** (systems pillar), not a novel honesty axis | Refutes any "fp64-honesty is an unoccupied benchmark axis." Cite as the axis owner; claim only the GPU-runtime-guarantee instantiation. |
| **CLEVR** (Johnson et al., CVPR 2017) | **REVIEW-ONLY** (PDF 403; framing confirmed via 2 snippets) | The canonical diagnostic-dataset template: isolate one capability, remove confounds, reveal model limitations — headline came from opening a NEW capability axis + catalyzing a subfield, **not** from "isolation" per se or a novel statistic | Genre precedent for Option 1: licenses "diagnostic lens + empirical finding" as headline-eligible, while conceding the lens is established | Confirms isolation is table-stakes (niche #3). Use to justify Option 1's *finding-led* framing; do not claim isolation as our novelty. |
| **DERAIL / diagnostic-split QA** (2012.01365 FULL-TEXT; 2509.22983) | FULL-TEXT | "Each task tests one dimension so failures pinpoint where an algorithm needs improvement" is routine methodology | Corroborates Option 1's genre acceptance | None load-bearing; supporting citation. |

---

## 4. 落地 (How to wire it into the paper — minimal, pivot-cheap)

The same assets serve all viable options, so the pivot costs a re-titling, not a re-experiment (consistent
with strategy doc §6c):

1. **Lead structure = Option 1 over Option 2.** Two co-equal pillars: (A) fp64-honest in-process GPU CO
   kernel [systems]; (B) inner-CO-necessity benchmark whose *value is the empirical finding* [diagnostic].
   Option 2's bounded-novelty language is the *related-work defense* of pillar B, not its headline claim.
2. **Promote the Feynman-34 audit to a named result** (it is DONE), framed set-independent. This is the
   ready-now half of pillar B and the strongest "failure-mode of an existing benchmark" hook.
3. **Build the 4-arm study** (the one real gap) with ≥1 external method (Operon end-to-end / PySR) to make
   "which methods fail" non-self-referential. Reuse committed Study A as the clean-isolation panel and the
   honesty guard (0 vs 33) as pillar A's figure.
4. **Add the SRBench++ confrontation paragraph and CLEVR genre-precedent** to related work; carry first-ness
   only as "unoccupied to the best of an extensive search, bounded by SRBench++/SRBench 2.0/2412.02126/
   Oliveira 2018."
5. **Update the strategy doc's §1c ledger**: the criterion, corpus, and Feynman audit move from PLANNED to
   DONE; reviewer-#1 FATAL objections 1/2/3 and risks R-a/R-b (criterion half) are RESOLVED. The novelty
   ceiling (R-d/R-f) and the 4-arm/real-world gaps (R-b end-to-end, R-e) remain.

---

## 5. 必须避免 (Must-NOT, carried forward)

- Never "prior benchmarks lack inner constants" (FALSE: 2412.02126, BRACIS 2015).
- Never "novel statistical test" (it is the extra-sum-of-squares F-test).
- Never "first to show inner CO matters" (Kommenda 2020 owns it as motivation).
- Never "unoccupied honesty benchmark axis" (Kronberger 2022 owns it).
- Never a bare "exactly 2" Feynman survivors — always the set-independent statement + the 0–11 sweep.
- Never bare "first"; never call the diagnostic-isolation lens itself the novelty (table-stakes genre).

Key paths (absolute): `/home/weish/hao/CuSR/experiments/e3_admit_criterion/criterion.py` (the BUILT rule);
`/home/weish/hao/CuSR/experiments/e3_admit_criterion/run_feynman34.py` + `out/feynman34.json` (the DONE
finding); `/home/weish/hao/CuSR/experiments/e3_admit_criterion/out/corpus_manifest.json` (21-problem corpus);
`/home/weish/hao/CuSR/experiments/e3_admit_criterion/REPORT.md` (full e3 report, §5 set-independence);
`/home/weish/hao/CuSR/results/operon_baseline__20260617/report.json` (Study A + honesty guard, committed);
`/home/weish/hao/CuSR/docs/research/co_benchmark_strategy.md` (the doc this builds on; §1c ledger to update).
