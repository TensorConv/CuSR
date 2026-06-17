# Adversarial review of the baseline (2026-06-17)

A 12-lens adversarial review workflow (`review_workflow.js`: anti-cheat, setup-fairness,
fake-report, per-engine code correctness, statistical validity, fast-math fairness,
devil's-advocate, reproducibility, general/free) → adversarial verification of every
high/critical finding against the raw data → synthesis. 15 agents, ~1.3M tokens.

## Verdict: **SOUND-WITH-FIXES** (now fixed)

The machinery is trustworthy: **every load-bearing number reproduces to the decimal** from
the raw per-cell JSONs; the neutral fp64 arbiter is correct; the one *attributable*
comparison — controlled **AD-vs-FD** — is Simpson-free (AD ≥ FD in 84/85 problem×gen cells;
the "no derivative bug" near-tie holds *within every one of the 17 problems*). No number
was wrong; no conclusion reversed in **direction**.

Notably, the review found the report was **more often too hard on the kernel than
self-flattering** — most overstatements cut *against* the kernel. But it caught real
honesty defects, including a fabricated-provenance phrase in the report's own anti-cheat
finding. All confirmed items are now fixed:

| # | finding | direction | fix |
|---|---|---|---|
| 1 | "kernel comparable to Operon" pools an easy (~0) + hard (0.10–0.45) cluster; per-problem ordering flips | self-serving | per-problem median-ratio table (g0+g100) + ordering breakdown (FD: Op-better 7 / kernel 5 / tie 5; AD: Op 6 / AD 7 / tie 4); claim restated as a magnitude effect from a few high-K problems + one-sided tail columns |
| 2 | "robustness gap widens with bloat" is a 6/17 artifact (11 narrow) | self-penalizing (false generalization) | finding restated to "6 widen / 11 narrow"; `robustness_gap.png` redrawn per-problem; `robustness_gap_widen_narrow` emitted |
| 3 | "verified on the >100× subset" cited a check with no code (in the anti-cheat finding) | fabricated provenance | committed `probes/probe_fp32_lie.py` that recomputes & classifies the subset and asserts the totals; wording → "recomputed by probe" |
| 4 | status==2 "NaN" mislabels λ-blowup stalls (99.8% return finite best-so-far; only 22 each truly NaN) | self-penalizing | new finding F6; analyze tracks `*_nan_true`; relabeled "failed-to-converge (stall/NaN)" |
| 5 | headline led with the +42344 status-conv gap (6.1× the neutral net) | self-serving | leads with the neutral net **+6959** (324082 vs 317123); 46574 kept as a *mechanism* diagnostic with its counterweight (`ad_newfail`) |
| 6 | `driver.py` default mi=500 doesn't reproduce the committed kernel data (1000) | reproducibility | split `OPERON_MAX_ITER=500` / `KERNEL_MAX_ITER=1000`; defaults now reproduce; removed the misleading single-cell "<3% maxiter" comment |
| 7 | quality comparisons silently drop 62k–86k trees; composition undisclosed | self-penalizing (disclosure) | `quality_set_composition` emitted; F7 states the set EXCLUDES one-sided rescues + the drop counts |
| 8 | `analyze.py` docstring "FD: all such are real lies" is false (38%) | code-comment | docstring → 38% (367/961) lies + the remaining 594 worse-in-own-fp32; `*_convworse_ownmetric` tracked |
| nit a | env labeled Operon "fp64" (it is fp32 eval + fp64 solve) | mildly self-serving | caveat softened; "don't over-read the precision gap" |
| nit b | hand-typed env strings (max_iter) not sourced | latent desync | `operon_iters_max` / `operon_near_cap_frac` sourced from data |

## Checked and found CLEAN (verified against raw data)

Controlled AD>FD is not a Simpson artifact (84/85 cells); "no derivative bug" holds within
every problem; median is the correct central statistic (means are inf-poisoned, handled);
tie band symmetric; aggregation faithful (reproduces to the decimal); `loss_start` alignment
exact across all 85 cells (Probe B byte-identical everywhere); fp64-scoring of fast-math fp32
coefficients is fair, not an anti-kernel artifact (the kernel-vs-Operon tail is a genuine
LM/basin gap, not scoring inflation); `lib.py` evaluator mirrors the kernel op set; Operon
worse-than-start (33) counted, never clamped.

Run the review: `Workflow({scriptPath: "results/operon_baseline__20260617/review_workflow.js"})`.
