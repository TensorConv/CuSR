# Kernel tech-debt ledger

Known limitations we accepted deliberately, with the trigger for revisiting each.

---

## TD-1 — reverse-AD `revad_safe_mul` silently returns 0 at removable singularities

**Where:** `cusr/kernel/revad_interp.cuh` — `revad_safe_mul(a,b) = (a==0||b==0)?0:a*b`, used at the
3 backward-pass adjoint-push sites of `eval_tree_vjp_d`.

**What:** The 0-annihilating multiply (added 2026-06-22 to kill `0·∞ / inf·0 / 0·nan` NaN
contamination) also silently returns **0** at *removable* singularities — where `0·∞` mathematically
cancels to a finite **nonzero** limit. Canonical case (Codex P2): `pow(sqrt(c0), x0)` at `c0=0, x0=2`
is `(√c0)²=c0`, true right-derivative **1**, but the kernel reports **0**.

**Why we accepted it (not a regression, doesn't occur here):**
- It is a *fundamental* first-order-AD limitation: forward AD, the pre-mask reverse, AND central
  finite-differences all give **NaN** at that point. Only symbolic simplification or one-sided FD
  recovers the 1. The mask trades loud-wrong (NaN) for silent-wrong (0); it did not create the gap.
- A purely-local mask **cannot** distinguish "genuine zero" (true deriv 0, common) from "removable
  nonzero" (Codex, rare) — both are locally `0·∞` (separating needs per-column dependency tracking).
- **0 occurrences in the real workload.** `tests/_probe_corpus_silent.cu` over the full corpora
  (inner-const 27.43M + early-gen 45.41M elements): every one of the 9000+3000 mask-affected entries
  agrees with an in-domain FD oracle (all are genuine zeros); **SILENT-WRONG = 0**.
- For the LM solve, a conservative 0 is *safer* than NaN (one NaN poisons a constant's whole
  Gauss-Newton column; a 0 only under-weights one data point).

**Regression guard:** `tests/_probe_corpus_silent.cu` (host, full corpus). Re-run if the workload,
the operator set, or the constant initialization changes. If it ever reports `SILENT-WRONG > 0`, this
debt has come due.

**Possible fixes (only if the guard fires) — all heavier, with trade-offs:**
- *value-ε regularization* of sqrt/log/inv/pow (floors the value, makes the limit resolve) — but
  changes forward semantics → must re-validate fit quality; ε is arbitrary.
- *one-sided-FD fallback* for entries that hit a `0·(non-finite)` annihilation — adds kernel branch
  divergence and reintroduces fp32 FD inaccuracy.
- *symbolic simplification* (`pow(sqrt(c),2)→c` before differentiating) — a whole symbolic layer.

**Refs:** `experiments/revad_v5/FINDINGS_codex_review_shared_nan.md` (§"Codex P2 follow-up");
probes `tests/_probe_removable.cu` (the synthetic case) and `tests/_probe_corpus_silent.cu` (corpus scan).
