# operon_baseline__20260617

**Operon `LMOptimizer` (CPU) vs the GPU kernel (FD & AD) — constant optimization on
byte-identical pre-CO trees, all 17 problems.** The deferred "baseline" from the Operon
corpus work, now run as a three-way comparison after the AD-Jacobian kernel landed.

- **Read `report.html`** for the full bilingual (中/EN) writeup (theme-adaptive).
- Pipeline: `lib.py` (neutral fp64 arbiter) · `run_operon.py` / `run_kernel.py` (per-cell
  engines) · `driver.py` (orchestration) · `analyze.py` (all metrics) · `build_report.py`
  (report.json, no hand-transcription) · `plots.py` · `review_workflow.js` (adversarial review).
- `probes/` — the gate probes that had to pass *before* the harness was trusted:
  `probe_lm_gates.py` (Probe A same-start/objective/K + Probe B regenerated==committed bytes),
  `probe_lm_cost_semantics.py`, `probe_lm_keeps_best.py`.
- `data/` — per-cell `<prob>__{operon,fd,ad}.json` (per-tree losses/status) +
  `report_aggregates.json`. `plots/` — 4 figures.

**What makes it honest (the design):**
- All three engines start from the SAME harvested `c_init` (Probe A), optimize the SAME K
  coefficients with the SAME raw 0.5·Σ(f−y)² objective (no linear scaling), and their
  returned coefficients are re-scored by ONE neutral fp64 evaluator (`lib.half_sse`) — so
  the comparison is independent of any engine's internal cost convention.
- The Operon trees are *regenerated* deterministically and proven byte-identical to the
  committed corpus pop.bin (Probe B) — we are optimizing the exact same trees the kernel sees.
- **AD-vs-FD is controlled** (same fp32 LM, only the Jacobian differs) → attributable.
  **kernel-vs-Operon is an external yardstick** (fp32 fast-math + simple λ-damping vs fp64
  trust-region) → bounds the achievable, not attributable to one factor.
- Failures are never dropped from rates; quality is compared only where both engines
  genuinely improved (>1%); the "converged-but-worsened" (fp32) rate is reported separately.

**Caveat up front:** the kernel runs here on the *Operon distribution* (every leaf a
coefficient, ~2.8× denser than evogp) — a STRESS test, not the kernel's design workload.

Regenerate: see `report.json` → `reproduce`.
