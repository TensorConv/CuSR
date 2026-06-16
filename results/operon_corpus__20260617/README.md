# operon_corpus__20260617

**Operon pre-CO corpus + Operon→pop.bin adapter** — a second, independent authoritative
workload source (and prospective CPU baseline) alongside the evogp corpus, used to
cross-validate the rank-deficiency finding across two GP engines.

- **Read `report.html`** for the full bilingual (中/EN) writeup (theme-adaptive).
- Adapter + harvest pipeline: `cusr/benchmark/workload/{operon_adapter,operon_dump,operon_harvest,pop_io,pop_ref,sr_problems}.py`
  (TDD, 53 tests). Spec: `docs/kernel/OPERON_ADAPTER_SPEC.md`.
- Analysis scripts here: `mechanism_proof.py`, `summarize.py`, `cross_kernel.py`, `plots.py`,
  `build_report.py` (regenerate `report.json`; then `scripts/make_report.py`).
- `data/` — `summary.json`, `mechanism_proof.json`, `cross_kernel.json`, `drift.csv`.
- `plots/` — `fail_vs_gen.png`, `fail_vs_nodes.png`, `density_vs_gen.png`, `k0_fraction.png`.

**Corpus** (harvested by `operon_harvest.py`; snapshots under
`data/workload/snapshots/operon_*`, `.bin` gitignored, **manifests committed**):
17 problems × noise{0,1%} × cap{32,64} × seed{0,1,2}, dumped at gens {0,1,2,4,8,16,32,64,100}
→ **204 cells / 1836 snapshots / 7.34M trees @ pop=4000, N=1000. 0 dropped, 0 K-over.**
Operon GP runs with **inline CO OFF** (so the populations are pre-CO, comparable to evogp),
restricted grammar {+,−,*,/,sin,cos,tan} = evogp funcset, same problems + identical (X,y).

**Headlines:**
- **Adapter is exact**: every Operon leaf is an optimizable coefficient (Variable=weight·x_col,
  Constant=value), so `K = tree.CoefficientsCount`. Validated by round-trip vs Operon's own
  evaluator (6/6 hand cases + **1999/2000 evolved trees exact**) and `inspect` PASS.
- **Coefficient density**: Operon ~**2.78×** denser than evogp (range 1.7–5.7×); Operon has
  **0% constant-free (K=0) trees** vs evogp's 7.7–24.2%.
- **Rank-deficiency mechanisms are DIFFERENT** (proven, not "both ⇒ confirmed"): Operon's
  weighted-leaf product `(w_a·x_a)(w_b·x_b)` is singular at 7 nodes / zero bloat (Jacobian
  columns are exact scalar multiples; fp32 cond ≈ 1.7e9); evogp's `c0·x_a·x_b` is full-rank
  (cond 1.0) and singular only with a redundant constant.
- **Population rates** (same kernel build): both rise with generation; on the active-tree
  (K>0) base gen0 is Operon 8.5% vs evogp 13.1%; evogp's bloat-driven rate climbs higher.
  Operon's extra contribution is an *irreducible structural floor* — robustness damping must
  cover both mechanisms.

**Not included** (deferred): the timed Operon-`LMOptimizer`-vs-kernel CO baseline on identical
trees — pending discussion. This is workload characterization, not a perf/quality baseline.

Regenerate: see `report.json` → `reproduce`.
