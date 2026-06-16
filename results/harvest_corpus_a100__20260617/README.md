# harvest_corpus_a100__20260617

**Phase 1 real GP-snapshot corpus** harvested from evogp — workload characterization
(what real populations look like) + a kernel convergence/failure profile on them.

- **Read `report.html`** for the full bilingual (中/EN) writeup.
- `summarize.py` — aggregates all pop=4000 manifests + runs the kernel sample + plots.
- `data/` — `drift.csv`, `kernel_profile.json`, `summary.json`.
- `plots/` — `workload_drift.png`, `per_problem_K.png`, `kernel_vs_gen.png`.

**Corpus** (harvested by `cusr/benchmark/workload/harvest.py`, snapshots under
`data/workload/snapshots/`, `.bin` gitignored, **manifests committed**):
- 204 cells = 17 problems × noise{none,1%} × cap{32,64} × seed{0,1,2}, each dumped at
  gens {0,1,2,4,8,16,32,64,100} → **1836 snapshots** @ pop=4000, N=1000.
- 6 large-M snapshots @ pop=65536 (I.18.12 / I.6.2 / nguyen-5 × gen{8,100}).
- 204/204 OK, 0 failures, 5.9 min on GPUs 0–5 (6–7 left free).

**Headlines:**
- **max_tree_len is the dominant workload lever**: gen100 cap=64 → mean_K 7.9 / 53 nodes
  vs cap=32 → 3.2 / 26.5. K_max 24 vs 13, both < MAX_K=32 → **0 K-over** (cap=64 safe).
- **Drift**: trees bloat with generation (cap64 mean_nodes 6.9→53, mean_K 1.7→7.9). The
  synth "early-gen" (mean_K~2.8) ≈ real **gen8** — synth is a fair early proxy; the real
  corpus extends the operating region toward larger/higher-K.
- **Per-problem variety**: gen100 mean_K spans 5.5–12.3; constant-free Nguyen targets pack
  in the most constants (nguyen/5 ~11–12).
- **Rank-deficient Cholesky dominates the bloated tail** (fail_cholesky 10%→up to 94% by
  gen100/cap64), *except* nguyen/5 stays healthy. **Population-wide** number — GP selects
  the top-x%, so this is not necessarily an end-to-end defect (measure end-to-end on the
  selected fraction); it is the target for the robustness work (Levenberg trust-region damping).

Regenerate: see `report.json` → `reproduce`.
