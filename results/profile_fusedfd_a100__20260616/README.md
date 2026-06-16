# profile_fusedfd_a100__20260616

**Per-kernel CUDA-event profiling** of `batch_lm_fusedfd`'s LM loop on the A100.
Characterization (where does CO time go), not optimization. No root needed — this
is the no-profiler path that complements the (admin-blocked) ncu measured roofline.

- **Read `report.html`** for the full bilingual (中/EN) writeup.
- `sweep.py` — runs the `-DPROFILE` build across M, medians 3 runs, makes the plots.
- `data/` — `prof_M<M>.json`, `profile_summary.json`, `profile_breakdown.csv`.
- `plots/` — `loop_breakdown_frac.png` (100%-stacked loop time), `wall_decomposition.png`.

**How it's built:** `make -C cusr/kernel batch_lm_fusedfd_prof` compiles the fused
source with `-DPROFILE`. The profiling is `#ifdef PROFILE`-guarded, so the default
`batch_lm_fusedfd` is byte-behavior-identical (parity verified: status.bin +
c_final.bin bit-identical across frozen / rebuilt-default / prof binaries). The
prof binary emits one extra `PROFILE_JSON {...}` line. It is gitignored (not committed).

**Headline:**
- `fd_jacobian` (the K-redundant full-tree FD Jacobian) = **49–73% of LM-loop GPU
  time** at every scale → optimization target #1 (forward-mode AD / dual numbers).
- The loop is **GPU-bound, not host-bound** (host residual 1.8–4.2%, memcpy 3–10%) —
  this **corrects** the analytical roofline's host-orchestration hypothesis: the gap
  below the ceiling is GPU-kernel inefficiency, not round-trips.
- `fd_jacobian + eval` (two tree-interpreter passes) = **69–87%** of loop → optimize
  interpretation/FD, not the linear algebra (`solve` ≤2.4%).
- `build_jtj` (J round-trip) grows to **14%** at scale → confirms the roofline's
  mild-memory-bound point, but secondary to `fd_jacobian`.
- One-time **setup ~constant 359–488ms** (CUDA ctx + alloc); dominates small-M wall,
  amortized at scale (loop = 30%→78% of total wall). Per-gen CO cost = `loop_ms`.
  Loop-only throughput 41k (16k) → 64k (256k) trees/s, ~1.3× the Tier0 end-to-end #.

**Note on the inner-const-heavy ~64.8% figure** (seen in kernel comments): that is a
*strict-tolerance* recovery rate. GP selects the top-x% each generation, so CO quality
on the non-selected tail has limited end-to-end impact — the 64.8% does not represent
real SR outcomes. Quality should be measured end-to-end (see direction memo), not by
that strict number.

**TODO (needs admin/root):** the measured counterpart — ncu warp-stall / achieved
occupancy / DRAM throughput per kernel — needs `NVreg_RestrictProfilingToAdminUsers=0`.
This event-based split is the no-root proxy.

Regenerate: see `report.json` → `reproduce`.
