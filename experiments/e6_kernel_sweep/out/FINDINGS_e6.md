# e6 Kernel Sweep — Findings (DRAFT)

Heterogeneous batched LM constant-optimization (CO) kernel vs. Operon (CPU, multi-core),
swept over preset (population character) x M (rows/data points) x N (population size) x
backend variant / core count.

**Status: DRAFT.** GPU clocks are unlocked (no `nvidia-smi -lgc` pin), so absolute throughput
and all speedups carry run-to-run jitter. The relative *shape* of the results (M-scaling,
N-shrink, Operon core-scaling, iso-quality vs. speed-only) is robust; the exact multipliers
are not nailed.

Timing protocol: **GPU `loop_ms` vs. Operon `wall_core`** — both *setup-excluded*, i.e. the
in-loop per-generation CO cost only (CUDA init / data marshalling / .so load are not counted on
either side). Throughput = trees-optimized / second. `kernel_max_iter=50`, `operon_max_iter=200`
(Operon is given 4x the LM budget). GPU throughput = median over seeds 0/1/2; quality
(`med_loss_fp64`) from seed0; Operon = seed0.

---

## 1. Accounting

| bucket | count |
|---|---|
| expected (full matrix) | 270 |
| completed (`status=ok`) | 242 |
| skipped (logged, with reason) | 28 |
| missing (silent drop) | **0** |

242 + 28 + 0 = 270. Identity verified (`accounting_ok=true`). Completed split:
**80 GPU** (of 90 planned) + **162 Operon** (of 180 planned). All 242 completed records carry
valid finite/positive throughput, timing fields, and `.npy` loss sidecars; the device-Jacobian
guard passed at runtime for every GPU record (rules out a host-FD slow-path regression).

The 28 skips, by reason (counted from the actual `skipped[]` array — note `report.json`
`meta...n_skipped_mem=8` is a stale *pre-run plan estimate*; the real count is 10):

- **10 — `memory_ceiling`** (GPU only): estimated footprint > 40 GB. All at N=10000:
  M=256000 (3 presets x 2 variants, 105–162 GB) and the two borderline M=64000 cases
  (early-gen + inner-const-heavy, ~40.6 GB, just over).
- **18 — `operon_ncores1_largeM`** (Operon only): unconditional skip of `ncores=1` at
  M >= 64000 (single-core wall-clock would dominate, 14 s–5505 s estimated). All 3 presets x
  {64000, 256000} x {N=100,1000,10000}.

---

## 2. Headline — per-preset throughput tables at informative configs

Each cell: **speedup (GPU loop_ms / Operon wall_core)** with an explicit tag:
**[ISO]** = iso-quality win, GPU final loss within 1.05x of Operon's (`loss_ratio = kernel/operon <= 1.05`);
**[SPD]** = speed-only, GPU is faster but its loss exceeds the 1.05x quality gate (NOT a clean win).
`nc` = Operon core count. `n/a` = config skipped (no fabricated number).

### 2a. `late-gen-bloated`, M=256000, N=100 — the strongest clean win

GPU throughput: ad = 326k tree/s, fusedfd = 228k tree/s. Operon: 16c=12.9k, 64c=9.8k, 128c=10.3k.

| GPU variant | vs nc=1 | vs nc=16 | vs nc=64 | vs nc=128 |
|---|---|---|---|---|
| **ad** | n/a (skip) | 25.3x [ISO] | 33.2x [ISO] | 31.7x [ISO] |
| **fusedfd** | n/a (skip) | 17.7x [ISO] | 23.2x [ISO] | 22.2x [ISO] |

Read this carefully: the **31.7x headline is vs the 128-core EPYC**, and it is *larger* than the
25.3x vs Operon's actually-fastest config (16 cores) **precisely because Operon anti-scales past
16 cores** (see Finding 3). Against Operon's best honest config the clean iso-quality win is
**~25x (ad) / ~18x (fusedfd)**; the "~31x" is the 128-core comparison.

### 2b. `early-gen`, M=64000, N=1000 — a clean win, but AD-only

GPU: ad = 69k tree/s, fusedfd = 53k tree/s. Operon: 16c=9.0k, 64c=9.2k, 128c=8.5k.

| GPU variant | vs nc=1 | vs nc=16 | vs nc=64 | vs nc=128 |
|---|---|---|---|---|
| **ad** | n/a (skip) | 7.7x [ISO] | 7.5x [ISO] | 8.2x [ISO] |
| **fusedfd** | n/a (skip) | 5.9x [SPD] | 5.8x [SPD] | 6.3x [SPD] |

**The ~8x early-gen win is AD-specific.** At the identical config fusedfd is ~6x but
*speed-only* (loss_ratio 1.060 > gate); only AD clears iso-quality (loss_ratio 1.048). Do not
report "~8x at N=1000" as variant-agnostic.

### 2c. `inner-const-heavy`, M=256000, N=100 — fast but NOT a clean win

GPU: ad = 167k tree/s, fusedfd = 144k tree/s. Operon: 16c=11.5k, 64c=9.4k, 128c=8.9k.

| GPU variant | vs nc=1 | vs nc=16 | vs nc=64 | vs nc=128 | loss_ratio |
|---|---|---|---|---|---|
| **ad** | n/a (skip) | 14.5x [SPD] | 17.7x [SPD] | 18.8x [SPD] | 1.297 |
| **fusedfd** | n/a (skip) | 12.5x [SPD] | 15.3x [SPD] | 16.2x [SPD] | 1.377 |

Every cell is **speed-only**: GPU is 12–19x faster but converges to a 30–38% *worse* loss. This
is an intrinsic quality ceiling, not a measurement artifact (Finding 4). We refuse to present
these as wins.

---

## 3. Honest findings

**(1) Clean iso-quality wins exist and are large.**
The biggest is `late-gen-bloated` M=256000 N=100: **~31.7x vs 128c / ~25.3x vs 16c, [ISO]**.
Early-generation populations give a smaller but still clean win: `early-gen` M=16000–64000
N=1000 reaches **~8x [ISO]** — *via the AD variant only* (fusedfd is ~6x [SPD] at the same
points). These are the configs where the GPU advantage is real, not a quality trade.

**(2) The win shrinks sharply with N (data-point count), and N is the dominant axis.**
Isolating N at fixed M (`late-gen-bloated`, M=16000, ad, vs 64c):

| N | speedup | tag |
|---|---|---|
| 100 | 12.5x | [ISO] |
| 1000 | 6.4x | [ISO] |
| 10000 | 1.3x | [SPD] |

The headline-M sweep (M=256000 for N<=1000) shows the same collapse — 31.7x (N=100) ->
4.6x (N=1000); note the N=10000 leg drops to M=64000 because M=256000 N=10000 was
memory-skipped, so that 1.5x is not a same-M point. **Why:** large N amortizes Operon's
per-tree fixed overhead (parse/setup is paid once per tree regardless of N, so high N hides it),
while the GPU becomes purely work-bound — at N=10000 both backends spend their time on the same
O(M·N·K) residual/Jacobian arithmetic and the GPU's launch-amortization edge evaporates. The GPU
wins when there are *many cheap trees*, not when each tree is individually expensive.

**(3) Operon barely scales past ~16 cores.**
Median Operon throughput ratio **128c / 16c = 1.06x** (64c/16c = 1.10x): on the 128-core EPYC,
"128 cores" is effectively "16 cores." This is the median pattern, not a universal law — mean
ratio is 1.71x and ~49% of configs *do* gain something (max 7.5x at favorable (M,N)); scaling is
(M,N)-dependent and collapses on most but not all configs. Consequence: the strongest *honest*
CPU baseline is 16-core, and quoting GPU-vs-128c speedups (as in 2a) actually flatters Operon
less, not more.

**(4) `inner-const-heavy` is SPEED-ONLY — an intrinsic quality ceiling.**
Of the inner-const cells, the overwhelming majority fail the iso-quality gate (max speed-only
18.8x at M=256000 N=100, loss_ratio 1.30). The only iso-quality exceptions are at the smallest
problem (M=1000 N=100, ~2–4x). The kernel converges to a *worse* fp64 loss on these
populations because they are rank-deficient in the inner constants (documented in
MIXED_PRECISION_PROBE.md) — more LM iterations / higher precision cannot recover a degenerate
normal-equation system. Speed here buys nothing.

**(5) `ad` beats `fusedfd` by ~1.25x median; `fusedfd` wins a few low-N/high-M shapes.**
Paired over 40 GPU configs, median ad/fusedfd throughput = **1.251x** (mean 1.31x); ad is faster
in 92.5% of configs. The 3 exceptions where fusedfd wins are all low-N / high-M
(early-gen M=64000 N=100: ad/fd=0.84; inner-const M=4000 N=100: 0.87; inner-const M=64000 N=100:
0.96) — at low N the fused finite-difference Jacobian's lower per-element cost beats AD's extra
tangent work. AD is the better default; fusedfd is a situational win.

**(6) M is a design choice — the GPU unlocks the large-M regime, and the advantage grows with M.**
At `late-gen-bloated` N=100, ad vs 128c, all [ISO]:

| M | speedup |
|---|---|
| 1000 | 3.1x |
| 4000 | 9.5x |
| 16000 | 14.4x |
| 64000 | 23.4x |
| 256000 | 31.7x |

Operon throughput is roughly flat in M (~9–13k tree/s) while GPU throughput climbs from 30k to
326k tree/s. M (how many candidate trees / data rows are pushed through CO per generation) is a
knob the SR driver sets, not a fixed property of the problem: the kernel makes large M cheap,
which is what makes the large-M regime usable at all.

(Counting unit for "iso vs speed-only": per (Operon-record x GPU-variant) cell — 294 such cells,
of which 133 clear the iso gate and 161 are speed-only. Collapsing core-counts to per-config
gives smaller denominators; the *classification* per cell is what matters and is preserved
throughout the tables above.)

---

## 4. Caveats

- **DRAFT — clocks unlocked.** No GPU clock pin; absolute throughput and exact speedups jitter
  run-to-run. Trust the shape, not the third significant figure.
- **Timing is setup-excluded on both sides.** GPU `loop_ms` (in-loop per-generation CO) vs.
  Operon `wall_core`; neither counts CUDA init, .so load, or data marshalling. (End-to-end
  `total_ms` / `wall_e2e` are recorded per-record but are not the headline metric.) Operon also
  runs 200 LM iters vs. the kernel's 50.
- **"Large M helps SR" is NOT shown here.** This sweep measures CO-kernel throughput and
  per-call convergence only. The claim that a larger M improves end-to-end symbolic-regression
  outcomes is a *demonstrator-level* claim (e2/e4), not demonstrated by e6.
- **Quality gate is one-sided & seed0-based.** [ISO]/[SPD] uses seed0 `med_loss_fp64` at a 1.05x
  tolerance; it certifies the GPU is not meaningfully *worse*, not that it is better.
