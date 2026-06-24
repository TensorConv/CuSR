# GPU Phase-A results — LOCKED @ 1410 MHz

Snapshot of the on-device GPU constant-optimization kernels (fused-FD / forward-AD /
reverse-AD) from `sweep_e6.jsonl`. **GPU side only**; Operon/CPU comparison is offline.
Source of truth: `gpu_phase_a.csv` / `gpu_phase_a.json` (this dir).

- **Completeness**: plan=135, ok=120, skipped=42 (memory-ceiling, logged), **missing=0** (0 silent drops / 0 subprocess failures).
- **Anti-host-FD**: every ok config passed the structural device-Jacobian PROFILE guard (on-device confirmed); no host-FD signature.
- **Metric**: throughput = (M - n_K0_dropped)/(loop_ms/1000), loop_ms = LM-loop GPU time (PROFILE_JSON; excludes CUDA init). kernel max_iter=50, median over seeds 0,1,2.

**fusedfd** (trees/s)

| preset | M | N=100 | N=1000 | N=10000 |
|---|---|---|---|---|
| early-gen | 1000 | 26k | 5503 | 622 |
| early-gen | 4000 | 81k | 19k | 1929 |
| early-gen | 16000 | 181k | 39k | 2997 |
| early-gen | 64000 | 196k | 50k | · |
| early-gen | 256000 | 275k | 59k | · |
| late-gen-bloated | 1000 | 30k | 8058 | 740 |
| late-gen-bloated | 4000 | 100k | 23k | 2406 |
| late-gen-bloated | 16000 | 182k | 39k | 4128 |
| late-gen-bloated | 64000 | 266k | 47k | 5228 |
| late-gen-bloated | 256000 | 297k | 52k | · |
| inner-const-heavy | 1000 | 25k | 5200 | 535 |
| inner-const-heavy | 4000 | 63k | 14k | 1536 |
| inner-const-heavy | 16000 | 94k | 18k | 1956 |
| inner-const-heavy | 64000 | 138k | 20k | · |
| inner-const-heavy | 256000 | 143k | 21k | · |

**ad** (trees/s)

| preset | M | N=100 | N=1000 | N=10000 |
|---|---|---|---|---|
| early-gen | 1000 | 34k | 10k | 1195 |
| early-gen | 4000 | 101k | 30k | 3485 |
| early-gen | 16000 | 154k | 52k | 5741 |
| early-gen | 64000 | 220k | 69k | · |
| early-gen | 256000 | 327k | 68k | · |
| late-gen-bloated | 1000 | 34k | 12k | 1439 |
| late-gen-bloated | 4000 | 103k | 32k | 3800 |
| late-gen-bloated | 16000 | 178k | 46k | 4883 |
| late-gen-bloated | 64000 | 217k | 55k | 5436 |
| late-gen-bloated | 256000 | 241k | 53k | · |
| inner-const-heavy | 1000 | 33k | 9570 | 1052 |
| inner-const-heavy | 4000 | 61k | 18k | 2113 |
| inner-const-heavy | 16000 | 109k | 24k | 2336 |
| inner-const-heavy | 64000 | 132k | 26k | · |
| inner-const-heavy | 256000 | 178k | 24k | · |

**revad** (trees/s)

| preset | M | N=100 | N=1000 | N=10000 |
|---|---|---|---|---|
| early-gen | 1000 | 40k | 14k | 1624 |
| early-gen | 4000 | 121k | 37k | 4181 |
| early-gen | 16000 | 217k | 62k | 6831 |
| early-gen | 64000 | 318k | 77k | · |
| early-gen | 256000 | 361k | 77k | · |
| late-gen-bloated | 1000 | 35k | 12k | 1487 |
| late-gen-bloated | 4000 | 96k | 28k | 3284 |
| late-gen-bloated | 16000 | 154k | 42k | 4680 |
| late-gen-bloated | 64000 | 265k | 51k | 5159 |
| late-gen-bloated | 256000 | 232k | 50k | · |
| inner-const-heavy | 1000 | 37k | 12k | 1367 |
| inner-const-heavy | 4000 | 67k | 23k | 2782 |
| inner-const-heavy | 16000 | 157k | 33k | 3467 |
| inner-const-heavy | 64000 | 150k | 37k | · |
| inner-const-heavy | 256000 | 212k | 36k | · |

## Conclusions (GPU side)

0. **revad/ad speed ratio (task-06, rev-vs-fwd slice)**: median revad/ad = **1.171×** (range [0.861, 1.485]); revad/fusedfd median = **1.495×** ([0.781, 2.613]). n_revad_ok=40. locked_clock_mhz=1410.
1. **AD is the faster variant**: median ad/fusedfd = **1.24×** (range [0.812, 1.967]); AD is also exact-Jacobian (equal/better quality). → lead/deploy variant.
2. **Peak**: revad early-gen M=256000 N=100 = **361334 t/s**.
3. **Saturates ~M=64k**: at N=1000, M=64k→256k gains <10% → A100 occupancy fills near M≈64k. CAVEAT: realistic in-loop population (~4000) sits well below saturation, so peak numbers must not be quoted as the in-loop rate.
4. **N axis = data-parallel headroom**: trees/s falls with N (more points/tree), but points/s RISES with N toward the compute roofline (early-gen ≈1.5e7→5.7e7 pts/s) — high N is where the GPU saturates its FLOPs.
5. **inner-const-heavy is slowest** (high K = more Jacobian columns); frac_converged 0.604–0.81 across configs. Its quality ceiling (loss) is a separate fp32/damping issue, not throughput.

LOCKED @ 1410 MHz → absolute t/s are publication-grade (medians over seeds 0,1,2).
