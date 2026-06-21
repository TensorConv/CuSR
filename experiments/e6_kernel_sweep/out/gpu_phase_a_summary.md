# e6 Phase-A (GPU) results — DRAFT (clocks unlocked)

Snapshot of the on-device GPU constant-optimization kernels (fused-FD + forward-AD)
from `sweep_e6.jsonl`. **GPU side only**; Operon/CPU comparison is Phase B (pending).
Source of truth: `gpu_phase_a.csv` / `gpu_phase_a.json` (this dir).

- **Completeness**: plan=90, ok=80, skipped=10 (memory-ceiling, logged), **missing=0** (0 silent drops / 0 subprocess failures).
- **Anti-host-FD**: every ok config passed the structural device-Jacobian PROFILE guard (on-device confirmed); no host-FD signature.
- **Metric**: throughput = (M - n_K0_dropped)/(loop_ms/1000), loop_ms = LM-loop GPU time (PROFILE_JSON; excludes CUDA init). kernel max_iter=50, median over seeds 0,1,2.

**fusedfd** (trees/s)

| preset | M | N=100 | N=1000 | N=10000 |
|---|---|---|---|---|
| early-gen | 1000 | 23k | 5598 | 622 |
| early-gen | 4000 | 79k | 19k | 1921 |
| early-gen | 16000 | 142k | 37k | 4124 |
| early-gen | 64000 | 267k | 53k | · |
| early-gen | 256000 | 306k | 57k | · |
| late-gen-bloated | 1000 | 27k | 7960 | 742 |
| late-gen-bloated | 4000 | 80k | 22k | 2395 |
| late-gen-bloated | 16000 | 140k | 37k | 4089 |
| late-gen-bloated | 64000 | 209k | 49k | 5267 |
| late-gen-bloated | 256000 | 228k | 50k | · |
| inner-const-heavy | 1000 | 22k | 5259 | 535 |
| inner-const-heavy | 4000 | 62k | 14k | 1530 |
| inner-const-heavy | 16000 | 114k | 18k | 1954 |
| inner-const-heavy | 64000 | 139k | 20k | · |
| inner-const-heavy | 256000 | 144k | 22k | · |

**ad** (trees/s)

| preset | M | N=100 | N=1000 | N=10000 |
|---|---|---|---|---|
| early-gen | 1000 | 29k | 10k | 1202 |
| early-gen | 4000 | 89k | 30k | 3489 |
| early-gen | 16000 | 154k | 51k | 5740 |
| early-gen | 64000 | 225k | 69k | · |
| early-gen | 256000 | 330k | 68k | · |
| late-gen-bloated | 1000 | 30k | 10k | 1446 |
| late-gen-bloated | 4000 | 88k | 32k | 3792 |
| late-gen-bloated | 16000 | 158k | 46k | 4893 |
| late-gen-bloated | 64000 | 220k | 55k | 5423 |
| late-gen-bloated | 256000 | 326k | 50k | · |
| inner-const-heavy | 1000 | 29k | 9345 | 1053 |
| inner-const-heavy | 4000 | 54k | 18k | 2110 |
| inner-const-heavy | 16000 | 136k | 25k | 2337 |
| inner-const-heavy | 64000 | 133k | 26k | · |
| inner-const-heavy | 256000 | 167k | 24k | · |

## Conclusions (GPU side)

1. **AD is the faster variant**: median ad/fusedfd = **1.251×** (range 0.844–1.968×); AD is also exact-Jacobian (equal/better quality). → lead/deploy variant.
2. **Peak**: ad early-gen M=256000 N=100 = **330338 t/s**.
3. **Saturates ~M=64k**: at N=1000, M=64k→256k gains <10% → A100 occupancy fills near M≈64k. CAVEAT: realistic in-loop population (~4000) sits well below saturation, so peak numbers must not be quoted as the in-loop rate.
4. **N axis = data-parallel headroom**: trees/s falls with N (more points/tree), but points/s RISES with N toward the compute roofline (early-gen ≈1.5e7→5.7e7 pts/s) — high N is where the GPU saturates its FLOPs.
5. **inner-const-heavy is slowest** (high K = more Jacobian columns); frac_converged 0.604–0.811 across configs. Its quality ceiling (loss) is a separate fp32/damping issue, not throughput.

DRAFT: clocks unlocked → medians, absolute t/s indicative. Locked-clock + ncu roofline later.
