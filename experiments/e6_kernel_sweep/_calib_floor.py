#!/usr/bin/env python
"""_calib_floor.py — SCRATCH (not committed). Measure the GENUINE on-device
throughput at the lowest-work matrix corners so GPU_TPUT_FLOOR can be set BELOW
the real minimum (and clearly ABOVE the ~1.3-2k host-FD signature). Applies the
"measure, don't guess" lesson: the dryrun only tests early-gen, but the full
matrix has late-gen-bloated / inner-const-heavy at M=1000 too."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sweep_e6 as sw  # noqa: E402  (same dir)

GPU = 0
CONFIGS = []
for preset in sw.ALL_PRESETS:
    for M in (1000,):
        for N in (100, 1000, 10000):
            CONFIGS.append((preset, M, N))

print("Calibration: genuine throughput at lowest-work corners (n_rep=3 median)")
print("-" * 78)
mins = {"fusedfd": float("inf"), "ad": float("inf")}
scratch = Path(__file__).resolve().parent / "out" / "calib_scratch"
for variant in ("fusedfd", "ad"):
    for preset, M, N in CONFIGS:
        pop = sw.get_pop(preset, M, N, 0)
        outd = scratch / f"{variant}_{preset}_{M}_{N}"
        k = sw.run_gpu_kernel(pop, variant, sw.KERNEL_MAX_ITER, outd, GPU, n_rep=3)
        t = k["throughput"]
        mins[variant] = min(mins[variant], t)
        print(f"  {variant:8s} {preset:18s} M={M} N={N:5d}: "
              f"tput={t:8.0f} tr/s  loop_ms={k['loop_ms']:7.2f}  "
              f"K_max={int(pop['K_max'])}  med_loss={k['med_loss_fp64']:.3e}")
        sw.evict_pop(preset, M, N, 0)

print("-" * 78)
overall_min = min(mins.values())
print(f"min throughput: fusedfd={mins['fusedfd']:.0f}  ad={mins['ad']:.0f}  "
      f"OVERALL={overall_min:.0f} tr/s")
print(f"host-FD signature ~1.3-2k; genuine min ~{overall_min:.0f}. "
      f"Suggested floor ~{0.6*overall_min:.0f} (0.6x genuine min, still "
      f">{0.6*overall_min/2000:.1f}x the 2k host-FD ceiling).")
