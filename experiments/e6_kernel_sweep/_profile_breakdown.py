#!/usr/bin/env python
"""_profile_breakdown.py — SCRATCH. Measure host_residual vs GPU-compute split via the
-DPROFILE binaries' PROFILE_JSON, at small vs large M, both variants. Decides the
on-device-loop payoff (OPTIMIZATION_BACKLOG §0.1/§3.1): if host_residual dominates the
LM loop at small M and shrinks at large M, the round-trip is the small-M bottleneck."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sweep_e6 as sw  # noqa: E402
from cusr.benchmark import popio  # noqa: E402

KDIR = Path(__file__).resolve().parents[2] / "cusr" / "kernel"
BIN = {"fusedfd": KDIR / "batch_lm_fusedfd_prof", "ad": KDIR / "batch_lm_ad_prof"}
scratch = Path(__file__).resolve().parent / "out" / "prof_scratch"
scratch.mkdir(parents=True, exist_ok=True)

CONFIGS = [("early-gen", 4000, 1000), ("early-gen", 64000, 1000),
           ("inner-const-heavy", 4000, 1000), ("inner-const-heavy", 64000, 1000)]

print(f"{'variant':8}{'preset':18}{'M':>7}{'N':>6} | "
      f"{'loop_ms':>9}{'gpu_ms':>9}{'host_res':>9}{'host%':>7} | top GPU cats (ms)")
print("-" * 110)
for variant in ("fusedfd", "ad"):
    for preset, M, N in CONFIGS:
        pop = sw.get_pop(preset, M, N, 0)
        outd = scratch / f"{variant}_{preset}_{M}_{N}"
        outd.mkdir(parents=True, exist_ok=True)
        pb = outd / "pop.bin"
        popio.save_pop_bin(pop, pb)
        r = subprocess.run([str(BIN[variant]), str(pb), str(outd),
                            "--max-iter", "50", "--quiet"],
                           capture_output=True, text=True, timeout=600)
        line = next((l for l in r.stdout.splitlines()
                     if l.startswith("PROFILE_JSON")), None)
        if not line:
            print(f"{variant:8}{preset:18}{M:>7}{N:>6} | NO PROFILE_JSON: "
                  f"{(r.stderr or r.stdout)[-150:]}")
            continue
        p = json.loads(line[len("PROFILE_JSON"):])
        loop = p["loop_ms"]; gpu = p["gpu_sum_ms"]; hr = p["host_residual_ms"]
        cats = p["cats"]
        top = sorted(cats.items(), key=lambda kv: -kv[1]["ms"])[:3]
        topstr = ", ".join(f"{k}:{v['ms']:.0f}" for k, v in top)
        print(f"{variant:8}{preset:18}{M:>7}{N:>6} | "
              f"{loop:>9.1f}{gpu:>9.1f}{hr:>9.1f}{100*hr/loop:>6.0f}% | {topstr}")
        sw.evict_pop(preset, M, N, 0)
print("-" * 110)
print("host% = fraction of the LM loop NOT in GPU ops (launch latency + host accept/reject")
print("+ memcpy-wait). High host% at small M + low at large M => the per-iter host round-trip")
print("is the small-M bottleneck => on-device accept/reject (BACKLOG §3.1) is the lever.")
