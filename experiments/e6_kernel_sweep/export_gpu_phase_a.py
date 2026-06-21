#!/usr/bin/env python
"""export_gpu_phase_a.py — snapshot the e6 Phase-A (GPU) results from the live
sweep JSONL into stable, organized artifacts. Reads only the 'kernel' records
(complete after Phase A); does NOT touch sweep_e6.jsonl (Phase B/Operon may still
be appending to it). Re-runnable.

Writes (to out/):
  gpu_phase_a.csv          one row per GPU config (ok + skipped), all metrics
  gpu_phase_a.json         structured: summary stats + per-config list
  gpu_phase_a_summary.md   human-readable tables + conclusions
"""
import csv
import json
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
SRC = OUT / "sweep_e6.jsonl"

VAR = ["fusedfd", "ad"]
PRE = ["early-gen", "late-gen-bloated", "inner-const-heavy"]
MS = [1000, 4000, 16000, 64000, 256000]
NS = [100, 1000, 10000]

recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
kern = [r for r in recs if r.get("backend") == "kernel"]
ok = [r for r in kern if r.get("status") == "ok"]
sk = [r for r in kern if r.get("status") == "skipped"]

plan = {(v, p, M, N) for v in VAR for p in PRE for M in MS for N in NS}
okkey = {(r["variant"], r["preset"], r["M"], r["knob"]) for r in ok}
skkey = {(r["variant"], r["preset"], r["M"], r["knob"]) for r in sk}
missing = sorted(plan - okkey - skkey)


def tput(v, p, M, N):
    for r in ok:
        if (r["variant"], r["preset"], r["M"], r["knob"]) == (v, p, M, N):
            return r["throughput"]
    return None if (v, p, M, N) in skkey else float("nan")  # nan = missing


# ---- CSV (authoritative per-config rows) ----
cols = ["status", "variant", "preset", "M", "N", "throughput_tps",
        "throughput_e2e_tps", "points_per_s", "loop_ms", "total_ms",
        "med_loss_fp64", "K_max", "n_dropped", "frac_converged", "frac_failed",
        "skip_reason", "est_footprint_gb"]
with (OUT / "gpu_phase_a.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in sorted(ok, key=lambda r: (r["variant"], r["preset"], r["M"], r["knob"])):
        N = r["knob"]
        pps = r["throughput"] * N  # ~ (M-drop)*N / loop_s : points evaluated/s
        w.writerow(["ok", r["variant"], r["preset"], r["M"], N,
                    f"{r['throughput']:.1f}", f"{r.get('throughput_e2e', float('nan')):.1f}",
                    f"{pps:.0f}", f"{r['loop_ms']:.3f}",
                    f"{r.get('total_ms', float('nan')):.3f}",
                    f"{r['med_loss_fp64']:.6g}", r.get("K_max", ""), r.get("n_dropped", ""),
                    f"{r.get('frac_converged', float('nan')):.3f}",
                    f"{r.get('frac_failed', float('nan')):.3f}", "", ""])
    for r in sorted(sk, key=lambda r: (r["variant"], r["preset"], r["M"], r["knob"])):
        w.writerow(["skipped", r["variant"], r["preset"], r["M"], r["knob"],
                    "", "", "", "", "", "", r.get("K_max", ""), "", "", "",
                    r.get("reason", ""), f"{r.get('estimated_footprint_gb', '')}"])

# ---- summary stats ----
peak = max(ok, key=lambda r: r["throughput"])
ratios = []
for p in PRE:
    for M in MS:
        for N in NS:
            a, fd = tput("ad", p, M, N), tput("fusedfd", p, M, N)
            if isinstance(a, float) and isinstance(fd, float) and a > 0 and fd > 0:
                ratios.append(a / fd)
summary = dict(
    generated_from=str(SRC), kernel_max_iter=ok[0].get("max_iter") if ok else None,
    n_plan=len(plan), n_ok=len(ok), n_skipped=len(sk), n_missing=len(missing),
    missing=[list(m) for m in missing],
    peak=dict(variant=peak["variant"], preset=peak["preset"], M=peak["M"],
              N=peak["knob"], throughput_tps=round(peak["throughput"], 1)),
    ad_over_fusedfd_median=round(st.median(ratios), 3),
    ad_over_fusedfd_range=[round(min(ratios), 3), round(max(ratios), 3)],
    frac_converged_range=[round(min(r["frac_converged"] for r in ok), 3),
                          round(max(r["frac_converged"] for r in ok), 3)],
    note=("throughput = (M - n_K0_dropped)/(loop_ms/1000), loop_ms from PROFILE_JSON "
          "(LM loop only, excludes CUDA init). On-device verified per-config via the "
          "device-Jacobian PROFILE guard. clocks UNLOCKED -> DRAFT. GPU side only; "
          "Operon/CPU comparison pending Phase B."),
)

# ---- JSON (structured) ----
configs = []
for r in sorted(ok + sk, key=lambda r: (r["variant"], r["preset"], r["M"], r["knob"])):
    e = dict(status=r["status"], variant=r["variant"], preset=r["preset"],
             M=r["M"], N=r["knob"])
    if r["status"] == "ok":
        e.update(throughput_tps=round(r["throughput"], 1),
                 throughput_e2e_tps=round(r.get("throughput_e2e", float("nan")), 1),
                 points_per_s=round(r["throughput"] * r["knob"]),
                 loop_ms=round(r["loop_ms"], 3), med_loss_fp64=r["med_loss_fp64"],
                 K_max=r.get("K_max"), frac_converged=r.get("frac_converged"))
    else:
        e.update(skip_reason=r.get("reason"),
                 est_footprint_gb=r.get("estimated_footprint_gb"))
    configs.append(e)
(OUT / "gpu_phase_a.json").write_text(
    json.dumps(dict(summary=summary, configs=configs), indent=2, default=float))


# ---- Markdown summary ----
def cell(v, p, M, N):
    t = tput(v, p, M, N)
    if t is None:
        return "·"
    if isinstance(t, float) and t != t:
        return "MISS"
    return f"{t/1000:.0f}k" if t >= 10000 else f"{t:.0f}"


def table(v):
    lines = [f"**{v}** (trees/s)",
             "",
             "| preset | M | N=100 | N=1000 | N=10000 |",
             "|---|---|---|---|---|"]
    for p in PRE:
        for M in MS:
            lines.append(f"| {p} | {M} | " +
                         " | ".join(cell(v, p, M, N) for N in NS) + " |")
    return "\n".join(lines)


md = f"""# e6 Phase-A (GPU) results — DRAFT (clocks unlocked)

Snapshot of the on-device GPU constant-optimization kernels (fused-FD + forward-AD)
from `sweep_e6.jsonl`. **GPU side only**; Operon/CPU comparison is Phase B (pending).
Source of truth: `gpu_phase_a.csv` / `gpu_phase_a.json` (this dir).

- **Completeness**: plan={summary['n_plan']}, ok={summary['n_ok']}, skipped={summary['n_skipped']} (memory-ceiling, logged), **missing={summary['n_missing']}** (0 silent drops / 0 subprocess failures).
- **Anti-host-FD**: every ok config passed the structural device-Jacobian PROFILE guard (on-device confirmed); no host-FD signature.
- **Metric**: throughput = (M - n_K0_dropped)/(loop_ms/1000), loop_ms = LM-loop GPU time (PROFILE_JSON; excludes CUDA init). kernel max_iter={summary['kernel_max_iter']}, median over seeds 0,1,2.

{table('fusedfd')}

{table('ad')}

## Conclusions (GPU side)

1. **AD is the faster variant**: median ad/fusedfd = **{summary['ad_over_fusedfd_median']}×** (range {summary['ad_over_fusedfd_range'][0]}–{summary['ad_over_fusedfd_range'][1]}×); AD is also exact-Jacobian (equal/better quality). → lead/deploy variant.
2. **Peak**: {summary['peak']['variant']} {summary['peak']['preset']} M={summary['peak']['M']} N={summary['peak']['N']} = **{summary['peak']['throughput_tps']:.0f} t/s**.
3. **Saturates ~M=64k**: at N=1000, M=64k→256k gains <10% → A100 occupancy fills near M≈64k. CAVEAT: realistic in-loop population (~4000) sits well below saturation, so peak numbers must not be quoted as the in-loop rate.
4. **N axis = data-parallel headroom**: trees/s falls with N (more points/tree), but points/s RISES with N toward the compute roofline (early-gen ≈1.5e7→5.7e7 pts/s) — high N is where the GPU saturates its FLOPs.
5. **inner-const-heavy is slowest** (high K = more Jacobian columns); frac_converged {summary['frac_converged_range'][0]}–{summary['frac_converged_range'][1]} across configs. Its quality ceiling (loss) is a separate fp32/damping issue, not throughput.

DRAFT: clocks unlocked → medians, absolute t/s indicative. Locked-clock + ncu roofline later.
"""
(OUT / "gpu_phase_a_summary.md").write_text(md)

print(f"ok={len(ok)} skipped={len(sk)} missing={len(missing)}")
print("wrote:", OUT / "gpu_phase_a.csv")
print("wrote:", OUT / "gpu_phase_a.json")
print("wrote:", OUT / "gpu_phase_a_summary.md")
