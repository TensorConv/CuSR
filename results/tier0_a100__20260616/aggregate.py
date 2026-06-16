"""Aggregate Tier0 throughput M-sweep -> CSV + log-log plot.
Merges the per-variant sweep files + the baseline@16k refill + the 256k extension.
"""
import json, csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

def load(fn):
    p = OUT / fn
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

# variant -> {M: row}, later files override earlier (refill/extension win)
variants = {"baseline": ["tp2_baseline.jsonl"],
            "devjac":   ["tp2_devjac.jsonl"],
            "fusedfd":  ["tp2_fusedfd.jsonl"]}

# laptop reference (RERUN_A100.md), M=1000 fixture
LAPTOP = {"baseline": 1319, "devjac": 3304, "fusedfd": 3812}

merged = {}
for v, files in variants.items():
    byM = {}
    for f in files:
        for r in load(f):
            if not r.get("ok"):
                continue
            # use synth points for the curve; keep real1000 separately
            byM[(r["point"], r["M"])] = r
    merged[v] = byM

# CSV
csv_path = OUT / "tier0_throughput.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["variant", "point", "M", "tps_med", "tps_p10", "tps_p90", "t_med_s", "sm_clock_mhz", "n"])
    for v, byM in merged.items():
        for (point, M), r in sorted(byM.items(), key=lambda kv: kv[0][1]):
            w.writerow([v, point, M, f"{r['tps_med']:.1f}", f"{r.get('tps_p10',0):.1f}",
                        f"{r.get('tps_p90',0):.1f}", f"{r['t_med']:.3f}", r.get("sm_clock_med","?"), r.get("n","?")])
print(f"wrote {csv_path}")

# Plot: trees/s vs M (synth points only, log-log)
plt.figure(figsize=(8, 5.5))
colors = {"baseline": "tab:red", "devjac": "tab:orange", "fusedfd": "tab:green"}
for v, byM in merged.items():
    pts = sorted([(M, r["tps_med"], r.get("tps_p10", r["tps_med"]), r.get("tps_p90", r["tps_med"]))
                  for (point, M), r in byM.items() if point.startswith("synth")])
    if not pts:
        continue
    Ms = [p[0] for p in pts]; tps = [p[1] for p in pts]
    lo = [p[1]-p[2] for p in pts]; hi = [p[3]-p[1] for p in pts]
    plt.errorbar(Ms, tps, yerr=[lo, hi], marker="o", capsize=3, color=colors[v], label=f"{v} (A100)")
    # laptop M=1000 reference marker
    if v in LAPTOP:
        plt.scatter([1000], [LAPTOP[v]], marker="x", s=80, color=colors[v], alpha=0.6)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("M (population size, # trees)  [N=1000 points, synth early-gen]")
plt.ylabel("throughput (trees / s)")
plt.title("A100 CO throughput vs batch size (fp32)\nx = laptop 5070Ti @ M=1000 reference")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plot_path = OUT / "tier0_throughput.png"
plt.tight_layout(); plt.savefig(plot_path, dpi=130)
print(f"wrote {plot_path}")

# console summary
print("\n=== throughput summary (trees/s, median) ===")
allM = sorted({M for byM in merged.values() for (pt, M) in byM if pt.startswith("synth")})
hdr = "M".ljust(9) + "".join(v.ljust(12) for v in merged)
print(hdr)
for M in allM:
    row = str(M).ljust(9)
    for v in merged:
        r = merged[v].get(("synth"+str(M), M)) or next((rr for (pt,mm),rr in merged[v].items() if mm==M and pt.startswith("synth")), None)
        row += (f"{r['tps_med']:.0f}".ljust(12) if r else "-".ljust(12))
    print(row)
