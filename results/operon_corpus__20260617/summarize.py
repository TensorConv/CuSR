"""summarize.py — Operon pre-CO corpus characterization + cross-corpus (Operon vs
evogp) density comparison, from the committed manifests (no GPU).

Outputs (results/operon_corpus__20260617/):
  data/summary.json   corpus totals, drop totals, per-problem density at key gens,
                      matched-cell Operon-vs-evogp density (mean_K, mean_nodes, K_max)
  data/drift.csv      corpus, problem, cap, gen, mean_K, mean_nodes, K_max
Honesty: aggregates ALL operon cells; reports drop totals (K-over must be 0); the
representation-density gap (Operon K ~= #leaves vs evogp K = #const-leaves) is the
headline workload difference, complementary to the mechanism_proof + cross_kernel.
"""
import csv, glob, json, statistics
from pathlib import Path

ROOT = Path("/home/weish/hao/CuSR")
SNAP = ROOT / "data/workload/snapshots"
OUTD = ROOT / "results/operon_corpus__20260617/data"
PROBLEMS = ["feynman/I.12.1","feynman/I.18.12","feynman/I.27.6","feynman/I.6.2",
    "feynman/I.12.2","feynman/I.13.12","feynman/II.3.24",
    "nguyen/1","nguyen/2","nguyen/3","nguyen/4","nguyen/5","nguyen/6","nguyen/7",
    "nguyen/8","nguyen/9","nguyen/10"]

def load_cell(corpus, prob, cap, seed=0, noise="0"):
    safe = prob.replace("/", "_")
    base = f"{safe}_pop4000_noise{noise}_len{cap}_seed{seed}"
    mf = SNAP / (f"operon_{base}" if corpus == "operon" else base) / "manifest.json"
    return json.loads(mf.read_text()) if mf.exists() else None

# ---- 1. full Operon corpus totals + drop accounting (ALL cells) ----
op_cells = sorted(glob.glob(str(SNAP / "operon_*" / "manifest.json")))
tot = dict(cells=0, snapshots=0, trees_in=0, trees_kept=0,
           dropped_unsupported=0, dropped_k_over=0, dropped_bad_type=0, dropped_nonfinite=0)
kmax_global = 0
for mf in op_cells:
    m = json.loads(Path(mf).read_text()); tot["cells"] += 1
    for s in m.get("snapshots", []):
        tot["snapshots"] += 1
        tot["trees_in"] += s.get("n_in", 0); tot["trees_kept"] += s.get("n_kept", 0)
        for k in ("dropped_unsupported","dropped_k_over","dropped_bad_type","dropped_nonfinite"):
            tot[k] += s.get(k, 0)
        kmax_global = max(kmax_global, s.get("K_max", 0))
tot["K_max_global"] = kmax_global

# ---- 2. matched-cell density: Operon vs evogp (cap32 & cap64, seed0, noise0) ----
rows = []          # drift.csv
density = []        # per (problem, cap, gen) both corpora
for cap in (32, 64):
    for prob in PROBLEMS:
        for corpus in ("operon", "evogp"):
            m = load_cell(corpus, prob, cap)
            if not m: continue
            for s in m.get("snapshots", []):
                rows.append(dict(corpus=corpus, problem=prob, cap=cap, gen=s["gen"],
                                 mean_K=s.get("mean_K"), mean_nodes=s.get("mean_nodes"),
                                 K_max=s.get("K_max")))
# pivot density at key gens for the report table
def at(corpus, prob, cap, gen, field):
    for r in rows:
        if r["corpus"]==corpus and r["problem"]==prob and r["cap"]==cap and r["gen"]==gen:
            return r[field]
    return None
for cap in (32, 64):
    for prob in PROBLEMS:
        for gen in (0, 100):
            o_k, e_k = at("operon",prob,cap,gen,"mean_K"), at("evogp",prob,cap,gen,"mean_K")
            o_n, e_n = at("operon",prob,cap,gen,"mean_nodes"), at("evogp",prob,cap,gen,"mean_nodes")
            if o_k is None or e_k is None: continue
            density.append(dict(problem=prob, cap=cap, gen=gen,
                operon_meanK=o_k, evogp_meanK=e_k,
                density_ratio=round(o_k/e_k, 2) if e_k else None,
                operon_meannodes=o_n, evogp_meannodes=e_n))

# aggregate density ratio at gen100 cap64 (the bloated regime)
ratios = [d["density_ratio"] for d in density if d["cap"]==64 and d["gen"]==100 and d["density_ratio"]]
agg = dict(mean_density_ratio_gen100_cap64=round(statistics.mean(ratios),2) if ratios else None,
           n=len(ratios))

OUTD.mkdir(parents=True, exist_ok=True)
with open(OUTD/"drift.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["corpus","problem","cap","gen","mean_K","mean_nodes","K_max"])
    w.writeheader(); [w.writerow(r) for r in rows]
json.dump(dict(corpus_totals=tot, density=density, density_summary=agg),
          open(OUTD/"summary.json","w"), indent=1)

print("=== Operon corpus totals ==="); print(json.dumps(tot, indent=1))
print(f"\n=== density ratio (Operon mean_K / evogp mean_K), gen100 cap64 ===")
print(f"  mean over {agg['n']} problems = {agg['mean_density_ratio_gen100_cap64']}x")
print(f"\n=== sample density (gen100 cap64) ===")
for d in density:
    if d["cap"]==64 and d["gen"]==100:
        print(f"  {d['problem']:15} operon K={d['operon_meanK']:.1f}/{d['operon_meannodes']:.0f}n  "
              f"evogp K={d['evogp_meanK']:.1f}/{d['evogp_meannodes']:.0f}n  ratio {d['density_ratio']}x")
