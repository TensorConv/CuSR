"""plots.py — figures for the Operon pre-CO corpus report. Reads the committed
data/{cross_kernel,drift.csv,summary,mechanism_proof}.json and writes plots/.
No GPU; matplotlib only.
"""
import csv, json, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/home/weish/hao/CuSR/results/operon_corpus__20260617")
ck = json.loads((D/"data/cross_kernel.json").read_text())
ck = [r for r in ck if "fail_chol_pct" in r and r["fail_chol_pct"] is not None]

C = {"operon": "#d1495b", "evogp": "#30638e"}

# ---- 1. fail_cholesky% vs generation (mean over problems, both corpora) ----
byg = {c: defaultdict(list) for c in ("operon","evogp")}
for r in ck: byg[r["corpus"]][r["gen"]].append(r["fail_chol_pct"])
fig, ax = plt.subplots(figsize=(7,4.5))
for c in ("operon","evogp"):
    gs = sorted(byg[c]); mean=[statistics.mean(byg[c][g]) for g in gs]
    lo=[min(byg[c][g]) for g in gs]; hi=[max(byg[c][g]) for g in gs]
    ax.plot(gs, mean, "o-", color=C[c], label=f"{c} (mean over 17 problems)")
    ax.fill_between(gs, lo, hi, color=C[c], alpha=0.15)
ax.set_xlabel("generation"); ax.set_ylabel("fail_cholesky %  (rank-deficient JtJ)")
ax.set_title("LM rank-deficiency vs generation — Operon vs evogp pre-CO populations\n"
             "(same kernel build, cap=32, seed=0; shaded = problem spread)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(D/"plots/fail_vs_gen.png", dpi=130); plt.close(fig)

# ---- 2. fail_cholesky% vs mean_nodes (the "money" conditioning-vs-size plot) ----
fig, ax = plt.subplots(figsize=(7,4.5))
for c in ("operon","evogp"):
    pts=[(r["mean_nodes"], r["fail_chol_pct"]) for r in ck if r["corpus"]==c and r.get("mean_nodes")]
    pts.sort()
    ax.scatter([p[0] for p in pts],[p[1] for p in pts], color=C[c], alpha=0.6, label=c, s=28)
ax.set_xlabel("mean tree size (nodes)"); ax.set_ylabel("fail_cholesky %")
ax.set_title("Rank-deficiency vs tree size — both corpora\n"
             "(each point = one problem×generation; cap=32, seed=0)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(D/"plots/fail_vs_nodes.png", dpi=130); plt.close(fig)

# ---- 3. coefficient density: mean_K vs generation (Operon ~2.8x evogp) ----
rows=list(csv.DictReader(open(D/"data/drift.csv")))
dk={c:defaultdict(list) for c in ("operon","evogp")}
for r in rows:
    if r["cap"]=="64" and r["mean_K"]:
        dk[r["corpus"]][int(r["gen"])].append(float(r["mean_K"]))
fig, ax = plt.subplots(figsize=(7,4.5))
for c in ("operon","evogp"):
    gs=sorted(dk[c]); mean=[statistics.mean(dk[c][g]) for g in gs]
    ax.plot(gs, mean, "o-", color=C[c], label=f"{c} mean_K (cap=64)")
ax.set_xlabel("generation"); ax.set_ylabel("mean #optimizable coefficients K / tree")
ax.set_title("Coefficient density — Operon weights every leaf (K≈#leaves),\n"
             "evogp only explicit constants (K=#const-leaves); cap=64, mean over 17 problems")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(D/"plots/density_vs_gen.png", dpi=130); plt.close(fig)

# ---- 4. K=0 (constant-free) tree fraction: striking structural difference ----
# evogp k0 from kernel k0_skip; operon = 0 by construction. Use gen-0 cap32 cross_kernel.
k0={c:defaultdict(list) for c in ("operon","evogp")}
for r in ck:
    if r.get("M") and r.get("k0_skip") is not None:
        k0[r["corpus"]][r["gen"]].append(100*r["k0_skip"]/r["M"])
fig, ax = plt.subplots(figsize=(7,4.5))
for c in ("operon","evogp"):
    gs=sorted(k0[c]); mean=[statistics.mean(k0[c][g]) for g in gs]
    ax.plot(gs, mean, "o-", color=C[c], label=f"{c}")
ax.set_xlabel("generation"); ax.set_ylabel("K=0 (constant-free) trees %  [skipped by LM]")
ax.set_title("Constant-free trees — Operon has NONE (every leaf is a coefficient);\n"
             "evogp has many bare-variable trees. cap=32, mean over 17 problems")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(D/"plots/k0_fraction.png", dpi=130); plt.close(fig)

print("wrote 4 plots to", D/"plots")
