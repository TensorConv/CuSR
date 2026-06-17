"""plots.py — figures for the Operon-vs-kernel CO baseline. Reuses analyze.cell_records
(single source of the loader/alignment gate + metric definitions). Run with matplotlib:
    /home/weish/hao/CuSR/.venv/bin/python results/operon_baseline__20260617/plots.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("/home/weish/hao/CuSR/results/operon_baseline__20260617")
sys.path.insert(0, str(D))
from analyze import cell_records, PROBLEMS, GENS, IMPROVE   # noqa: E402

C = {"op": "#2a9d8f", "fd": "#e76f51", "ad": "#264653"}
NAME = {"op": "Operon", "fd": "kernel FD", "ad": "kernel AD"}

pool = {g: defaultdict(list) for g in GENS}
for prob in PROBLEMS:
    for g, r in cell_records(prob):
        opt = r["K"] > 0; s = r["loss_start"]
        pool[g]["s"].append(s[opt])
        for name, lf in (("op", r["op_loss"]), ("fd", r["fd_loss"]), ("ad", r["ad_loss"])):
            pool[g][name].append(lf[opt])
        pool[g]["fd_st"].append(r["fd_status"][opt]); pool[g]["ad_st"].append(r["ad_status"][opt])

def cat(g, k):
    return np.concatenate(pool[g][k]) if pool[g][k] else np.array([])

gens = [g for g in GENS if pool[g]["s"]]

def med_ratio(g, name):
    s = cat(g, "s"); f = cat(g, name); m = (s > 0) & np.isfinite(f)
    return float(np.median(f[m] / s[m])) if m.any() else np.nan

# ---- 1. efficacy anchor: median loss-ratio per engine vs gen (log y) + kernel status ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
for name in ("op", "fd", "ad"):
    ax1.plot(gens, [med_ratio(g, name) for g in gens], "o-", color=C[name], label=NAME[name])
ax1.set_yscale("log"); ax1.set_xlabel("generation")
ax1.set_ylabel("median  loss_final / loss_start   (fp64; lower = deeper fit)")
ax1.set_title("CO efficacy (POOLED median over 17 problems)\n"
              "per-problem the ordering flips — see the per-problem table in the report")
ax1.legend(); ax1.grid(alpha=0.3, which="both")
for eng in ("fd", "ad"):
    conv = [100 * (cat(g, f"{eng}_st") == 0).mean() for g in gens]
    chol = [100 * (cat(g, f"{eng}_st") == 4).mean() for g in gens]
    ax2.plot(gens, conv, "o-", color=C[eng], label=f"{eng.upper()} converged")
    ax2.plot(gens, chol, "s--", color=C[eng], label=f"{eng.upper()} Cholesky-fail", alpha=0.7)
ax2.set_xlabel("generation"); ax2.set_ylabel("% of optimizable (K>0) trees")
ax2.set_title("Kernel LM status vs generation (FD vs AD — controlled)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(D / "plots/efficacy_and_status.png", dpi=130); plt.close(fig)

# ---- 2. robustness gap PER PROBLEM (review #2: pooled "widens" is a 6/17 artifact) ----
# COMBINED (fd+ad)/2 gap per problem — SAME definition as analyze.py so the widen/narrow
# count matches finding #4 (6/11). Per-problem lines (red=widen, grey=narrow) + pooled bold.
def cell_gap(r):   # combined (FD+AD) gap fraction for one cell
    opt = r["K"] > 0; s = r["loss_start"]
    opi = opt & (s > 0) & np.isfinite(r["op_loss"]) & (r["op_loss"] < s * IMPROVE)
    den = int(opi.sum()) or 1
    fdg = int((opi & ~(np.isfinite(r["fd_loss"]) & (r["fd_loss"] < s * IMPROVE))).sum())
    adg = int((opi & ~(np.isfinite(r["ad_loss"]) & (r["ad_loss"] < s * IMPROVE))).sum())
    return 100 * (fdg + adg) / (2 * den)
per_prob_gap = defaultdict(dict)
for prob in PROBLEMS:
    for g, r in cell_records(prob):
        per_prob_gap[prob][g] = cell_gap(r)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
widen = narrow = 0; pooled = defaultdict(lambda: [0, 0])
for prob, gd in per_prob_gap.items():
    xs = [g for g in gens if g in gd]; ys = [gd[g] for g in xs]
    if len(xs) >= 2:
        w = ys[-1] > ys[0] + 1e-9; widen += int(w); narrow += int(ys[-1] < ys[0] - 1e-9)
        ax.plot(xs, ys, "-", color=("#e76f51" if w else "#9aa7b0"), alpha=0.5, lw=1)
    for g in xs:
        pooled[g][0] += gd[g]; pooled[g][1] += 1
ax.plot(gens, [pooled[g][0] / pooled[g][1] for g in gens], "o-", color="#111", lw=2.4,
        label="mean over 17 problems")
ax.set_xlabel("generation")
ax.set_ylabel("% of Operon->>1%-improved trees the kernel (FD+AD avg) did NOT improve >1%")
ax.set_title(f"Robustness gap to Operon — PER PROBLEM (combined FD+AD)\n"
             f"{widen}/17 widen (red), {narrow}/17 narrow (grey) g0→g100; the pooled mean hides the split")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(D / "plots/robustness_gap.png", dpi=130); plt.close(fig)

# ---- 3. achieved-loss ratio CDF (kernel vs Operon) on trees BOTH improved >1% ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, eng in zip(axes, ("fd", "ad")):
    for g in (0, 16, 100):
        if g not in gens: continue
        s = cat(g, "s"); op = cat(g, "op"); k = cat(g, eng)
        m = (s > 0) & np.isfinite(op) & np.isfinite(k) & (op > 0) & (k > 0) \
            & (op < s * IMPROVE) & (k < s * IMPROVE)
        if m.sum() == 0: continue
        rr = np.sort(np.log10(k[m] / op[m])); ys = np.linspace(0, 1, len(rr))
        ax.plot(rr, ys, label=f"gen {g} (n={len(rr)})")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(f"log10( {eng.upper()} loss / Operon loss )   (<0: kernel lower)")
    ax.set_title(f"kernel {eng.upper()} vs Operon — achieved 0.5·SSE (both improved >1%; CDF)")
    ax.set_xlim(-3, 3); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axes[0].set_ylabel("cumulative fraction of trees")
fig.tight_layout(); fig.savefig(D / "plots/loss_ratio_cdf.png", dpi=130); plt.close(fig)

# ---- 4. AD-vs-FD status migration heatmap (controlled, K>0, all gens) ----
labels = {0: "conv", 1: "maxit", 2: "NaN", 3: "K=0", 4: "Chol"}; codes = [0, 1, 2, 4]
M = np.zeros((len(codes), len(codes)))
for g in gens:
    fs = cat(g, "fd_st"); as_ = cat(g, "ad_st")
    for i, fc in enumerate(codes):
        for j, ac in enumerate(codes):
            M[i, j] += ((fs == fc) & (as_ == ac)).sum()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.log10(M + 1), cmap="viridis")
ax.set_xticks(range(len(codes))); ax.set_xticklabels([labels[c] for c in codes])
ax.set_yticks(range(len(codes))); ax.set_yticklabels([labels[c] for c in codes])
ax.set_xlabel("AD status"); ax.set_ylabel("FD status")
ax.set_title("AD-vs-FD status migration (controlled, K>0, all gens)\noff-diagonal = changed outcome; 4→0 = AD rescues FD Cholesky-fail")
for i in range(len(codes)):
    for j in range(len(codes)):
        ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center",
                color="w" if M[i, j] < M.max() / 2 else "k", fontsize=9)
fig.colorbar(im, label="log10(#trees+1)")
fig.tight_layout(); fig.savefig(D / "plots/ad_fd_migration.png", dpi=130); plt.close(fig)
print("wrote 4 plots to", D / "plots")
