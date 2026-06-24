#!/usr/bin/env python
"""plot_section2.py — C2 evaluation figures from the locked-clock e7 sweep.

Reads sweep_e6.jsonl (kernel + reused Operon records) + iso_quality.json and emits
the four C2 figures required by tasks/05 plus a parameter-sensitivity table:
  fig_throughput_surface.png  trees/s vs M, per variant (N=1000 slice)
  fig_n_collapse.png          trees/s vs N at fixed M=64000, per variant
  fig_large_m.png             trees/s vs M to 256k (saturation), best variant per preset
  fig_crossover.png           GPU (best iso variant) vs Operon {1,16,64,128}c, trees/s vs M
  sensitivity_table.md        throughput sensitivity to preset/M/N + variant ratios

Pure-data: every point is a record in the JSONL; missing cells are drawn as gaps,
never interpolated. DRAFT vs locked is taken from locked_clock_mhz on the records.

Usage:
  uv run python experiments/e7_section2/plot_section2.py --out experiments/e7_section2/out \
    --figs experiments/e7_section2/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

VARIANTS = ["fusedfd", "ad", "revad"]
PRESETS = ["early-gen", "late-gen-bloated", "inner-const-heavy"]
MS = [1000, 4000, 16000, 64000, 256000]
NS = [100, 1000, 10000]
NCORES = [1, 16, 64, 128]
MARK = {"fusedfd": "o", "ad": "s", "revad": "^"}


def load(out: Path):
    recs = [json.loads(l) for l in (out / "sweep_e6.jsonl").read_text().splitlines()
            if l.strip()]
    kern = {(r["variant"], r["preset"], r["M"], r["knob"]): r for r in recs
            if r.get("backend") == "kernel" and r.get("status") == "ok"}
    oper = {(r["preset"], r["M"], r["N"], r["knob"]): r for r in recs
            if r.get("backend") == "operon" and r.get("status") == "ok"}
    return kern, oper


def ktput(kern, v, p, M, N):
    r = kern.get((v, p, M, N))
    return r["throughput"] if r else None


def kband(kern, v, p, M, N):
    """(median, lo, hi) trees/s over the 3 seeds. lo/hi = min/max (n=3 → honest
    non-parametric range, NOT a Gaussian CI). median == record throughput (verified
    0/120 mismatches). Returns None if the cell is missing/skipped."""
    r = kern.get((v, p, M, N))
    if not r:
        return None
    med = r["throughput"]
    s = r.get("med_tput_seeds")
    if isinstance(s, list) and s:
        return med, min(s), max(s)
    return med, med, med


WHISKER_NOTE = "whiskers: min–max over 3 seeds"


def _series(kern, v, p, MS_or_NS, fixed, axis):
    """Collect (xs, ys, yerr_lo, yerr_hi) for errorbar; axis='M' varies M at N=fixed,
    axis='N' varies N at M=fixed. Missing cells are skipped (drawn as gaps)."""
    xs, ys, lo, hi = [], [], [], []
    for x in MS_or_NS:
        b = kband(kern, v, p, x, fixed) if axis == "M" else kband(kern, v, p, fixed, x)
        if b:
            med, l, h = b
            xs.append(x); ys.append(med); lo.append(med - l); hi.append(h - med)
    return xs, ys, lo, hi


def fig_throughput_surface(kern, figs: Path, locked):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, p in zip(axes, PRESETS):
        for v in VARIANTS:
            xs, ys, lo, hi = _series(kern, v, p, MS, 1000, "M")
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[v] + "-", capsize=3,
                            markersize=5, label=v)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(p); ax.set_xlabel("M (population)"); ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("trees/s (N=1000)")
    fig.suptitle(f"GPU CO throughput vs M (locked {locked} MHz)" if locked
                 else "GPU CO throughput vs M (DRAFT, unlocked)")
    fig.text(0.995, 0.005, WHISKER_NOTE, ha="right", va="bottom", fontsize=7, alpha=.55)
    fig.tight_layout(); fig.savefig(figs / "fig_throughput_surface.png", dpi=130)
    plt.close(fig)


def fig_n_collapse(kern, figs: Path, locked):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, p in zip(axes, PRESETS):
        for v in VARIANTS:
            xs, ys, lo, hi = _series(kern, v, p, NS, 64000, "N")
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[v] + "-", capsize=3,
                            markersize=5, label=v)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(p); ax.set_xlabel("N (points/tree)"); ax.grid(True, alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("trees/s (M=64000)")
    fig.suptitle("trees/s collapses with N (more points per tree)")
    fig.text(0.995, 0.005, WHISKER_NOTE, ha="right", va="bottom", fontsize=7, alpha=.55)
    fig.tight_layout(); fig.savefig(figs / "fig_n_collapse.png", dpi=130)
    plt.close(fig)


def fig_large_m(kern, figs: Path, locked):
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in PRESETS:
        for v in VARIANTS:
            xs, ys, lo, hi = _series(kern, v, p, MS, 1000, "M")
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[v] + "-", alpha=.7,
                            capsize=2, elinewidth=.8, markersize=4, label=f"{p[:8]}/{v}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M"); ax.set_ylabel("trees/s (N=1000)")
    ax.set_title("Large-M saturation (all variants/presets)")
    ax.grid(True, alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.text(0.995, 0.005, WHISKER_NOTE, ha="right", va="bottom", fontsize=7, alpha=.55)
    fig.tight_layout(); fig.savefig(figs / "fig_large_m.png", dpi=130)
    plt.close(fig)


def _crossover_panel(kern, oper, figs: Path, locked, p, N, fname, regime_label):
    """One crossover panel: GPU variant trees/s vs M (with min–max-over-seed whiskers)
    overlaid on Operon {1,16,64,128}c flat envelopes, at preset p / points N."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for v in VARIANTS:
        xs, ys, lo, hi = _series(kern, v, p, MS, N, "M")
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[v] + "-", capsize=3,
                        markersize=5, label=f"GPU {v}")
    for nc in NCORES:
        xs, ys = [], []
        for M in MS:
            r = oper.get((p, M, N, nc))
            if r:
                xs.append(M); ys.append(r["throughput"])
        if xs:
            ax.plot(xs, ys, "--", alpha=.6, label=f"Operon {nc}c")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M"); ax.set_ylabel(f"trees/s ({p}, N={N})")
    ax.set_title(f"GPU vs Operon crossover — {regime_label}\n({p}, N={N})")
    ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.text(0.995, 0.005, "GPU whiskers = min–max/3 seeds; Operon = reused e6 (no per-rep spread)",
             ha="right", va="bottom", fontsize=6.5, alpha=.55)
    fig.tight_layout(); fig.savefig(figs / fname, dpi=130)
    plt.close(fig)


def fig_crossover(kern, oper, figs: Path, iso, locked):
    """TWO regimes, so the figure leads with the defensible number:
      - STABLE  (early-gen N=1000): low sentinel drift, the ~9.6× headline cell.
      - NOISY   (late-gen-bloated N=100): where the 28× max lives but reused Operon
        timing is least reliable — explicitly flagged 'do not lead' (Phase 6 + e6 sentinels)."""
    _crossover_panel(kern, oper, figs, locked, "early-gen", 1000,
                     "fig_crossover_stable.png", "STABLE regime (defensible ~9.6x headline)")
    _crossover_panel(kern, oper, figs, locked, "late-gen-bloated", 100,
                     "fig_crossover.png", "NOISY regime — do NOT lead (Operon timing least reliable)")


def sensitivity_table(kern, figs: Path, out: Path, locked):
    """trees/s sensitivity: per (variant), how throughput moves across M and N at a
    fixed preset (early-gen). The 'selection knob' here is the design choice M/N."""
    lines = [f"# Parameter-sensitivity (trees/s) — {'locked %d MHz' % locked if locked else 'DRAFT'}",
             "", "## throughput vs M (early-gen, N=1000)", "",
             "| variant | " + " | ".join(f"M={M}" for M in MS) + " |",
             "|---|" + "---|" * len(MS)]
    for v in VARIANTS:
        cells = []
        for M in MS:
            t = ktput(kern, v, "early-gen", M, 1000)
            cells.append(f"{t:.0f}" if t else "·")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", "## throughput vs N (early-gen, M=64000)", "",
              "| variant | " + " | ".join(f"N={N}" for N in NS) + " |",
              "|---|" + "---|" * len(NS)]
    for v in VARIANTS:
        cells = []
        for N in NS:
            t = ktput(kern, v, "early-gen", 64000, N)
            cells.append(f"{t:.0f}" if t else "·")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    (out / "sensitivity_table.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/e7_section2/out")
    ap.add_argument("--figs", default="experiments/e7_section2/figs")
    args = ap.parse_args()
    out, figs = Path(args.out).resolve(), Path(args.figs).resolve()
    figs.mkdir(parents=True, exist_ok=True)
    kern, oper = load(out)
    locked = next((r.get("locked_clock_mhz") for r in
                   ([] if not kern else kern.values())), None)
    iso_path = out / "iso_quality.json"
    iso = json.loads(iso_path.read_text()) if iso_path.exists() else {}
    fig_throughput_surface(kern, figs, locked)
    fig_n_collapse(kern, figs, locked)
    fig_large_m(kern, figs, locked)
    fig_crossover(kern, oper, figs, iso, locked)
    sensitivity_table(kern, figs, out, locked)
    print(f"kernel cells={len(kern)} operon cells={len(oper)} locked={locked}")
    print("wrote figs to", figs)


if __name__ == "__main__":
    main()
