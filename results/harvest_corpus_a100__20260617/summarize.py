"""summarize.py — Phase 1 harvest corpus summary (workload characterization).

Reads every pop=4000 snapshot manifest, aggregates the K/size drift across
generation x problem x cap x noise, and (with --kernel-gpu G) runs
batch_lm_fusedfd on a representative sample to profile convergence/failure vs
generation on REAL trees. Emits report.json (bilingual) + CSV + plots; render
with scripts/make_report.py.

  uv run python results/harvest_corpus_a100__20260617/summarize.py --kernel-gpu 5
"""
import argparse, csv, glob, json, os, re, statistics, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = ROOT / "data" / "workload" / "snapshots"
KERNEL = ROOT / "cusr" / "kernel" / "batch_lm_fusedfd"
GENS = [0, 1, 2, 4, 8, 16, 32, 64, 100]
DONE_RE = re.compile(
    r"converged=(\d+) \(([\d.]+)%\).*?maxiter=(\d+).*?fail_nan=(\d+).*?"
    r"fail_cholesky=(\d+).*?k0_skip=(\d+)")


def load_rows():
    rows = []
    for mf in sorted(SNAP.glob("*pop4000_noise*/manifest.json")):  # harvest naming; excludes smoke
        m = json.loads(mf.read_text())
        for s in m["snapshots"]:
            rows.append(dict(dataset=m["dataset"], noise=float(m["noise"]),
                             cap=int(m["max_tree_len"]), seed=int(m["seed"]), gen=int(s["gen"]),
                             mean_K=s["mean_K"], mean_nodes=s["mean_nodes"], K_max=s["K_max"],
                             max_nodes=s["max_nodes"], total_c=s["total_c"], M=s["M"],
                             n_kover=s["n_kover"]))
    return rows


def mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def kernel_profile(gpu):
    probs = ["feynman/I.18.12", "feynman/I.6.2", "nguyen/5", "feynman/I.13.12"]
    gens = [0, 4, 16, 64, 100]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs("/tmp/cusr_ksample", exist_ok=True)
    prof = []
    for ds in probs:
        d = SNAP / f"{ds.replace('/', '_')}_pop4000_noise0_len64_seed0"
        for g in gens:
            f = d / f"pop_gen{g:04d}.bin"
            if not f.exists():
                continue
            p = subprocess.run([str(KERNEL), str(f), "/tmp/cusr_ksample"],
                               capture_output=True, text=True, env=env)
            mt = DONE_RE.search(p.stdout)
            if mt:
                conv, pct, maxit, fnan, fchol, k0 = mt.groups()
                prof.append(dict(dataset=ds, gen=g, conv_pct=float(pct),
                                 fail_chol=int(fchol), fail_nan=int(fnan),
                                 k0=int(k0), maxiter=int(maxit), M=4000))
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-gpu", type=int, default=-1, help="GPU for the kernel sample; -1 skips")
    args = ap.parse_args()
    rows = load_rows()
    datasets = sorted({r["dataset"] for r in rows})
    (HERE / "data").mkdir(exist_ok=True); (HERE / "plots").mkdir(exist_ok=True)
    print(f"loaded {len(rows)} snapshot records from {len(datasets)} problems")

    # ---- aggregate drift: mean over (problem,seed,noise) per (cap, gen) ----
    drift = {}
    for cap in (32, 64):
        drift[cap] = {}
        for g in GENS:
            sub = [r for r in rows if r["cap"] == cap and r["gen"] == g]
            drift[cap][g] = dict(mean_K=mean([r["mean_K"] for r in sub]),
                                 mean_nodes=mean([r["mean_nodes"] for r in sub]),
                                 K_max=max([r["K_max"] for r in sub], default=0),
                                 max_nodes=max([r["max_nodes"] for r in sub], default=0))
    # corpus-wide K-over (cap=64 high-K trees exceeding kernel MAX_K=32)
    kover_total = sum(r["n_kover"] for r in rows)

    # ---- per-problem mean_K at gen 100 (cap64, noise0, avg seeds) ----
    per_prob = {}
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds and r["cap"] == 64
               and r["noise"] == 0.0 and r["gen"] == 100]
        per_prob[ds] = dict(mean_K=mean([r["mean_K"] for r in sub]),
                            mean_nodes=mean([r["mean_nodes"] for r in sub]),
                            K_max=max([r["K_max"] for r in sub], default=0))

    # ---- CSV ----
    with open(HERE / "data" / "drift.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cap", "gen", "mean_K", "mean_nodes", "K_max", "max_nodes"])
        for cap in (32, 64):
            for g in GENS:
                d = drift[cap][g]
                w.writerow([cap, g, f"{d['mean_K']:.3f}", f"{d['mean_nodes']:.3f}", d["K_max"], d["max_nodes"]])

    # ---- plot 1: drift (mean_K & mean_nodes vs gen, cap32 vs cap64) ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for cap, c in [(32, "#0969da"), (64, "#d1495b")]:
        a1.plot(GENS, [drift[cap][g]["mean_K"] for g in GENS], "o-", color=c, label=f"max_tree_len={cap}")
        a2.plot(GENS, [drift[cap][g]["mean_nodes"] for g in GENS], "o-", color=c, label=f"max_tree_len={cap}")
    for a, t in [(a1, "mean K (constants/tree)"), (a2, "mean nodes/tree")]:
        a.set_xscale("symlog"); a.set_xlabel("generation"); a.set_ylabel(t); a.grid(alpha=.3); a.legend()
    fig.suptitle("Real GP workload drift across generations (corpus avg over 17 problems x 3 seeds x noise)")
    fig.tight_layout(); fig.savefig(HERE / "plots" / "workload_drift.png", dpi=130)

    # ---- plot 2: per-problem mean_K @ gen100 ----
    items = sorted(per_prob.items(), key=lambda kv: kv[1]["mean_K"])
    fig, ax = plt.subplots(figsize=(8.4, 6))
    ax.barh([k.replace("feynman/", "F:").replace("nguyen/", "N") for k, _ in items],
            [v["mean_K"] for _, v in items], color="#66a182")
    ax.set_xlabel("mean K (constants/tree) @ gen 100, cap=64"); ax.grid(alpha=.3, axis="x")
    ax.set_title("Per-problem constant load at gen 100 — corpus variety")
    fig.tight_layout(); fig.savefig(HERE / "plots" / "per_problem_K.png", dpi=130)

    # ---- optional kernel convergence/failure profile ----
    prof = []
    if args.kernel_gpu >= 0 and KERNEL.exists():
        prof = kernel_profile(args.kernel_gpu)
        (HERE / "data" / "kernel_profile.json").write_text(json.dumps(prof, indent=2))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
        for ds in sorted({p["dataset"] for p in prof}):
            ps = [p for p in prof if p["dataset"] == ds]
            gs = [p["gen"] for p in ps]
            a1.plot(gs, [p["conv_pct"] for p in ps], "o-", label=ds.replace("feynman/", "F:").replace("nguyen/", "N"))
            a2.plot(gs, [100 * p["fail_chol"] / p["M"] for p in ps], "o-", label=ds)
        a1.set_ylabel("converged %"); a2.set_ylabel("fail_cholesky % (rank-deficient)")
        for a in (a1, a2):
            a.set_xscale("symlog"); a.set_xlabel("generation"); a.grid(alpha=.3)
        a1.legend(fontsize=8)
        fig.suptitle("Kernel behaviour on real trees vs generation (cap=64, noiseless, seed0)")
        fig.tight_layout(); fig.savefig(HERE / "plots" / "kernel_vs_gen.png", dpi=130)

    # ---- console summary ----
    print("\ncap  gen  mean_K  mean_nodes  K_max  max_nodes")
    for cap in (32, 64):
        for g in (0, 8, 32, 100):
            d = drift[cap][g]
            print(f"{cap:>3} {g:>4} {d['mean_K']:>7.2f} {d['mean_nodes']:>11.2f} {d['K_max']:>6} {d['max_nodes']:>10}")
    print(f"\ncorpus K-over (trees with K>32 skipped): {kover_total}")
    print(f"per-problem mean_K @gen100 range: {min(v['mean_K'] for v in per_prob.values()):.2f} "
          f"- {max(v['mean_K'] for v in per_prob.values()):.2f}")
    if prof:
        print("\nkernel sample (conv% / fail_chol%):")
        for ds in sorted({p['dataset'] for p in prof}):
            ps = [p for p in prof if p['dataset'] == ds]
            print(f"  {ds}: " + "  ".join(f"g{p['gen']}={p['conv_pct']:.0f}%/{100*p['fail_chol']/p['M']:.0f}%" for p in ps))

    json.dump(dict(drift=drift, per_problem=per_prob, kover_total=kover_total, prof=prof),
              open(HERE / "data" / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
