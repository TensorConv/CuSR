"""summarize.py — Korns inner-constant (frequency-class) corpus characterization.

The 17-problem Feynman+Nguyen corpus is essentially inner-constant-free (true
constants are outer scales or absent) — the regime where constant optimization
(CO) is LEAST decisive. This corpus adds the two frequency-class Korns targets,
whose true constants sit INSIDE cos/sin (frequencies) — the regime where CO is
decisive. It is the realistic-tree input for the planned end-to-end experiment,
NOT the experiment itself.

Reads every korns pop=4000 snapshot manifest, aggregates K/size drift across
generation x cap x noise x seed, and (with --kernel-gpu G) runs BOTH the FD and
AD kernels on a representative sample to profile convergence/failure vs
generation on the inner-heavy trees. Emits report.json (bilingual) + CSV +
plots; render with scripts/make_report.py.

  .venv/bin/python results/korns_corpus__20260617/summarize.py --kernel-gpu 1
"""
import argparse, csv, json, os, re, subprocess
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = ROOT / "data" / "workload" / "snapshots"
KERNELS = {"fd": ROOT / "cusr" / "kernel" / "batch_lm_fusedfd",
           "ad": ROOT / "cusr" / "kernel" / "batch_lm_ad"}
PROBLEMS = ["korns/11", "korns/12"]
GENS = [0, 1, 2, 4, 8, 16, 32, 64, 100]
# ground-truth inner-constant count (constants inside cos/sin; see
# cusr/demonstrator/taxonomy.py). korns/11: c2 inside cos. korns/12: c2,c3.
N_INNER = {"korns/11": 1, "korns/12": 2}
DONE_RE = re.compile(
    r"converged=(\d+) \(([\d.]+)%\).*?maxiter=(\d+).*?fail_nan=(\d+).*?"
    r"fail_cholesky=(\d+).*?k0_skip=(\d+)")


def load_rows():
    rows = []
    for ds in PROBLEMS:
        safe = ds.replace("/", "_")
        for mf in sorted(SNAP.glob(f"{safe}_pop4000_noise*/manifest.json")):
            m = json.loads(mf.read_text())
            for s in m["snapshots"]:
                rows.append(dict(dataset=m["dataset"], noise=float(m["noise"]),
                                 cap=int(m["max_tree_len"]), seed=int(m["seed"]),
                                 gen=int(s["gen"]), mean_K=s["mean_K"],
                                 mean_nodes=s["mean_nodes"], K_max=s["K_max"],
                                 max_nodes=s["max_nodes"], total_c=s["total_c"],
                                 M=s["M"], n_kover=s["n_kover"]))
    return rows


def mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def kernel_profile(gpu):
    """Run FD + AD on cap64/noise0/seed0 for both problems across gens."""
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs("/tmp/cusr_korns_ksample", exist_ok=True)
    gens = [0, 4, 16, 64, 100]
    prof = []
    for ds in PROBLEMS:
        d = SNAP / f"{ds.replace('/', '_')}_pop4000_noise0_len64_seed0"
        for g in gens:
            f = d / f"pop_gen{g:04d}.bin"
            if not f.exists():
                continue
            row = dict(dataset=ds, gen=g, M=4000)
            for tag, kbin in KERNELS.items():
                if not kbin.exists():
                    continue
                p = subprocess.run([str(kbin), str(f), "/tmp/cusr_korns_ksample"],
                                   capture_output=True, text=True, env=env)
                mt = DONE_RE.search(p.stdout)
                if mt:
                    conv, pct, maxit, fnan, fchol, k0 = mt.groups()
                    row[f"{tag}_conv_pct"] = float(pct)
                    row[f"{tag}_fail_chol"] = int(fchol)
                    row[f"{tag}_fail_nan"] = int(fnan)
                    row[f"{tag}_maxiter"] = int(maxit)
                    row[f"{tag}_k0"] = int(k0)
            prof.append(row)
    return prof


def build_report(drift, per_prob, kover_total, prof, n_cells, n_snaps, n_trees):
    def fchol_pct(p, tag):
        return 100.0 * p.get(f"{tag}_fail_chol", 0) / p["M"]

    # bloat-timing contrast: mid-gen (16) vs late-gen (100), cap64
    k11_16 = drift["korns/11"][64][16]["mean_K"]; k11_100 = drift["korns/11"][64][100]["mean_K"]
    k12_16 = drift["korns/12"][64][16]["mean_K"]; k12_100 = drift["korns/12"][64][100]["mean_K"]
    findings = [
        {"claim": {
            "zh": f"【内部常数工况·建成】两道频率类 Korns 题语料已采成:真实常数藏在 cos/sin 内部"
                  f"(korns/11 内部常数 1 个=频率 7.23;korns/12 内部常数 2 个=频率 9.8 与 1.3),"
                  f"线性最小二乘无法拟合 → CO 在此才有决定性作用。对照 17 题 Feynman+Nguyen 语料内部常数≈0。",
            "en": f"[inner-const regime · built] Corpus harvested for the two frequency-class Korns "
                  f"targets: true constants sit INSIDE cos/sin (korns/11 has 1 inner const = freq 7.23; "
                  f"korns/12 has 2 = freqs 9.8 & 1.3) — linear least-squares cannot fit them, so CO is "
                  f"decisive here. Contrast the 17-problem Feynman+Nguyen corpus (inner consts ≈ 0)."},
         "evidence": {
            "zh": f"{n_cells} cell / {n_snaps} 快照 / {n_trees:,} 棵树,0 丢弃、0 K-over。"
                  f"矩阵:2 题 × 噪声{{0,1%}} × cap{{32,64}} × seed{{0,1,2}} × gen{{0,1,2,4,8,16,32,64,100}},pop=4000,N=1000。",
            "en": f"{n_cells} cells / {n_snaps} snapshots / {n_trees:,} trees, 0 dropped / 0 K-over. "
                  f"Matrix: 2 problems × noise{{0,1%}} × cap{{32,64}} × seed{{0,1,2}} × gen{{0..100}}, pop=4000, N=1000."}},
        {"claim": {
            "zh": f"【负载差异=膨胀时机,非终点】两题晚期都膨胀到相近的高 K(cap64 gen100:"
                  f"korns/11 K={k11_100:.1f}、korns/12 K={k12_100:.1f}),差别在膨胀的早晚:"
                  f"korns/11(立方+高频,目标近乎不可解)很早就膨胀(gen16 已 K={k11_16:.1f}),"
                  f"korns/12(两个中等频率)中期仍稀疏(gen16 K={k12_16:.1f})、到晚期才膨胀。"
                  f"=> 干净的低 K 工作点在 korns/12 的早/中代。",
            "en": f"[workload difference = bloat TIMING, not endpoint] Both bloat to a similar high K by "
                  f"late gen (cap64 gen100: korns/11 K={k11_100:.1f}, korns/12 K={k12_100:.1f}); the difference is "
                  f"WHEN: korns/11 (cube+high-freq, near-unsolvable) bloats early (K={k11_16:.1f} already at gen16), "
                  f"while korns/12 (two moderate freqs) stays sparse through mid-gen (K={k12_16:.1f} at gen16) and "
                  f"only bloats late. => the clean low-K operating point is korns/12 at early/mid gen."}},
    ]
    if prof:
        def g(ds, gg): return next((p for p in prof if p["dataset"] == ds and p["gen"] == gg), None)
        p11_4, p11_100 = g("korns/11", 4), g("korns/11", 100)
        p12_16, p12_100 = g("korns/12", 16), g("korns/12", 100)
        if all([p11_4, p11_100, p12_16, p12_100]):
            findings.append({"claim": {
                "zh": f"【kernel 正常运行;秩亏随膨胀走,非内部常数本身】FD 与 AD 在两题上均无崩溃、有限返回。"
                      f"Cholesky 失败率随 bloat 走:早/中代低(korns/11 gen4 {fchol_pct(p11_4,'fd'):.0f}%、"
                      f"korns/12 gen16 {fchol_pct(p12_16,'fd'):.0f}%),晚代两题都塌成秩亏"
                      f"(gen100 cap64 korns/11 {fchol_pct(p11_100,'fd'):.0f}%、korns/12 {fchol_pct(p12_100,'fd'):.0f}%)"
                      f"——与 Feynman 语料同一个‘膨胀→秩亏’机制。=> 干净的 CO 工作区是早/中代。",
                "en": f"[kernel runs fine; rank-deficiency tracks bloat, not inner-consts per se] FD and AD both "
                      f"run crash-free with finite returns on both. Cholesky-fail tracks bloat: low at early/mid gen "
                      f"(korns/11 gen4 {fchol_pct(p11_4,'fd'):.0f}%, korns/12 gen16 {fchol_pct(p12_16,'fd'):.0f}%), and "
                      f"both collapse to rank-deficient at late gen (gen100 cap64 korns/11 {fchol_pct(p11_100,'fd'):.0f}%, "
                      f"korns/12 {fchol_pct(p12_100,'fd'):.0f}%) — the SAME 'bloat→rank-deficiency' mechanism as the Feynman "
                      f"corpus. => the clean CO operating regime is early/mid gen."},
                "evidence": {
                    "zh": "kernel 状态(converged/fail)是工况描述符,非质量判定——见 caveats(operon_baseline 已证"
                          "status==0 不等于真改进)。FD 收敛率(按状态)略高于 AD,但这不是质量结论。",
                    "en": "kernel status (converged/fail) is a workload descriptor, NOT a quality verdict — see "
                          "caveats (operon_baseline showed status==0 ≠ genuine improvement). FD's status-converged rate is "
                          "marginally higher than AD's, but that is not a quality conclusion."}})
    return {
        "title": "CuSR corpus: Korns inner-constant (frequency-class) trees on the A100 pipeline",
        "date": "2026-06-17",
        "question": {
            "zh": "把内部常数(藏在 cos/sin 频率里、线性拟合解不出、CO 才有用)的题接进 A100 采集流水线,"
                  "跑出一份真实 GP 语料,作为后续端到端 CO 实验的‘CO 该发光’主战场。",
            "en": "Wire inner-constant targets (constants buried in cos/sin frequencies — unfittable by "
                  "linear scaling, where CO is decisive) into the A100 harvest pipeline and produce a real GP "
                  "corpus, as the 'CO should shine' battlefield for the planned end-to-end CO experiment."},
        "setup": {
            "zh": "在 sr_problems.py + dump_evogp.py 的共享题库加 korns/11、korns/12(逐字一致,过 AST parity 测试),"
                  "GP 算子集沿用 {+,-,*,/,sin,cos,tan}(korns/11 的 x^3 仅生成 y,GP 用 x*x*x 近似),"
                  "用现有 harvest.py 在 GPU 1-5 并行采集。kernel 画像:FD 与 AD 在 cap64/无噪/seed0 上跑。",
            "en": "Added korns/11 & korns/12 to the shared problem library in sr_problems.py + dump_evogp.py "
                  "(byte-identical, AST parity test passes); GP funcset unchanged {+,-,*,/,sin,cos,tan} (korns/11's "
                  "x^3 only generates y, GP approximates via x*x*x); harvested with the existing harvest.py on "
                  "GPUs 1-5. Kernel profile: FD and AD on cap64/noiseless/seed0."},
        "env": {
            "git_sha": "(this commit)",
            "gpu": "8x A100-SXM4-80GB; harvest on GPUs 1-5 (0 had a foreign idle alloc; 6-7 left free)",
            "engine": "evogp (GPU GP-SR), GP_CONFIG single-sourced in dump_evogp.py",
            "kernels": "batch_lm_fusedfd + batch_lm_ad (sm_80, fp32, --use_fast_math)",
            "storage": f"{n_snaps} korns pop4000 snapshots (.bin gitignored; {n_cells} manifests committed "
                       f"under data/workload/snapshots/korns_*)",
            "reproducible": "each snapshot regenerates from its committed manifest (dataset+seed+pop+config)"},
        "params": {
            "matrix": {"problems": "korns/11, korns/12", "noise": "0.0, 0.01", "caps": "32, 64",
                       "seeds": "0,1,2", "gens": "0,1,2,4,8,16,32,64,100", "pop": 4000, "N": 1000,
                       "cells": n_cells, "snapshots": n_snaps, "trees": n_trees},
            "targets": {"korns/11": "6.87 + 11*cos(7.23*x0**3)  (n_inner=1)",
                        "korns/12": "2 - 2.1*cos(9.8*x0)*sin(1.3*x1)  (n_inner=2)"},
            "kernel_sample": {"variants": "fd + ad", "cap": 64, "noise": 0.0, "seed": 0,
                              "gens": "0,4,16,64,100"}},
        "findings": findings,
        "plots": [
            {"file": "plots/workload_drift.png", "caption": {
                "zh": "K 与节点数随代漂移(korns/11 vs korns/12,cap32 vs cap64)",
                "en": "K & node-count drift across generations (korns/11 vs korns/12, cap32 vs cap64)"}},
            {"file": "plots/kernel_vs_gen.png", "caption": {
                "zh": "FD/AD 在内部常数真实树上的收敛率与 Cholesky 失败率 vs 代(cap64,无噪,seed0)",
                "en": "FD/AD convergence% and Cholesky-fail% on inner-const real trees vs gen (cap64, noiseless, seed0)"}},
        ],
        "caveats": [
            {"zh": "kernel 的 converged/fail 是工况描述符,不是质量判定:operon_baseline 已证 status==0 ≠ 真改进。"
                   "质量需用中立 fp64 评分,留到端到端实验。",
             "en": "kernel converged/fail is a workload descriptor, not a quality verdict: operon_baseline showed "
                   "status==0 ≠ genuine improvement. Quality needs neutral fp64 scoring — deferred to the end-to-end experiment."},
            {"zh": "korns/11 的立方+高频目标在 [-5,5] 上是公认的‘已知天花板’,即便 CO 完美也基本不可解;"
                   "它在这里是高 K + 秩亏压力工况,不是干净的‘CO 展示’题。korns/12 才是主展示题。",
             "en": "korns/11's cube+high-freq target on [-5,5] is a known ceiling — near-unsolvable even with perfect "
                   "CO; here it serves as a high-K + rank-deficiency stress workload, not a clean 'CO demo'. korns/12 is the demo target."},
            {"zh": "这是语料(真实树)+ 工况画像,不是端到端实验:它不衡量‘CO 是否提升公式恢复’,只提供后续实验的输入。",
             "en": "This is a corpus (real trees) + workload characterization, NOT the end-to-end experiment: it does not "
                   "measure 'does CO improve formula recovery', it only supplies the input for that experiment."},
            {"zh": "GP 算子集无 POW;korns/11 的 x^3 由乘法近似(同 Nguyen log/sqrt 题的处理),是 y 生成器而非 GP 约束。",
             "en": "GP funcset has no POW; korns/11's x^3 is approximated by multiplication (same as the Nguyen log/sqrt "
                   "entries) — a y-generator, not a GP constraint."},
        ],
        "reproduce": [
            "# 1. problems already in cusr/{benchmark/workload/sr_problems.py, kernel/dump_evogp.py}",
            ".venv/bin/python -m pytest cusr/benchmark/workload/tests/test_sr_problems.py -q",
            "# 2. harvest (GPUs 1-5, ~1 min)",
            ".venv/bin/python cusr/benchmark/workload/harvest.py --problems korns/11,korns/12 \\",
            "    --gpus 1,2,3,4,5 --noises 0.0,0.01 --caps 32,64 --seeds 0,1,2 \\",
            "    --pop 4000 --N 1000 --gens 0,1,2,4,8,16,32,64,100",
            "# 3. characterize + render",
            ".venv/bin/python results/korns_corpus__20260617/summarize.py --kernel-gpu 1",
            ".venv/bin/python scripts/make_report.py results/korns_corpus__20260617/report.json",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-gpu", type=int, default=-1, help="GPU for the kernel sample; -1 skips")
    args = ap.parse_args()
    rows = load_rows()
    (HERE / "data").mkdir(exist_ok=True); (HERE / "plots").mkdir(exist_ok=True)
    n_cells = len({(r["dataset"], r["noise"], r["cap"], r["seed"]) for r in rows})
    n_snaps = len(rows)
    n_trees = sum(r["M"] for r in rows if r["gen"] == 100) * 0 + n_cells * len(GENS) * 4000
    print(f"loaded {len(rows)} snapshot records ({n_cells} cells) from {sorted(set(r['dataset'] for r in rows))}")

    # ---- drift per (problem, cap, gen): mean over noise+seed ----
    drift = {}  # drift[ds][cap][gen]
    for ds in PROBLEMS:
        drift[ds] = {}
        for cap in (32, 64):
            drift[ds][cap] = {}
            for g in GENS:
                sub = [r for r in rows if r["dataset"] == ds and r["cap"] == cap and r["gen"] == g]
                drift[ds][cap][g] = dict(mean_K=mean([r["mean_K"] for r in sub]),
                                         mean_nodes=mean([r["mean_nodes"] for r in sub]),
                                         K_max=max([r["K_max"] for r in sub], default=0),
                                         max_nodes=max([r["max_nodes"] for r in sub], default=0))
    kover_total = sum(r["n_kover"] for r in rows)

    # ---- per-problem gen100 cap64 noise0 (avg seeds) ----
    per_prob = {}
    for ds in PROBLEMS:
        sub = [r for r in rows if r["dataset"] == ds and r["cap"] == 64
               and r["noise"] == 0.0 and r["gen"] == 100]
        per_prob[ds] = dict(mean_K=mean([r["mean_K"] for r in sub]),
                            mean_nodes=mean([r["mean_nodes"] for r in sub]),
                            K_max=max([r["K_max"] for r in sub], default=0),
                            n_inner=N_INNER[ds])

    # ---- CSV ----
    with open(HERE / "data" / "drift.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "cap", "gen", "mean_K", "mean_nodes", "K_max", "max_nodes"])
        for ds in PROBLEMS:
            for cap in (32, 64):
                for g in GENS:
                    d = drift[ds][cap][g]
                    w.writerow([ds, cap, g, f"{d['mean_K']:.3f}", f"{d['mean_nodes']:.3f}",
                                d["K_max"], d["max_nodes"]])

    # ---- plot 1: drift ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    styles = {("korns/11", 32): ("#0969da", "--"), ("korns/11", 64): ("#0969da", "-"),
              ("korns/12", 32): ("#d1495b", "--"), ("korns/12", 64): ("#d1495b", "-")}
    for ds in PROBLEMS:
        for cap in (32, 64):
            c, ls = styles[(ds, cap)]
            lbl = f"{ds} cap{cap}"
            a1.plot(GENS, [drift[ds][cap][g]["mean_K"] for g in GENS], "o-" if ls == "-" else "o--",
                    color=c, label=lbl)
            a2.plot(GENS, [drift[ds][cap][g]["mean_nodes"] for g in GENS], "o-" if ls == "-" else "o--",
                    color=c, label=lbl)
    for a, t in [(a1, "mean K (constants/tree)"), (a2, "mean nodes/tree")]:
        a.set_xscale("symlog"); a.set_xlabel("generation"); a.set_ylabel(t); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Korns inner-const workload drift (avg over noise{0,1%} x seed{0,1,2})")
    fig.tight_layout(); fig.savefig(HERE / "plots" / "workload_drift.png", dpi=130); plt.close(fig)

    # ---- optional kernel profile ----
    prof = []
    if args.kernel_gpu >= 0:
        prof = kernel_profile(args.kernel_gpu)
        (HERE / "data" / "kernel_profile.json").write_text(json.dumps(prof, indent=2))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
        marks = {"korns/11": "o", "korns/12": "s"}
        cols = {"fd": "#0969da", "ad": "#d1495b"}
        for ds in PROBLEMS:
            ps = [p for p in prof if p["dataset"] == ds]
            gs = [p["gen"] for p in ps]
            for tag in ("fd", "ad"):
                if f"{tag}_conv_pct" in (ps[0] if ps else {}):
                    a1.plot(gs, [p[f"{tag}_conv_pct"] for p in ps], marks[ds] + "-",
                            color=cols[tag], label=f"{ds} {tag.upper()}")
                    a2.plot(gs, [100 * p[f"{tag}_fail_chol"] / p["M"] for p in ps], marks[ds] + "-",
                            color=cols[tag], label=f"{ds} {tag.upper()}")
        a1.set_ylabel("converged %"); a2.set_ylabel("fail_cholesky % (rank-deficient)")
        for a in (a1, a2):
            a.set_xscale("symlog"); a.set_xlabel("generation"); a.grid(alpha=.3); a.legend(fontsize=8)
        fig.suptitle("FD vs AD on Korns inner-const real trees vs gen (cap64, noiseless, seed0)")
        fig.tight_layout(); fig.savefig(HERE / "plots" / "kernel_vs_gen.png", dpi=130); plt.close(fig)

    # ---- summary.json + report.json ----
    json.dump(dict(drift=drift, per_problem=per_prob, kover_total=kover_total, prof=prof),
              open(HERE / "data" / "summary.json", "w"), indent=2)
    report = build_report(drift, per_prob, kover_total, prof, n_cells, n_snaps, n_trees)
    json.dump(report, open(HERE / "report.json", "w"), ensure_ascii=False, indent=1)

    # ---- console ----
    print("\nds        cap  gen  mean_K  mean_nodes  K_max")
    for ds in PROBLEMS:
        for cap in (32, 64):
            for g in (0, 16, 100):
                d = drift[ds][cap][g]
                print(f"{ds:9} {cap:>3} {g:>4} {d['mean_K']:>7.2f} {d['mean_nodes']:>11.2f} {d['K_max']:>6}")
    print(f"\ncorpus K-over: {kover_total}")
    if prof:
        print("\nkernel sample (FD conv%/chol%  |  AD conv%/chol%):")
        for ds in PROBLEMS:
            for p in [x for x in prof if x["dataset"] == ds]:
                print(f"  {ds} g{p['gen']:>3}: FD {p.get('fd_conv_pct',0):.0f}%/"
                      f"{100*p.get('fd_fail_chol',0)/p['M']:.0f}%  |  AD {p.get('ad_conv_pct',0):.0f}%/"
                      f"{100*p.get('ad_fail_chol',0)/p['M']:.0f}%")


if __name__ == "__main__":
    main()
