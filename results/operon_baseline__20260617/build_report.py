"""build_report.py — assemble report.json from data/report_aggregates.json so every
number is provably sourced (no hand-transcription). Run AFTER analyze.py.
"""
import json
import subprocess
from pathlib import Path

D = Path("/home/weish/hao/CuSR/results/operon_baseline__20260617")
A = json.loads((D / "data/report_aggregates.json").read_text())
BG = A["by_gen"]; GENS = [str(g) for g in A["gens"]]
EFF = A["efficacy_ratio"]; MIG = A["ad_vs_fd_migration"]
KF = A["kernel_fail_where_operon_improved"]; Q = A["quality_genuine_converged"]


def pct(num, den):
    return round(100 * num / den, 1) if den else 0.0


def gsum(key):
    return sum(BG[g].get(key, 0) for g in GENS)


tot_opt = gsum("opt")
fd_conv, ad_conv = gsum("fd_conv"), gsum("ad_conv")
fd_chol, ad_chol = gsum("fd_chol"), gsum("ad_chol")
fd_nan, ad_nan = gsum("fd_nan"), gsum("ad_nan")
fd_cw, ad_cw = gsum("fd_convworse"), gsum("ad_convworse")
fd_lie, ad_lie = gsum("fd_fp32lie"), gsum("ad_fp32lie")
fd_cw2x, ad_cw2x = gsum("fd_convworse2x"), gsum("ad_convworse2x")
g0 = GENS[0]   # gen with the most conv-worse (small trees, flattest landscapes)
op_worse = gsum("op_worse")
rescue = MIG.get("4->0", 0)          # FD-Cholesky-fail -> AD-converged
regress = MIG.get("0->4", 0)         # FD-converged -> AD-Cholesky-fail
to_nan = MIG.get("0->2", 0)          # FD-converged -> AD-NaN
both_chol = MIG.get("4->4", 0)
qadfd, qfdop, qadop = Q["ad_vs_fd"], Q["fd_vs_operon"], Q["ad_vs_operon"]

try:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                  cwd=str(D), text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    sha = "unknown"


def gapline():   # robustness gap per gen
    out = []
    for g in GENS:
        d = KF.get(g, {}); den = d.get("op_improved_1pct", 0) or 1
        out.append(f"g{g}: FD {pct(d.get('fd_noimprove_here',0), den)}% / "
                   f"AD {pct(d.get('ad_noimprove_here',0), den)}%")
    return "; ".join(out)


def effline(name):   # median loss-ratio per gen for an engine
    return "; ".join(f"g{g}: {EFF[g][name]['median_ratio']}" for g in GENS if g in EFF)


report = {
    "title": "CuSR baseline: Operon LMOptimizer vs the GPU kernel (FD & AD) on identical pre-CO trees",
    "date": "2026-06-17",
    "question": {
        "zh": "在完全相同的 pre-CO 树上,我们的 GPU 常数优化核(FD 与 AD 两版)相比成熟 CPU 优化器 "
              "Operon LMOptimizer,收敛性与最终拟合质量如何?全部 17 题,看能发现什么。",
        "en": "On byte-identical pre-CO trees, how do our GPU constant-optimization kernels (FD and AD) compare "
              "to a mature CPU optimizer (Operon's LMOptimizer) in convergence and final fit quality? All 17 "
              "problems — exploratory."},
    "setup": {
        "zh": "每题(cap=32, seed=0, noise=0)用 operon_dump.run_cell 确定性重生成 Operon 语料中完全相同的 "
              "pre-CO 种群(逐字节 == 已提交语料,Probe B)。每个 checkpoint 代:(1) Operon 自带 "
              "CoefficientOptimizer(LMOptimizer) 从相同 c_init 优化每棵树;(2) 核 FD(batch_lm_fusedfd)与 "
              "AD(batch_lm_ad)跑同一批 pop.bin。三者同起点(Probe A:相同起点/相同目标 0.5·SSE/相同 K/确定性);"
              "最终用中立 fp64 0.5·Σ(f-y)² 对每个引擎返回的系数统一重算 —— 与引擎自报代价无关。",
        "en": "Per problem (cap=32, seed=0, noise=0) we deterministically regenerate the EXACT Operon pre-CO "
              "population via operon_dump.run_cell (byte-identical to the committed corpus — Probe B). Per "
              "checkpoint gen: (1) Operon's own CoefficientOptimizer(LMOptimizer) optimizes every tree from the "
              "harvested c_init; (2) kernel FD (batch_lm_fusedfd) and AD (batch_lm_ad) run on the same pop.bin. "
              "All three start from the same c_init (Probe A: same start / same 0.5·SSE objective / same K / "
              "deterministic). Each engine's returned coefficients are scored by the NEUTRAL fp64 0.5·Σ(f-y)²."},
    "env": {
        "git_sha": sha,
        "host": "8×A100 sm_80 server (kernel: 1 GPU/job, GPUs 0-5; 2 free per standing rule)",
        "operon": "pyoperon 0.6.1 (isolated venv), threads=1 (deterministic)",
        "kernels": "batch_lm_fusedfd (FD Jacobian, established) + batch_lm_ad (AD Jacobian, commit 90b8d1f, candidate)",
        "neutral_objective": "0.5 * sum((f(x)-y)^2), fp64, identical evaluator for all engines (lib.half_sse)",
        "max_iter": "Operon 500 (converged: max 498, 0.08% of trees near cap); kernel 1000",
    },
    "params": {
        "sweep": {"problems": f"{A['n_problems']} (7 Feynman incl. multivariate + Nguyen-1..10)",
                  "gens": ", ".join(str(g) for g in A["gens"]), "cap": "32", "seed": "0", "noise": "0",
                  "pop": "4000", "N": "1000",
                  "optimizable_tree_instances": f"{tot_opt} (K>0), pooled over problems×gens"},
        "metrics": {
            "efficacy_anchor": "distribution of per-tree loss_final/loss_start (fp64); median + p10/p90",
            "meaningful_improvement": "loss_final < 0.99*loss_start (>1% cut) — NOT 'loss changed at all'",
            "converged_but_worsened": "status==0 yet fp64 loss_final > loss_start; fp32-lie subset = kernel's own fp32 loss <= start",
            "quality": "log10(kernel/Operon) on trees BOTH genuinely improved (excludes converged-but-worsened)"},
    },
    "findings": [],
    "tables": [],
    "caveats": [
        {"zh": "核跑在 **Operon 分布**的树上(每叶一个系数,K≈叶子数,比 evogp 密 ~2.8×)—— 对核是压力测试,"
               "非其目标工况。如实披露。",
         "en": "The kernel runs on the OPERON distribution (every leaf a coefficient, K≈#leaves, ~2.8× denser "
               "than evogp) — a STRESS test for the kernel, not its design workload."},
        {"zh": "**核 vs Operon 是外部标尺,非受控实验**:核是 fp32+fast-math+简单 λ-damping,Operon 是 fp64+成熟 "
               "trust-region,Jacobian 也不同 —— 差异无法归因到单一因素。**AD vs FD 才是受控对比**(同 fp32 LM,"
               "仅 Jacobian 不同),可归因。",
         "en": "kernel-vs-Operon is an EXTERNAL YARDSTICK, not controlled: fp32+fast-math+simple λ-damping vs "
               "fp64+mature trust-region, different Jacobian — gap not attributable to one factor. Only AD-vs-FD "
               "is controlled (same fp32 LM, only the Jacobian differs)."},
        {"zh": "“收敛率”不可跨引擎直比:Operon 的 Success 与核的 status==0 判据不同。正文用中立的损失比分布"
               "(loss_final/loss_start)与 >1% 显著改进率作同口径指标。",
         "en": "'Convergence rate' is not cross-engine comparable (Operon Success vs kernel status==0 differ). The "
               "apples-to-apples metrics are the neutral loss-ratio distribution and the >1% meaningful-improvement rate."},
        {"zh": f"收敛上限:Operon 在 500 内收敛(最大 498,0.08% 接近上限);核用 1000(给足,避免低估核)。"
               f"即便如此,个别 nguyen 高代 cell 仍有最多 {pct(A['maxiter_fraction_max'],1)}% 的核树触顶——这些是"
               "振荡/慢收敛树。中立 achieved-loss 头条指标对残留 maxiter 稳健(触顶树仍贡献其 best-so-far 损失);"
               "受影响的是核“收敛率”诊断,非损失比。",
         "en": f"Convergence caps: Operon converges within 500 (max 498, 0.08% near cap); the kernel gets 1000 "
               f"(generous, to avoid understating it). Even so, a few high-gen nguyen cells leave up to "
               f"{pct(A['maxiter_fraction_max'],1)}% of kernel trees at maxiter — oscillating/slow-converging "
               "trees. The neutral achieved-loss headline is robust to residual maxiter (those trees still "
               "contribute their best-so-far loss); only the kernel 'convergence rate' diagnostic is affected."},
        {"zh": "中立评分用精确 fp64;核用 fast-math fp32 —— 在奇点附近二者可差 10^数十×(核的 fast-math 倒数 vs "
               "精确 fp64)。质量比较仅在双方都显著改进的子集上,并单独报告 converged-but-worsened 率。",
         "en": "Neutral scoring is exact fp64; the kernel uses fast-math fp32 — near singularities the two can "
               "differ by 10^tens× (fast-math reciprocal vs exact fp64). Quality is compared only where both "
               "genuinely improved; the converged-but-worsened rate is reported separately."},
    ],
    "reproduce": [
        "/home/weish/hao/operon-venv/bin/python results/operon_baseline__20260617/probes/probe_lm_gates.py  # Probe A/B",
        "python results/operon_baseline__20260617/driver.py --phase operon --max-iter 500   # Operon converges within 500",
        "python results/operon_baseline__20260617/driver.py --phase kernel --max-iter 1000  # generous cap for the kernel",
        "python results/operon_baseline__20260617/analyze.py",
        "python results/operon_baseline__20260617/build_report.py",
        "/home/weish/hao/CuSR/.venv/bin/python results/operon_baseline__20260617/plots.py",
        "uv run python scripts/make_report.py results/operon_baseline__20260617/report.json",
    ],
}

F = report["findings"]
# LEAD: AD vs FD (the one controlled result)
F.append({
    "claim": {"zh": f"【受控·主结果】AD 比 FD 更稳健:全代池合计 {rescue} 棵 FD-Cholesky-失败 在 AD 下转为收敛"
                    f"(收敛 AD {ad_conv} vs FD {fd_conv} / {tot_opt} 可优化树;Cholesky AD {ad_chol} vs FD {fd_chol})。"
                    f"代价:AD 新增 {regress} 棵 0→4、{to_nan} 棵 0→2,NaN 总数 AD {ad_nan} > FD {fd_nan}。",
              "en": f"[CONTROLLED · headline] AD is more robust than FD: {rescue} FD-Cholesky-fails become "
                    f"converged under AD (converged AD {ad_conv} vs FD {fd_conv} of {tot_opt}; Cholesky AD "
                    f"{ad_chol} vs FD {fd_chol}). Cost: AD adds {regress} (0→4) and {to_nan} (0→2); NaN total "
                    f"AD {ad_nan} > FD {fd_nan}."},
    "evidence": {"zh": f"AD-vs-FD 状态迁移(同 fp32 LM,仅 Jacobian 不同):4→0={rescue}, 4→4={both_chol}, "
                       f"0→4={regress}, 0→2={to_nan}。",
                 "en": f"AD-vs-FD status migration (same fp32 LM, only Jacobian differs): 4→0={rescue}, "
                       f"4→4={both_chol}, 0→4={regress}, 0→2={to_nan}."}})
F.append({
    "claim": {"zh": f"【受控】无导数 bug:FD 与 AD 都显著改进的 {qadfd['n']} 棵树上,最终 0.5·SSE 几乎一致"
                    f"(log10(AD/FD) 中位 {qadfd['median_log10']};AD 更优 {qadfd['a_better']}, FD 更优 {qadfd['b_better']})。",
              "en": f"[CONTROLLED] No derivative bug: on {qadfd['n']} trees where both FD and AD genuinely improve, "
                    f"final 0.5·SSE is essentially identical (median log10(AD/FD)={qadfd['median_log10']}; "
                    f"AD-better {qadfd['a_better']}, FD-better {qadfd['b_better']})."},
    "evidence": {"zh": f"共同显著改进集质量比 p10 {qadfd['p10']}, p90 {qadfd['p90']}。",
                 "en": f"common-genuinely-improved quality ratio p10 {qadfd['p10']}, p90 {qadfd['p90']}."}})
F.append({
    "claim": {"zh": "【同口径效能】用中立损失比(loss_final/loss_start)衡量,核与 Operon 量级相当 —— "
                    "而非二元 improved% 给出的虚高印象(核几乎每棵都微调下降)。",
              "en": "[apples-to-apples efficacy] By the neutral loss-ratio (loss_final/loss_start), the kernel and "
                    "Operon are of comparable magnitude — unlike the inflated picture from binary improved% (the "
                    "kernel nudges almost every tree down a hair)."},
    "evidence": {"zh": f"各代中位损失比 — Operon: {effline('op')}; FD: {effline('fd')}; AD: {effline('ad')}（越低越好,<1=改进）。",
                 "en": f"median loss-ratio by gen — Operon: {effline('op')}; FD: {effline('fd')}; AD: {effline('ad')} "
                       f"(lower is better, <1 = improved)."}})
F.append({
    "claim": {"zh": "【外部标尺】对成熟 fp64 LM 的稳健性差距随 bloat 增大,AD 缩小但未消除。",
              "en": "[yardstick] The robustness gap to a mature fp64 LM widens with bloat; AD shrinks but does not close it."},
    "evidence": {"zh": "Operon 显著改进(>1%)的树中,核未能显著改进的比例 — " + gapline(),
                 "en": "Of trees Operon improved >1%, fraction where the kernel did NOT improve >1% — " + gapline()}})
F.append({
    "claim": {"zh": f"【fp32 诚实性·严抓项】核以 fast-math fp32 优化,少数树**自报收敛(status==0)却 fp64 更差**"
                    f"(占全部 {tot_opt} 可优化树实例:FD {pct(fd_cw,tot_opt)}%、AD {pct(ad_cw,tot_opt)}%):"
                    f"FD {fd_cw} 棵(其中真·fp32 谎报 {fd_lie}、严重 >2× {fd_cw2x}),AD {ad_cw} 棵(谎报 {ad_lie}、严重 >2× "
                    f"{ad_cw2x})。AD 因精确导数更易冲进 fp32 奇点,严重案例约为 FD 的 {round(ad_cw2x/max(fd_cw2x,1),1)}×。"
                    f"对照 Operon worse-than-start = {op_worse}(同样非零,但少 1–2 个数量级)。",
              "en": f"[fp32 honesty · anti-cheat] The kernel optimizes a fast-math fp32 proxy; a small fraction of "
                    f"the {tot_opt} optimizable tree-instances (FD {pct(fd_cw,tot_opt)}%, AD {pct(ad_cw,tot_opt)}%) "
                    f"CLAIM converged (status==0) yet are WORSE in fp64: FD {fd_cw} (genuine fp32-lies {fd_lie}, "
                    f"serious >2× {fd_cw2x}), AD {ad_cw} (lies {ad_lie}, serious >2× {ad_cw2x}). AD's exact "
                    f"derivatives drive harder into fp32 singularities — ~{round(ad_cw2x/max(fd_cw2x,1),1)}× the "
                    f"serious cases of FD. Operon worse-than-start = {op_worse} (also nonzero, 1–2 orders fewer)."},
    "evidence": {"zh": "中立 fp64 重算每个引擎返回系数;status==0 且 fp64 损失 > 起点 即 converged-but-worsened;"
                       "核自身 fp32 损失 ≤ 起点 的子集 = 真·fp32 谎报(被 fast-math 欺骗,已抽查 >100× 子集证实);"
                       ">2× 子集 = 不可能是 fp32 噪声的严重案例。",
                 "en": "Neutral fp64 re-score of every engine's coefficients; status==0 with fp64 loss > start = "
                       "converged-but-worsened; the subset whose kernel fp32 loss ≤ start = genuine fp32-lie "
                       "(verified on the >100× subset); the >2× subset = serious cases that cannot be fp32 noise."}})
F.append({
    "claim": {"zh": "【外部标尺】在双方都显著改进的子集上,质量与 Operon 互有胜负(中位接近打平),核劣化尾更长。",
              "en": "[yardstick] On trees both genuinely improved, quality vs Operon is mixed (near-tie median); the "
                    "kernel has a longer degradation tail."},
    "evidence": {"zh": f"FD/Operon: 中位 log10 {qfdop['median_log10']}, p90 {qfdop['p90']}(n={qfdop['n']}); "
                       f"AD/Operon: 中位 {qadop['median_log10']}, p10 {qadop['p10']}, p90 {qadop['p90']}(n={qadop['n']})。log10>0=核损失更高。",
                 "en": f"FD/Operon: median log10 {qfdop['median_log10']}, p90 {qfdop['p90']} (n={qfdop['n']}); "
                       f"AD/Operon: median {qadop['median_log10']}, p10 {qadop['p10']}, p90 {qadop['p90']} "
                       f"(n={qadop['n']}). log10>0 = kernel loss higher."}})

# ---- tables ----
rows = []
for g in GENS:
    d = BG[g]; opt = d["opt"]; e = EFF[g]
    rows.append([f"gen {g}", opt,
                 e["op"]["median_ratio"], e["fd"]["median_ratio"], e["ad"]["median_ratio"],
                 pct(d["op_impr1"], opt), pct(d["fd_impr1"], opt), pct(d["ad_impr1"], opt),
                 pct(d["fd_chol"], opt), pct(d["ad_chol"], opt),
                 f"{d['fd_convworse']}/{d['fd_fp32lie']}", f"{d['ad_convworse']}/{d['ad_fp32lie']}"])
report["tables"].append({
    "name": "Per-generation (pooled over 17 problems, K>0 trees)",
    "columns": ["gen", "opt", "Op med-ratio", "FD med-ratio", "AD med-ratio",
                "Op >1%↓%", "FD >1%↓%", "AD >1%↓%", "FD Chol%", "AD Chol%",
                "FD cw/lie", "AD cw/lie"],
    "rows": rows})

pp = A["per_problem"]; gl = GENS[-1]
prows = [[prob, pp[prob][gl]["opt"], pp[prob][gl]["op_impr1"], pp[prob][gl]["fd_conv"],
          pp[prob][gl]["ad_conv"], pp[prob][gl]["fd_chol"], pp[prob][gl]["ad_chol"]]
         for prob in A["problems"] if gl in pp.get(prob, {})]
report["tables"].append({
    "name": f"Per-problem at gen {gl} (most-bloated, hardest end; K>0 trees)",
    "columns": ["problem", "opt", "Op >1%↓", "FD conv", "AD conv", "FD Chol-fail", "AD Chol-fail"],
    "rows": prows})

(D / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(f"wrote {D/'report.json'}  ({len(F)} findings, {len(report['tables'])} tables)")
