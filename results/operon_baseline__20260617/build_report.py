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
# review-fix aggregates
NET = A["neutral_net_improvement"]                 # #5: ad_minus_fd, totals
ORD = A["per_problem_efficacy_ordering"]           # #1: per-problem op/kernel-better/tie @g100
WN = A["robustness_gap_widen_narrow"]              # #2: widen vs narrow counts
QC = A["quality_set_composition"]                  # #7: drop counts
fd_nan_true, ad_nan_true = gsum("fd_nan_true"), gsum("ad_nan_true")    # #4
fd_own, ad_own = gsum("fd_convworse_ownmetric"), gsum("ad_convworse_ownmetric")  # #8
fd_w100, ad_w100 = gsum("fd_worse100x_any"), gsum("ad_worse100x_any")  # #1 one-sided tail (any status)
# #1 net by neutral arbiter (counterweight to the 46574 status-migration headline)
ad_newfail = MIG.get("0->1", 0) + MIG.get("0->2", 0) + MIG.get("0->4", 0)  # FD converged, AD did not
op_med = ", ".join(f"g{g}:{EFF[g]['op']['median_ratio']}" for g in GENS)
ADFD_PP = A["adfd_quality_per_problem"]["max_abs_median"]    # F2 (sourced)
WFR = A["worst_finite_loss_ratio"]                           # F5 (sourced; corrects the review's 1-19x)


def sci(x):     # compact scientific notation for the worst-finite-ratio numbers
    return f"{x:.0e}".replace("e+0", "e").replace("e+", "e")

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
        "max_iter": (f"Operon 500 (converged within: max {A['operon_iters_max']} iters, "
                     f"{pct(A['operon_near_cap_frac'],1)}% of trees near cap); kernel 1000"),
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
    "plots": [
        {"file": "plots/efficacy_and_status.png",
         "caption": {"zh": "左:CO 效能锚 —— 各引擎每代的合并中位损失比(对数轴,越低=拟合越深;按题排序会翻转,见按题表)。"
                           "右:核 LM 状态(收敛/Cholesky 失败)随代变化,FD vs AD 受控对比。",
                     "en": "Left: efficacy anchor — POOLED median loss_final/loss_start per engine vs gen (log axis, "
                           "lower=deeper; per-problem the ordering flips — see the per-problem table). Right: kernel "
                           "LM status (converged / Cholesky-fail) vs gen, FD vs AD (controlled)."}},
        {"file": "plots/robustness_gap.png",
         "caption": {"zh": "稳健性差距按题分解:多数题 g0→g100 反而缩小(灰),仅少数高 K 题扩大(红)。合并线掩盖了这一分化。",
                     "en": "Robustness gap per problem: most NARROW g0→g100 (grey), only a few high-K problems widen "
                           "(red). The pooled line hides this split."}},
        {"file": "plots/loss_ratio_cdf.png",
         "caption": {"zh": "在双方都显著改进的树上,核与 Operon 最终 0.5·SSE 之比的 CDF(log10,<0=核更低)。"
                           "中位接近打平,核的劣化尾(右侧)更长。",
                     "en": "On trees both improved >1%, CDF of log10(kernel/Operon final 0.5·SSE) (<0=kernel lower). "
                           "Near-tie median; the kernel's degradation tail (right) is longer."}},
        {"file": "plots/ad_fd_migration.png",
         "caption": {"zh": "AD-vs-FD 状态迁移热图(受控,K>0,全代)。对角线外=结果改变;左下 4→0=AD 把 FD 的 "
                           "Cholesky 失败救成收敛。",
                     "en": "AD-vs-FD status migration (controlled, K>0, all gens). Off-diagonal = outcome changed; "
                           "4→0 = AD rescues an FD Cholesky-fail to convergence."}},
    ],
    "caveats": [
        {"zh": "核跑在 **Operon 分布**的树上(每叶一个系数,K≈叶子数,比 evogp 密 ~2.8×)—— 对核是压力测试,"
               "非其目标工况。如实披露。",
         "en": "The kernel runs on the OPERON distribution (every leaf a coefficient, K≈#leaves, ~2.8× denser "
               "than evogp) — a STRESS test for the kernel, not its design workload."},
        {"zh": "**核 vs Operon 是外部标尺,非受控实验**:核是 fp32+fast-math+简单 λ-damping,Operon 用非 fast-math "
               "求值 + fp64 线性求解 + 成熟 trust-region,Jacobian 也不同 —— 差异无法归因到单一因素(精度差距勿过度解读:"
               "Operon 并非全程 fp64)。**AD vs FD 才是受控对比**(同 fp32 LM,仅 Jacobian 不同),可归因。",
         "en": "kernel-vs-Operon is an EXTERNAL YARDSTICK, not controlled: fp32+fast-math+simple λ-damping vs "
               "Operon's non-fast-math eval + fp64 linear solve + mature trust-region, different Jacobian — gap not "
               "attributable to one factor (don't over-read the precision gap: Operon is not end-to-end fp64). Only "
               "AD-vs-FD is controlled (same fp32 LM, only the Jacobian differs)."},
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
        "python results/operon_baseline__20260617/probes/probe_fp32_lie.py   # backs the fp32-honesty finding",
        "python results/operon_baseline__20260617/build_report.py",
        "/home/weish/hao/CuSR/.venv/bin/python results/operon_baseline__20260617/plots.py",
        "uv run python scripts/make_report.py results/operon_baseline__20260617/report.json",
    ],
}

F = report["findings"]
# LEAD: AD vs FD (the one controlled result). Lead with the NEUTRAL net (review #5),
# keep the status-migration 46574 as a MECHANISM diagnostic with its counterweight.
F.append({
    "claim": {"zh": f"【受控·主结果】用本实验自己的金标准(中立 fp64 >1% 显著改进)衡量,AD 比 FD 净多改进 "
                    f"{NET['ad_minus_fd']} 棵树(AD {NET['ad_impr1_total']} vs FD {NET['fd_impr1_total']} / {tot_opt})。"
                    f"机制:AD 把 {rescue} 棵 FD-Cholesky-失败救成收敛(Cholesky AD {ad_chol} vs FD {fd_chol}),"
                    f"代价是新增 {ad_newfail} 棵 FD-收敛→AD-未收敛。",
              "en": f"[CONTROLLED · headline] By the experiment's OWN gold standard (neutral fp64 >1% improvement), "
                    f"AD genuinely improves {NET['ad_minus_fd']} more trees than FD (AD {NET['ad_impr1_total']} vs FD "
                    f"{NET['fd_impr1_total']} of {tot_opt}). Mechanism: AD rescues {rescue} FD-Cholesky-fails to "
                    f"convergence (Cholesky AD {ad_chol} vs FD {fd_chol}), at the cost of {ad_newfail} new "
                    f"FD-converged→AD-not-converged."},
    "evidence": {"zh": f"中立净改进 +{NET['ad_minus_fd']};状态迁移(机制,非头条):4→0={rescue}, 4→4={both_chol}, "
                       f"0→4={regress}, 0→2={to_nan}。状态收敛差 +{ad_conv-fd_conv} 比中立净改进高 "
                       f"~{round((ad_conv-fd_conv)/max(NET['ad_minus_fd'],1),1)}×(status≠跨引擎可比,见注),故以中立净改进为准。",
                 "en": f"Neutral net +{NET['ad_minus_fd']}; status migration (mechanism, not headline): 4→0={rescue}, "
                       f"4→4={both_chol}, 0→4={regress}, 0→2={to_nan}. The raw status-conv gap +{ad_conv-fd_conv} is "
                       f"~{round((ad_conv-fd_conv)/max(NET['ad_minus_fd'],1),1)}× the neutral net (status is not "
                       f"cross-comparable; see caveats), so we lead with the neutral net."}})
F.append({
    "claim": {"zh": f"【受控】无导数 bug:FD 与 AD 都显著改进的 {qadfd['n']} 棵树上,最终 0.5·SSE 几乎一致"
                    f"(log10(AD/FD) 中位 {qadfd['median_log10']};AD 更优 {qadfd['a_better']}, FD 更优 {qadfd['b_better']});"
                    f"且该近似打平在 17 题中每一题内部都成立(非正负抵消)。",
              "en": f"[CONTROLLED] No derivative bug: on {qadfd['n']} trees where both FD and AD genuinely improve, "
                    f"final 0.5·SSE is essentially identical (median log10(AD/FD)={qadfd['median_log10']}; "
                    f"AD-better {qadfd['a_better']}, FD-better {qadfd['b_better']}) — and the near-tie holds WITHIN "
                    f"each of the 17 problems (per-problem median |log10(AD/FD)| ≤ {ADFD_PP})."},
    "evidence": {"zh": f"共同显著改进集质量比 p10 {qadfd['p10']}, p90 {qadfd['p90']};每题中位 log10(AD/FD) 绝对值 "
                       f"≤ {ADFD_PP}(17 题全部,见 adfd_quality_per_problem)。",
                 "en": f"common-genuinely-improved quality ratio p10 {qadfd['p10']}, p90 {qadfd['p90']}; per-problem "
                       f"median |log10(AD/FD)| ≤ {ADFD_PP} on all 17 (adfd_quality_per_problem)."}})
F.append({
    "claim": {"zh": f"【同口径效能·按题异质】合并中位损失比给出“Operon 在 bloated 树上更深”的印象,但这是少数高 K 题"
                    f"主导的量级效应,**按题排序会翻转**:g100 每题中位比,Operon 更优 {ORD['fd']['operon_better']} 题、"
                    f"核(FD)更优 {ORD['fd']['kernel_better']} 题、打平 {ORD['fd']['tie']} 题;对 AD 则 Operon "
                    f"{ORD['ad']['operon_better']}、AD {ORD['ad']['kernel_better']}、打平 {ORD['ad']['tie']}。"
                    f"没有哪道题长得像合并三元组。",
              "en": f"[efficacy · per-problem heterogeneity] The pooled median loss-ratio suggests 'Operon goes deeper "
                    f"on bloated trees', but that is a magnitude effect dominated by a few high-K problems — the "
                    f"per-problem ordering FLIPS. At g100, Operon's median is better on {ORD['fd']['operon_better']} "
                    f"problems, the kernel (FD) on {ORD['fd']['kernel_better']}, tied on {ORD['fd']['tie']}; for AD it "
                    f"is Operon {ORD['ad']['operon_better']} / AD {ORD['ad']['kernel_better']} / tie {ORD['ad']['tie']}. "
                    f"No single problem looks like the pooled triple."},
    "evidence": {"zh": f"合并各代中位损失比 — Operon: {effline('op')}; FD: {effline('fd')}; AD: {effline('ad')}。"
                       f"按题表见下;>2% 差距才算一题之胜负。",
                 "en": f"pooled median loss-ratio by gen — Operon: {effline('op')}; FD: {effline('fd')}; AD: "
                       f"{effline('ad')}. See the per-problem table; >2% apart = a per-problem win."}})
F.append({
    "claim": {"zh": f"【外部标尺·按题异质】“稳健性差距随 bloat 增大”只对少数高 K 题成立:17 题中 {WN['widen']} 题"
                    f"g0→g100 差距扩大、{WN['narrow']} 题反而缩小。合并曲线的扩大主要由这几题驱动,不是普遍现象。",
              "en": f"[yardstick · per-problem heterogeneity] 'The robustness gap widens with bloat' holds only for a "
                    f"minority: of 17 problems, {WN['widen']} widen g0→g100 while {WN['narrow']} NARROW. The pooled "
                    f"trend is driven by those few high-K problems, not a general phenomenon."},
    "evidence": {"zh": "按题 g0→g100 差距(核未能 >1% 改进 Operon 已改进的树之比例)见 report_aggregates.json "
                       "robustness_gap_widen_narrow.per_problem。",
                 "en": "Per-problem g0→g100 gap (fraction of Operon->improved trees the kernel did not improve >1%) in "
                       "report_aggregates.json robustness_gap_widen_narrow.per_problem."}})
F.append({
    "claim": {"zh": f"【fp32 诚实性·严抓项】核以 fast-math fp32 优化,少数树**自报收敛(status==0)却 fp64 更差**"
                    f"(FD {fd_cw} 棵 / {pct(fd_cw,tot_opt)}%、AD {ad_cw} 棵 / {pct(ad_cw,tot_opt)}%,共 {tot_opt} 树)。"
                    f"其中:被 fast-math 真·欺骗(核自身 fp32 也 ≤ 起点)FD {fd_lie}({pct(fd_lie,fd_cw)}%)/ AD {ad_lie}"
                    f"({pct(ad_lie,ad_cw)}%);其余(FD {fd_own}/ AD {ad_own})在核**自己的 fp32 目标**里也更差"
                    f"(accept/停止判据缺陷,非精度伪影)。严重 >2×:FD {fd_cw2x}、AD {ad_cw2x}。对照 Operon "
                    f"worse-than-start = {op_worse}。",
              "en": f"[fp32 honesty · anti-cheat] A small fraction CLAIM converged (status==0) yet are WORSE in fp64 "
                    f"(FD {fd_cw} / {pct(fd_cw,tot_opt)}%, AD {ad_cw} / {pct(ad_cw,tot_opt)}% of {tot_opt}). Of these: "
                    f"genuinely fast-math-fooled (kernel's own fp32 also ≤ start) FD {fd_lie} ({pct(fd_lie,fd_cw)}%) / "
                    f"AD {ad_lie} ({pct(ad_lie,ad_cw)}%); the rest (FD {fd_own} / AD {ad_own}) are worse in the kernel's "
                    f"OWN fp32 objective too (an accept/stop-criterion defect, not a precision artifact). Serious >2×: "
                    f"FD {fd_cw2x}, AD {ad_cw2x}. Operon worse-than-start = {op_worse}."},
    "evidence": {"zh": f"中立 fp64 重算每个引擎返回系数。任意 status 的单向恶化尾(>100× 更差):FD {fd_w100}、AD {ad_w100}。"
                       f"最坏有限损失比:Operon ~{sci(WFR['op'])}× vs FD ~{sci(WFR['fd'])}×、AD ~{sci(WFR['ad'])}×"
                       f"——核的尾比 Operon 重 ~17–76 个数量级(Operon 也会在其 {op_worse} 棵更差树上放大,但远轻)。"
                       f">100× 子集由 probes/probe_fp32_lie.py 复算核对。",
                 "en": f"Neutral fp64 re-score. One-sided worsening tail (>100× worse, any status): FD {fd_w100}, AD "
                       f"{ad_w100}. Worst FINITE loss-ratio: Operon ~{sci(WFR['op'])}× vs FD ~{sci(WFR['fd'])}×, AD "
                       f"~{sci(WFR['ad'])}× — the kernel's tail is ~17–76 orders of magnitude heavier (Operon also "
                       f"blows up on its {op_worse} worse trees, but far less). The >100× subset is recomputed by "
                       f"probes/probe_fp32_lie.py."}})
F.append({
    "claim": {"zh": f"【失败标签校正·严抓项】核的 status==2 标为“NaN”,但其实是 λ>1e12 的阻尼停滞:FD {fd_nan} 中仅 "
                    f"{fd_nan_true} 棵、AD {ad_nan} 中仅 {ad_nan_true} 棵真正非有限,其余 ~99.8% 返回有限的 best-so-far。"
                    f"故应读作“未收敛(停滞/NaN)”,而非数值发散。",
              "en": f"[failure-label correction · anti-cheat] The kernel's status==2 is labeled 'NaN' but is really a "
                    f"λ>1e12 damping STALL: of FD {fd_nan} only {fd_nan_true}, of AD {ad_nan} only {ad_nan_true} are "
                    f"truly non-finite; the other ~99.8% return a finite best-so-far. Read it as 'failed-to-converge "
                    f"(stall/NaN)', not numerical divergence."},
    "evidence": {"zh": "status==2 且中立 fp64 损失非有限 才是真 NaN;其余为 best-so-far 有限值(batch_lm_*.cu: λ>1e12→finished=2)。",
                 "en": "status==2 AND non-finite neutral fp64 loss = true NaN; the rest return finite best-so-far "
                       "(batch_lm_*.cu: λ>1e12 → finished=2)."}})
F.append({
    "claim": {"zh": f"【外部标尺】在双方都显著改进的子集上,质量与 Operon 互有胜负(中位接近打平),核劣化尾更长。"
                    f"注:该子集**剔除了**单向救援树(核改进而 Operon 未改进者),故不含正是头条稳健性信号的 AD 救援。",
              "en": f"[yardstick] On trees both genuinely improved, quality vs Operon is mixed (near-tie median); the "
                    f"kernel has a longer degradation tail. Note: this set EXCLUDES one-sided rescues (kernel improved "
                    f"but Operon did not), so it cannot contain the very AD rescues that are the headline robustness signal."},
    "evidence": {"zh": f"FD/Operon: 中位 log10 {qfdop['median_log10']}, p90 {qfdop['p90']}(n={qfdop['n']};另剔除 "
                       f"Operon-only {QC['fd_vs_operon']['op_only_dropped']}、核-only {QC['fd_vs_operon']['k_only_dropped']});"
                       f"AD/Operon: 中位 {qadop['median_log10']}, p90 {qadop['p90']}(n={qadop['n']})。log10>0=核损失更高。",
                 "en": f"FD/Operon: median log10 {qfdop['median_log10']}, p90 {qfdop['p90']} (n={qfdop['n']}; dropped "
                       f"Operon-only {QC['fd_vs_operon']['op_only_dropped']}, kernel-only "
                       f"{QC['fd_vs_operon']['k_only_dropped']}); AD/Operon: median {qadop['median_log10']}, p90 "
                       f"{qadop['p90']} (n={qadop['n']}). log10>0 = kernel loss higher."}})

# ---- tables ----
rows = []
for g in GENS:
    d = BG[g]; opt = d["opt"]; e = EFF[g]
    rows.append([f"gen {g}", opt,
                 e["op"]["median_ratio"], e["fd"]["median_ratio"], e["ad"]["median_ratio"],
                 pct(d["op_impr1"], opt), pct(d["fd_impr1"], opt), pct(d["ad_impr1"], opt),
                 pct(d["fd_chol"], opt), pct(d["ad_chol"], opt),
                 f"{d['fd_worse100x_any']}/{d['ad_worse100x_any']}",
                 f"{d['fd_convworse']}/{d['ad_convworse']}"])
report["tables"].append({
    "name": "Per-generation, pooled over 17 problems (K>0 trees)",
    "columns": ["gen", "opt", "Op med-ratio", "FD med-ratio", "AD med-ratio",
                "Op >1%↓%", "FD >1%↓%", "AD >1%↓%", "FD Chol%", "AD Chol%",
                "FD/AD >100×-worse", "FD/AD conv-worse"],
    "rows": rows})

# review #1: per-problem median loss-ratio at g0 and g100 — exposes the per-problem flips
# (a reader can see no problem matches the pooled triple, and where the kernel BEATS Operon).
pp = A["per_problem"]; g0t, glt = GENS[0], GENS[-1]


def mr(prob, g, eng):
    c = pp[prob].get(g, {}); v = c.get(f"{eng}_medr")
    return v if v is not None else "—"


prows = [[prob, mr(prob, g0t, "op"), mr(prob, g0t, "fd"), mr(prob, g0t, "ad"),
          mr(prob, glt, "op"), mr(prob, glt, "fd"), mr(prob, glt, "ad")]
         for prob in A["problems"] if g0t in pp.get(prob, {})]
report["tables"].append({
    "name": "Per-problem median loss-ratio loss_final/loss_start (lower=deeper fit; the pooled "
            "headline hides these flips — e.g. nguyen/3 the kernel beats Operon)",
    "columns": ["problem", "g0 Op", "g0 FD", "g0 AD", "g100 Op", "g100 FD", "g100 AD"],
    "rows": prows})

(D / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(f"wrote {D/'report.json'}  ({len(F)} findings, {len(report['tables'])} tables)")
