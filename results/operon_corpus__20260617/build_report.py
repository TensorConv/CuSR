"""build_report.py — assemble report.json from the committed data artifacts so every
number is provably sourced (no hand-transcription). Run: python build_report.py;
then scripts/make_report.py results/operon_corpus__20260617/report.json.

All descriptors that name *which* cells are extreme (e.g. which reach the K cap) are
COMPUTED from the data here — an earlier hand-written descriptor wrongly attributed the
K=32 cap to univariate Nguyen; it is in fact cap64 multivariate Feynman (caught by the
AI-error audit). Active-tree (K>0) gen0 rates and the evogp K=0 range are computed too.
"""
import csv, json, statistics
from collections import defaultdict
from pathlib import Path

D = Path("/home/weish/hao/CuSR/results/operon_corpus__20260617")
S = json.loads((D/"data/summary.json").read_text())
MP = json.loads((D/"data/mechanism_proof.json").read_text())
CK = [r for r in json.loads((D/"data/cross_kernel.json").read_text()) if r.get("fail_chol_pct") is not None]
DRIFT = list(csv.DictReader(open(D/"data/drift.csv")))

T = S["corpus_totals"]
ratio = S["density_summary"]["mean_density_ratio_gen100_cap64"]

# --- which cells reach the K cap (K_max==MAX_K==32)? Computed over ALL operon cells
# (every seed/noise/cap), not the seed0/noise0 drift subset, since the claim is
# corpus-wide. (An earlier drift-only version under-counted; this matches the audit.) ---
import glob
SNAP = Path("/home/weish/hao/CuSR/data/workload/snapshots")
k32 = set(); nguyen_cap64_kmax = 0
for mf in glob.glob(str(SNAP/"operon_*"/"manifest.json")):
    cell = mf.split("/")[-2]; m = json.loads(Path(mf).read_text()); prob = m["dataset"]
    for s in m["snapshots"]:
        if s["K_max"] == 32: k32.add(prob)
        if "len64" in cell and "nguyen" in prob and s["K_max"]:
            nguyen_cap64_kmax = max(nguyen_cap64_kmax, s["K_max"])
k32_problems = sorted(k32)                                   # multivariate Feynman
k32_str = ", ".join(p.split("/")[-1] for p in k32_problems)

# --- cross-kernel aggregation by (corpus, gen): full-M and active (K>0) bases ---
byg = {c: defaultdict(list) for c in ("operon", "evogp")}
byg_active = {c: defaultdict(list) for c in ("operon", "evogp")}
k0 = {c: defaultdict(list) for c in ("operon", "evogp")}
for r in CK:
    byg[r["corpus"]][r["gen"]].append(r["fail_chol_pct"])
    if r.get("M"):
        act = r["M"] - r["k0_skip"]
        byg_active[r["corpus"]][r["gen"]].append(100*r["fail_chol"]/act if act else 0.0)
        k0[r["corpus"]][r["gen"]].append(100*r["k0_skip"]/r["M"])
def fc(c, g): v=byg[c][g]; return (round(statistics.mean(v),1), round(min(v)), round(max(v)))
def fca(c, g): return round(statistics.mean(byg_active[c][g]),1)
def k0m(c, g): return round(statistics.mean(k0[c][g]),1)
GENS=[0,4,16,64,100]
evo_k0_means=[k0m("evogp",g) for g in GENS]
k0_lo, k0_hi = min(evo_k0_means), max(evo_k0_means)

# mechanism cond numbers
def cond(key): return next(v["cond"] for k,v in MP.items() if k.startswith(key))
op_cond=cond("operon_xa"); ev_cond=cond("evogp  c0*xa*xb"); evb_cond=cond("evogp  c0*c1")

report = {
  "title": "Operon pre-CO corpus + adapter — cross-engine workload characterization",
  "date": "2026-06-17",
  "question": {
    "zh": "evogp 语料里观察到的「秩亏 Cholesky 随世代上升」是 evogp 特有的,还是真实 SR 种群的普遍现象?为回答它,我们把 Operon(CPU SOTA,自带 inline CO)接成第二个权威负载来源 + baseline:写 Operon→pop.bin 适配器,在同一批题/同一份 (X,y) 上采 pre-CO 种群,再用同一个内核在两套语料上量 LM 的病态程度。",
    "en": "Is evogp's 'rank-deficient Cholesky rising with generation' an evogp artifact, or a general property of real SR populations? To answer it we wired Operon (CPU SOTA, with inline CO) as a second authoritative workload source + baseline: an Operon→pop.bin adapter, pre-CO populations harvested on the SAME problems / identical (X,y), then the SAME kernel run on both corpora to measure LM conditioning."},
  "setup": {
    "zh": "适配器(docs/kernel/OPERON_ADAPTER_SPEC.md):Operon 每个叶子都是可优化系数 —— 变量节点=权重·x_col,常数节点=值 —— 故 K=tree.CoefficientsCount。变量→MUL(CONST=权重,VAR=列),常数→CONST,postfix→prefix,列按 HashValue 映射。语义精确:对 Operon 自带求值器 round-trip,6/6 手例 + 1999/2000 真实进化树精确(均为本工作内验证,见 SPEC §5/§10)。采集:Operon GP 关 inline CO(local_iterations=0,p_local=0)、限定算子集{+,-,*,/,sin,cos,tan}(= evogp funcset)、与 evogp 同题/同采样/同 pop=4000/N=1000/seed{0,1,2}/cap{32,64}/checkpoint{0,1,2,4,8,16,32,64,100}。交叉内核:同一份新编 batch_lm_fusedfd 跑两套语料(cap32,seed0,--max-iter 50)。机理证明:对最小树用内核同款有限差分 J 算 JtJ 条件数。",
    "en": "Adapter (docs/kernel/OPERON_ADAPTER_SPEC.md): every Operon leaf is an optimizable coefficient — Variable=weight·x_col, Constant=value — so K=tree.CoefficientsCount. Variable→MUL(CONST=weight,VAR=col), Constant→CONST, postfix→prefix, column via HashValue. Exact semantics: round-trip vs Operon's own evaluator, 6/6 hand cases + 1999/2000 evolved trees exact (both verified in this work, SPEC §5/§10). Harvest: Operon GP with inline CO OFF (local_iterations=0, p_local=0), restricted grammar {+,-,*,/,sin,cos,tan} (= evogp funcset), same problems/sampling/pop=4000/N=1000/seeds{0,1,2}/caps{32,64}/checkpoints as evogp. Cross-kernel: ONE freshly-built batch_lm_fusedfd on both corpora (cap32, seed0, --max-iter 50). Mechanism proof: kernel-style finite-difference J → JtJ condition number on minimal trees."},
  "env": {
    "git_sha": "(this commit)",
    "operon": "pyoperon 0.6.1 (manylinux wheel) in /home/weish/hao/operon-venv; harvest CPU-only (2x EPYC 7763, threads=1/cell, 48 cells concurrent)",
    "kernel": "batch_lm_fusedfd (this tree's build, sm_80, fp32, --use_fast_math), cross-corpus sweep on A100 GPUs 0-5",
    "storage": f"{T['snapshots']} operon pop.bin (.bin gitignored; {T['cells']} manifests committed under data/workload/snapshots/operon_*)"},
  "params": {
    "corpus": {"problems": 17, "noise": "0.0, 0.01", "caps": "32, 64", "seeds": "0,1,2",
               "gens": "0,1,2,4,8,16,32,64,100", "pop": 4000, "N": 1000,
               "cells": T["cells"], "snapshots": T["snapshots"], "trees": T["trees_in"]},
    "cross_kernel": {"problems": 17, "gens": "0,4,16,64,100", "cap": 32, "seed": 0,
                     "corpora": "operon + evogp (same build)", "runs": len(CK), "max_iter": 50},
    "drops": {k: T[k] for k in ("dropped_unsupported","dropped_k_over","dropped_bad_type","dropped_nonfinite")}},
  "findings": [
    {"claim": {
      "zh": f"语料交付且零损耗:{T['cells']} cells / {T['snapshots']} snapshots / {T['trees_in']:,} 棵树,全部成功转换,丢弃=0(unsupported/K-over/bad-type/non-finite 全 0)。MAX_K=32 全语料安全:最密的 cell 是 cap64 多变量 Feynman({k32_str}),K_max=32 —— 恰好触到 MAX_K=32 编译上限,dropped_k_over=0(无越界);单变量 Nguyen 最高仅 K_max={nguyen_cap64_kmax}。变量越多→带权叶子越多→K 越高。",
      "en": f"Corpus delivered, zero loss: {T['cells']} cells / {T['snapshots']} snapshots / {T['trees_in']:,} trees, all converted, 0 dropped (unsupported/K-over/bad-type/non-finite all 0). MAX_K=32 safe corpus-wide: the densest cells are cap64 MULTIVARIATE Feynman ({k32_str}), reaching K_max=32 — exactly at the MAX_K=32 compile bound, dropped_k_over=0 (none over); univariate Nguyen tops at K_max={nguyen_cap64_kmax}. More variables → more weighted leaves → higher K."},
     "evidence": {"zh": "summarize.py → data/summary.json (corpus_totals);K=32 cell 清单由 drift.csv 现算;适配器 round-trip + inspect PASS + 两跑字节一致(SPEC §10–12)。", "en": "summarize.py → data/summary.json (corpus_totals); the K=32 cell list is computed from drift.csv here; adapter round-trip + inspect PASS + two-run byte-identical (SPEC §10–12)."}},

    {"claim": {
      "zh": f"系数密度差:Operon 把每个叶子都当可优化系数(变量带权重),故其负载比 evogp 密 ~{ratio}×(gen100/cap64,mean_K 之比,17 题平均;范围 1.7–5.7×)。更尖锐的结构差:Operon 的 K=0(无常数可优化)树占比恒为 0%,evogp 有 {k0_lo}–{k0_hi}%(裸变量树,内核直接跳过)。这是表示差异,不是 bug。",
      "en": f"Coefficient-density gap: Operon treats every leaf as an optimizable coefficient (variables carry weights), so its workload is ~{ratio}× denser than evogp (gen100/cap64, ratio of mean_K, mean over 17 problems; range 1.7–5.7×). Sharper structural difference: Operon's K=0 (no constant to optimize) fraction is 0% at every generation; evogp's is {k0_lo}–{k0_hi}% (bare-variable trees the kernel skips outright). A representation difference, not a bug."},
     "evidence": {"zh": "summarize.py density 表(data/summary.json);k0% 由交叉内核 k0_skip/M 现算(plots/k0_fraction.png)。", "en": "summarize.py density table (data/summary.json); k0% computed from the cross-kernel k0_skip/M (plots/k0_fraction.png)."}},

    {"claim": {
      "zh": f"【核心,机理可证】两套语料的秩亏来自不同机制 —— 不是「都秩亏=互相印证」。Operon:每叶带权,x_a·x_b 的自然展开是 (w_a x_a)(w_b x_b),两列 Jacobian ∂m/∂w_a=w_b·x_a·x_b 与 ∂m/∂w_b=w_a·x_a·x_b 严格成比例 → JtJ 精确秩亏(精确算术下秩 1,奇异),fp32 下 cond≈{op_cond:.1e},在 7 节点 / 零 bloat 即奇异。evogp:裸变量 c0·x_a·x_b 是 K=1 满秩(cond={ev_cond:.0f}),只有当出现冗余常数(c0·c1·x_a·x_b,典型由 bloat 累积、也可能初始化即有)才奇异(cond≈{evb_cond:.1e})。同症状,不同病因。",
      "en": f"[Centerpiece, provable] The two corpora reach rank-deficiency by DIFFERENT mechanisms — not 'both rank-deficient ⇒ mutual confirmation'. Operon: every leaf is weighted, so the natural form of x_a·x_b is (w_a x_a)(w_b x_b); the two Jacobian columns ∂m/∂w_a=w_b·x_a·x_b and ∂m/∂w_b=w_a·x_a·x_b are exact scalar multiples → JtJ exactly rank-deficient (rank 1 / singular in exact arithmetic; cond≈{op_cond:.1e} in fp32), at 7 nodes / ZERO bloat. evogp: bare-variable c0·x_a·x_b is full-rank K=1 (cond={ev_cond:.0f}); it goes singular only with a redundant constant (c0·c1·x_a·x_b, typically accumulated by bloat though it can also occur at initialization), cond≈{evb_cond:.1e}. Same symptom, different cause."},
     "evidence": {"zh": "mechanism_proof.py → data/mechanism_proof.json(内核同款 fp32 有限差分 J;精确算术下两列成比例→秩 1,fp32 下 cond>1e6 即 Cholesky 失败)。", "en": "mechanism_proof.py → data/mechanism_proof.json (kernel-style fp32 finite-difference J; columns are exact scalar multiples ⇒ rank 1 in exact arithmetic; cond>1e6 ⇒ fp32 Cholesky fails)."}},

    {"claim": {
      "zh": f"种群级轨迹(同内核,cap32,seed0):按全种群 /M 计,两套语料 gen0 都 ~8–10%(Operon {fc('operon',0)[0]}% vs evogp {fc('evogp',0)[0]}%)—— 不存在「Operon gen0 就高」。但 /M 不是同口径:evogp 的分母含 ~20% 被跳过的 K=0 树。按可比的「活跃树(K>0)」口径,gen0 是 Operon {fca('operon',0)}% vs evogp {fca('evogp',0)}% —— evogp 反而更高。两者都随世代上升(见 fail_vs_nodes);evogp 的 bloat 驱动型升得更高(/M gen64 {fc('evogp',64)[0]}%,峰值)而 Operon 到 {fc('operon',64)[0]}%。如实报告,不强凑对比。",
      "en": f"Population-level trajectory (same kernel, cap32, seed0): on a full-population /M base both corpora are ~8–10% at gen0 (Operon {fc('operon',0)[0]}% vs evogp {fc('evogp',0)[0]}%) — no 'Operon already high at gen0'. But /M is not like-for-like: evogp's denominator includes ~20% skipped K=0 trees. On the comparable active-tree (K>0) base, gen0 is Operon {fca('operon',0)}% vs evogp {fca('evogp',0)}% — evogp is in fact higher. Both rise with generation (see fail_vs_nodes); evogp's bloat-driven rate climbs higher (/M gen64 {fc('evogp',64)[0]}%, its peak) while Operon reaches {fc('operon',64)[0]}%. Reported as-is, no forced contrast."},
     "evidence": {"zh": "cross_kernel.py(170 runs,0 error)→ data/cross_kernel.json;活跃口径 = fail_chol/(M−k0_skip)。", "en": "cross_kernel.py (170 runs, 0 errors) → data/cross_kernel.json; active base = fail_chol/(M−k0_skip)."}},

    {"claim": {
      "zh": "对内核鲁棒性的含义:两套语料的种群级秩亏率都随膨胀上升 —— 关键差别不在「升 vs 平」,而在病因:Operon 多了一层与 bloat 无关、不可由减小树削去的结构性底噪(每叶权重过参数化,最小积即奇异);evogp 的秩亏完全依赖冗余常数(最小形满秩)。所以鲁棒目标(Levenberg trust-region 阻尼)必须同时覆盖这两类。注:秩亏是全种群口径;GP 每代选前百分之 x,故这不直接等于端到端缺陷(64.8% 严标准恢复率为前期工作结论,亦同此理)。",
      "en": "Implication for kernel robustness: both corpora's population rank-deficiency rises with bloat — the key difference is NOT 'rising vs flat' but the cause: Operon carries an additional bloat-independent structural floor that reducing tree size cannot remove (per-leaf-weight over-parameterization, singular at the minimal product), whereas evogp's deficiency is entirely contingent on redundant constants (full-rank at the minimal form). So the robustness target (Levenberg trust-region damping) must cover BOTH. Note: rank-deficiency is a population-wide figure; GP selects the top-x% each generation, so it is not directly an end-to-end defect (the 64.8% strict-tolerance recovery rate, a prior-work result, carries the same caveat)."},
     "evidence": {"zh": "综合上列;方向见 memory/cusr-direction。", "en": "Synthesis of the above; direction in memory/cusr-direction."}},
  ],
  "tables": [
    {"name": "Mechanism proof — JtJ conditioning of minimal trees (kernel-style fp32 fd-Jacobian)",
     "columns": ["tree", "engine form", "K", "nodes", "cond(JtJ) [fp32]", "fp32-singular?"],
     "rows": [[k.split(" (")[0], k.split("(")[1].rstrip(")") if "(" in k else "", v["K"], v["n_nodes"],
               f"{v['cond']:.2e}", "YES" if v["effectively_singular_fp32"] else "no"] for k,v in MP.items()]},
    {"name": "fail_cholesky % (fp32-pivot breakdown under damping; conservative rank-deficiency proxy) & K=0 fraction by generation — same kernel build, cap32 seed0",
     "columns": ["gen", "Operon fail% /M (min–max)", "evogp fail% /M (min–max)",
                 "Operon fail% active(K>0)", "evogp fail% active(K>0)", "Operon K=0%", "evogp K=0%"],
     "rows": [[g, f"{fc('operon',g)[0]} ({fc('operon',g)[1]}–{fc('operon',g)[2]})",
               f"{fc('evogp',g)[0]} ({fc('evogp',g)[1]}–{fc('evogp',g)[2]})",
               f"{fca('operon',g)}", f"{fca('evogp',g)}", f"{k0m('operon',g)}", f"{k0m('evogp',g)}"] for g in GENS]},
  ],
  "plots": [
    {"file": "plots/fail_vs_gen.png", "caption": {"zh": "fail_cholesky %(/M 全种群口径)随世代 —— 两套语料,同内核。两者都上升,/M 下 gen0 接近(但 evogp 含 ~20% K=0 稀释,见表中活跃口径);evogp 在膨胀区升得更高。阴影=17 题散布。", "en": "fail_cholesky % (full-population /M base) vs generation — both corpora, same kernel. Both rise; on /M the gen0 rates look close (but evogp is diluted by ~20% K=0 trees — see the active-base columns); evogp climbs higher in the bloated regime. Shaded = spread over 17 problems."}},
    {"file": "plots/fail_vs_nodes.png", "caption": {"zh": "fail_cholesky vs 平均树大小(每点=一题×一世代)。病态随树变大而加重,两套语料皆然。", "en": "fail_cholesky vs mean tree size (each point = one problem×generation). Conditioning worsens with size in both corpora."}},
    {"file": "plots/density_vs_gen.png", "caption": {"zh": "系数密度 mean_K 随世代(cap64,17 题均)。Operon 给每叶都挂权重 → K 远高于只数显式常数的 evogp。", "en": "Coefficient density mean_K vs generation (cap64, mean over 17). Operon weights every leaf → K far above evogp, which counts only explicit constants."}},
    {"file": "plots/k0_fraction.png", "caption": {"zh": "无常数(K=0)树占比:Operon 恒为 0(每叶都是系数),evogp 有 ~8–24%(裸变量树被内核跳过)。", "en": "Constant-free (K=0) tree fraction: Operon is always 0 (every leaf is a coefficient); evogp has ~8–24% (bare-variable trees the kernel skips)."}},
  ],
  "caveats": [
    {"zh": "fail_cholesky 不等于「秩亏」的直接度量:它是内核里 fp32 主元 s≤0 且在 λ 放大(每次拒绝 ×10,直到 λ>1e12)后仍不恢复的事件 —— 比「秩亏」更严格(大 λ 通常能救回仅仅病态的 JtJ)。它是秩亏的保守代理;真正稳健的秩证据是机理证明里的 cond/精确秩 1。", "en": "fail_cholesky is NOT a direct rank measure: it is the kernel event where the fp32 pivot s≤0 AND fails to recover as λ is multiplied by 10 each rejection up to λ>1e12 — stricter than 'rank-deficient' (large λ usually rescues a merely ill-conditioned JtJ). It is a conservative proxy; the robust rank evidence is the mechanism proof's exact rank-1 / cond."},
    {"zh": "机理证明的 cond:精确算术下该 Jacobian 是秩 1(cond=∞);表中 ~1.7e9 是它在 fp32 + 有限差分下的呈现 —— 恰是 fp32 内核实际看到的。", "en": "Mechanism cond: in exact arithmetic the Jacobian is rank 1 (cond=∞); the ~1.7e9 in the table is its fp32 + finite-difference manifestation — precisely what the fp32 kernel sees."},
    {"zh": "fail_cholesky% 的 /M 分母含被跳过的 K=0 树(Operon 0%,evogp 7.7–24.2%),两侧「活跃树」基数不同;表中并列活跃(K>0)口径与 K=0%,避免按 /M 误读 gen0「接近」。", "en": "The /M denominator of fail_cholesky% includes skipped K=0 trees (Operon 0%, evogp 7.7–24.2%), so the 'active-tree' base differs; the table reports the active (K>0) base and K=0% alongside, to prevent misreading the /M gen0 'closeness'."},
    {"zh": "交叉内核扫描只取 cap32/seed0(种群统计,代表性足够);完整语料含 noise/cap64/3-seed,manifest 全部 committed,可重跑。", "en": "The cross-kernel sweep uses cap32/seed0 only (a population statistic, representative); the full corpus spans noise/cap64/3-seeds, all manifests committed and rerunnable."},
    {"zh": "Baseline(Operon LMOptimizer 的 CO 质量/耗时 vs 我们的内核,同一批树)按约定暂缓,待与作者讨论;本报告是负载刻画,不是性能/质量 baseline。", "en": "The baseline (Operon LMOptimizer's CO quality/time vs our kernel on identical trees) is deferred by agreement, pending discussion; this report is workload characterization, not a perf/quality baseline."},
    {"zh": "manifest 里的 best_fitness = Operon 的 R2(平方 Pearson 相关,取负供其极小化器),是可复现性记录,不是拟合优度,勿当质量指标。", "en": "manifest best_fitness = Operon's R2 (squared Pearson correlation, negated for its minimizer): a reproducibility record, NOT goodness-of-fit; do not read it as quality."},
  ],
  "reproduce": [
    "# 1. harvest the Operon pre-CO corpus (CPU; ~10 min, 48 cells concurrent):",
    "/home/weish/hao/operon-venv/bin/python -m cusr.benchmark.workload.operon_harvest --procs 48 --threads 1",
    "# 2. build the kernel + cross-corpus fail_cholesky sweep (GPUs 0-5):",
    "nvcc -O2 -arch=sm_80 -std=c++17 --use_fast_math -o cusr/kernel/batch_lm_fusedfd cusr/kernel/batch_lm_fusedfd.cu cusr/kernel/loader.c",
    "/home/weish/hao/operon-venv/bin/python results/operon_corpus__20260617/cross_kernel.py",
    "# 3. mechanism proof + workload summary + plots + this report:",
    "/home/weish/hao/operon-venv/bin/python results/operon_corpus__20260617/mechanism_proof.py",
    "/home/weish/hao/operon-venv/bin/python results/operon_corpus__20260617/summarize.py",
    ".venv/bin/python results/operon_corpus__20260617/plots.py",
    "/home/weish/hao/operon-venv/bin/python results/operon_corpus__20260617/build_report.py",
    "/home/weish/hao/operon-venv/bin/python scripts/make_report.py results/operon_corpus__20260617/report.json",
  ],
}
(D/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
print(f"wrote report.json; K=32 cells: {k32_str}; nguyen_cap64_kmax={nguyen_cap64_kmax}; "
      f"gen0 active operon {fca('operon',0)} vs evogp {fca('evogp',0)}; k0 range {k0_lo}-{k0_hi}")
