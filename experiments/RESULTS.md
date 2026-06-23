# 实验结果总账 (e1–e6) — pilot ledger

> **2026-06-22 reset:** e1-e6 全部降级为 **pilot / infrastructure / failure-mode discovery**。
> 它们告诉我们哪些变量重要、哪些测法会误导、哪些 claim 有希望; 但不再作为 paper 的最终主证据。
> 最终 paper 实验必须按 [`../docs/research/experiment_plan.md`](../docs/research/experiment_plan.md) 重做。
>
> 本页只保留旧实验台账和 pilot lessons。旧数字可用于规划, 不能直接搬进 paper headline。

实验目录是一条主线的几代地层(EvoGP SR + GPU 批量常数优化 CO)。新的 paper 主张见
[`../docs/research/contributions.md`](../docs/research/contributions.md)。本页只回答:
旧实验各自还能作为哪类 pilot lesson。

---

## 0. 一眼台账

| 实验 | 用途 | 状态 | 头条(细节见权威文档) |
|---|---|---|---|
| **e1** `operator_bench` | 算子层 CO benchmark harness(同一 pop 喂 scipy/Operon/PySR/kernel×3,按质量档比吞吐) | laptop sanity 完;A100 口径见 e6 | 变体阶梯 + frontier 叙事;laptop **2.4–29× vs scipy**(fp32-vs-fp64,**仅 sanity,已被 e6 的同质 A100 口径取代**) |
| **e2** `demonstrator` | EvoGP 树 → pop.bin → kernel CO 的桥 + SR 恢复 benchmark 基础设施(e4 的底座) | 接好 + parity 闸门过 | 377 棵活树 **0% 函数发散 / 0 常数错配**;E2 跑的就是 E1 测的同一 kernel 路径 |
| **e3** `admit_criterion` | "非内部 CO 不可"题的确定性准入判据(5 机制,TDD + 红队) | **committed**,稳 | AI-Feynman 34 题 → **冻结集下 2**(跨集 0–11,set-independent:**几乎无 Feynman 内部常数需超出可达常数的非线性 CO**);构造语料 21 收 / 11 对照 0 漏 |
| **e4** `study_b` | 端到端 EvoGP+CO 四臂(no_co / sparse_gpu / cpu_every / gpu_every),固定代数协议 | **committed** (6087d3a),audit 过 | **C1** kernel 健全(1060 cell 0 崩);**C2** CO 提升**符号恢复**(gpu 16 vs no_co 0,McNemar **p=3.1e-5**),**仅 single-inner**;质量与 CPU 持平;每代 CO 无额外收益;multi-inner **未测非证伪** |
| **e5** `strong_baseline` | Operon(具名强 CPU baseline)吞吐扫描,de-risk 旧 "2.4–29×" | **未提交,已被 e6 取代** ⚠️ | **头条被推翻**:误测了**慢的 host-FD `.so`**;"GPU 在任一 M 打不过并行 Operon" **仅对 host-FD 成立**。**存活结论**:Operon 并行效率随 K 升(inner 并行最好)——已被 e6 证实 |
| **e6** `kernel_sweep` | 改正后的 kernel 吞吐扫描(fused/ad × preset × M × N × Operon 核数) | **未提交,当前权威 kernel 吞吐结果** | late-gen-bloated M=256k N=100:**~25× iso vs 16 核(最诚实基线)/ ~31.7× vs 128 核** [ISO];early-gen **~8× iso(仅 AD)**;**inner-const 只快不等质量**(fp32 秩亏天花板);**N 主导**(100→10k 胜势崩);**M 是设计旋钮**(优势随 M 增);ad 中位 **1.25×** over fd |

**权威 FINDINGS 文档**:
e3 → [`e3_admit_criterion/REPORT.md`](e3_admit_criterion/REPORT.md) ·
e4 → [`e4_study_b/STUDY_B_FINDINGS.md`](e4_study_b/STUDY_B_FINDINGS.md) ·
e5 → [`e5_strong_baseline/FINDINGS.md`](e5_strong_baseline/FINDINGS.md)(**SUPERSEDED**,留作 host-FD 教训 + PE-by-K) ·
e6 → [`e6_kernel_sweep/out/FINDINGS_e6.md`](e6_kernel_sweep/out/FINDINGS_e6.md)(当前权威) ·
e1 laptop → [`../docs/archive/RESULTS_laptop.md`](../docs/archive/RESULTS_laptop.md)(**laptop sanity,吞吐已被 e6 取代**) ·
e2 桥 → [`../docs/archive/RESULTS_kernel_bridge.md`](../docs/archive/RESULTS_kernel_bridge.md)。
速度优化路线图 → [`../docs/kernel/OPTIMIZATION_BACKLOG.md`](../docs/kernel/OPTIMIZATION_BACKLOG.md)。

---

## 1. Pilot lessons after reset

- **Workload definition must come first.** e3 suggests that many nominal inner-constant problems are actually
  foldable/canonical/linear-scaling-sufficient, but this must be rerun as a broader benchmark audit.
- **Fixed-tree apples-to-apples is mandatory.** e4 and GP-vs-GP comparisons are useful deployment pilots, but
  backend performance must be measured on the same frozen trees, same `X/y`, same `c_init`.
- **Synthetic scaling is useful but not enough.** e6 shows which axes matter (M, N, K, nodes, quality gate),
  but it used `gen_synth`, not real dumps.
- **Parameter choices are not frozen.** `M`, `N`, `K`, tree size/depth, and `max_iter` need sensitivity sweeps;
  especially `N`, because it can change both throughput and CPU/GPU ranking.
- **High-K inner-heavy is a boundary, not an automatic win.** Pilot results suggest rank/conditioning can make
  speed-only results fail quality gates.
- **Iteration budget is unresolved.** Pilot discussion suggests `max_iter=50` may be conservative; paper needs
  `iter.bin`, Operon iteration stats, and budget/warm-start sweeps.
- **Measurement discipline matters.** e5 measured the wrong kernel path; future experiments need structural
  binary/profile guards.

## 2. Open items that must be rerun

- Benchmark audit table over standard suites.
- Blind baseline validation for constructed admitted/control corpus.
- Real-dump fixed-tree CO replay.
- Locked synthetic scaling rerun.
- Parameter sensitivity table/figure for `M`, `N`, `K`/tree size, and iteration budget.
- Equal-wall-clock EvoGP deployment experiment.
- Iteration diagnostics and warm-start/budget sweep.

## 3. Pilot assets → new contribution plan

- **Contribution 1: workload/benchmark framework** ← e3 as prototype only. Needs benchmark audit + blind baselines.
- **Contribution 2: GPU SR-CO primitive** ← e2/kernel/e6 as infrastructure and pilot. Needs real-dump replay +
  locked synthetic scaling.
- **Contribution 3: empirical characterization/deployment** ← e4/e6 as pilot. Needs equal-wall-clock loop +
  iteration/warm-start diagnostics.

详见 [`../docs/research/contributions.md`](../docs/research/contributions.md) 的逐条诚实边界。

---

## 4. 全局口径与红线(读任何头条前先看)

- **Do not cite old headline numbers as final.** Use them only to choose rerun points.
- **Do not inherit old contribution order.** The reset order is workload/benchmark -> system primitive ->
  empirical characterization/deployment.
- **Do not treat e6 as real workload evidence.** It is calibrated synthetic pilot.
- **Do not treat e4 as hardware performance evidence.** It is deployment pilot.
