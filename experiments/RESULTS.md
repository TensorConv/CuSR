# 实验结果总账 (e1–e6) — 单一现状入口

> **这是"我们的实验都测出了什么 + 现在算数的是哪一份"的唯一汇总页。**
> 本页只做 **台账 + 链接**:每条实验给 用途 / 状态 / 头条 / 权威文档。**数字以各实验的
> FINDINGS 为准**(本页只复述头条,细节点链接,避免二次抄录漂移)。最后整理 2026-06-21。

实验目录是一条主线的几代地层(EvoGP SR + GPU 批量常数优化 CO,冲 HPEC 系统向)。
配套的**论文主张**在 [`../docs/research/contributions.md`](../docs/research/contributions.md)(三条贡献 + 先验红线);
本页是**实测数据**侧,把主张落到证据上。

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
e1 laptop → [`../docs/RESULTS_laptop.md`](../docs/RESULTS_laptop.md)(**laptop sanity,吞吐已被 e6 取代**) ·
e2 桥 → [`../docs/RESULTS_kernel_bridge.md`](../docs/RESULTS_kernel_bridge.md)。
速度优化路线图 → [`../docs/kernel/OPTIMIZATION_BACKLOG.md`](../docs/kernel/OPTIMIZATION_BACKLOG.md)。

---

## 1. 已坐实(settled)

- **kernel 是健全的系统件**:fp32 迭代 + fp64 验收保证(交付解 ≤ 初值),1060 cell **0 崩**(e4 C1)。
- **大 M 等质量吞吐胜势真实且大**:late-gen-bloated **~25× vs 16 核(Operon 最诚实配置)**,~31.7× vs 128 核,均 [ISO](e6)。
- **胜势由 N 主导,M 是设计旋钮**:N 100→10k 胜势从 ~12× 崩到 ~1.3×;优势随 M 增(1k→256k:3.1×→31.7×)。GPU 让大 M 变便宜,而大 M 是 SR driver 设的旋钮,非问题固有(e6)。
- **AD 是要部署的变体**:中位 1.25× 快过 fused-FD(范围 0.84–1.97),且 Jacobian 精确(e6)。
- **inner-const 只快不等质量 = 内在 fp32 秩亏天花板**,非测量假象、非 Jacobian 精度问题;damping / 更高精度 / 更多迭代**都救不回**(e5/e6 + [`../docs/MIXED_PRECISION_PROBE.md`](../docs/MIXED_PRECISION_PROBE.md))。**别再提 damping 当解法。**
- **准入判据建成并验证**;AI-Feynman **几乎不含**"需非线性内部 CO"的 regime(e3)。
- **集成 CO 提升 single-inner 符号恢复**,且与 CPU 质量持平(e4 C2)。
- **Operon 并行扩展随 K 升**:inner-const(高 K)16→64 核 **2.4×**,而 early/late(低 K)持平至反扩展(median 128/16:late 0.93,early 1.10,inner 2.39)。⟹ e6 "Operon 过 16 核几乎不扩展"(总中位 1.06×)是被两个低 K preset 拉的,inner-const 是例外——这正是 e5 PE-by-K 机制,已被 e6 更大扫描证实。

## 2. 未决 / 本批未证(open — 别写成结论)

- **multi-inner 未测**(e4 在 15 道 multi-inner 上统计功效近零:lenient discordant=0)。**开放前沿**,既不能写"CO 帮 multi-inner"也不能写"不帮"。
- **"大 M 改善 SR 端到端"未证**:e6 只测 CO kernel 吞吐 + 单次收敛,这是 demonstrator 级主张(e2/e4),e6 不证。
- **ncu roofline 待 sudo 锁频**:决定 compute-vs-memory bound 与 reverse-AD 的可达天花板(OPTIMIZATION_BACKLOG §0.2)。
- **reverse-mode AD 未实现**:已被 PROFILE 验证为 **#1 速度杠杆**(Jacobian 占 LM loop 60–66%),尚未动手。
- **DRAFT,时钟未锁**:绝对吞吐/加速比 run-to-run 抖动,信形状不信第三位有效数字。锁频后重测待 sudo。

## 3. 结果 → 三条贡献的映射

- **贡献一(系统 kernel)** ← e1(harness/变体阶梯)+ e2(桥 parity)+ e6(A100 吞吐)+ e4 C1(健全性)。
- **贡献二(准入判据 = 工具)** ← e3。
- **贡献三(用判据量出的实证:Feynman 失效分析 + 对照验证的解锁)** ← e3(Feynman 34→2)+ e4 C2(single-inner 解锁、对照不解锁)。

详见 [`../docs/research/contributions.md`](../docs/research/contributions.md) 的逐条诚实边界。

---

## 4. 全局口径与红线(读任何头条前先看)

- **e6 计时 setup-excluded 两侧对齐**:GPU `loop_ms`(代内 per-gen CO)vs Operon `wall_core`;都不计 CUDA init / .so load / marshaling。Operon 给 200 LM 迭代 vs kernel 50。等质量门 = GPU 末 loss ≤ Operon × 1.05,**fp64 重算**、seed0、单侧(只证"不显著更差")。
- **e4 是固定代数协议**:**不出任何速度/硬件主张**,信号是**恢复率**(solved + symbolic),不是 R²(single-inner 上 R² 已饱和)。
- **"first" 措辞守 `contributions.md` 红线**:LM 非新算法(系统贡献);诚实性轴归 Kronberger 2022;GPU 异构树 CO 的相邻先验是 Kozax(de Vries 2025,一阶 JAX)。
- **laptop ≠ A100**:`RESULTS_laptop.md` 的 2.4–29× 是 fp32-vs-fp64 的笔记本 sanity,**不是论文数字**;论文吞吐口径以 e6(A100、同质 apples-to-apples、等质量门)为准。
