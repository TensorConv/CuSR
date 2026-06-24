# 论文事实卡 / PAPER_FACTS —— 写论文随身带(离线版)

> **为什么有这份**:很多关键事实原本只存在 Claude 的 memory(`~/.claude/...`)里——那**不在 repo 里**,
> 下载到 MacBook 就没了。这份把**写论文真正要用的关键事实**固化进 repo:当前正确数字、红线、引用陷阱、
> 诚实边界。细节仍在各专门文档里(见下「文档地图」),这里只放"容易记错 / 只在记忆里 / 必须随手有"的部分。
>
> 更新 **2026-06-25**。数字都对应 `experiments/e7_section2/` 下的 on-disk 产物,过了对抗审计 + codex。

---

## 文档地图(在 MacBook 上写论文,先看这几个)

| 文档 | 装什么 |
|---|---|
| `docs/PAPER.md` | 待办 + 进度(「现在到哪了」表,2026-06-25) |
| `docs/research/paper_plan.md` | 6 页怎么拼、三块贡献怎么分、C1/C3 可选材料、退路 |
| `docs/research/C2_draft.md` | **C2 正文初稿**(设计节=完整初稿;评测节=锁定骨架+数齐) |
| `docs/research/contributions.md` | 三条 claim 的边界 + **prior-art 对照表** + safe intro bullets |
| `docs/OUTLINE.md` | 全文骨架(9 节)+ 实验栈 A–E |
| `experiments/e7_section2/FINDINGS.md` | **section-2 每个数的出处** + 审计 + 「没做什么」 |
| `docs/kernel/REVAD_V5.md` / `INTERNAL.md` | kernel 设计;`lm_algorithm.tex` / `fig_architecture.tex` = 算法图(tectonic 编译) |

---

## 当前正确数字(cheat sheet)

> ⚠️ 用这些,别用旧的。最大的一处更新:crossover 基线已从 128 核改成**一个 64 核 EPYC**(见下一节)。

| 量 | 值 | 出处 |
|---|---|---|
| 峰值吞吐 | **361,334 trees/s**(revad, early-gen M=256k N=100, 3-seed 中位, 锁1410) | `out/gpu_phase_a.json` |
| **revad vs 一个 64 核 EPYC**(iso-quality, in-loop) | **≈5.8×**(稳定区, 精确 5.76×)/ **≈14.8×**(臃肿区)——**是上界** | `out/determinism/crossover_clean.json` |
| revad / fd(Jacobian 方法之争) | **1.50×** 中位 | `out/gpu_phase_a.json` |
| revad / ad(反向 vs 前向) | 1.17× 中位(范围 0.86–1.49) | `out/gpu_phase_a.json` |
| 反向AD vs 前向AD(kernel 级) | Jacobian **1.8×**(高K)/ 1.48×(低K);整 loop 1.27×/1.15× | `docs/kernel/REVAD_V5.md` |
| ncu 瓶颈 | 非 FLOP(FMA≤11%, fp64=0%)、非带宽(DRAM≤23%, 多<10%)→ **发射/片上内存 bound**;stall = `wait`(依赖链)+ `long_scoreboard`(访存延迟) | `out/ncu_summary.md` |
| instruction roofline | 146–333 GIPS = 发射峰值的 **24–55%**(同时 <1% FLOP/HBM 峰值) | `out/ncu_instruction_roofline.json` |
| 排序等价 | **Spearman(revad,fd)=0.959**, top-10% 选择重合 95.5%;revad≈ad 0.995;"AD 比 FD 差"已反证 | `out/ranking_revad_vs_fd.txt` |
| 质量门(parity vs scipy fp64) | within-1.05×:fusedfd 92.1% / ad·revad 86.6%;2× 与 10× 档全 match | `out/parity_gate_*.txt` |
| C1 审计 | Feynman 34 道 → **2 道**(随 canonical 集在 0–11 间);判据 5 个机制 | e3(admit criterion) |

---

## ⚠️ 基线决定(2026-06-25):128 核数据**不可用**,用一个 64 核 EPYC

- 机器是**双路 2×EPYC 7763**(NUMA node0=0–63, node1=64–127)。重负载 co-tenant **钉在 node0**。
- **128 核 = 跨两个 socket**,必然撞上 node0 的 co-tenant → `cv_operon.json` 实测 128c **6,711** vs 64c **10,802** t/s(−38% ≫ 5–6% 同会话 CV)=**争用,不是 Operon scaling**。
- ⇒ **所有 `vs 128c` 的数(9.6× / 28× / 9.27×)作废,论文里别引 128 核。** 旧表在 FINDINGS 里保留为"测过什么"的审计记录,但不再是基线。
- **基线 = 一个 64 核 EPYC 7763**(= 一颗 CPU;标准的"一块 A100 vs 一颗服务器 CPU"比法;64c 还待在 node1、躲开 co-tenant → 现有最干净)。
- **5.8× 仍是上界**:那次 Operon 仍在共享机上(tenant_overlap=1.0)→ 把比值往**上**偏 → 真·空闲机的值 ≤ 5.8×。论文写 "≈5.8×、上界、跨天 ±30%、精确数需空闲机",别写成精确值。GPU 侧锁频独占,CV<0.5%。

---

## 🚫 红线 —— 永远不要 claim(每条都有 full-text 反证)

- ❌ **"first GPU CO"** —— Kozax(de Vries, GECCO 2025)已做 GPU 梯度 CO。只能 claim 下面那条窄切片。
- ❌ **新优化器** —— LM 是 1944/1963;我们是系统实现,不是算法新意。
- ❌ **"CO 让 SR 变好 / CO matters"** —— Kommenda 2020 已证。C3 只说"装进去用得省",不碰这句。
- ❌ **"现有 benchmark 没有 inner constants"** —— 假(de Melo 2015 的 Korns f11/f12;arXiv 2412.02126 Table 1 有)。安全说法:"广泛使用的题库里,真要非线性优化**非正则**内部常数的题**很少**"。
- ❌ **"新统计判据"** —— 那是 extra-sum-of-squares / nested-model F 检验,SRBench 2.0 已用。
- ❌ **"fp64-honesty guard 是新机制"** —— textbook return-best/迭代精化;只能 claim "first GPU-SR 系统去**测/报**它 + 那个 `delivered ≤ init` 的保证"。
- ❌ **永远别写 "first / 首个 / 唯一"** —— novelty 是有限(且偏美)搜索的 absence-of-evidence,置信上限 = MEDIUM。
- ✅ **能站住的 claim(就报这个窄切片)**:*second-order batched LM + 显式 per-tree Jacobian(reverse-mode AD)+ 结构异构 且 异构 K 的树,单次自定义 CUDA launch* —— 这是 **SYSTEMS/工程**贡献。

---

## 📌 引用陷阱 + 正确归属(高危,容易写错)

- **Kozax**(GPU-SR、做 CO 的最近 prior art)= **Sigur de Vries, Sander W. Keemink, Marcel A.J. van Gerven**,arXiv **2502.03047**,GECCO 2025 Companion。**不是 "de Wolff"!**(旧 notes 标错过)。它是 **JAX/vmap**(XLA 之下才是 CUDA,非手写 kernel);CO = **一阶**(autodiff 梯度)+ 一个简单 GA,**无 LM/Gauss-Newton**。→ 我们的差异:二阶 LM + per-tree Jacobian + 异构树 + 单 CUDA kernel。
- **BRACIS 2015 的 CO 方法 benchmark** = **de Melo, Fowler, Banzhaf**,"Evaluating Methods for Constant Optimization of Symbolic Regression Benchmark Problems",DOI 10.1109/BRACIS.2015.55,pp.25–30。**不是 "Kommenda 2015 BRACIS"**(反复出现的误标;连搜索摘要都标错过)。它的发现:6 个优化器在 Korns **f11/f12 全失败** = "inner constants 难"的好证据。
- **正确的 CO 历史引用集**:de Melo et al. **2015**(方法 benchmark)+ Kommenda et al. **2013 GECCO** Companion(CO by NLS in SR)+ Kommenda et al. **2020 GPEM**("CO matters" / 频率检测)。
- 其余 prior-art 区分见 `contributions.md` 的对照表:vs **Gpufit / JAXFit**(单一固定模型、同构 NLS)、vs **Kronberger et al.**(conditioning/honesty)、vs **Keijzer scaling / SRBench filters**(linear-scaling / benchmark 过滤)。

---

## 诚实边界 / 必带的 caveat(审稿一定会问)

- **Workload 是「合成-可恢复」**:结构对齐 3 个真 Feynman 快照(I.18.12 gen5 / I.12.1 gen50 / I.6.2 gen30),但 target = `tree(x;c)+1% 噪声`、有已知近全局最优——**这是故意的**(做干净的"性能+精度"基准),**不是"拟合真实数据"的 claim**。措辞:"在真实的结构分布下,kernel 吞吐是 X"。
- **高 K 的质量天花板是内在的**(damping / fp64 / floor / QR **全证伪** —— `docs/MIXED_PRECISION_PROBE.md`;别再提"加 damping")。inner-const 这档是 **SPEED-only**,不是质量赢。
- **Operon 基线在共享机上**(co-tenant)→ crossover 倍数是**上界 / 近似**(跨天 ±30%);要精确得空闲机重测。GPU 侧锁频独占,CV<0.5%——这是**不对称**的:只有 CPU 侧吵。
- **赢面 N-主导**:~22×@N=100 → ~1.4×@N=10k(e5/e6)。别从单个 N 外推;评测里要有 N 轴。
- **warm-start(C3)还没实现**:可行性已核(EvoGP 交叉/变异**逐字节复制常数** → 自动继承,只缺一个"写回 node_value"的钩子,**不是新算法**)。跑出来干净 C3 才写满,否则缩成一小节。

---

## 三块贡献(一句话;详见 `paper_plan.md`)

- **C1** = workload/benchmark:说清"哪种 SR 题真要在函数内部拟合常数",并证明现有题库这种题**很少**(判据 + 构造题集 + Feynman 审计表)。
- **C2 = 主角**(占 6 页的一半):GPU kernel —— 异构批 LM + **reverse-AD** + fp64 兜底 + 性能剖析(ncu SOL+stall 为主、instruction roofline 作图)+ **vs 一个 64 核 EPYC** 同条件对比。
- **C3** = 装进 EvoGP 用得省:进程内调用(数据不来回搬)+ warm-start。**红线:不说"CO 有用"**,只说"用得起、用得省"。
- 死规矩:**三块不等于三块一样大**。C2 占一半;C1/C3 各留一个最硬的结果;塞不下就 C3 缩成一小节(退回 C1+C2 两块)。
