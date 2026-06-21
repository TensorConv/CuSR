# CuSR — 三条贡献(诚实版,论文 intro 可直接取用)

**用途**:论文的 contribution 陈述。每条给 **(i) 论文可直接用的一句话 → (ii) 实质 → (iii) 诚实边界(红线)**。
本文件按**真实情况**写,不受任何外部草稿措辞约束。

**诚实纪律(沿用 `co_benchmark_strategy.md` / `co_benchmark_contribution_plan.md`)**:`access_level`
报告且不上调;绝不写裸 "first";absence-of-evidence 永远只当 absence-of-evidence;每一处拔高都用具名
先验框边界。

**结构说明(为什么是这三条,而不是"算法/系统/大对比表")**:
- "**CO 有用 / 把 CO 嵌进进化环**"**不是贡献**——前者是 Kommenda 2020 的先验,后者是标准 memetic GP。
  所以"integrated EvoGP+CO system"降级为**演示载体**,第三条改成**由判据量出的实证发现**。
- "**诚实性 / 奇点附近 conditioning**"**不是独立贡献**——Kronberger 2022 占了这个轴。它只当贡献一的
  正确性特性。
- 真实三条 = **系统 kernel(收成工程) / 选题判据(只 claim SR 操作化) / 实证发现(失效分析+对照有效性)**。

---

## 贡献一 ── 系统:GPU 上异构树的二阶批量 LM kernel

> **我们给出一个自定义 CUDA kernel,在单次 kernel launch 内,对结构各异、且常数个数 K 也各异的一批
> 表达式树,并发执行二阶 Levenberg–Marquardt 常数优化(核内 forward-mode AD 显式逐树构造 Jacobian),
> 并把"交付解不劣于初值"作为 fp32-迭代 / fp64-验收 的运行时保证强制执行。**

**实质**
- 变长 K 的异构批量法方程 `(JᵀJ + λI)` 在一次 launch 内的打包求解(每棵树系统维度不同)。
- 核内 forward-mode AD 逐树精确 Jacobian。
- in-process 嵌入,CUDA context 只付一次;工程数 CO ~13.2×、端到端 ~2.3×(**项目内部数,标注清楚**)。

**诚实边界(红线)**
- **不是新算法**——LM 是 1944/1963 的教科书方法。这是**系统/工程贡献**(让它在新场景跑起来的新实现),
  不是 "novel optimizer"。
- 差异化只 claim 这个**窄存在**(restricted existential,MEDIUM,absence-of-evidence):
  - vs **Gpufit / JAXFit**(arXiv 2208.12187):它们批量拟合**单一固定模型**、同构。
  - vs **Kozax(de Vries, Keemink & van Gerven 2025, GECCO Companion;arXiv 2502.03047)**:异构但
    **一阶 AD 梯度 + 简单 GA、JAX/vmap、非手写 CUDA、无 LM**(2026-06-21 原文核实)。
  - 我们 = **二阶 LM + 显式 per-tree Jacobian + 异构(结构与 K 皆变)+ 手写 CUDA、单次 launch**。
  - 措辞:"first custom CUDA kernel to run batched second-order LM with explicit per-tree Jacobians over
    structurally heterogeneous trees in a single launch",绝不写 "first GPU CO" / "first batched LM on GPU"。
- **诚实性只当正确性特性**:引用 **Kronberger et al. 2022("Local Optimization Often is Ill-conditioned…",
  arXiv 2209.00942, IEEE SYNASC, FULL-TEXT)** 为 conditioning/honesty-near-singularities 轴的主人;只
  claim "第一个在 GPU SR CO kernel 里把它作为运行时保证强制执行/报告",绝不写"新机制 / 未被占的诚实轴"
  (机制 = return-best LM + 混合精度细化,均为老的;JAXFit 在 GPU 上已单调)。

---

## 贡献二 ── 方法:隔离"非内部常数优化不可"regime 的可复用选题判据

> **我们提出一个确定性的"两侧线性缩放差距"准入判据,并据此构造一个语料:一道题被收入,当且仅当
> Keijzer 线性缩放后的 `y ≈ a + b·f(x)` 对真实最小骨架拟合 R² 低、而完整非线性 CO 拟合 R² 高。
> 它是一把可对任意 SR 基准施用的工具,用来判定该基准到底有没有真正考查"内部常数的精确恢复"。**

**实质**
- 判据 = **工具**(不止是一份数据集)+ 构造语料 + 配套对照集。
- 评分沿用 **SRBench Def 4.1 / R² > 0.999**,并加**分级 per-constant recovery `|ĉ − c| / |c|`**,弃用 TED。

**诚实边界(红线)**
- 底层统计是教科书的**嵌套模型 extra-sum-of-squares / F 检验**(R1,stat462 FULL-TEXT)。
- **单侧**线性基线过滤已发表(SRBench 2.0 / "Call for Action",arXiv 2505.03977,排除线性 R²>0.99 的题;
  Oliveira 2018 的 linearity meta-feature,arXiv 1805.10365)。
- "靠构造隔离某能力"是既有范式(SRSD;McDermott 2012;Instance Space Analysis)。
- 所以只 claim **SR 上的具体操作化**(**两侧、缩放后**的"纳入"规则当作 benchmark 构造工具,对比他们
  的单侧"排除平凡"规则),绝不写"新统计检验";first-ness 只写"在一次广泛检索下未见占用,边界见相关工作"。
  **MEDIUM**。

---

## 贡献三 ── 实证:用这把判据量出来的两件事(都不是"CO 有用")

> **(a) 对现有基准的失效分析**:把判据施于学界标准的 **AI-Feynman** 题集,发现它**几乎不含**这个 regime
> (冻结规则下 34 题中最多约 2 道;跨题集 0–11,set-independent 报告)——即标准 SR 基准其实没有按大家
> 默认的方式考查"内部常数精确恢复"。
> **(b) 工具有效性(对照验证)**:在构造语料上,内部 CO 把**准入题**从近零恢复率"解锁",而结构匹配的
> **对照题**(内部 CO 本不该起作用)**不被解锁**——这个"准入解锁、对照不解锁"的对比,证明判据精准框住
> 了它声称的 regime。

**诚实边界(红线)**
- **绝不**写"我们证明 CO 有价值"——那是 **Kommenda et al. 2020**(GPEM;Poly-10 R² 0.537→>0.8)的先验,
  只当**动机**引用。
- 解锁效应**限于 single-inner**;**multi-inner 目前所有方法都解不开**(Study B:各臂 ~4–7%),如实写成
  **开放前沿**,**绝不**写"收益集中在多内部常数题"(与我们自己的数据相反)。
- (b) 待 zoo-crash bug 修复后的**干净重跑**坐实,并以**统计交互检验**给出"准入差距显著大于对照差距";
  (a) 已做但有边界,措辞 set-independent。

---

## 速查:相邻先验各归谁(防止再次踩雷)

| 别人占的 | 谁 | 给我们的红线 |
|---|---|---|
| "CO/非线性常数优化有用" | Kommenda 2020 | 只当动机,绝不当成果 |
| 诚实性 / 奇点附近 conditioning / SVD 秩 | Kronberger 2022 (2209.00942) | 只 claim "GPU-SR 第一个强制/报告";秩检查只当对照工具 |
| GPU 上对异构树做 CO | Kozax = **de Vries** 2025 (2502.03047) | 一阶 JAX;我们只 claim 二阶 LM + per-tree Jacobian + CUDA 这一窄片 |
| GPU 批量 NLS | Gpufit / JAXFit | 单一固定模型;我们 claim 异构树 |
| 判据的统计内核 | 嵌套模型 F 检验 (R1) | 只 claim SR 操作化 |
| 单侧线性基线过滤 | SRBench 2.0 (2505.03977) | 只 claim 两侧、缩放后的纳入规则 |
| BRACIS 2015 CO 基准 | de Melo/Fowler/Banzhaf(**非 Kommenda**) | 引用别写错 |
