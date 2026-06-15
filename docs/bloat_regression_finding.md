# 强配置反而回归:bloat 在控制题上咬人(2026-06-08,部分结果)

把配置往 EvoGP 官方 SR 例子(`uci_sr.py`)对齐后重跑子集,**控制题(Nguyen,n_inner=0)
的恢复率不升反降**。这是一条"更强 ≠ 更好"的反例,记下来免得回头忘了这版为什么回归。

## 配置 diff

| 轴 | 弱配置(上一版) | 强配置(本版) |
|---|---|---|
| max_tree_len / layer | 32 / 4 | **64 / 6** |
| selection | Default(0.3, 0.01) | **Tournament(20, 0.5, 0.1)** |
| const 采样 | 6 个固定值 | **`[-5,5]` × 10000** |
| mutation_rate | 0.2 | 0.1 |
| funcs | 8(无 log/tanh) | 10(补 log/tanh) |
| **parsimony** | **无** | **无** ← 关键:两版都没有 |

## 结果(控制题,lenient 列)

- **上一版:5/14**(nguyen_1, 2, 6, 8, 10 恢复)
- **本版:1/12**(只剩 nguyen_8 = `sqrt(x0)`;nguyen_1/2/6/10 全丢)
- best_expr 普遍**膨胀**:`nguyen_1` 本该是干净的 `x³+x²+x`,变成
  `x0 - 0.41202·(4·x0 + (2·x0-0.636)·(2·x0-…))` 一团。

控制题集体回归、且 best_expr 明显膨胀 → **没有 parsimony 时,放大容量(大树 + 富采样 +
高留存的 Tournament)让"拟合好但形式错"的膨胀树赢过了干净小形式。** 弱配置没回归,
是因为它树太小、压根 bloat 不起来。

## 确认 vs 未确认

- **确认**:这版控制题回归,机制是 bloat(完整、非采样噪声)。
- **未确认(inner-常数行还在跑)**:korns/feynman 这些**真正重要**的题,在强配置下是
  变好还是变坏。它们的结构可能因别的原因(如 2 变量交互搜索没碰到)找不到,与 bloat 无关。
- **未测**:parsimony 是否真能修好——这是**领先假设,不是结论**。

## 一条值得查的 sub-hypothesis:CO 在加速 bloat

本版 `co_fits` 极高(多数题 460–480,即每 2 代对 top-16 拟到底)。CO 给膨胀树配上好常数
→ 它们 fitness 上去 → 主导种群。**所以 bloat 可能是 CO 加速的,而不只是 GP 自带的**——
这正好是"朴素激进 CO → 锁死"叙事的一部分。要证需在本配置下跑 CO-off 对比(`diag_co_off_on.py`
的做法):若 CO-off 膨胀明显更轻,则 CO 是 bloat 加速器。

## 决策门(等 inner-常数行落地)

- inner 题**比弱配置有起色** → 容量改动值 → "保留改动 + 加尺寸控制"。
- inner 题**还是 0 / 更糟** → 容量改动不值 → **部分回退**,而不是在改坏的配置上硬糊 parsimony。

## 注意:别把 base 卫生和配方 C 混了

若结论是"base 需要尺寸控制",要把它和**配方 C 的自适应/多目标 parsimony(被测对象)**分开——
否则把 parsimony 放进 base 会模糊 A/B/C 的自变量。base 用标准 GP 卫生(EvoGP 自带
DeleteMutation / hoist / `enable_pareto_front`),配方 C 才是那个**和 CO 动态耦合的**强化版。
