# EvoGP in-loop: 能不能直接当 CO backend 用

记一下从 memetic SR loop(另一个实验,把 LM 当"边进化边精调常数"的局部优化器)
试着拿这个 kernel 当常数优化(constant optimization, CO)后端时,摸出来的边界。
对应 [Roadmap](INTERNAL.md#路线图--roadmap) 里 "EvoGP in-loop 集成" 那条长期项 —— 下面是真去接之前要知道的事。

日期:2026-06-08。结论先行:**结构上能吃任意 EvoGP 树,但 EvoGP 论文那档强配置
(`max_tree_len=512` / `max_layer_cnt=9`)会顶到两个编译期上限,外加一个跟 FD 绑死的性能软肋。**

## 现成就能用的部分

- **树解释器是通用栈机**(`batch_lm.cu` `eval_tree_d`):吃任意 opcode 树,不是写死某个 NLS 模型。
  大树只是更大的树。
- **heterogeneous-K 原生**:每棵树的常数个数 K 在 `TreeMeta` 里,batch 内 K 不齐不触发 warp divergence。
- **解算器(LM + Cholesky)与 opcode 无关**,只认 Jacobian。

所以"能不能吃 EvoGP 的树"——能,这本来就是它的活。

## 强配置会撞的三件事

| 项 | 现状 | EvoGP 强配置下 | 修法 |
|---|---|---|---|
| `MAX_K` | 32(编译期,`batch_lm.cu`) | 512 节点的树常数叶可能 50–100+,**超 32 触发 runtime guard** | 调常数重编译;显存代价小(`JtJ` 是 K_max²/树) |
| `MAX_STACK` | 64(编译期) | 深 9 的树栈深大概率 <64,**估计够**,但要验 | 不够也是重编译 |
| FD Jacobian 随 K 线性涨 | `ε=1e-3` 固定,每次 LM 迭代 `K_max` 次前向 eval | 大树=常数多=FD 列多 → 正好戳强配置的痛点 | 见下 |

第三条改不掉(是 FD 的本性):**K 个常数 = 每次 LM 迭代 K 次前向 eval**。一棵 bloat 出 80 个
常数的树,Jacobian 成本是 1 常数树的 80×。而强配置(无 parsimony,`enable_pareto_front=False`)
**正是会 bloat 出这种树的**。已列 Roadmap 的 fused-FD / K-bucket dispatch / analytical Jacobian
都是冲这个去的。

## bloat ↔ parsimony:对这个 kernel 是双重动机

符号回归的目标是找回真公式,真公式常数很少(本 benchmark 里每题 0–5 个)。GP 放开树大小后会
**bloat**——长出一堆不改善拟合的废子树,连带几十上百个常数叶。这些常数:

- 对**恢复**是噪声(永远凑不出那个小的真形式);
- 对**这个 CUDA kernel** 是纯浪费 FD 列(每个常数每代多一次前向 eval)。

所以 GP loop 里的 **parsimony / 简约惩罚不只是精度旋钮,也是 CO 的算力控制阀**:把树压小 → K 变小
→ 这个 kernel 的 FD 成本才压得住。这条对 TorchLM(autograd Jacobian)也成立但没这么剧烈——
**FD-列浪费是 CUDA backend 特有的软肋**,也因此 parsimony 对 kernel 路径的价值更大。

## 接入状态

- 现在是**离线 batch**:`pop.bin` → subprocess → `c_final.bin` / `status.bin`,无 in-process Python 绑定。
- memetic loop 侧留了 `CudaKernelLM`(stub,`NotImplementedError`)占位;in-loop 集成 = 每代
  dump→subprocess→读回的 glue。
- 当前 in-loop 实际跑的是 torch-native LM(autograd Jacobian,无 K 上限,任意树任意 K 都吃,
  per-candidate 循环、大树慢)。CUDA kernel 是 perf 那一程的后手。

## 相关 Roadmap 项

- **fused FD kernel** / **K-bucket dispatch**:直接缓解上面第三条。
- **Analytical Jacobian(tree autodiff)**:终极解,FD 随 K 线性涨的问题根除。
- **动态 `MAX_K`**:强配置下 fail-loud 会更频繁,届时再说。
- **EvoGP in-loop 集成**:本文即这条的前置 feasibility 记录。
