# batch_lm v5 — Reverse-mode AD Jacobian(设计 + 实现)

**状态**:已实现、已验证、已提交(`chore/a100-server-bringup`,未 push)。
**代码**:`cusr/kernel/revad_interp.cuh` + `cusr/kernel/batch_lm_revad.cu`。
**它替代谁**:v4 forward-mode AD(`ad_interp.cuh` / `batch_lm_ad.cu`,设计见已归档的 [`../archive/kernel/AD_JACOBIAN_V4.md`](../archive/kernel/AD_JACOBIAN_V4.md))。
**日期**:2026-06-22(NaN-safety 同日补);本文 2026-06-23 补写。

一句话:v5 把算 Jacobian 的方式从 forward AD 换成 reverse AD(反向传播)。同样精确,但**和常数个数 K 无关地快**——这正是 LM 循环里最贵的那块(实测 Jacobian 占整个循环 60–66%)。

---

## 1. 为什么用 reverse mode

每棵树是一个标量函数:输入 K 个常数,输出 1 个值 y。要的是 Jacobian——y 对每个常数的偏导 ∂y/∂c_j,一共 K 列。

- **forward AD(v4)**:一次前向带着一束切向量推。一束宽 AD_W(默认 8),所以要跑 `ceil(K/8)` 趟。成本 **∝ K**。
- **reverse AD(v5)**:一次前向(把每个节点的本地偏导记下来)+ 一次反向扫描,**一趟就出全部 K 列**。成本 **和 K 无关**(K-flat)。

数学上:输出是标量、输入是多个,reverse 就是对的模式(就是反向传播)。K 越大,v5 相对 v4 赢得越多——forward 趟数随 K 涨,reverse 不涨。

v4 当初选 forward 是因为它直击 FD 的"K 趟重复解释树"瓶颈(见 v4 文档)。v5 更进一步:连 forward 那 `ceil(K/8)` 趟也省成一趟。

---

## 2. 怎么工作(两趟)

核心是 `eval_tree_vjp_d`(`revad_interp.cuh:66`),单点、一趟出全部 K 列。算子规则**逐条照抄 `ad_interp.cuh`**(同样的 math、同样的奇点行为),所以在有限处 reverse 的值和偏导跟 forward 逐元素一致。**只 include 复用 ad_interp.cuh,绝不改它。**

**前向趟**(`i = n_nodes-1 → 0`,和原来的 `eval_tree_d` 同序):用值栈 `sv` 算每个节点的值,同时把**本地偏导**记进 tape:
- UFUNC:`d1[i] = df/da`(比如 sin → cos(a),exp → exp(a),sqrt → 1/(2√a))
- BFUNC:`d1[i] = ∂o/∂l`,`d2[i] = ∂o/∂rv`(比如 mul → d1=rv, d2=l;div → d1=1/rv, d2=-o/rv)

**反向趟**(`i = 0 → n_nodes-1`,正好是前向的逆序):用伴随栈 `sa`,push/pop 镜像前向的值栈。起点 `sa = [1.0]`(= ∂y/∂y)。每到一个节点,先弹出它输出的 adjoint,再:
- VAR:丢弃(变量不是要拟合的常数)
- CONST:`out_grad[ci[i]] += adj`(梯度在这里累加)
- UFUNC:push `adj * d1[i]`(操作数的 adjoint)
- BFUNC:先 push `adj * d2[i]`(rv 的,压深),再 push `adj * d1[i]`(l 的,在顶)——和前向的弹序对上

`out_val` 不输出(要值用 `eval_tree_val_host`)。

---

## 3. NaN-safety:`revad_safe_mul`

反向趟里 adjoint × 本地偏导这一步,用 `revad_safe_mul`(`revad_interp.cuh:45`)而不是直接 `*`:

```c
revad_safe_mul(a, b) = (a == 0.0f || b == 0.0f) ? 0.0f : a * b;
```

**为什么**:如果某条边的两个因子里有一个**恰好是 0**,这条边对梯度的真实贡献就是 0(0 adjoint = 该节点不影响输出;0 偏导 = 该操作数不影响父节点)。直接乘会撞上 `0*inf` / `inf*0` / `0*nan` = NaN,把整列毒化掉。数学上真零因子湮灭乘积,跟另一个(可能是 inf 的)因子取什么值无关,所以返回 0 是对的。

只在"**恰为 0.0f**"时湮灭:极小的非零(denormal)仍走 `a*b`(此时积也极小,在容差内);真正的 inf 偏导配非零 adjoint(真奇点、真梯度确实是 inf)仍给 inf。

这让 v5 在**值有限但朴素 AD 会出 NaN**的奇点上给出正确的有限值——这是比 forward AD 更正确的一处,scipy fp64 核对确认(见 §6)。

**已知局限(TD-1)**:可去奇点。如果 `0·∞` 实际收敛到非零有限极限(例如 `pow(sqrt(c0),2)` 在 c0=0,真右导=1),这个 mask 会**沉默地给 0** 而不是真值。这是一阶 AD 的固有局限——forward、未加 mask 的 reverse、中心差分在这点同样失败(都给 NaN),只有符号化简或单侧差分能拿到真值。**我们的语料里 0 例**(`tests/_probe_corpus_silent.cu`:27.4M + 45.4M 个元素,全部被 mask 影响的元素真梯度都是 0)。决定:保留 mask(对真实数据 100% 正确,且 0 比 NaN 对 LM 安全),记为 TECH_DEBT 的 TD-1,留 `_probe_corpus_silent.cu` 当回归哨兵。

---

## 4. kernel + 接进 LM 循环

`rev_jacobian_kernel`(`revad_interp.cuh:166`)是 `ad_jacobian_kernel` / `fd_jacobian` 的**逐字节 drop-in**:

- 布局:warp-per-tree,32 个 lane 跨 N(`for i = lane; i < N; i += 32`)。
- 输出契约完全一致:`Jm = J + m*K_max*N`;`Jm[j*N + i] = ∂y_i/∂c_j`(j < K[m], i < N)。列 j ≥ K[m] 不写;K==0 的树直接 return。
- 和 forward 的区别:**没有外层 `col_lo += AD_W` 分桶**——一次 point-pass 就出全部 K 列。

**接法是 drop-in**(`batch_lm_revad.cu` = `batch_lm_ad.cu` 只换 Jacobian 那一发):
- `batch_lm_revad.cu:376` 把 Jacobian launch 换成 `rev_jacobian_kernel`。
- 其余全部复用 `lm_core.cuh`:JtJ 组装、Cholesky 解、residual/loss/eval、host 端 LM 控制循环、accept/reject、λ 调度——**一字不改**。
- **fp64-honesty guard 保留打开**(`batch_lm_revad.cu:514`,`#ifndef NO_FP64_GUARD`):`fp64_boundary_audit` 把"诚实 fp64 下 c_final 非有限、或比 c_init 还差"的树退回 c_init。保证交付 ≤ 起点。

**内存**:每 lane 的 tape = `d1[MAX_NODES] + d2[MAX_NODES]` 加值栈/伴随栈 `sv/sa[MAX_STACK]`,约 **1.5KB/lane** local mem——比 forward AD 的 `st[MAX_STACK*AD_W]` ≈ 2.3KB **还省**。`MAX_NODES=128`(实测最大 n_nodes=32,留 4× 余量);`batch_lm_revad.cu:253` 在 load 期断言 `max(n_nodes) ≤ MAX_NODES`,超了报错让你增大重编。

---

## 5. 实测性能

(median-5,2 块独立 A100 + 自测一致;`batch_lm_ad_prof` vs `batch_lm_revad_prof`,`--max-iter 50`)

| 负载 | Jacobian kernel | 整个 LM 循环 |
|---|---|---|
| inner-const(高 K,K_max=13) | **~1.8×** | ~1.27× |
| early-gen(低 K) | ~1.48× | ~1.15× |

- **K 越大赢越多**:forward 趟数 ∝ K,reverse 与 K 无关。
- 循环加速 < Jacobian 加速,是 Amdahl:Jacobian 提速后从占循环 ~50% 缩到 ~30%,其余(eval、build_jtj)v5 不碰。
- NaN-safety 的开销:Jacobian +2.3%(inner-const)/ +3.6%(early-gen),都 < 5%。

---

## 6. 正确性怎么验的

- **scipy fp64 oracle**:NaN 修复后样本 **404/404 一致,0 残留**;全语料 host 探针(`_check_residual.cu`,所有 27.4M + 45.4M 元素)**0 个 reverse 非有限、0 污染**。
- **reverse 比 forward 更正确**:forward 的分块 JVP 有 NaN 污染 bug(`NaN*0=NaN`,一个非有限本地偏导毒化整束切向量、连累兄弟列);reverse 的逐节点 adjoint 把列**隔离**开,严格更少 NaN、**从不更差**(rev-worse=0)。scipy 在 68/68 个有争议元素上判 reverse 对。
- **mutation test 10/10**:故意改坏代码(改符号、换 d1/d2、改 push 序、删 zero-init……)全被测试抓到。
- **anti-cheat**:7-agent 对抗 workflow + 一次 Codex review。Codex 抓到了我自己 anti-cheat 漏掉的一个 overclaim(reverse 自己也会在 `0*inf` 处 co-NaN),当天用 `revad_safe_mul` 修掉并重验。
- 测试:`tests/test_revad_{host,jac,parity}.cu`(+ `_nofast` / `_tiny` 变体)。产物在 `experiments/revad_v5/`(FINDINGS、scipy 对比、mutation、Python fp32 镜像 `reverse_vjp_py.py`)。

---

## 7. 边界 / 注意

- **v5 是速度,不是质量**。它不修高 K 的**秩亏天花板**(那是 workload 的内在难度,damping/fp64/floor/QR 全证伪,见 [`../MIXED_PRECISION_PROBE.md`](../MIXED_PRECISION_PROBE.md))。fp32-honesty 的 caveat 和 v4 一样,靠 fp64 guard 兜。
- **TD-1 可去奇点**(§3):真实语料 0 例,但代码里是真实存在的一阶 AD 局限,见 `TECH_DEBT.md`。
- 对论文(C2):e6 里的 `ad` 是 **forward v4**;C2 的吞吐数字要用 **`batch_lm_revad`** 重跑。v4→v5 本身(更快 + 修了 forward 的 NaN 污染)就是一个干净的系统结果,值得报。

---

## 8. 文件 / 构建 / 提交

**文件**
- `cusr/kernel/revad_interp.cuh` —— VJP 解释器 + `rev_jacobian_kernel`(本设计的核心)
- `cusr/kernel/batch_lm_revad.cu` —— LM 驱动(= batch_lm_ad.cu 换 Jacobian 一发 + load 期 MAX_NODES 断言 + fp64 guard)
- `cusr/kernel/tests/test_revad_{host,jac,parity}.cu` —— 单测
- `experiments/revad_v5/` —— 验证产物(scipy、mutation、FINDINGS)

**构建**(`cusr/kernel/Makefile`)
```bash
make batch_lm_revad        # 部署用
make batch_lm_revad_prof   # -DPROFILE,带 Jacobian 段计时
```
flags 同基线:`-O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math`。

**提交**(`chore/a100-server-bringup`,未 push):
- `0908af2` v5 Jacobian(drop-in、更快、修 forward NaN 污染)
- `a9a1197` 验证产物
- `4db3f11` NaN-safety(`revad_safe_mul` + gate 修复 + 探针 + TECH_DEBT)
- `bfe7ef2` NaN-safety 验证产物
