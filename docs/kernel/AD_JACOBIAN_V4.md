# batch_lm v4 — Forward-mode AD Jacobian (设计 + 实现 spec)

**状态**: 设计冻结, 待实现 (TDD). 目标产物 `cusr/kernel/batch_lm_ad.cu`.
**前置**: v3 `batch_lm_fusedfd.cu` (本文件的 drop-in 基线).
**日期**: 2026-06-17.

---

## 1. 动机 — 为什么 v4 是 AD, 不是路线图原定的 device-side LM state

路线图 (`INTERNAL.md:504-550`) 把 **device-side LM state / 干掉 H↔D 同步** 列为"下一刀",
理由是省同步开销。但 **A100 实测 (`results/profile_fusedfd_a100__20260616/`) 把这个前提否了**:

| 量 | 实测 | 结论 |
|---|---|---|
| fd_jacobian | **49–73%** of LM-loop GPU 时间 | 真正的瓶颈 |
| fd_jacobian + eval (两趟树解释) | **69–87%** | 解释器 dispatch 主导 |
| host residual + 全部 memcpy | **≤ ~13%** | device-side state 天花板只有这么高 |

`fd_jacobian` 贵的根: **K 趟冗余有限差分** —— 每个常数列扰动后, 把整棵树在 N 个点上**重新解释一遍**。
profiling 明确说瓶颈是解释器 dispatch / 栈机 (local-mem stack), 不是 FLOP。

**Forward-mode AD 直击这一点**: 树只解释**一遍** (或分桶后 `ceil(K/W)` 遍), 一次前向传播同时算出
值和对所有常数的导数。dispatch 次数从 `1+K` 降到 `ceil(K/W)`:

| 负载 | FD walk 数 (1+K) | AD walk 数 (W=8) | walk 降幅 |
|---|---|---|---|
| early-gen mean_K≈3 | 4 | 1 | ~4× |
| 真实后期 mean_K≈8 | 9 | 1 | ~9× |
| Feynman K=17 | 18 | 3 | ~6× |

注意 profiling 跑在 **early-gen 低 K synth** 上, **低估**了真实高 K 负载的收益 (FD 跑 9–18 趟, AD 1–3 趟)。

**附带的质量红利** (契合 demand-driven 方向, 不是纯性能):
- 精确 Jacobian → 去掉 `eps_fd` 整套相对步长机制 + 假收敛风险 (`INTERNAL.md:493-497` 花力气治的病, 这里根治)。
- 预期收敛率**持平或上升** → 直接服务端到端 SR 质量。

---

## 2. 范围 (重要 — 控制 parity 表面积)

**v4 核心 = 单 kernel 替换**:
- `fd_jacobian_fused_kernel` → `ad_jacobian_kernel`, **输出同一个 `d_J`**, 逐字节 drop-in。
- `build_jtj_jtr_kernel` / `solve_kernel` / `eval` / `residual` / `loss` / host LM 控制循环 / accept-reject / λ 调度 **全部不动**。
- `eps_fd` 在 AD kernel 里不再使用 (保留 `main()` 签名稳定, 参数空置)。

**显式推迟到 v4.1** (不在本版):
- **J → JtJ 融合** (省掉 `build_jtj` 的 14% + J 的 N×M×K global 往返)。需要 per-warp K×K 归约,
  per-thread/per-warp 寄存器压力大 (K=32 → JtJ 累加器 1024 floats), 单独立项。
- **device-side LM state** (正交, v5)。
- **reverse-mode tape** (只有当 profiling 显示高 K 树在 chunked-forward 下仍是热点才上)。

drop-in 的好处: 收益可干净归因 (fd_jacobian 时间塌缩, build_jtj 不变), 且下游全部复用已 parity 验证的代码。

---

## 3. d_J 输出契约 (必须逐字节对齐 fd_jacobian)

来自 `batch_lm_fusedfd.cu:234, 256, 271`:

```
J 基址 (per tree m):  Jm = J + (size_t)m * K_max * N
写入:                  Jm[(size_t)j * N + i] = ∂y_i / ∂c_j      for j < K[m], i < N
列 j >= K[m]:          不写 (build_jtj 也不读)
K == 0 的树:           直接 return (无常数可微)
```

FD 写的是 `(eval(c_j+h) - y_base) / h ≈ ∂y_i/∂c_j` (前向差分)。AD 写**精确** `∂y_i/∂c_j` —— 同一个量, 同一个槽。
布局 = warp-per-tree, 32 lane 跨 N 跨步 (`for i = lane; i < N; i += 32`), 与 fd_jacobian 一致。

---

## 4. Forward-mode JVP 解释器

把 `eval_tree_d` (`batch_lm_fusedfd.cu:109-157`) 的标量栈机升级成**带切向量**的栈机。
每个栈槽从 `float v` 变成 `(float v, float t[W])`, 其中 `t[w] = ∂v/∂c_{col_lo+w}`。

**签名** (放在 `ad_interp.cuh`, 标 `__host__ __device__` 以便 host 单测):
```c
// 算出值 + 对常数列 [col_lo, col_lo+W) 的切向量, 单点。
// W 是 compile-time chunk 宽 (AD_W). 返回值写 *out_val, 切向量写 out_t[0..W)。
__host__ __device__ void eval_tree_jvp_d(
    const int *nt, const float *nv, int n_nodes, const int *ci,
    const float *x, const float *c, int col_lo, int W,
    float *out_val, float *out_t);
```

**叶子 seed**:
- `N_VAR`:   push (x[v], 0)
- `N_CONST` (k = ci[i]): push (c[k], e), 其中 `e[w] = (k == col_lo+w) ? 1 : 0`

**算子导数规则表** (全 23 个, 逐 case 加一行向量更新; `da/dl/dr` 是已弹出的切向量):

UFUNC `(a, da) → (r, dr)`:
| op | r | dr[w] |
|---|---|---|
| SIN | sin a | cos(a)·da[w] |
| COS | cos a | −sin(a)·da[w] |
| TAN | tan a | (1+r²)·da[w] |
| SINH | sinh a | cosh(a)·da[w] |
| COSH | cosh a | sinh(a)·da[w] |
| TANH | tanh a | (1−r²)·da[w] |
| LOG | log a | da[w]/a |
| EXP | exp a | r·da[w] |
| INV | 1/a | −r²·da[w] |
| NEG | −a | −da[w] |
| ABS | \|a\| | sign(a)·da[w]  (sign(0):=0) |
| SQRT | √a | da[w]/(2r)  (r==0 → inf, 与 eval 一致) |

BFUNC `(l,dl),(r,dr) → (o, do)` (解释器先弹 l 再弹 rv; `o=f(l,rv)`):
| op | o | do[w] |
|---|---|---|
| ADD | l+rv | dl[w]+dr[w] |
| SUB | l−rv | dl[w]−dr[w] |
| MUL | l·rv | l·dr[w] + rv·dl[w] |
| DIV | l/rv | (dl[w] − o·dr[w]) / rv |
| POW | powf(l,rv) | rv·powf(l,rv−1)·dl[w] + (any dr≠0 ? o·logf(l)·dr[w] : 0) |
| MAX | fmaxf(l,rv) | (l>=rv) ? dl[w] : dr[w] |
| MIN | fminf(l,rv) | (l<=rv) ? dl[w] : dr[w] |
| LT/GT/LE/GE | 0或1 | 0 |

**POW 注意**: 指数常为定常数 (其 `dr==0`)。此时**不要**算 `logf(l)` (l≤0 会假 NaN) —— 用
`dr≠0` 守卫跳过第二项。base/exponent 都可能含参的一般情形才走完整公式。

**奇点一致性**: LOG(a≤0) / SQRT(a<0) / INV(a==0) / DIV(rv==0) 的**值**本来就经 `logf/sqrtf/1.0f/a`
走到 inf/nan; 切向量同样走到 inf/nan, 与 eval 的 NaN 传播一致 → 下游 `isfinite` 判据 (`:576`) 照常兜底。
**不需要**额外特判 (POW 的 log 守卫除外)。

---

## 5. ad_jacobian_kernel (chunked forward, drop-in)

```c
// AD_W: compile-time chunk 宽, 默认 8 (-DAD_W=N 可调)。
// 每个 warp 一棵树, 32 lane 跨 N; 外层按 AD_W 分桶遍历 K 列。
__global__ void ad_jacobian_kernel(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N,
    const float *c_all, int K_max, float *J)   // 注意: 无 eps, 无 y_base
{
    // ... 同 fd_jacobian 的 warp/tree/指针 setup (m, lane, nt/nv/ci/c, Jm = J + m*K_max*N) ...
    int K = meta.K;  if (K == 0) return;
    for (int col_lo = 0; col_lo < K; col_lo += AD_W) {
        int W = min(AD_W, K - col_lo);
        for (int i = lane; i < N; i += 32) {
            float val, t[AD_W];
            eval_tree_jvp_d(nt, nv, meta.n_nodes, ci, xs + i*n_vars, c, col_lo, W, &val, t);
            for (int w = 0; w < W; w++)
                Jm[(size_t)(col_lo + w) * N + i] = t[w];   // 精确 ∂y_i/∂c_{col_lo+w}
        }
    }
}
```

**per-thread local-mem 预算** (W=8): 栈值 `[MAX_STACK]` 64 + 栈切向量 `[MAX_STACK][W]` 512 ≈ **576 floats ≈ 2.3 KB**。
(对比真·单趟 W=K_max=32: 64×33 ≈ 8 KB/thread, 占用率崩 → 所以默认 chunk, 不做满宽单趟。)

**调用点** (`batch_lm_fusedfd.cu:498-502` 处): 把 `fd_jacobian_fused_kernel<<<...>>>(... d_y, eps_fd, K_max, d_J)`
换成 `ad_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, K_max, d_J)`。
其余循环体一字不改。

### 5.1 文件 / include 结构 (避免重定义 — 关键)

`ad_interp.cuh` 是以下符号的**唯一来源** (single source of truth):
- `enum NTypeE` (N_VAR..N_TFUNC) 与 `F_*` 算子枚举 (F_SIN..F_GE) — 普通 enum
- `#define MAX_STACK / MAX_K / AD_W` (用 `#ifndef` 守卫, AD_W 默认 8)
- `__host__ __device__ eval_tree_jvp_d(...)` (JVP 解释器)
- `__global__ ad_jacobian_kernel(...)` (Rung 2 加入)
- 一个**改名的** value-only 中心差分 oracle (如 `eval_tree_val_host`), 仅供 host 单测 —— **不要**命名为 `eval_tree_d` (避免与 batch_lm_ad.cu 里现有的冲突)

头文件自带 `#include "pop_format.h"` (提供 TreeMeta)。include 顺序按 rung:
- **Rung 1-2**: 测试文件 (`tests/test_jvp_host.cu` / `tests/test_ad_jac.cu`) **只** include `ad_interp.cuh`。**完全不碰** `batch_lm_ad.cu` —— 它仍是 FD 拷贝, 独立编译, 枚举各自一份, 不同编译单元无冲突。
- **Rung 3**: `batch_lm_ad.cu` 顶部 (在 `pop_format.h` 之后) `#include "ad_interp.cuh"`, 并**删掉** batch_lm_ad.cu 里现在重复的 `enum NTypeE` / `F_*` 枚举 / `#define MAX_STACK` / `#define MAX_K` (改由头文件提供); **保留**它自己的 value-only `eval_tree_d` (eval_kernel_batched 仍用它)。头文件**不得**定义同名 `eval_tree_d`。

---

## 6. TDD 阶梯 (workflow 按此小步迭代)

起点: `batch_lm_ad.cu` = `batch_lm_fusedfd.cu` 的拷贝 (已用 FD 通过全部测试 → baseline green)。
然后逐级把 FD 换成 AD, **新测试驱动新代码, 旧测试守护替换**。

| Rung | 测试 (RED→GREEN) | 验收 | GPU? |
|---|---|---|---|
| **1. JVP 规则** | `tests/test_jvp_host.cu` (nvcc host 编译): 23 算子 + VAR/CONST seed, 每个建最小树, `eval_tree_jvp_d` 切向量 vs 中心差分 (`eval_tree_d`), max rel err < 1e-3 | 全算子 < tol | 否 (纯 host) |
| **2. J 正确性** | `-DDUMP_JAC` 调试build 在 iter0 后 dump `d_J`; 用解析 J 的小 fixture (y=c0·x+c1, y=c0·sin(c1·x), y=exp(c0·x)) 比对 | max rel err < 1e-3 | 是 |
| **3. 集成 (换 kernel 后旧测试守护)** | `BATCH_LM=../batch_lm_ad` 跑 test_fixture / _op / _scale / _stress | status==0 ∧ c_rel<5e-3 (scale tier-1); stress 见下注 | 是 |
| **4. parity gate (验收)** | `BATCH_LM=../batch_lm_ad test_parity_gate.py` 真实 pop.bin vs scipy fp64 | loss_down≥93% ∧ within_1.05x≥90% ∧ within_10x≥99%, **且 ≥ fusedfd** | 是 |
| **5. perf (可选)** | `batch_lm_ad_prof` profiling build, 对比 fusedfd 的 fd_jacobian 时间塌缩 + 端到端 speedup @ M∈{16k,64k} | J 生产时间显著下降 | 是 |

**stress 注**: `test_fixture_stress` 的状态码是 **FD 时代**的期望 (如 `log_sub_K1 → FAIL_NAN`)。AD 精确导数可能
**合理地**改变某些状态 (FD NaN 处 AD 收敛, 或反之)。改变时**以 scipy 为真值**判定, 并在 truth.json 注明理由 ——
**禁止**为迁就 FD 怪癖去阉割 AD 实现。

---

## 7. 风险 / 验证要点

- **POW 与奇点**: 最易错。Rung 1 host 单测必须覆盖 `c0^2`、`x^c0` (含参指数)、`log(c0·x)` 在 c0·x≤0 时的 NaN 传播。
- **chunk 边界**: `col_lo+w` 索引、`W=min(AD_W, K-col_lo)` 的尾桶、K 不被 AD_W 整除的树。Rung 2 用 K=3、K=9 fixture 卡。
- **占用率/lmem**: W=8 起步; 若 Rung 5 显示深树 lmem 拖垮占用, 调 AD_W 或上 reverse-mode (v4.1)。
- **fast-math 一致性**: host 单测用普通 fp32 math, device 用 `--use_fast_math` 快内联; Rung 1 验**规则代数**
  (tol 1e-3 容差吸收), Rung 4 验**端到端**含 fast-math。两层分离, 不混。
- **byte-parity 边界**: 仅 J 生产 kernel 变了; 可选断言 status.bin/c_final.bin 与 fusedfd 的**收敛趋势**一致
  (不要求 bit-identical —— AD≠FD, 但 parity-vs-scipy 应持平或更好)。

---

## 8. 构建

`cusr/kernel/Makefile` 加 (镜像 fusedfd 的两个 target):
```makefile
batch_lm_ad: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ batch_lm_ad.cu loader.c
batch_lm_ad_prof: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -o $@ batch_lm_ad.cu loader.c
```
flags 与基线一致: `-O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math`。GPU pin: `CUDA_VISIBLE_DEVICES=0`。

---

## 9. 实测结果 (2026-06-17, A100, pop.bin 1000 棵)

**实现状态**: Rung 1-3 全绿。host JVP 单测 33/33; 解析 J GPU 单测 4/4; 集成 fixture (fixture/op/scale tier-1/stress) 全过, c_rel ~1e-7。`eps_fd` 已退役 (AD 精确), scale 极端量级 1e±6 不再依赖相对步长调参即过。

> ⚠️ **2026-06-17 更新 — 下方旧 (pre-fix) 数字已被 stop-criterion 修复 (commit `8b9234b`) 取代。**
> 当时的 "AD 净更鲁棒 494>432" 是**假收敛膨胀**: LM accept/reject 把 uphill step 当收敛提交了 (没查
> `loss_try<=loss`, 见 `tests/test_convergence_honesty.py`)。两引擎已修 (canonical LM 顺序, h_loss
> 单调非增)。**以下全部为修复后实测。**

**parity gate (post-fix, vs scipy fp64, pop.bin)**:

| 指标 | AD (v4) | fusedfd (老版, 同机) | 阈值 |
|---|---|---|---|
| loss_down | 93.3% | 94.0% | ≥93 ✅/✅ |
| within_1.05× | **84.9%** ❌ | 92.1% ✅ | ≥90 |
| within_10× | 100.0% | 99.7% | ≥99 ✅/✅ |
| converged | 339 | 347 | — |

(修复前曾是 AD 80.8 / FD 92.4, converged 494/432 — 那 494 的"领先"是假收敛。)

**集合受控判定 (决定性, post-fix)** — AD&FD&scipy 都收敛的同一 stable 集上:
- pop.bin (stable 258): AD within_1.05× **91.1%** ≈ FD **89.9%** (AD +1.2pp)。
- 高 K feyn_I.18.12 (stable 986): AD **81.9%** ≈ FD **82.3%** (-0.3pp)。
→ **两个 corpus 同集合零质量回归**: 两边都收敛的树, AD 找的极小点和 FD 一样好。

AD raw within_1.05× 没过 (84.9) **纯是集合构成**: AD 多收 73 棵难树 (within_1.05× 仅 56.2%) 稀释了分母 (stable 91.1% → 84.9%)。**FD 的高分靠放弃这些难树。** within_1.05× **不是集合不变量**, 不应据此判 AD 劣于 FD。

**性能**: pop.bin 是 small-M, trees/s 噪声大 (post-fix 实测 AD ~2.3–2.6k vs FD ~2.1–2.2k; 代表性吞吐见下方 benchmark corpus 表 ~1.23–1.29×, 别引 pop.bin 单点)。AD kernel 44 寄存器 (FD 45), lmem 2304 B/thread (AD_W=8 single-chunk 覆盖 K_max=8 的真实负载)。rung-5 profiling (-DPROFILE, pop.bin): **J 生产段 88.7→29.7ms (3.0× 塌缩)**, loop 131→77ms (1.70×); J 生产从 FD loop 的 67.5% 降到 AD 的 38.6%, 两趟树解释 (ad_jac+eval=57ms) 现占 74% → v4.1 融合 eval 是下一杠杆。(脚本在 `tests/v4_validation/`; 原始 `.txt` 输出 gitignore, 跑脚本重生。)

**真实负载吞吐 (benchmark corpus, 2026-06-17; 脚本 `tests/v4_validation/{operon_sweep,throughput_sweep}.py`)**: 端到端 AD/FD **~1.23–1.29×, 跨 scale 稳定, 不随 M 放大**:

| 负载 | M | mean K | AD/FD (端到端 trees/s) |
|---|---|---|---|
| synth early-gen | 64k | 1.7 | 1.22× |
| synth early-gen | 256k | 1.7 | 1.12× |
| Operon pre-CO (len32) | 4k | 6–11 | 1.23× (中位, 15 cells) |
| nguyen_5 (EvoGP) | **65k** | 11 | **1.27×** |

**为什么不随 scale 放大** (反直觉, 重要): loop speedup 在全 K 区间稳定 ~1.27–1.3×, 因为 J 生产的**加速比**与其**占 loop 比例**反向抵消。J 生产单核加速随 K **非单调**: K~2.8 (EvoGP 设计负载) **3.0×**, K~1.7 1.76×, K~11 (Operon dense) **仅 1.49×** —— 高 K 时 chunked-forward 每趟携带 ~K 个切向量, **算术主导**, walk-count 优势 (FD `1+K` 趟 → AD `ceil(K/8)` 趟) 被吃掉。且 build_jtj (高 K 下占 AD loop ~20%) + eval 是 AD 不碰的共享开销。

→ **AD 的甜点是 K~2-5 的 EvoGP 设计负载**; Operon dense (K 12-22) 是 stress test (operon_baseline README 自述 "~2.8× denser than evogp, not the kernel's design workload")。高 K 大 M 要再快必须 **v4.1**: 融合 J→JtJ 免物化 d_J (砍掉那 ~20% build_jtj) + 按 K 分桶调 AD_W (高 K 时 AD_W=8 携带过宽)。

**12 角度 adversarial review**: 0 个被确认的 high/critical bug。两个 HIGH (POW 负底数未测 / 高 K 跨 chunk 未测) 均被复核**驳回** (NaN 是正确契约且与 FD 一致 / 跨 chunk 干扰结构上不可能)。jvp-rules / warp-lane / chunk-boundary / djac-layout / lm-loop-semantics 全 clean。

**收敛权衡 (post-fix, 诚实)**: 修复前的 "AD 净 +62 更鲁棒" 是假收敛, 已撤。诚实数字: pop.bin AD 339 ≈ FD 347 (FD 略多); 高 K feyn FD 收敛数明显多 (2051 vs 1393)。但 `coverage_probe.py` 证明高 K 这个差距 **~92% 是标签假象**: FD 多收的 957 棵里, AD 在 **91.8%** 上找到了**同样好的 fp64 拟合** (中位 ratio 1.0), 只是诚实判据下被标 MAXITER。真·FD 更好的仅 ~8% (78 棵), 反向 AD 真赢 ~62 棵 → 净尾巴基本对消。

**结论**: v4 AD **正确**; **交付拟合质量处处 ≈ FD** (set-controlled 双 corpus); 真实优势 = **~1.25–1.3× 吞吐 + 精确 J 免 eps**。旧 "净更鲁棒" 已证伪 (假收敛), "高 K 覆盖差" ~92% 是标签。**评判用 set-controlled / 交付 loss, 勿用收敛数。**

**采纳决策 (post-fix, 已 informed)**: 诚实重测后 AD 交付质量处处 ≈ FD (set-controlled 双 corpus), 更快, 精确免 eps; 挡 AD 的两条保留 (鲁棒性领先 / 高 K 覆盖) 一条是假收敛、一条 ~92% 是标签 → **AD 在 EvoGP 设计负载 (低 K) 是干净的 adopt, 高 K 上交付质量也站得住** (但见下方 fp32-honesty caveat: AD 对 fast-math 近奇点欺骗暴露更大, 采纳建议配 fp64 guard 或先上 trust-region)。raw within_1.05× gate (90%) 不是集合不变量, 不应作唯一闸门。
- **gate 建议**: within_1.05× 评判改到 set-controlled stable 集 (`setcontrolled.py` 已 env 参数化, 任意 corpus 可跑)。
- **复现**: `cusr/kernel/tests/v4_validation/` (setcontrolled / coverage_probe / headtohead / *sweep)。
- **仍开 — fp32-honesty (#1, status==0 yet fp64-worse)**: 见 `tests/fp32_honesty_regression/` (基线 fusedfd 57 / ad 208 serious)。stop-criterion 修复**部分缓解**: post-fix **ad 208→141, fd 57→28** (fp32-可见的 uphill 现诚实报 status!=0); 但**纯 fast-math 欺骗** (fp32 说改善、fp64 灾难, 近奇点; AD 最坏 1.5e16×) 残留, 且 **AD 暴露远高于 FD (141 vs 28)** —— 精确导数把树推奇点更狠。需 **trust-region / fast-math audit / fp64 边界审计** 根治 (同时治两引擎)。**这是 AD 采纳的真实 caveat**: 交付质量 ≈ FD, 但 fp32-欺骗暴露更大, 建议配 fp64-honesty guard。
- **v4.1**: 融合 J→JtJ 免物化 d_J (砍 build_jtj ~20%) + 按 K 分桶 AD_W。
