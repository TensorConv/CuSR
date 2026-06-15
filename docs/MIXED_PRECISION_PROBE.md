# 混合精度 K×K 解探针 — 实测 (refuted RESULTS 结论 #3)

> 2026-06-13. 5070Ti laptop. 起因: `RESULTS_laptop.md` 结论 #3 主张
> "inner-const-heavy tier-B 65% 软肋, 根因 fp32 高 K Cholesky 崩, 下一步混合精度 K×K 解".
> per-tree 交叉表 (见下) 把这条**证伪**了.

## 方法

* kernel 加编译期开关 `-DSOLVE_FP64` (`008/batch_lm_fusedfd.cu` solve_kernel):
  cast fp32 JtJ/JtR → double, double 里分解+回代, δ cast 回 fp32. per-point
  eval/Jacobian 仍 fp32. damping 形式不动 (relative `A[jj]*=(1+λ)`), 隔离纯精度效应.
* A = 无 flag (fp32 基线), B = `-DSOLVE_FP64`. 同源同 flag 同机现编
  (`-O2 -arch=native -std=c++17 -lineinfo --use_fast_math`).
* gate #1: 编辑后 fp32-A 复现冻结基线 → **PASS** (chol=652, tier-B=2566, bit 对齐).
* 探针 `probe_fp64solve.py`: A/B × preset, 读 per-tree raw status + c_final,
  fp64 interp 重算 loss, classify_tiers, 出 A-vs-B 交叉表.

## 结果

| preset | tier-B A→B | Cholesky崩 A→B | 净 tier-B |
|---|---|---|---|
| inner-const-heavy (真实) | 2566 → 2558 | 652 → 682 | **−8** |
| early-gen (真实) | 3206 → 3203 | 475 → 471 | −3 |
| late-gen-bloated (真实) | 2513 → 2513 | 2456 → 2457 | +0 |
| synth-inner-const-heavy | 3224 → 3208 | 267 → 293 | −16 |

**fp64 K×K 解没救回任何 tier-B, 净值微负** (churn). chol 崩数反而 +30
(fp32 偶尔把负主元舍入成小正数蒙混过关; fp64 老实判 s≤0, 暴露更多).

inner-const-heavy 那 520 棵 chol-miss 在 B 里的归宿:

* → 救回 tier-B: **6 (1.2%)**
* → 仍 Cholesky 崩 s≤0: **501 (96.3%)**
* → 收敛却仍差 5%: 9 (1.7%)
* → 其它: 4

## 诊断

* 这 501 棵在 **λ 已 >1e12** (FAIL_CHOLESKY 的判据) 且 **fp64** 下仍 s≤0.
* relative damping 下主元 ~ `A[jj]·(1+λ)`. λ>1e12 还 s≤0 ⟹ `A[jj]≈0`
  (该常数曲率 = Σ_i J[i,j]² ≈ 0) ⟹ **死常数 / 不可辨识常数** (高 K bloated 树
  里冗余常数组合普遍: 如 `c0·c1·x` 只有乘积可辨识, `sin(c0+c1)` 只有和可辨识).
* 这是**秩亏**, 不是 fp32 条件数. fp64 无精度可恢复 (0 在任何精度都是 0).
* 现行代码 s≤0 → 整个 solve abort → δ=0 (所有常数都不动) → 树停滞 → miss tier-B.
  **不是因为拟合不可能, 是因为求解器一遇退化方向就整步放弃.**

## 结论

* **RESULTS 结论 #3 的处方 (混合精度 K×K 解) 证伪** —— 对 tier-B 软肋无效.
  需把 RESULTS_laptop.md 结论 #3 订正 (实测数字, 非推测).

## rung C: pivot-floor (放行退化方向, 实测)

`-DPIVOT_FLOOR=ε`: Cholesky 主元 `s = max(s, ε)` 不 abort, 让 solve 继续优化
可辨识方向 (纯 fp32, damping 形式不动, 只碰退化主元). ε 两点防单点误导.

| 档 | tier-B | Δ | chol崩 | 良态树churn | 520 chol-miss: →tier-B / →收敛短 / →maxit短 / →NaN |
|---|---|---|---|---|---|
| A (基线) | 2566 (64.7%) | — | 652 | — | — |
| **F9 (ε=1e-9)** | **2813 (70.9%)** | **+6.2pp** | 0 | **7** ✓干净 | 246 (47%) / 206 (40%) / 67 (13%) / 3 |
| F3 (ε=1e-3) | 3186 (80.3%) | +15.6pp | 0 | **410** ✗污染 | 269 (52%) / 196 (38%) / 55 (11%) / 0 |

* **F9 是干净判别档** (只动 7 棵良态树): 放行退化方向**确实救回约一半** chol-miss
  (246/520 → tier-B), tier-B 64.7→70.9%. 0 NaN, 数值安全.
* **另一半不可约**: 206+67=273 棵放行后迁移到"收敛/maxiter 却仍差 5%" —— 这些树
  常数放行后**仍拟合不好** (fp32 Jacobian 天花板 / 高 K EvoGP 树本就拟不上目标).
* **F3 vs F9 = 归因差异, 不是真假**: F3 的 +15.6pp 也是真 tier-B (3186 棵 fp64 重算
  loss 确在 5% 内, 合法 fit). 但 F9 的 +247 几乎全来自那 520 目标集 (干净隔离"放行
  退化方向"), F3 的 +620 只有 269 来自 520, 另 ~351 来自 A 本就处理好的树 → F3 是
  **更宽的全局 ridge**, 不是同一修法的更干净版. 所以可修量是**区间**, 不是单点.

## 净结论 (实测, 给 paper)

1. tier-B 65% 软肋**不是精度问题** (fp64 解 +0), 是**秩亏 + 求解器整步放弃退化方向**.
2. 可修量是**区间 +6pp ~ +15pp** (F9 定向放行下界 / 更激进 floor 连边际树一起的上界);
   其余是 fp32 Jacobian / 不可辨识常数的拟合地板. 比"用混合精度修好"诚实.
   **病根与正确修法 (代数订正, 勿用 "column-scaled 阻尼" 简写)**:
   * kernel `A[jj]*=(1+λ)` 恒等于 `JᵀJ + λ·diag(JᵀJ)` = `JᵀJ + λD²` (D²=diag JᵀJ = 列范数²).
     在**零对角线**树 (死常数, J 列≡0, ∂out/∂c=0) 上 λ·0=0 抬不起主元 → Cholesky s≤0.
     ⟹ "换成 column-scaled `JᵀJ+λD²`" 是 **no-op** (kernel 已是), 修不了. (相关列/off-diag
     退化 relative damping 本就能修, det>0; 失败的是零对角线那类.)
   * 真正两条修法:
     - **(便宜, 留现架构) 绝对阻尼** `JᵀJ + λI` 或主元地板 `s=max(s,ε)`: 给死方向加绝对正量,
       Cholesky 能解, 死方向 δ→0 无害, 可辨识常数照常优化. F9/F3 即此族, ε/floor 尺度是 wart.
     - **(鲁棒, 像 oracle) QR-on-J + 列主元** (MINPACK lmder 真实做法): 不形成 JᵀJ (不平方
       条件数 κ(JᵀJ)=κ(J)²), rank-revealing 优雅把不可辨识方向置 0. 这才是 scipy 同批树拿
       好 L* 的真正原因 —— 不是 λD² 阻尼项 (那 kernel 已有). 代价: warp-per-tree QR 实现量大.
3. **是否采纳进 kernel = 策略决定, 非本轮 inline 能定**:
   * 扰动每条轨迹 → 动 008 parity gate. 采纳门是**全 preset no-harm + parity**, 不是只
     inner-const-heavy (floor 只在此 preset 测过). F3 已证 floor 会动良态树, 而扰动的
     **符号可能 per-preset 不同** (这里救了, 早期 preset 可能反而回退) → 必须全 preset 重测.
   * 冻结 demonstrator recentering 下, 要不要动 kernel tier-B 本身是不是 critical path, 待用户定.
   本轮只做到"实测点名正确方向 + 量化可修/不可约区间", 不擅自采纳.
* 吞吐: --quiet 抑制了 binary 自报时间, 未测; 解法既已转向, 吞吐对比留到选定方案后.

## rung D 调研: QR-on-J / fp64-JtJ + rank-reveal (写 kernel 前的 gating 测量)

用户倾向"修法二 (QR-on-J)". 写 kernel 前先用 fp64 numpy 量条件数 (探针
`probe_conditioning.py`), 结论是**两条修法都不该建** —— 问题不在求解器:

* **失败树是结构秩亏, 不是有限近奇异.** 520 chol-miss 里 99% rank<K (典型 deficit
  3-4 = 高 K 树半数常数冗余); "满秩近奇异" = 0 (注: 此数是 rank 阈值的同义反复, 真证据
  是 deficit 分布 + 下条). ⟹ QR 的招牌好处 (避免 κ 平方保住有限小奇异值) **无对象可作用**.
* **fp32 累加噪声才是 F9/F3 两难的根**: 1000 项 fp32 累加把真·秩亏方向的主元埋到 ~1e-4
  相对量级, 和合法小方向重叠. fp64 累加压到 ~1e-13. (这本是支持 fp64-JtJ 而非 QR 的理由.)
* **但 gating 测量否决了 fp64-JtJ+rank-reveal 能干净分离**: 健康达标树 (绝不能扰动的)
  **自己也普遍秩亏** —— κ(J) 中位 **1.3e10**, 92% κ>1e6. 失败树与健康树的谱**完全重叠**
  (都横跨 κ 1e3–1e44). ⟹ **没有 rank-reveal 容差能只截断失败树而不动健康树.**
* 零对角线 (死常数) 这条更软的轴: 失败树 67% 有零对角线 (可检测+pin), 但健康达标树
  也 4.4% 有 → 连这条都非零误伤. 根本重叠仍在.

**净结论 (调研, 给 paper)**: inner-const-heavy 的 tier-B 65% **不是可干净修的 solve bug**.
高 K bloated GP 树**普遍常数秩亏** (健康树和失败树一样), 任何足以救回失败的正则化都会扰动
working-but-rank-deficient 的多数树. 是 **recovery↔churn 权衡** (F9 +6pp/0.2%churn ↔
F3 +15pp/12%churn), 不是 clean win. **QR-on-J 逃不掉** (重叠在问题里不在求解器) → QR 双重
overkill. 三个落点:
  1. **最干净可得**: 检测+pin 零对角线死常数 ≈ +6pp, ~4% 健康 churn. 便宜 (对角线检查),
     无 QR, 吞吐~免费. 仍非零误伤.
  2. **+6→+15pp**: 需 Schur 主元截断, 重度 churn working 树. 不 clean.
  3. **判据应是选择保真度 (§3c) 不是 tier-B 数 / bit-parity** —— 已测, 化解了"权衡":

     | 档 | tier-B | Spearman | top10% | top25% |
     |---|---|---|---|---|
     | A | 2566 | 0.815 | 0.553 | 0.765 |
     | F9 | 2813 | 0.867 | 0.838 | 0.866 |
     | F3 | 3186 | 0.868 | 0.866 | 0.882 |

     * 选择保真度**不降反升** (Spearman 0.81→0.87, top10% 0.55→0.84): 救回的树现在排对了位,
       健康树 churn 对排序无害. 所谓 recovery↔churn 权衡在**正确判据 (选择) 下消失** = 净赢.
     * **F9 (保守/低 churn) 拿到几乎全部收益; F3 (+15pp tier-B) 对选择 0 额外贡献** (0.867 vs 0.868)
       → 保守档是甜点, 激进正则化不值.
     * ⟹ 当时结论: floor 在选择判据下是净赢, 不需要 QR. **但下文产品化时被 W0 gate 否决** ——
       固定地板会在别的数据集上巨步炸树; 见"产品化尝试"节. floor 仅 6 preset 选择保真度好不够.
* 吞吐: fp64-*形成* (非解) 改 build_jtj (O(NK²)) + 2× buffer 流量, **非免费** (未测);
  太贵则 Kahan 补偿求和兜底. 但既然 solve 改动整体不 clean, 这点已次要.

## 产品化验证 (保守 floor 全 preset, `validate_floor_allpresets.py`)

A vs F9 (保守 floor ε=1e-9), 判据 = 选择保真度 no-harm (§3c, 非 bit-parity):

| preset | tier-B A→F9 | Spearman | top10% | gate |
|---|---|---|---|---|
| early-gen | 3206→3357 | 0.935→0.967 | 0.895→0.938 | PASS |
| late-gen-bloated | 2513→2627 | 0.828→0.828 | 0.430→**0.372** | (见下) |
| inner-const-heavy | 2566→2813 | 0.815→0.867 | 0.553→0.838 | PASS |
| synth-early-gen | 3299→3310 | 0.943→0.952 | 0.958→0.958 | PASS |
| synth-late-gen-bloated | 2634→2688 | 0.983→0.996 | 0.960→0.989 | PASS |
| synth-inner-const-heavy | 3224→3260 | 0.913→0.925 | 0.889→0.894 | PASS |

* **Spearman (稳健全局排序) 全 6 preset 不退化** (late-gen 持平, 5 个提升); tier-B 全升.
* late-gen top10 唯一"退化" (0.43→0.37) = **tie-噪声, 非真退化**: 其 top-10% 280 棵 loss 全在
  [9.41e-11, 9.47e-11] (max/min=1.01, 完全并列在机器零附近), 选哪 280 棵由亚 1% 噪声定; 顶部
  1832 棵都 ≤1.05×最优. Spearman 持平佐证全局排序未动. (对照 inner-const top10 loss 跨 3.7×,
  那里 0.55→0.84 是真提升.) note 早记过无噪声真实 preset top10 = 并列排序噪声.
* ⟹ **no-harm gate 通过** (稳健判据). floor 在高 K (真实痛点) 是真赢, 别处无害.

## 产品化尝试 → W0 parity gate 否决 (floor 不安全, 已撤销)

把 floor 设默认后跑 008 W0 parity gate (独立 `data/pop.bin`, vs scipy fp64, 非调过的 6 preset):

| build | loss-down | within-1.05× | within-10× | gate |
|---|---|---|---|---|
| nofloor (基线) | 94.0% | 92.4% | 99.8% | 3/3 PASS (复现校准基线) |
| floor ε=1e-9 | 97.6% | **86.6%** | **96.7%** | **FAIL** |

诊断 (vs nofloor 逐树, fp64 重算 loss): floor 让 **123 棵树变差, ratio 中位 59×** (101 棵>2×,
76 棵>10×, 最坏 26000×). **其中 91 棵是 nofloor 的 MAXITER 树** (从没 s≤0) + 30 棵 chol崩.
机制**未完全诊断**: 对 MAXITER 树, `max(s,ε)` 是把小正主元**抬高** → 步长反而**变小**, 所以
"b/√ε 巨步"只解释那 30 棵真 s≤0 的. MAXITER 那批更可能是 floor 让一个 marginal 步被**接受**,
λ 就不再升到那些树本依赖的重阻尼 → 偏离 (**已追到底, 见末节 trace: 实测确认正是这条 (B), 非 fp32
精度问题; fp64/fp32 在 floor 终点 = 1.00×**). 经验事实: floor 炸 W0, 主要伤 MAXITER 树.

⟹ **floor 不安全, 撤销产品化**. kernel 默认改回安全基线 (整步放弃, 过 W0 gate); floor 降为
`-DPIVOT_FLOOR` opt-in ablation. **关键教训: 6 preset 选择保真度验证不足以当 gate —— 独立的
W0 数据集才暴露退化. W0 (既有的 canonical 质量闸) 应是任何 solve 改动的第一道关, 不是最后一道.**

**候选修法 (未验证, 非平凡)**: 绝对 Levenberg 阻尼 `JᵀJ + λI`. 但它改**所有**树的阻尼, 而健康树
也普遍秩亏 → 可能同样炸 W0; 需重调 λ schedule (现 relative `*=(1+λ)`-tuned); MAXITER 树受损也
不是"换 λ 形式就好"的强证据. **列为候选, 不是答案.** 本会话已连续证伪/否决 3 个"修法"
(混合精度 +0 / 固定地板炸 W0 / λI 未验证), 每个都倒在下一次测量 → 不再带信心命名"下一步".
RESULTS_laptop.md 结论 #3 订正: 混合精度证伪, "下一步=floor"也证伪 (kernel 是冻结 demonstrator,
tier-B 未必在 critical path; 是否做 λI 待用户定).

## 复现

* kernel rung: `008/batch_lm_fusedfd.cu` 的 `SOLVE_FP64` / `PIVOT_FLOOR` 编译开关
  (默认 = 冻结 fp32 基线, 无 flag 时 bit 级不变, gate#1 已验).
* binary: `008/` 下 `nvcc ... [-DSOLVE_FP64|-DPIVOT_FLOOR=1e-9f] -o batch_lm_fusedfd_{A,B,F9,F3}`.
* 探针: `probe_fp64solve.py` (A vs B), `probe_pivotfloor.py` (A vs F9/F3),
  `trace_maxiter.py` (floor 把树搞差的机制判别, 需 kernel 加 `loss_final.bin` 写出 = 自报 fp32 loss).

## MAXITER 机制追到底 + best-seen 证伪 (2026-06-13 trace)

用户选 "best-seen + 先追 MAXITER 机制". 两半都有裁决.

**best-seen = no-op (逻辑 + 实测双证).** kernel 的 accept 是 `h_loss_try < h_loss` 严格下降才
更新 h_c (batch_lm_fusedfd.cu:534), 且 MAXITER 树从不走收敛分支 (528 会设 finished=1) → 它的
h_c 末值**就是整条轨迹 fp32-loss 最低的迭代** = best-seen-by-fp32. 在 W0 (`data/pop.bin`, M=1000)
上 floor 比 nofloor 更差的 **118 棵**里, **104 (88%) 是 MAXITER** → best-seen 对它们什么都不改.

**机制 = (B) 真更差盆地, 不是 (A) fp32 撒谎.** 判别量 = floor 的 c_final 同一点上 `floor_fp64 /
floor_fp32` (interp fp64 vs kernel 自报 fp32):

| 量 | 值 |
|---|---|
| floor 比 nofloor 更差 (fp64, >1.05×) | 118 棵, 中位 **64×**, 最坏 26056× |
| 这些树 floor-side status | MAXITER 104 / CONV 13 / NAN 1 |
| **判别 floor_fp64 / floor_fp32** | 中位 **1.00×** (0.73–1.1), >2×(A) = **0** 棵, ≤2×(B) = **118** 棵 |
| kernel 自己 fp32 账本就显示 floor 差 | 106/118 (不是被 fp64 才看出) |
| 这 118 棵 nofloor 拟合质量 (fp64<1e-3) | **118/118 全是好拟合** |
| 其中 nofloor 自己 chol-崩但 best-finite 已好 | 28/118 |

读法: fp32 和 fp64 在 floor 终点**完全一致** (1.00×) → kernel 没被 fp32 骗, 它忠实地最小化了
忠实的指标, 只是 floor 把树带进了**真的更差的盆地**. nofloor 在这 118 棵全拟合到 <1e-3 (好);
floor 全搞坏. 机制: nofloor 撞退化主元 → 拒整步 → 升 λ → 重阻尼步进**好盆地** (28 棵甚至 nofloor
chol-崩, best-finite 已停在好点); floor 把小主元抬成 ε → 放行贪心中步 → λ 不升 → 落进更差盆地.
**floor 短路了那个本该做正则的 λ-升挡; chol-abort + best-finite 本来就是对的鲁棒行为, floor 不是
修复而是破坏好树.** floor 在 inner-const-heavy 上的 tier-B "收益"是错觉/净负 —— 那个 preset 全是
难树没有好树可破坏, 聚合 tier 计数掩盖了伤害; W0 有好树 + 逐树量才暴露.

⟹ **best-seen 死路 (no-op); floor 死路 (净负, 非精度问题). 安全基线已是对的解.** tier-B 缺口对
高 K 结构秩亏树是 intrinsic. 剩余候选 (λI 等) 要想赢, 得**击败 λ-升挡** —— 而 λ-升挡对良态多数
已经正确, 任何全局改动都得先过 W0 不破坏这 118+ 棵好树. → **建议 bank**. (kernel `loss_final.bin`
诊断写出是 additive, 不碰 solve/不影响 gate; 是否保留/commit 待定.)
