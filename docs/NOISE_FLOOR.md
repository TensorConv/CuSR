# 噪声地板 — 合成 preset 的绝对参照 + oracle 基底吸引域发现

> 2026-06-13. 5070Ti laptop. PROTOCOL §3c "合成 preset 另报噪声地板" 的兑现.
> 实现: `gen_synth.py` 落 `c_true` sidecar (`*.ctrue.npy`); `runner.py` 遇 sidecar
> 即算 `interp.loss_pop(pop, c_true)` = 地板, 与 oracle / 各后端走同一条 fp64 loss 路径.

## 什么是噪声地板

合成树的目标 `y = f(x; c_true) + 1% 噪声`. 把**真常数** `c_true` 代回算的 loss =
注入噪声的能量, 是统计下界: 任何 c 比它更低都是在拟合噪声. 真实 EvoGP preset 无
`c_true` (那些树多数表达不了目标), 故地板**仅合成 preset** 有.

oracle (scipy) 在 c 空间搜索, `c_true` 是其中一个候选点 ⟹ 数学上 `L* ≤ loss(c_true)`.
所以"oracle 落在地板之上 >0.5%"只可能是 **oracle 没找到 c_true 那个盆地**.

## 数字 (M=4000, fp32 kernel = fusedfd)

| preset | 地板 loss 中位 | oracle 够地板 ≤1.005× | kernel 够地板 ≤1.005× | 差 |
|---|---|---|---|---|
| synth-early-gen | 9.45e-02 | 79.7% | 78.4% | 1.3pp |
| synth-late-gen-bloated | 3.58e+00 | 84.4% | 82.1% | 2.3pp |
| synth-inner-const-heavy | 5.27e-02 | **48.1%** | **46.7%** | 1.4pp |

**头条 = 不变量**: kernel 够到真极小的比例 ≈ oracle, 全 preset 差 ≤2.3pp. 残余高 K
"软肋" (inner-const-heavy 48%) 是**对 fp64 CPU reference 同样成立的 problem-hardness**,
不是 kernel 缺陷. 地板是相对 tier 的**绝对兜底**: tier-B 说"kernel 追平 scipy 的 LM",
地板说"kernel 够到真极小的频次和 scipy 一样" —— 正面回答"你 oracle 自己就不是最优,
tier 没意义"那类质疑.

## oracle 为何够不到地板 = 基底吸引域 (实测, 非预算/精度)

证伪两个朴素解释 (`probe_floor_basin.py`, 每 preset 抽 150 棵 converged-miss):

* **不是 max_nfev=200 预算**: 够不到的树绝大多数 status=converged, 非 iter-limit
  (conv-miss : cap-miss = 625:95 / 387:43 / **1648:408**).
* **不是 c_true 不是真极小**: scipy 从 `c_true` 重启 → **100% 守住地板** (median L/floor
  0.998–0.999, 即有限样本 MLE 比地板低 ~K/N, 健康; 也反证 sidecar 对齐无误).
* 从 `c_init` 重启复现 miss (median L/floor ~1e4, 0% 够地板).

⟹ 从**演化式 ±30% 初值** (`c_init = c_true·U(0.7,1.3)+N(0,0.02)`) 出发, LM 收敛到
另一个差得多的局部点 —— 基底吸引域问题. (不区分真·多峰与 FD-Jacobian 坏步; kernel
同样用 FD, 两种解释下结论都成立. 想区分可给解析 Jacobian 重跑, 收益低未做.)

## 读法注意

* **reach% 随初值散布而变**, ±30% 是生成器假设 —— 报数字必须带初值口径, 否则会被读成
  "scipy 只能解一半 CO". 真正初值无关的是 oracle≈kernel 的**跟随**, 主推这个.
* **与秩亏发现 (MIXED_PRECISION_PROBE) 区分, 别合并**: 那是**真实** inner-const-heavy 的
  tier-B 软肋, 根因秩亏/死常数 (求解器遇退化方向整步放弃 — 此**机制** kernel 特有, scipy 用 QR
  不放弃; 但它**暴露的秩亏是 shared 的**, 健康达标树同样秩亏). 这里是**合成**
  inner-const-heavy 相对**地板**, 根因基底吸引域 (对 oracle 同样成立, shared). 不同 workload、
  不同主导机制. 锐化对比: 合成 oracle≈kernel 仅 ~1pp ⟹ kernel 的退化-放弃惩罚在合成上**小**
  (合成树没真实 bloated 树那么病态) ⟹ 合成的 gap ≈ 纯 shared 基底吸引域, 真实的 gap 多出的那截
  来自 kernel 整步放弃这个特有**机制**. 但注意: 该机制反映的秩亏本身是 shared 的 (健康树同样秩亏),
  且**修不干净** (pivot-floor 救它就炸 W0, 见 MIXED_PRECISION_PROBE 产品化节) ⟹ 不是"可修的
  kernel bug", 实质仍是高 K 树秩亏的 workload 性质.

## 复现

* sidecar: `gen_synth.py --all` 落 `workload/synth/*.ctrue.npy` (与 .bin 同 seed, 同序).
* runner 报地板: `runner.py --pop preset:synth-* --backends kernel_fusedfd` 出"噪声地板"节.
* 基底验证: `probe_floor_basin.py` (oracle 缓存 + c_true 重启 vs c_init 重启).
* 不变量测试: `test_harness.py::test_noise_floor` (长度/有限/地板≪初值/确定性/roundtrip).
