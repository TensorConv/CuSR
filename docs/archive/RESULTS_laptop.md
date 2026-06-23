# E1 本地 sanity 结果 (5070 Ti laptop, 全 6 preset, M=4000)

> 2026-06-13. 协议 v2 (`PROTOCOL.md`). 7 后端 × 6 preset, repeats=3, fp32 kernel vs fp64 oracle。
> **本机数字仅 sanity, 未饱和 (M=4000 在 5070Ti laptop 上不喂饱 A100); 论文数字按 `008/RERUN_A100.md` 在 A100 重测。**
> 原始 json/md 在 `_out/` (gitignored); torch 未跑 (M=4000 上 ~88 min/preset 不现实)。

## fusedfd (最优变体) vs scipy

| preset | K_mean | scipy e2e | fusedfd e2e | e2e 加速 | fusedfd tput-B (t/s) | kernel tier-A | kernel tier-B | kernel fail |
|---|---|---|---|---|---|---|---|---|
| early-gen (真实) | 2.8 | 5.87s | 0.49s | 12× | 6591 | 83.7% | 90.1% | 528 |
| late-gen-bloated (真实) | 1.2 | 1.07s | 0.44s | 2.4× | 5699 | 88.9% | 89.7% | 2528 |
| inner-const-heavy (真实) | 6.9 | 19.97s | 0.69s | **29×** | 3716 | 60.0% | 64.8% | 671 |
| synth-early-gen | 2.8 | 3.79s | 0.49s | 7.7× | 6727 | 90.0% | 93.1% | 72 |
| synth-late-gen-bloated | 1.2 | 1.35s | 0.43s | 3.1× | 6131 | 93.0% | 95.4% | 199 |
| synth-inner-const-heavy | 6.9 | 16.13s | 0.71s | 23× | 4539 | 74.0% | 81.4% | 293 |

## 变体阶梯 (吞吐 t/s, tput-B; baseline 在 laptop PCIe 受限被放大, A100 预期收窄)

| preset | baseline kernel | devjac | fusedfd |
|---|---|---|---|
| early-gen | 849 | 4440 | 6591 |
| inner-const-heavy | 510 | 2685 | 3716 |

## 对手在 frontier 上的位置 (没人通吃)

* **scipy** fp64: tier 100% 但 198–2619 t/s (高 K 时 20s/preset)
* **pysr200** fp64: tier-A 95–99% 但 11–185 t/s — 高 K 时 **294–342 秒** (278–383 迭代/树)
* **pysr** (8-iter): 快些但质量塌 (inner-const-heavy tier-A 仅 27.8%)
* **operon** fp32: 1700–1900 t/s 但 tier-A **15%–88% 乱跳** (Eigen LM 早停, 中位 1–5 迭代, 不可靠)
* **kernel** fp32: 3700–6700 t/s @ tier-B 65–95% — 快且质量中上, frontier 上的甜点

## 选择保真度 (kernel, Spearman / top-25% / top-10%)

| preset | Spearman | top-25% | top-10% |
|---|---|---|---|
| early-gen (真实, 无噪声) | 0.935 | 92% | 90% |
| late-gen-bloated (真实) | 0.828 | 53% | 43% |
| inner-const-heavy (真实) | 0.815 | 76% | 55% |
| synth-early-gen (噪声 1%) | 0.943 | 96% | 96% |
| synth-late-gen-bloated | 0.983 | 96% | 96% |
| synth-inner-const-heavy | 0.913 | 87% | 89% |

top-10% 在无噪声真实 preset 偏低 = 顶部"并列满分"(L*≈1.5e-10 一坨)的排序噪声, 非质量缺陷
(misrank 树真实名次全 ≤50/249); 带噪声合成 preset 顶部拉得开 → 89–96%。

## 噪声地板 (仅合成 preset; 详见 `NOISE_FLOOR.md`)

地板 = 真常数 `c_true` 代回算的 loss = 注入 1% 噪声的能量 (统计下界). 它是相对 tier 的**绝对兜底**。

| preset | oracle 够地板 ≤1.005× | kernel 够地板 ≤1.005× | 差 |
|---|---|---|---|
| synth-early-gen | 79.7% | 78.4% | 1.3pp |
| synth-late-gen-bloated | 84.4% | 82.1% | 2.3pp |
| synth-inner-const-heavy | 48.1% | 46.7% | 1.4pp |

* **头条 = 不变量**: kernel 够到真极小的频次 ≈ oracle (差 ≤2.3pp) → 残余高 K 软肋是**对 fp64 CPU
  reference 同样成立**的 problem-hardness, 不是 kernel 缺陷。tier-B "kernel 追平 scipy 的 LM" +
  地板 "kernel 够到真极小和 scipy 一样频繁" 两句合起来, 堵掉"你 oracle 自己不最优"的质疑。
* oracle 够地板 48–85% 偏低 = **从演化式 ±30% 初值的基底吸引域** (实测: 够不到的树多数 status=
  converged 非预算上限; 从 `c_true` 重启 100% 守地板 med 0.998 → c_true 是真极小)。reach% 随初值
  散布而变, 报数必带此口径, 否则被读成"scipy 只能解一半"。

## 结论 (给 paper)

1. **加速比是 K-曲线, 2.4×–29× (本机/未饱和/fp32-vs-fp64)**, 非单一数字 → 印证 frontier 叙事。
2. **kernel 赢得最狠处 = CO 最贵处 (高 K, scipy 20s/37 迭代)**; CO 简单 (K≈1, scipy 1 迭代) 优势小。
3. **质量曲线 tier-B 65%–95%**; 真实高 K 的 65% 是软肋。**[6-13 订正]** 原推测 "fp32 Cholesky → 混合精度 K×K 解" 已被实测**证伪**(见 `MIXED_PRECISION_PROBE.md`):fp64 解救回 **+0** 棵,真因是**秩亏 / 死常数**(不可辨识方向,0 在任何精度都是 0)+ 求解器遇退化方向**整步放弃**。pivot-floor(放行退化方向)实测可修 **+6~+15pp**,但该 floor 一上独立的 008 W0 parity gate 即炸(within-1.05× 92.4%→86.6% FAIL,主要连累 MAXITER 健康树),已降为 `-DPIVOT_FLOOR` opt-in ablation。列缩放 `JᵀJ+λD²` 本就是 **no-op**(kernel `A[jj]*=(1+λ)` 已等价于它,零对角线死常数上 λ·0 抬不起主元),**无干净 solve 修法**:健康树与失败树条件数谱**完全重叠**(QR-on-J 也分不开,重叠在问题里不在求解器),候选 `JᵀJ+λI` 未验证、大概率同样炸 W0。根因 = 高 K bloated GP 树**普遍秩亏的 workload 性质**,非 solver bug;kernel 维持冻结安全基线(整步放弃 → 返回 best-finite,过 W0 gate),此 tier-B 软肋是 intrinsic 难度,详见 `MIXED_PRECISION_PROBE.md`。与噪声地板发现互证(kernel 够到真极小的频次 ≈ oracle,差 ≤2.3pp,见 `NOISE_FLOOR.md`)。
4. **fusedfd 主推**, 稳定比 baseline ~8–13× (laptop PCIe), 比 devjac ~1.5×。
5. **绝对参照 (噪声地板, 合成 preset)**: kernel 够到真极小的频次 ≈ fp64 oracle (差 ≤2.3pp);
   高 K 残余软肋是 shared problem-hardness (演化初值的基底吸引域, 对 oracle 同样成立), 非 kernel
   缺陷。与 #3 的真实-preset 秩亏软肋是**不同 workload / 不同机制**, 别合并。见 `NOISE_FLOOR.md`。
6. 待 A100: 饱和 M 扫描 (现在全未饱和) + 上述各项按 `RERUN_A100.md` 重测。
