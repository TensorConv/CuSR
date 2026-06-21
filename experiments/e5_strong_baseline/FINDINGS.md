# e5 — 强基线吞吐量扫描 (DRAFT, 2026-06-21)

> ⚠️ **头条已被推翻 (2026-06-21, 见 `out/fused_vs_fd_confirm.txt`)**:本扫描的 in-process
> `libcusr_co_fd.so` 是**慢的 host-FD 基线**(每 FD 步 H↔D 往返),**不是**部署用的 fused
> on-device FD kernel。实测 fused 比它快 **12–21×、质量逐位相同**;换 fused 后 GPU 在 M=64k
> **反超 64 核 EPYC 2–6.5×**(early-gen 6.5×、inner 2.1×)。下文"任一 M 都打不过 32/64 核 Operon"
> 的结论**仅对 host-FD 变体成立**。待办:建 `libcusr_co_fusedfd.so` 接入 co_inproc 后重跑。
> **质量天花板(inner-const)不受影响**(fused==fd 数值),仍需 damping 修复。

GPU CO kernel vs **Operon** (具名强 CPU baseline, C++/Eigen LM), 同一份 pop / 同一批树
(apples-to-apples)。M ∈ {1k,4k,16k,64k}, N=1000, seed=0, A100 GPU0, 256 核。
目的: de-risk OUTLINE §五A 旧串 "2.4–29×" — 那是本机粗测 + 弱/内部基线。

## 口径 (两个被修正的混淆)
1. **kernel 用 in-process `.so`** (`co_inproc.get_inproc_co`, CUDA ctx 一次性), **不是**子进程
   binary。timed region 包住 `.optimize()`(每代真实的 re-coerce + H2D/D2H, 不外提)。
2. **warm vs warm**: kernel 先 1 次不计时 warm + 取 ≥3 中位; Operon 持久池先付启动再计时。
3. **各自调好的工作点**: kernel `max_iter=50`; Operon `max_iter=200` (逐树早停, cap 罕中)。
   诊断: kernel wall 50→200 ≈ 线性(批跑满 cap、无逐树回收) ⇒ 50 是诚实点。iter-Pareto
   (inner 16k) med_loss 5.78@25→5.05@50→4.89@100→4.80@200, 在 Operon 3.63 之上**封顶**。
4. **等质量门**: loss 一律 fp64 重算 (`interp.loss_pop`); speedup 只在 med_loss 匹配 (≤×1.05)
   时才算干净。R2 / per-constant recovery |ĉ−c|/|c| 也一并 fp64 重算。
5. 时钟**未锁** ⇒ DRAFT, 取中位; ncu roofline 待 sudo 锁频。

## 头条 (吞吐量 trees/s vs M) — **prelim 的"扫到大 M 找 GPU 胜点"假说被数据推翻**

| preset | K̄ | kernel tput (1k→64k) | op-1核 | op-32核 | op-64核 | **vs 1核** | **vs 32核** | **vs 64核** | 等质量? |
|---|---|---|---|---|---|---|---|---|---|
| early-gen        | 2.8 | 1533→1909→1841→1684 | 561 / 737 / 514 / *omit* | 5931→7764 | 4907→6680 | 2.6–3.6× | 0.22–0.33× | 0.25–0.35× | **否 (+5.9~9.2%)** |
| late-gen-bloated | 1.2 | 1521→1801→1855→1411 | 1488/1169/1323/ *omit* | 7977→6690 | 8204→6566 | 1.0–1.5× | 0.19–0.33× | 0.19–0.33× | **是 (≤+3.8%)** |
| inner-const-heavy| 6.9 | 1241→1383→1303→1462 | 247/232/229/**231(投影)** | 4573→7320 | 4290→8824 | **5.0–6.0× (非等质量)** | 0.20–0.27× | 0.17–0.29× | **否 (+31~43%)** |

(tput = (M−n_dropped)/wall_median, trees/s; "1k→64k" = 四个 M 的 kernel tput; op 列示首末或全列。)

## 诚实结论

1. **kernel 吞吐量基本与 M 无关 (flat-to-declining), 不随 M 上升而饱和爬升 —— 推翻 prelim 假说。**
   1k→64k 64× 的 M 范围内 kernel tput 峰值早早出现在 4k–16k 后回落 (late-gen 在 64k −24%)。
   A100 在 M=4000 **并非 starved**: per-tree + marshaling 开销 + 批跑满 cap ⇒ 吞吐量近似 M-不变。
   (audit 证实 kernel tput 非单调, 物理可信 = DRAM/cache 饱和 + 每树开销主导, 非造假。)
2. **kernel 在任一 M 都打不过 32/64 核 Operon, 且差距随 M 持平或扩大。** 所有 vs32/vs64 ∈
   0.17–0.35 (即慢 3–5×)。同时 Operon 并行吞吐量随 M **上升** (inner 64 核 4290→8824)。
   ⇒ **量程内不存在 GPU 追平并行 Operon 的 M, 趋势上也不会有。** "扫到大 M 找胜点"不成立。
   (注: audit 之一声称 "kernel 比 32 核还快" 是**事实错误** —— 实测每个 vs32 都 < 1, 已核对 JSON 否决。)
3. **唯一干净 (等质量) 的结果是 late-gen-bloated**: kernel vs 单核 1.0–1.5× (M=1k 时几乎打平),
   被并行 3–5× 压制。early-gen / inner-const **每个 M 都没过 loss 门**, 其 speedup 一律带"非等质量"标。
4. **vs-64 核相对胜负随 K 反转 (Operon 对高 K 并行最好)** —— 关键发现:
   - vs **单核**: 高 K 赢最多 (inner 5–6×, 因单核 Operon 在 inner 上每树 LM 代价最高)。
   - vs **64 核**: 反转, inner **最差** (vs64 0.17–0.29)。机理 = 并行效率 PE64 随 K 升:
     inner 0.27–0.46 ≫ early 0.14–0.16 ≫ late 0.07–0.09。
   - 一句话: **GPU 对弱基线看起来最强的 regime, 正是强基线扩展性最好的 regime。**
     旧 "越难赢越多" 框架只在**单核弱基线**下成立, 换强基线即失效。
5. **"29×" 不成立**: 活在 M=64k + 弱/内部基线; 换 Operon + 等质量门后, 干净的只剩 late-gen ≤1.5× vs 单核。
6. **64k 单核处理**: 按规则 (每核吞吐量 spread >25% 则不投影):
   - early-gen spread 43.3% ⇒ **omit** (JSON 记 reason, 表中 *omit*);
   - late-gen spread 27.3% ⇒ **omit**;
   - inner-const spread 7.8% ⇒ **投影** 231 trees/s。该 64k vs-1核既是投影、又非等质量 (**双重标注**)。

## Pareto (M=16000, inner-const-heavy, kernel max_iter 25/50/100/200)
kernel med_loss 5.78@25→5.05@50→4.89@100→4.80@200, 全部封顶在 Operon-1核 3.63 之上 ⇒
更多迭代换不回质量, wall 却近线性上涨。**等质量在此 regime 不可达**, Pareto 前沿不交。

## 质量天花板 (inner-const-heavy) — fp32 条件数上限, 非 Jacobian 精度问题
kernel 中位 loss 比 Operon 高 31–43% (全 M), R2 ~0.96–0.97 vs Operon ~0.98。诊断:
- **精确 (forward-mode) Jacobian (AD 变体) 只关闭 ~1.9% loss** (fd=5.02 vs ad=4.93), 仍 +37% over Operon
  ⇒ 缺口**不是 Jacobian 精度问题**, 是 **fp32 条件数天花板**。
- iter-Pareto 也封顶 (见上): 加迭代降不下来。
- 机理: `batch_lm.cu` solve_kernel 的 LM 阻尼是 **乘性对角 (Marquardt, A[j,j]*=(1+lam))**, 把近零主元
  乘 (1+lam) 抬不离零 ⇒ 无法正则化秩亏/病态 JtJ, 这类树落到 Cholesky 非正主元路径 (status=-1, 不更新)。
- Operon wheel 也是 fp32, 故差距非纯精度而是**条件数鲁棒性**。论文 §五D 须诚实写 "kernel 在此 regime
  质量逊于 Operon", 不能只甩锅给题目。

## R2 / recovery 细节 (诚实补色)
- **early-gen**: 中位 recovery 几乎打平 (kernel ~0.102 vs operon ~0.103), R2 双方 ~0.999 (饱和)。
  ⇒ +5.9~9.2% 的 loss 缺口是**尾部少数难树**拉的, 不是系统性偏差。**但 loss 门基于 loss, early-gen
  仍判 NOT-matched** (recovery 等质量不改门的判定)。
- **inner-const**: recovery kernel ~0.174 vs operon ~0.140, R2 kernel ~0.96–0.97 vs operon ~0.98
  —— 这里 R2/recovery 都能区分, 与 loss 缺口一致 (真实质量逊)。

## 已知 caveats
- **DRAFT, 时钟未锁** —— 所有绝对 wall/吞吐量为草测, 取中位; 锁频后 (sudo 待批) 重跑干净版 + ncu roofline。
- kernel 吞吐量**非单调** (随 M 持平至下降); 任何 "kernel 线性扩展" 的表述都错。
- Operon **并行效率差** (PE64 7–46%), 64 核普遍劣于 32 核 (early/late/inner 多处 64<32) ⇒ Operon 的
  并行加速是**通信/同步受限**, 非算法线性。这意味着 256 核线性投影更不可信 —— **报实测 32/64 核, 不报满核**。
- **late-gen K0 丢弃**: 两后端都丢 ~30% K0 平凡树 (对称), 故绝对 trees/s 是在 ~70% 非平凡子集上, 比率干净。
- 早/inner 的 vs-单核 speedup **不是干净 speedup** (loss 没追平), 表中已逐格标"非等质量"。

数据: `out/sweep_report.json` (含 per-tree fp64 loss .npy), `out/sweep_stdout.txt`, `out/sweep.DONE`。
harness: `sweep.py` (复用 `prelim.py` 合约)。run 完整性: n_configs=48 expected=48 missing=[]。
