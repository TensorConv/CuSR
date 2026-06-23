# 性能剖析：ncu SOL + warp-stall + instruction roofline

- **编号**：04
- **状态**：未开始
- **来自**：`docs/PAPER.md` 第二节第 4 条
- **需要**：GPU + sudo（profiling，已批）

## 目标

用 ncu 测出 kernel 真正的瓶颈（发射/指令-bound），产出：热核的 Speed-of-Light 分解 + top warp-stall 原因 + 一张 instruction roofline。坐实"不是带宽/算力 bound"，给真实天花板，并据此判定后面要不要碰显存布局那类优化。

## 背景

kernel 实测只用到算力峰值和带宽峰值的 **<1%**（两条都不到），所以经典 FLOP-roofline 画出来是"远在两条线下方一个点"、没诊断力。重点核：**Jacobian（占 LM 循环 60–66%**；fusedfd 的 fd_jacobian / ad 的 JVP / revad 的反向 VJP）、build_jtj、eval。

## 验收标准（做完要能逐条勾上）

- [ ] 三个 variant 的热核都有 SOL（compute% / memory% / issue% 利用率）
- [ ] top 几个 warp-stall 原因（预计：MIO/SFU throttle=超越函数、execution dependency=栈机依赖链、branch divergence）
- [ ] instruction roofline 点（GIPS vs 指令强度），显示贴近发射 roof
- [ ] 一句反证成立——"不是带宽/算力 bound"（→ 显存布局优化不用做）

## 怎么做（步骤；这是测量任务，不强求 TDD）

1. 锁频后（见 `tasks/05`），挑一张安静的卡。
2. 写个小 ncu 驱动脚本，对 fusedfd / ad / revad 各跑一次，**用精确 kernel 名**（别用宽正则——`eval` 会误抓 `eval_loss_fp64_kernel`）：
   `sudo ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:fd_jacobian_fused_kernel|ad_jacobian_kernel|rev_jacobian_kernel|build_jtj_jtr_kernel|eval_kernel_batched' -c 2 --section SpeedOfLight --section WarpStateStats --section SchedulerStats --section InstructionStats -o <out> <binary> <pop> --max-iter 2`
3. **两档 M**：主测小 M（~2–4k，每核多采几个 launch 看 stall）；再加**一档 M≈64k**（减 section、只取 SOL）确认大 M（论文 claim 的工作点）结论一致——别只拿小 M 下大 M 的结论。
4. 两个 regime：inner-const-heavy（高 K）+ early-gen。
5. 解析 ncu（`--csv` 或 `--import` .ncu-rep）→ SOL 柱 + stall 表 + instruction-roofline 点。
6. 自检：驱动脚本对 ncu 输出的解析加个小测试（喂样例 csv，断言字段解析对），不用真 GPU。

## 关联文件

- `cusr/kernel/batch_lm_{fusedfd,ad,revad}_prof` — 被剖析的二进制
- 新建：`experiments/e6_kernel_sweep/ncu_profile.py`（驱动 + 解析）+ 其单测
- `results/roofline_a100__20260616/` — 旧的 analytical roofline，只当 "<1% 双峰值" 反证的来源
