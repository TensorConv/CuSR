# codex 审查（gpt-5.5 / xhigh）— section 二 收窄计划

2026-06-24。让 codex 压测收窄版的 section 二 计划（`tasks/04`、`tasks/05`、`docs/PAPER.md` 4–6 条）。它读了 sweep、gen_synth、e6 输出，还实测对了 pop hash。下面是它的意见 + 我的处理。

## codex 确认 sound 的（核心收窄成立）

- **复用 Operon 的机制成立**：把 e6 的 `status=ok` Operon 记录 + loss 侧车拷进新输出目录，sweep 的 resume 按 `(operon, N{N}, preset, M, ncores)` 认成已完成 → 跳过 Operon、只跑 GPU。✓
- **pop 可确定性重生成**：codex 实测——`gen_synth` 重生成的 pop 和当年 GPU 记录里的 `pop_sha256` **逐字节一致**，重叠记录全对上。✓ 所以"同一个 pop"前提成立。
- **Operon 计时复用科学上可接受**（GPU 锁频/反向求导不影响 CPU 计算），风险只是机器状态计时漂移，不是算法失效。
- **ncu SOL + warp-stall 主诊断方向对**；instruction roofline 可从 ncu 指标拿到（但内置 roofline 是 FLOP 向的，GIPS/强度要自己解析）。

## codex 找到的真问题 + 我的处理（已折进 tasks）

| # | 问题 | 处理 |
|---|---|---|
| 1 | Operon 记录没存 pop-hash，复用靠 gen_synth 确定性兜底、没强校验 | 配对时加 pop-hash 断言 + 跑哨兵（已写进 `tasks/05`） |
| 2 | **"seed0 出面 + 补种子误差棒" harness 不支持**（记录按 seed 聚合取中位、key 里没 seed） | 改：不强行收窄 seed，直接 `--full` 跨 3 seed 取中位（不改代码）；误差棒留作可选的 per-seed 改动（`tasks/05` + `PAPER.md` 已改） |
| 3 | **只锁空卡 + 全卡跑会混进没锁的卡**（调度器用 0..n-1、覆盖 CUDA_VISIBLE_DEVICES） | 改：等机器空、锁全 8 张、`--n-gpus 8`，并记录 GPU id + 频率（`tasks/05` 已改） |
| 4 | 复用 Operon = 跳过 phase B，revad 的 iso-quality gate 不会被算 | 加一步：离线用 GPU fp64 loss + Operon loss 侧车重算 gate（`tasks/05` 已加） |
| 5 | CPU 计时跨天/机器状态可能漂 | 跑 3–5 个 Operon 哨兵和旧数对一下（`tasks/05` 已加） |
| 6 | `export_gpu_phase_a.py` 只半 revad-aware（VAR 有了，但 summary 比值/表/文字还只 ad-vs-fusedfd） | 扩 export 出 revad/ad 比值（第 6 条要的）（`tasks/05` 已写） |
| 7 | ncu 宽正则 `eval` 会误抓 `eval_loss_fp64_kernel`；`-c 3 -s 5` 可能采不全 | 改用精确 kernel 名（fd_jacobian_fused / ad_jacobian / rev_jacobian / build_jtj_jtr / eval_kernel_batched）（`tasks/04` 已改） |
| 8 | 只小 M ncu 不足以支撑大 M 的 claim | 加一档 M≈64k（减 section）确认（`tasks/04` 已加） |
| 9 | 计数口径：180 = 162 完成 + 18 跳过；侧车 162 不是 164 | 数字已更正（`tasks/05`） |

## 还需注意（非阻塞）

- seed0-only 的 GPU-vs-Operon 对比只能当"seed0 配对吞吐包络"讲（Operon 本来就 seed0-only）；GPU 跨 seed 取中位是稳的，但不量化 CPU/质量的方差。要更强就得给 Operon 也补种子（代价大，暂不）。

codex 全程没改文件。完整输出在本次会话。
