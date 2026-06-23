# 规模扫描（锁频干净版）+ 复用 Operon 同条件对比

- **编号**：05
- **状态**：未开始
- **来自**：`docs/PAPER.md` 第二节第 5、6 条
- **需要**：GPU（+ 可选 sudo 锁频）

## 目标

锁频干净版规模扫描，三个 variant（fusedfd / ad / revad），**复用 e6 已有的 Operon** 做同条件对比。出论文最主要那批图：吞吐面（trees/s vs M，各 variant）、N 轴 collapse、大 M 行为、**GPU 大 M vs Operon N 核 iso-quality crossover**、选参敏感性表；顺带 revad-vs-ad 速度切片（第 6 条）。

## 背景（"收窄"的几条 + codex 复查后的修正，见 `tasks/CODEX_REVIEW_section2.md`）

- **Operon 免跑**：e6 已有 **162 条完成的 Operon 记录（+18 条按设计跳过）+ 162 个 fp64 loss 侧车**。Operon 是 CPU 跑的，跟 GPU 锁不锁频、用不用反向求导都无关 → 直接配对、不重跑。codex 实测确认 `gen_synth` 重生成的 pop 和当年**逐字节一样**（hash 对得上），"同一个 pop"的前提成立。
  - ⚠️ 补两步：(1) 配对时加 pop-hash 断言（Operon 记录当年没存 hash，靠 gen_synth 确定性兜底）；(2) **跑几个 Operon 哨兵**和 e6 旧数对一下，确认 CPU 计时没因机器状态漂移。
- **GPU 跑法（修正）**：harness **没有**"只跑 seed0"或"按 seed 出误差棒"的模式（记录按 seed 聚合、key 里没 seed）——要 seed0-only 或 per-seed 误差棒都得改代码。所以**不强行收窄 seed**：直接 `--full` 跨 3 seed 取中位（135 个配置，稳、不改代码）；误差棒留作"要的话再加 per-seed 记录"。
- **锁频（修正）**：调度器用 0..n-1 的 GPU id、且覆盖 `CUDA_VISIBLE_DEVICES`，**没法干净地只在空卡子集上跑**——"只锁空卡 + 全卡跑"会混进没锁的卡。正经跑法：**等机器空、锁全部 8 张、`--n-gpus 8`**，并把锁定频率 + GPU id 记进每条记录（harness 小改）。
- **同条件 gate 要离线重算**：复用 Operon = 跳过 phase B，harness 不会给 revad 算 iso-quality gate（现有 gate 只有 ad/fusedfd）。写个小脚本**离线**用 GPU fp64 loss + Operon loss 侧车重算 revad/ad/fusedfd vs Operon 的同条件配对。

## 验收标准（做完要能逐条勾上）

- [ ] GPU 全面跑完（fusedfd / ad / revad，跨 3 seed 取中位、锁频），每点过 fp64 质量门
- [ ] Operon 哨兵对得上 e6 旧数（确认可复用）
- [ ] 离线重算 iso-quality，crossover 出来：GPU 大 M vs Operon {1,16,64,128} 核
- [ ] 图：吞吐面 / N-collapse / 大 M / Operon crossover + 选参敏感性表
- [ ] 第 6 条：export 出 revad/ad 速度比值 + 质量等价（已做 `analyze_ad_vs_fd_ranking`）

## 怎么做（步骤）

1. **等机器空、锁全 8 张**：`python -m cusr.benchmark.gpu_clocks --lock --gpu 0..7`，记下锁定 MHz，跑完 `--unlock`。给 harness 加一步：GPU id + 锁定频率记进每条记录。
2. **pop**：用 e6 那批 pop（已存在的直接用，缺的按 seed 确定性生成；codex 已验证逐字节一致）。
3. **复用 Operon**（codex 确认机制 sound）：把 e6 `sweep_e6.jsonl` 的 Operon `status=ok` 记录 + loss 侧车拷进本次输出目录；sweep 的 resume 按 `(operon, N{N}, preset, M, ncores)` + status=ok/有限 throughput/有 wall_core 认成已完成 → **跳过 Operon、只跑 GPU**。
4. **Operon 哨兵**：另跑 3–5 个 Operon 配置和 e6 旧数对一下，确认 CPU 计时没漂。
5. **跑 GPU**：`sweep_e6.py --full`（已接 revad），`--n-gpus 8`、n_rep=3、跨 3 seed 取中位。
6. **离线重算 iso-quality**：小脚本用 GPU fp64 loss + Operon loss 侧车，配对算 revad/ad/fusedfd vs Operon 的同条件 gate（phase B 被跳过、harness 不会自己给 revad 算）。
7. **出表/图**：扩 `export_gpu_phase_a.py` 出 **revad/ad 比值**（第 6 条）+ 现有 ad/fusedfd；画吞吐面 / N-collapse / 大 M / Operon crossover。

## 关联文件

- `experiments/e6_kernel_sweep/sweep_e6.py` — 扫描驱动（已接 revad）
- `experiments/e6_kernel_sweep/export_gpu_phase_a.py` — 出表（已认 revad）
- `experiments/e6_kernel_sweep/out/sweep_e6.jsonl` + `sweep_e6_losses/operon_*.npy` — **复用的 Operon 数据**
- `cusr/benchmark/gpu_clocks.py` — 锁频
- `experiments/revad_v5/analyze_ad_vs_fd_ranking.py` — revad-vs-ad 质量等价（已做）
