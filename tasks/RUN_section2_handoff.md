# 执行手册：CuSR 论文 section 二（C2 评测）— workflow 编排 + 严防造假 + codex 复核

> 这是给执行者（另一个 Claude）的任务书。你在 `/home/weish/hao/CuSR`（CUDA 符号回归；核心是批量 Levenberg–Marquardt 常数拟合 GPU kernel）。
> 目标：把 section 二 的三件事**真刀真枪跑出来**，产出可信、可复现、**每个数都能追到磁盘 artifact** 的结果。**用 Workflow 工具做多智能体编排。最后让 codex 复核。**

## 0. 先读这些（按顺序，别跳）

1. `docs/PAPER.md` — 总任务清单（第二节第 4 / 5 / 6 条 = 你要做的）
2. `tasks/04-ncu-profiling.md` — 第 4 条（ncu 性能剖析）详细 spec
3. `tasks/05-scaling-sweep.md` — 第 5+6 条（规模扫描 + 反向 vs 前向）详细 spec
4. `tasks/CODEX_REVIEW_section2.md` — codex 已挑出的坑 + 必须照办的修正（**全部按这个来**）
5. `docs/kernel/OPTIMIZATION_BACKLOG.md` §0.2 — 为什么剖析用 ncu SOL+stall+instruction roofline、不用经典 FLOP roofline；以及实测 <1% 双峰值的事实
6. `docs/research/paper_plan.md` — C2 在 6 页里占什么、要哪些图
7. `experiments/e6_kernel_sweep/sweep_e6.py` 顶部 docstring + `experiments/e6_kernel_sweep/test_sweep_e6.py` — **harness 的反作假合同**，遵守、不许绕过
8. `docs/PROTOCOL.md` — 实验诚信协议

## 1. 铁律（不可谈判 — 这是这个项目的命根子，"严防造假和虚构"）

0. **不准编数**。任何报出来的数字都要能追到磁盘 artifact（jsonl 记录 / .npy / .ncu-rep / .csv / 图）。没跑就没有——不许估、不许外推、不许"大概是"。
1. **不许绕过 harness 反作假 guard**：`assert_gpu_binary`（二进制白名单）、`assert_device_jacobian`（结构性 device-Jacobian 检查，N-invariant）、`FORBIDDEN_SUBSTRINGS`、fp64 质量门、`pop_hash`。guard 报警 = 停下报告，**不许 work around**。
2. **fp64 质量独立重算**（`cusr/benchmark/interp.loss_pop` / `cusr/kernel/verify.py`），永远不信后端自报 loss。
3. **锁频**：计时前锁全 8 卡（`python -m cusr.benchmark.gpu_clocks --lock --gpu <i>`），把锁定 MHz + GPU id 记进每条记录；锁不上就逐次记录频率并报告稳定性，没锁的数一律标 DRAFT。
4. **测时先确认 variant**：跑前核对二进制确实是目标 kernel（fusedfd / ad / revad）——别重蹈 e5 "测错 variant" 的覆辙。
5. **崩溃/失败如实报**：不许悄悄丢弃、不许把崩溃重新解释成"发现"（e4 study-B 那次"CO 有害"其实是崩溃 artifact，不是结论）。
6. **复用 Operon 必须**：(a) 校验 pop-hash 一致（gen_synth 重生成逐字节相同，codex 已验过）；(b) 跑 3–5 个哨兵和 e6 旧数对——漂了就别复用。
7. **null / 难看的结果照实写**（revad 某处没更快，就写没更快；AD≈FD，就写 AD≈FD，别美化）。

## 2. workflow 怎么搭（你写 JS；下面是建议相位）

⚠️ **GPU 争用**：扫描吃满 8 卡、ncu 要安静卡——**所有占 GPU 的步骤必须串行**，别让多个 agent 同时抢卡。workflow 的并行价值在**对抗性审计**和 CPU 侧分析/解析，**不在 fan-out GPU 作业**。

- **相位 0 · 完整性预检（串行）**：锁频（记 MHz）→ `make` 全部二进制 → 跑离线测试（`test_sweep_e6` / `test_gpu_clocks` / `test_revad_scipy_parity`）→ scipy 对拍确认 revad 数值对 → 确认 gen_synth 重生成的 pop 和 e6 的 `pop_sha256` 逐字节一致。**任一失败 → 停，报告，别硬跑。**
- **相位 1 · ncu 剖析（第 4 条，占 GPU 串行）**：照 `tasks/04`——精确 kernel 名（`fd_jacobian_fused_kernel` / `ad_jacobian_kernel` / `rev_jacobian_kernel` / `build_jtj_jtr_kernel` / `eval_kernel_batched`）、小 M + 一档 M≈64k、section = SpeedOfLight+WarpStateStats+SchedulerStats+InstructionStats。存 .ncu-rep/.csv，解析出 SOL + top stall + instruction-roofline 点。
- **相位 2 · 规模扫描（第 5 条，占 GPU 串行）**：照 `tasks/05`——先拷 e6 的 `status=ok` Operon 记录 + loss 侧车进新输出目录、**确认 resume 跳过 Operon**（别白跑一遍）；`sweep_e6.py --full --n-gpus 8`（这一步 harness 自己并行吃 8 卡，workflow 只包它 + 监控，**别再 fan-out**）；跑 Operon 哨兵对旧数。
- **相位 3 · 分析出图（CPU，可并行）**：离线重算 revad/ad/fusedfd vs Operon 的 iso-quality gate（phase B 被跳过、harness 不会自己给 revad 算）；吞吐面 / N-collapse / 大 M / Operon crossover；扩 `export_gpu_phase_a.py` 出 revad/ad 比值；revad-vs-ad 质量用现成的 `experiments/revad_v5/analyze_ad_vs_fd_ranking.py`。
- **相位 4 · 对抗性审计（反作假，重点并行）**：对**每个头条数字**派**独立 agent 去证伪**——从原始 artifact 重新推一遍、查 variant / 锁频 / 质量门 / pop-hash 有没有混、有没有崩溃 artifact 或静默跳过、单位/口径对不对。**多数 agent 确认才收下**，否则丢弃或标红。这一相位是这次 workflow 的核心价值。
- **相位 5 · codex 复核**：用 openai-codex 插件（**gpt-5.5 / xhigh**）复核整轮 + 结果，重点查造假 / 虚构 / 不稳 / 配混。命令示例：
  `CLAUDE_PLUGIN_ROOT=~/.claude/plugins/marketplaces/openai-codex/plugins/codex node "$CLAUDE_PLUGIN_ROOT/scripts/codex-companion.mjs" task --effort xhigh "<复核 prompt：让它对抗性审 section 二 的结果与 artifact>"`
  （node 在 `~/.nvm/versions/node/v24.14.1/bin`）。把 codex 的原话附进 findings。

## 3. 环境备忘

- `nvcc` / `ncu` 在 `/usr/local/cuda/bin/`（不在 PATH，自己 `export PATH=/usr/local/cuda/bin:$PATH`）。
- 建二进制：`make -C cusr/kernel batch_lm_fusedfd batch_lm_fusedfd_prof batch_lm_ad batch_lm_ad_prof batch_lm_revad batch_lm_revad_prof`；`_dump_jac_sample` 直接 `nvcc ... -o cusr/kernel/_dump_jac_sample cusr/kernel/_dump_jac_sample.cu cusr/kernel/loader.c`（无 make 规则）。
- 跑测试：`uv run python -m pytest <path> -q`。
- 复用的 Operon 数据：`experiments/e6_kernel_sweep/out/sweep_e6.jsonl`（operon 记录）+ `experiments/e6_kernel_sweep/out/sweep_e6_losses/operon_*.npy`。
- profiling sudo 已批（password-less for ncu / nvidia-smi）。8×A100 sm_80。

## 4. 产出

1. **findings 文档**（建议 `experiments/e7_section2/FINDINGS.md`）：**每个数字带 artifact 路径**；外加一节"**没做 / 跳过 / 失败了什么**"。
2. 所有原始 artifact 留磁盘（jsonl / npy / ncu-rep / csv / 图）。
3. 相位 4 审计结论 + 相位 5 codex 复核结论，**原样附上**。
4. 别 push（本机推不了）；commit 可以。

> 开工前先读完第 0 节，然后用 Workflow 工具把相位 0–5 编出来再跑。任何一步和铁律冲突，**停下来如实报告**，不要为了"跑完"而造数。
