# codex 审查（gpt-5.5 / xhigh，自由探索）— section 一

2026-06-23。让 codex 审查 section 一 全部工作 + 自由探索（不限路径）。它读了 kernel、扫描、测试，还实际跑了 scipy 对拍脚本。下面是它的发现 + 我核实后的处理。

## 确认的 bug（已修，本次提交）

| # | 发现 | 核实 | 处理 |
|---|---|---|---|
| 1 | `smoke()` 只跑 fusedfd/ad，没 revad（端到端冒烟没覆盖反向求导） | 真（`sweep_e6.py:478`） | 改成 `for variant in GPU_VARIANTS` |
| 2 | `export_gpu_phase_a.py` 报告/缺漏统计漏 revad | 真（`VAR=["fusedfd","ad"]`） | 加 `revad` |
| 3 | `gpu_clocks._run` 不检查 returncode——失败的 `sudo nvidia-smi -lgc` 被静默吞掉 | 真 | `_run` 非零返回即抛 `RuntimeError` |
| 5 | parity 测试漏 shared-nonfinite 桶（revad 若在良态点退回 NaN，原断言抓不到） | 真 | `classify()` 加 `rev_nan_val_finite` 计数 + 测试断言 `==0` |
| 9 | revad 不在 `_POINTS_PER_S`（成本估算回退默认值） | 真（仅影响成本估算，不影响正确性） | 加 `revad=1.4e7` |

## 更正我之前的过头话（codex 抓的，重要）

- **#4 parity 测试测的是 host，不是 GPU kernel。** `_dump_jac_sample` 里没有 kernel launch，跑的是共享的 `__host__ __device__` AD interp 的**主机执行**——它验证的是反向求导**算法**，不是 GPU kernel 外壳（launch / 显存布局 / 寄存器）。GPU kernel 本身由 `test_parity_gate.py` 端到端（loss 级）验证。已改测试 docstring 写清楚范围。
- **#7 "scipy-FD 度量假象"是合理推断，但这个排序脚本没把它"坐实"。** 脚本只比了 AD vs FD（证明 AD 不比 FD 差），没有任一方直接对 scipy。要真正坐实，得做 "scipy 用有限差分 Jacobian" vs "scipy 用解析/AD Jacobian" 的对比。现状：AD≈FD 排除了"AD 更差"，假象是**最合理的解释**，但未被这个脚本独立验证——别写成已证实。

## 留给正式质量评测（未修，flag）

| # | 发现 | 说明 |
|---|---|---|
| 6 | 排序分析只在 both-finite(ad&fd) 上算 | 真，但量很小：低 K 排除 6/840，高 K 排除 0/3959。正式评测里报一下单侧 nonfinite 的数。 |
| 7 | scipy-FD 假象未独立坐实 | 见上；正式评测做 scipy 两种 Jacobian 的对比。 |
| 8 | materiality floor 1e-6 是绝对值、未按目标尺度归一 | 正式当证据前，报 floor 敏感性 + 被剔除集的构成。 |
| 10 | `verify_locked` 紧跟锁频就读 `clocks.sm`，空载 GPU 可能误判 | 上机锁频时改用 `nvidia-smi -q -d CLOCK` 或先加负载再验。 |

codex 全程没改文件（review only）。它的完整输出在本次会话里。
