# 012_op_bench — E1 算子层 CO benchmark harness (W1)

> 想快速搞懂这块在干啥(人话、不冗长)→ 看 **`OVERVIEW.md`**。本文件是技术日志。

HPEC 论文核心图 E1 的测量工具: 同一份 `pop.bin` workload 喂给多个常数优化后端,
按 `PROTOCOL.md` 的口径比"达到质量档位的吞吐"。

## 文件

* `PROTOCOL.md` — E1 口径 (停机 / 质量档位 / 计时), W1 定稿, 改动要 bump version
* `popio.py` — pop.bin 读写 (008 格式)
* `interp.py` — 向量化 numpy 后缀解释器 (fp64 oracle 用) + infix 打印
* `backends.py` — 后端契约 + ScipyPop / TorchPop / OperonLM / PySRBFGS / CudaKernelPop
* `runner.py` — oracle 缓存 + tier 判定 + 结果表 (json/md → `_out/`)
* `test_harness.py` — TDD 套件 (`uv run python test_harness.py`)
* `smoke_pyoperon.py` — W2 day-1 冒烟 (pyoperon 算子级 LM 可行性)
* `smoke_pysr_co.py` — W3 day-1 冒烟 (PySR/SymbolicRegression.jl 算子级 CO, juliacall;
  顶部 12 条防雷规则是 PySRBFGS 后端的设计依据)
* `workload/` — W4: 收割驱动 + 刻画脚本 + `characterization.md` (66 份真实 EvoGP 快照
  在 `workload/snapshots/`, gitignored, 见 manifest.json); `gen_synth.py` 合成生成器
  (边际对齐真实快照, 确定性); `presets.json` 6 个 preset 钉档 (3 真实 + 3 合成, sha1 校验)

## 跑法

```bash
source ../../scripts/env.sh           # kernel 后端需要 CUDA 运行库
uv run python test_harness.py        # 全套自测
uv run python runner.py --pop ../../data/fixtures/pop.bin \
    --backends scipy,torch,kernel,kernel_devjac,kernel_fusedfd --limit 300 --smoke
uv run python runner.py --pop preset:inner-const-heavy --backends scipy,operon,kernel
uv run python workload/gen_synth.py --all     # 重新生成 3 个合成 preset (确定性)
```

## 状态

* 2026-06-11: W1 落地 — 口径 v1 + scipy/torch/kernel 三后端 + 自测 9/9 绿 + 真实 pop 冒烟
* 2026-06-11: W2 冒烟 GO — pyoperon 0.6.1 算子级 LM 可用 (LMOptimizer + CoefficientOptimizer),
  防雷规则见 smoke_pyoperon.py 顶部注释; wheel 是单精度构建
* 冒烟发现 (细节见 PROTOCOL §3 与 _out/): ① scipy 配向量化解释器后 CPU 基线强了 ~28×,
  旧 118× 口径对 CPU 不公平, E1 主轴 = 质量-吞吐 frontier 而非裸加速比;
  ② scipy 多进程池对 ~1% 退化树非确定 → oracle 串行 + tier-A 0.5% 容差;
  ③ kernel 在协议分母下 (全 eligible 树, 不只 both-converged): tier-B 71%, fail 18%
* 2026-06-11: W2 后半落地 — OperonLM 后端进 harness (programmatic 构树; Operon postfix
  节点序 = 右子树→左子树→父, 常数 rank 显式回映 + 构树后断言; 0.6.1 wheel 的 Fmax
  dispatch 实测算成 min → MAX/MIN 树级不支持). 11/11 测试绿. 300 树冒烟: e2e 0.056 s
  (CPU 满核, 比本机 kernel 还快 — M=300 远未饱和, crossover 论证素材), tier-B 53.4%
  (Eigen LM 自家准则早停, 中位 3 迭代) vs kernel 71.1%.
* 2026-06-11: W3 day-1 GO — `optimize_constants` 可经 juliacall 直调且确定性可复现;
  PySR 默认 8-iter 预算在 sin 频率参数上拟不回 (E1 故事素材: 原生预算 vs 放宽两档都报);
  暗坑: K=1 时静默换 Newton, GIL 全程持有 (并行要 Julia 侧 @threads). 见 smoke_pysr_co.py.
* 2026-06-11: W3 收尾 — PySRBFGS 后端进 harness (`pysr` 原生 8-iter 档 + `pysr200` 放宽档)。
  两个集成坑已修: ① parse_expression 在 EmptyModule (只有 Base) 解析符号 → safe_pow/
  safe_log/safe_sqrt 函数对象 Core.eval 进去; ② **Julia 不是 fork-safe** — pysr 载入后
  fork 的 mp.Pool 死锁 → 所有进程池改 spawn context + per-backend 持久池 (启动成本
  在计时窗口外, 由 warmup 吃掉)。测试 13/13 绿。
* 2026-06-11: W4 前半 — 66 份快照 + 刻画 + 3 preset 推荐 (见 workload/characterization.md);
  真实快照只有 7 个 opcode (四则 79% + sin/cos/tan, dump_evogp 的 USING_FUNCS 使然) →
  Operon/PySR 映射缺口对真实 workload 零影响。
* 2026-06-12: W4 后半收尾 — `gen_synth.py` 合成生成器 (每树一元预算法精确控
  trig 占比; nodes/K/trig 边际对齐真实快照 ±5%, 两个已知残差见文件注释) +
  `presets.json` 6 preset 钉档 + runner 支持 `--pop preset:NAME` (sha1 校验,
  mismatch 警告)。合成树 y = f(x;c_true)+1% 噪声 = "可恢复"问题, 与真实 preset
  口径不同, 论文分开报。测试 14/14 绿; synth-inner-const-heavy 三后端冒烟通
  (合成组各后端 tier 率整体高于真实组, 符合可恢复设计)。
* 2026-06-12: 坏树 / 深度调查 (回应"best-finite 该不该加")。
  ① **kernel 已隐式 best-finite**: c_final 从 h_c 写出 (batch_lm.cu:536),h_c 仅在接受步
  覆写,失败树返回的是"最后接受的最优值"。实测 (L300) 770 个常数**全有限 (0 NaN)**;
  46 棵 status=failed 树 loss 全有限,**21 棵已达 tier-B**,其余 25 棵是真·拟合不够
  (loss/tier-B阈值 中位 421×) 而非被排除。→ "加便宜 best-finite" 是 no-op,已在;真正
  没做的是 NaN 步 → reject+λ↑+retry (属阻尼重试族,按计划延后)。诊断 `_diag_failtrees.py`。
  ② **深度刻画**: warp-per-tree,32 lane 跨数据点,`eval_tree_d` 顺序走 n_nodes,每 lane
  一个 `float stack[64]`。主成本 = **节点数**(顺序长度,真实 ≤32 capped)而非树深;栈深定
  本地内存占用 (真实 ≤11,MAX_STACK=64 **过配 6×** → 缩到 ~16 是 memory-axis 便宜优化,
  待 A100 测)。异构节点数 (真实同 pop 内 1→32) = warp 尾延迟,主要 scaling 痛点。设计点
  ≤32 节点由 EvoGP max_tree_len 决定,与真实 dump 天然匹配 (是 workload 假设,需披露)。
* 待办: preset 上全量 E1 (论文数字 A100 重测)、Operon LMOptimizer 收敛准则
  可配性调查 (质量档位偏低是否可调)、kernel NaN→retry + MAX_STACK 缩配 (A100 上测增益)
