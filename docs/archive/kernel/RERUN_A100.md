# A100 重跑清单 / Numbers that must be re-measured on A100

> 2026-06-11 起, W0 质量修复 (相对 FD 步长 + xtol=1e-6) 之后的所有数字都是在
> **RTX 5070 Ti Laptop (12 GB)** 上测的 — 用 `-arch=sm_80` 二进制经 PTX JIT 跑.
> **投稿前必须在 A100 重测下列各项**; 本机数字只用于开发期对照, 不进论文.
>
> 跑前: `nvidia-smi` 确认无并发 GPU 任务; `source scripts/env.sh`; 三个变体全部重编.

## 0. 精度口径 (fp32 是既定目标, 别手痒开 fp64)

- **kernel 全程 fp32, 这是设计目标不是妥协**: EvoGP 本身 fp32, pop.bin 数据 (x/y/目标)
  也是 fp32 → 没有 fp64 信号可恢复, "真值"本就只有 fp32 精度; 下游用途是按拟合质量排序
  候选公式, 不需要 fp64 位数. **A100 重测一律 fp32**, 不要为了好看去开 fp64.
- **oracle 仍用 scipy fp64**: 它是"最优能拟到多低"的金标准 (gold), 与 kernel 用什么精度无关.
  fp32 kernel 的 tier 率 = "fp32 追平 fp64 金标准的比例", 是有意义的诚实数字.
- **fp64 仅一个选做场景**: 若做 SRSD realistic-range (常数跨 1e-11~1e11, W8 stretch) 遇到
  JᵀJ 病态 → 也是**混合精度** (只在 K×K 的 Cholesky 解里用 fp64, 其余 fp32, A100 上近免费),
  不是全程 fp64. 主线实验不碰. (注: kernel tier-A<scipy 的差距, **顶档并列那部分**是 fp32 分辨率;
  但**高 K tier-B 软肋经实测不是精度**而是秩亏/死常数 — fp64 解救回 +0, 经实测
  **无干净 solve 修法** (列缩放 `JᵀJ+λD²` 是 no-op kernel 已是; pivot-floor 炸独立 008
  W0 gate 已撤、降 opt-in; 健康/失败树秩亏谱重叠 QR 也分不开), kernel 维持冻结安全基线;
  详见 012 `MIXED_PRECISION_PROBE.md` 产品化节.)

## 1. 三变体吞吐 (data/fixtures/pop.bin, 1000 棵)

| 项 | 5070 Ti laptop (修复后) | A100 (修复后) | 备注 |
|---|---|---|---|
| `batch_lm` (baseline, host-FD) | 0.758 s / 1319 t/s | **待测** | 修复前 A100: 0.740 s / 1351 t/s |
| `batch_lm_devjac` | 0.303 s / 3304 t/s | **待测** | |
| `batch_lm_fusedfd` | 0.262 s / 3812 t/s | **待测** | |

注: laptop PCIe 窄, H↔D 重的 baseline 吃亏更狠 → **变体差距 (2.7-3.2×) 在本机被放大**,
A100 上预期收窄. 这本身是 design-space 章的素材 (互连带宽是 workload 参数之一).

## 2. parity 闸门 + 报告再生

```bash
# 从仓根跑
GATE_REPORT=docs/kernel/verify_report_a100.md uv run python cusr/kernel/tests/test_parity_gate.py
```

- laptop 修复后基线: loss-down **94.0%**, 1.05× **92.4%**, 10× **99.8%** (`docs/kernel/verify_report_5070ti_w0.md`)
- 阈值 (93/90/99) 按 laptop 校准, A100 ±1pp 漂移属正常, 更大要查
- ⚠️ 现有 `docs/kernel/verify_report.md` 是**修复前 A100** 的 (1.05×=89.9%, 10×=100%), 已 stale —
  重生成时旧文件先改名归档 (`verify_report_a100_prefix_eps_abs.md` 之类)

## 3. 量级扫描 (便宜, 顺手跑)

```bash
# 从仓根跑
uv run python cusr/kernel/tests/test_fixture_scale.py                       # baseline
BATCH_LM=cusr/kernel/batch_lm_devjac  uv run python cusr/kernel/tests/test_fixture_scale.py
BATCH_LM=cusr/kernel/batch_lm_fusedfd uv run python cusr/kernel/tests/test_fixture_scale.py
```

laptop: 三变体均 tier1 11/11 + tier2 5/5.

## 4. EvoGP 每代墙钟 (in-loop overhead 分母)

```bash
uv run python cusr/kernel/bench_evogp_gen.py
```

| pop | laptop per-gen mean | A100 | 备注 |
|---|---|---|---|
| 1000 | 1.4 ms | **待测** | |
| 4000 | 3.8 ms | **待测** | |

laptop 结论 (预计 A100 同量级): **EvoGP 进化一代 ≈ 1-4 ms, 全种群 CO 一次 ≈ 260-760 ms
→ in-loop 时间预算几乎全是 CO**. "GPU 上 CO 免费"不成立; 正确叙事是 "CO 是 memetic
GPU SR 的绝对瓶颈算子, 这正是优化它的理由". E2 排期按 runtime ≈ gens × p × CO_time 估.

## 5. 构建注意

- Makefile `NVCC_FLAGS` 默认 `-arch=sm_80` (A100 原生); 本机 sm_120 靠 compute_80 PTX JIT,
  首跑有 JIT 暖机 (计时前先跑一遍)
- devjac / fusedfd 不在 Makefile 里, 在 `cusr/kernel/` 里手动编:
  `cd cusr/kernel && nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math -o batch_lm_devjac batch_lm_devjac.cu loader.c`
