# `experiments/` —— 编号实验目录

> ⚠️ **现状入口看 [`RESULTS.md`](RESULTS.md)**(e1–e6 结果总账:用途/状态/头条/权威文档)。
> **下方"现有实验"表用的是旧的 `001`–`012` 编号,是历史地层**;当前实际目录是 **`e1`–`e6`**
> (HPEC 推进期的重编号)。本 README 的血缘叙事可读作背景,但**算数的现状以 `RESULTS.md` 为准**。

每一个**独立研究问题 / 复现任务**在这里占一个子目录，按 **数字前缀** 编号。每个子目录自包含：README + 数据 + 代码 + 产出。

## 为什么分实验子目录

- 每个实验独立、可删、可归档，不互相污染。
- 不要把平台 / 库代码放这里——那是 `bench/` 的事（跨实验复用）。
- 每个实验的 `scripts/run_*.py` 是**可复现入口**；`smoke/*.py` 是一次性 smoke（允许烂）。

## 现有实验

这 11 个不是 11 个独立方向，而是**一条主线在三次 pivot 中留下的几代地层**。
方向演化：EML SR（证伪）→ NLS 优化器质量（暂缓）→ memetic 假设 → **SR 符号发现 benchmark + GPU 批量 CO（当前）**。
按血缘分组看：

**① 复现期（早期，休眠）**

| 编号 | 主题 | 状态 |
|------|------|------|
| `001_evogp_repro` | EvoGP 论文复现，第一次接触代码库 | bootstrap 完成，休眠 |
| `002_eml_repro` | EML 论文（单算子完备基，Odrzywołek PNAS 2026）PyTorch demo 复现 | smoke 过，休眠 |

**② EML SR 方向（已证伪 / 归档）**

| 编号 | 主题 | 状态 |
|------|------|------|
| `005_eml_sr_dp` | EML 文法上 DP+beam+梯度精炼，曾想当 paper 方向 | **archived（2026-04-22）**，plan superseded |

**③ NLS 常数拟合「优化器质量」线（基础设施 + 暂缓）**

| 编号 | 主题 | 状态 |
|------|------|------|
| `003_nls_bench` | **NLS-for-CO 的 bench 平台**（库在顶层 `bench/`）：给定骨架，测优化器拟常数的质量/速度 | Phase 0 完成，`bench/` 仍被复用 |
| `006_nls_benchmark` | 9 条 Feynman 上 5 个 CPU optimizer 横评（GSL/NLopt + perf） | **deferred（2026-05-02）** |
| `004_nls_tutorial` | 从零学 NLS 算法（Newton→LM）的自包含 notebook，备课用 | 学习用；⚠️ 有 parallel-agent talk-prep 在动，勿动 |

**④ memetic GP × NLS（假设检验）**

| 编号 | 主题 | 状态 |
|------|------|------|
| `007_evogp_memetic` | 进化循环里嵌 LM polish 常数：更准更快 vs 锁死搜索？2×2×2 + 多 seed | current，已推到 step 7（结论作参考） |

**⑤ GPU 批量 LM kernel（CUDA capstone，核心资产）**

| 编号 | 主题 | 状态 |
|------|------|------|
| `008_batch_lm_sr` | 一个 CUDA kernel 同解 ~1000 个异构-K NLS（warp-per-tree，原生异构-K，三变体） | FD 质量修复 + parity 换代闸门就位；旧 135×/118× 加速口径已废，新口径见 012 |

**⑥ 当前前沿：SR 符号发现 benchmark + GPU 批量 CO**

| 编号 | 主题 | 状态 |
|------|------|------|
| `009_sr_benchmark` | 符号**恢复率** benchmark 基础设施（题集 + 判定器 + CO backend + pipeline） | current |
| `010_memetic_sweep` | A 无CO / B 朴素CO / C 配方 的正式 sweep | **降级**为「受控 host 上的机制研究」，以 011 为参考基准 |
| `011_co_headroom` | 锚点换成 PySR/Operon：论证「CPU 上 CO 被配给，GPU 去配给后进入 CPU 够不着的 regime」 | current，近期主攻 |
| `012_op_bench` | **HPEC E1 算子层 CO benchmark harness**：同一 pop.bin 喂 scipy/torch/kernel×3/(pyoperon)，按质量档位比吞吐（口径在 `PROTOCOL.md`） | current，W1 落地（2026-06-11） |

> ⚠️ **两个「NLS benchmark」含义不同，别混**：③（003/006）测「给定骨架，优化器拟常数的质量」（optimizer quality）；⑥（009/010/011）测「整条 GP 能不能找回真公式结构」（symbolic recovery）；`012` 回到 optimizer 侧但比的是**吞吐**（HPEC 系统口径，质量只作档位门槛）。
>
> **当前真正活跃**：`012`（HPEC E1 harness）+ `009`（infra）+ `008`（GPU kernel）+ `011`（E3 素材）；`003`/`bench` 作复用件。其余休眠 / 归档 / 暂缓 / 降级。HPEC 总计划：`discussion/hpec_paper_plan_v2_zh.md`。

## 新开一个实验

```bash
NN=004
NAME=my_topic
mkdir -p experiments/${NN}_${NAME}/{data,smoke,scripts,configs}
```

然后写 `experiments/${NN}_${NAME}/README.md`：

- **状态 + 日期**
- **上游依赖**（哪个 `upstream/XXX` 里的哪个 commit）
- **目标**（要证明 / 要复现 / 要测量什么）
- **验收标准**（什么结果算成功）

## 编号约定

- 编号不复用。删掉的实验保留编号，下一个用下一个数。
- 数字前缀 + 蛇形名字（`005_gpu_lm_bench`），全小写。

## 实验产出往哪放

- **小数据 / 配置**：`experiments/XXX/data/` 里 commit。
- **大产出 / log**：`experiments/XXX/runs/`、`experiments/XXX/logs/`、`experiments/XXX/outputs/` 都是 **gitignored**（见 `.gitignore`）。
- **跨实验共享数据**：`/data/`（根目录）或 `experiments/XXX/data/` + dataset registry（bench 用这种）。

## 实验用 bench 库

```python
from bench.sources.synthetic import SyntheticSource
from bench.backends.scipy_common import ScipyLSBackend
from bench.runner import Runner
```

`bench/` 在顶层、`[tool.pytest.ini_options] pythonpath=["."]` 保证 import 干净。实验代码不 `pip install` 任何东西。
