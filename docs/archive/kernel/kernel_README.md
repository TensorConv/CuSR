# batch-lm-sr — GPU batched Levenberg-Marquardt for SR populations

把符号回归 (Symbolic Regression) 每代产出的 ~1000 棵候选表达式树, **一次性在 GPU 上拟合各自的常数**
(非线性最小二乘, 每棵树形状 / 常数个数都不同). scipy 串行 / 16-worker 并行都得分钟级; 这里一个 CUDA
kernel 同时解 1000 个异构 NLS, GPU 上 **sub-second**, **量级上大概百倍**.

| 后端 | 用时 |
|---|---:|
| **batch-lm-sr** (GPU) | < 1 秒 |
| scipy LM (16 worker 并行) | ~分钟级 |

跟 scipy 比对, 89.9% 的树 `gpu_loss ≤ 1.05× sc_loss`, 100% 在 10× 范围内.
完整对比 [`data/verify_report.md`](data/verify_report.md).

> ⚠️ 当前是 baseline / MVP — 算法跑得通 + 有数值正确性诊断, 但超参数
> (`xtol`, `λ` 起点, FD `ε`, `max_iter`) 还没仔细调.

---

## Quick start

仓库自带 `data/pop.bin` (1000 棵真 EvoGP 种群, 4 MB), clone 完直接能跑.

```bash
make                                # 编译 ./batch_lm + ./inspect
./inspect data/pop.bin              # 体检, 期望 PASS, M_prob=1000
./batch_lm data/pop.bin             # GPU 拟合

# (可选) scipy 再跑一遍比对, 分钟级
uv venv && uv pip install -r requirements.txt
uv run python verify.py --nproc 16
```

跑回归测试 (3 套 fixture, 14 个 case):
```bash
uv run python tests/test_fixture.py
uv run python tests/test_fixture_op.py
uv run python tests/test_fixture_stress.py
```

---

## Environment

| 依赖 | 说明 |
|---|---|
| NVIDIA GPU + driver | 算力 ≥ 7.0, driver 支持 CUDA 12.x |
| CUDA Toolkit 12.x | 提供 `nvcc`. 推荐 conda 装: `conda create -n cuda-toolkit -c nvidia cuda-toolkit=12.4 -y && conda activate cuda-toolkit` |
| C 编译器 | `gcc` 或 `clang` (编 `inspect`) |
| Python 3.10+ + numpy + scipy | 跑 `verify.py` 用, 见 [`requirements.txt`](requirements.txt). 推荐 [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -r requirements.txt` |

**GPU 架构**: 默认 `Makefile` 用 `-arch=sm_80`. 其它 GPU 改这一行 —
查你卡的 compute capability: https://developer.nvidia.com/cuda-gpus.

**Windows**: 走 WSL2 — 装 Win NVIDIA driver, WSL Ubuntu, 然后跟 Linux 一样装 CUDA Toolkit.
clone 到 WSL 的 `~/`, 不要 `/mnt/c/` (跨文件系统访问慢 5-10×).

---

## Files

```
batch_lm.cu          GPU 主算法 (CUDA kernel + host 调度)
loader.{h,c}         pop.bin → host 内存
inspect.c            pop.bin 统计 dump (pure C, 无 CUDA 依赖)
pop_format.h         binary 格式定义 (C / Python 共用)
tree_interpreter.py  Python 端 tree forward eval (供 verify.py 用作 oracle)
verify.py            scipy 并行跑同一份种群, 写出 verify_report.md 做对比
dump_evogp.py        跑 EvoGP 生成 pop.bin (需 EvoGP, 见下面)
tests/               3 套 fixture: 算法基线 / 算子覆盖 / 边界情况
data/pop.bin         仓库自带的种群 (1000 棵, Feynman I.18.12, 4 MB)
```

数据流: `pop.bin → ./batch_lm → c_final.bin + status.bin → verify.py → verify_report.md`.
四个程序用 binary 文件松耦合, 不互相 import — 任意一步都能单独 rerun.

---

## Algorithm

每棵树独立跑 Levenberg-Marquardt 拟合常数 `c`, 让 `loss(c) = (1/2)·Σᵢ(y_pred(xᵢ; c) − yᵢ)²` 最小:

```
loop until converged or max_iter:
    J ← Jacobian of residual (有限差分, ε=1e-3)
    解 (JᵀJ + λ·diag(JᵀJ)) · δ = Jᵀ·r       (K×K Cholesky)
    if loss(c + δ) < loss(c): accept, λ /= 10
    else:                     reject, λ *= 10
    if ‖δ‖ < xtol·(‖c‖+xtol): break
```

GPU 上的布局:
- **Warp-per-problem**: 1 warp (32 thread) 处理 1 棵树, 0 warp divergence
- **Column-major Jacobian** `J[m, k, i]`, coalesced memory access
- Host 跑外层 accept/reject 调度 + `λ` 更新, kernel 跑数值热路径
- **Compile-time bounds**: `MAX_K=32` (常数数上限), `MAX_STACK=64` (interpreter 栈深)

`status.bin` 状态码 (每棵树一个 int32):

| code | 含义 |
|:---:|---|
| 0 | CONVERGED |
| 1 | MAXITER (用完迭代, 没收敛也没 fail) |
| 2 | FAIL_NAN (eval 在 c 处推到奇点) |
| 3 | K0_SKIP (K=0, 无常数可优化) |
| 4 | FAIL_CHOLESKY (JᵀJ 退化反复, `λ > 1e12`) |

---

## 集成 EvoGP — 自己生成 pop.bin

仓库自带的 `pop.bin` 是用 EvoGP 跑 Feynman I.18.12 (gen=20, pop=1000) 生成的, 4 MB.
要换数据集 / 参数, 跑 `dump_evogp.py`. 额外需要 PyTorch + EvoGP.

```bash
# 1. 装 EvoGP (https://github.com/EMI-Group/evogp)
git clone https://github.com/EMI-Group/evogp ../evogp_src
cd ../evogp_src && uv pip install -e . && cd -

# 2. 装 PyTorch (匹配你的 CUDA 版本, 见 https://pytorch.org)
uv pip install torch sympy

# 3. 跑 dump
uv run python dump_evogp.py \
    --dataset=feynman/I.18.12 --gen=20 --pop=1000 --N=1000 --seed=0 \
    -o data/pop.bin
```

`dump_evogp.py` 顶部 `PROBLEMS` dict 自带 4 道 Feynman 题作 sanity 用 — `--dataset=` 换成
下列任一即可, 不用下载任何外部数据集 (X, y 用 sympy lambdify 内存合成):

| dataset id | K | 算子焦点 | 物理 |
|---|:-:|---|---|
| `feynman/I.12.1` | 1 | 纯 MUL | `μ·N` (最简 baseline) |
| `feynman/I.18.12` | 1 | SIN | `r·F·sin(θ)` (默认, 仓库 pop.bin 来源) |
| `feynman/I.27.6` | 1 | 嵌套 DIV | `1/(1/d1 + n/d2)` (薄透镜) |
| `feynman/I.6.2`  | 4 | SQRT + EXP + POW | `exp(-(θ/σ)²/2)/(√(2π)·σ)` (高斯) |

加新题目: 在 `PROBLEMS` dict 加一条 (`skeleton_expr` 用 sympy 语法, 照现有条目的样子写),
不需要去找 Feynman 原始数据集.

---

## License

MIT. 见 [`LICENSE`](LICENSE).
