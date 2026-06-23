# batch-lm-sr — GPU batched Levenberg-Marquardt for SR populations

> 一句话: 把"给 1000 棵符号回归候选表达式各自拟合常数"这件事从 scipy 串行 (一代约 15 分钟) 搬到 GPU
> (1 个 kernel, 约 0.7 秒/代), 在 A100 上 **~135× 加速**, 数值跟 scipy 几乎等价。

> **研究方向 (2026-06-09)**: 本 kernel 现定位为 009 `co_backend` 多后端对比里的**一个 backend** (`CudaKernelLM`)。贡献重心见 [`../../discussion/hpec_batched_lm_benchmark_direction_zh.md`](../../discussion/hpec_batched_lm_benchmark_direction_zh.md) (草稿待 review): scipy / torch / cuda (/gpufit/padded) 同批对比 + roofline, SR 恢复降级为「冻结一个 CO 明显有用的朴素配置」demonstrator。本 INTERNAL 只管 kernel 实现, 不随方向变。

---

## 这是什么 / What it does

**符号回归 (Symbolic Regression, SR)** 用进化算法找一个"形状" `f(x)` 拟合 (x, y) 数据 —
比如要发现 `f(r, F, θ) = c · r · F · sin(θ)` 这种公式 (Feynman 力矩公式 I.18.12).

进化每一代会产出 ~1000 棵**候选表达式树** (e.g. `c0·sin(x) + c1·x²`, `exp(c0·x) + c1·x³`, ...).
对每棵树, 在评分之前先要**把里面的常数 `c` 调到让 MSE 最小** —
这是一个**非线性最小二乘 (NLS, Non-Linear least Squares)** 问题, 经典解法是 Levenberg-Marquardt (LM).

每棵候选树有两个挑战:
- **形状不同** (算子组合各异: sin / exp / div / pow / ...)
- **常数个数 K 不同** (heterogeneous-K, 这版数据集 K ∈ [0, 8])

scipy 一棵一棵串行跑 LM → 1000 棵需要约 15 分钟 (16 CPU worker 并行也得 ~100 秒).
本仓库写了一个 CUDA kernel **同时解 1000 个异构 NLS**, 在 1 张 A100 上 0.7 秒搞定.

> ⚠️ **当前定位是 baseline / MVP**: 算法跑得通 + 数值正确性有诊断,
> 但 `xtol` / `max_iter` / FD `ε` / `λ` 起点这些超参数都是粗设, 还没仔细调.
> 优化空间见下面 Roadmap.

---

## 性能数字 / Headline

跑 **1000 棵真 EvoGP 候选树** (Feynman I.18.12, gen=20, pop=1000, 每棵 N=1000 数据点):

| 后端 | 硬件 | 墙钟 | 吞吐 |
|---|---|---:|---:|
| **batch-lm-sr** (GPU) | 1×A100-80GB | **0.73 s** | **1353 trees/s** |
| scipy LM (CPU 并行) | 16 workers | 98.5 s | 8.5 trees/s |
| **加速比** | — | — | **~135×** |

> ⏱️ 0.73s 是 A100 上的数. 不同卡 (RTX 30xx / 40xx / H100) 会有差异, 通常 ±2× 以内. 见下面"GPU 架构"一节.

> ⚠️ **2026-06-11 kernel 质量修复** (相对 FD 步长 + xtol=1e-6, 见"已知限制"): 上表是**修复前** A100 的数.
> 修复后在 RTX 5070 Ti laptop 上的 parity: loss-down 94.0%, 1.05× **92.4%** (修复前 89.9%), 10× 99.8% —
> 见 [`verify_report_5070ti_w0.md`](../archive/kernel/verify_report_5070ti_w0.md). **A100 重测清单见 [`RERUN_A100.md`](../archive/kernel/RERUN_A100.md)**.

正确性诊断 (跟 scipy 同种群比较):

| 视角 | 数值 |
|---|---:|
| **loss 确实降了** (K>0 树里 `final_loss < initial_loss`) | **93.6%** |
| **跟 scipy 几乎等价** (`gpu_loss ≤ 1.05× sc_loss`) | 89.9% |
| **跟 scipy 差距 ≤ 2×** | 92.8% |
| **跟 scipy 差距 ≤ 10× (无 blow-up)** | **100%** |

完整报告: [`data/verify_report.md`](data/verify_report.md)

---

## 环境准备 / Prerequisites

| 项 | 要求 | 用途 |
|---|---|---|
| NVIDIA GPU | 算力 ≥ 7.0 | 跑 `./batch_lm` (CUDA kernel) |
| NVIDIA driver | 较新版本 (支持 CUDA 12.x) | `nvidia-smi` 能看到 GPU |
| CUDA Toolkit | 12.x | 提供 `nvcc` 编译器, 注意区分 driver (装 GPU 那个) vs toolkit (开发用) |
| C 编译器 | gcc 或 clang | 编译 `inspect` (pure C, 无 CUDA) |
| Python | 3.10+ | 跑 `verify.py` 用 |
| Python 包 | `numpy`, `scipy` | 见 `requirements.txt` |

> ℹ️ 仓库自带预跑好的 `data/pop.bin` (1000 棵种群, 4 MB), clone 完不用装 EvoGP 就能跑主路径.
> **只有重新 dump 自己的种群时才需要 EvoGP + PyTorch** (见"高级"一节).

---

## 环境安装 / Environment setup

### 选项 A — Linux native (最简, 推荐)

1. **NVIDIA driver** 一般 Linux 桌面/服务器已经装好. 验证:
   ```bash
   nvidia-smi
   ```
   能看到 GPU 即可.

2. **CUDA Toolkit 12.x** — 两种主流路线:

   **路线 A1 — conda (推荐, 不污染系统)**:
   ```bash
   conda create -n cuda-toolkit -c nvidia cuda-toolkit=12.4 -y
   conda activate cuda-toolkit
   nvcc --version  # 应该看到 release 12.4
   ```

   **路线 A2 — apt (Ubuntu, 系统级)**:
   ```bash
   # 跟 NVIDIA 官方指南 https://developer.nvidia.com/cuda-downloads
   # 别用发行版自带的 `nvidia-cuda-toolkit` (版本一般偏老),
   # 走 NVIDIA 的 repo 装新版.
   nvcc --version
   ```

3. **Python 依赖**. 推荐用 [uv](https://docs.astral.sh/uv/) (一行装好, 自动管 venv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh     # 装 uv (一次性)
   uv venv                                              # 建 .venv/
   uv pip install -r requirements.txt
   ```

   不想用 uv, 走传统 venv + pip 也行:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate     # Windows / WSL Windows shell: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   > ⚠️ Ubuntu 23.04+ / Debian 12+ 的系统 python 不让 `pip install` 直接装到 system site-packages
   > (PEP 668), 必须先建 venv 再装. 别加 `--break-system-packages` 这种参数硬冲, 走 venv 是正路.

### 选项 B — Windows + WSL2 (Windows 用户走这条)

Windows 上原生编译 CUDA 很折腾, 99% 的 ML 开发者用 WSL2.

1. **Windows 侧**: 装最新 NVIDIA driver — 任意近期版本 (≥ 525) 都自带 WSL2 GPU 转发支持.
   **不需要在 Windows 装 CUDA Toolkit**.

2. **装 WSL2 + Ubuntu**. PowerShell (管理员) 跑:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   重启, 进 WSL Ubuntu shell.

3. **在 WSL Ubuntu 里**, 先验证 GPU 可见:
   ```bash
   nvidia-smi   # 应该能看到 Windows 那张卡
   ```
   如果看不到: Win 侧 driver 太老, 升级到 ≥ 525.

4. **在 WSL Ubuntu 里装 CUDA Toolkit**. 跟选项 A 完全一样 (conda 或 apt 都行).
   走 NVIDIA 官方的 [WSL 安装指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   也可以 — 但注意装的是 **WSL 专用 deb 包**, 不是 Linux 通用版.

5. 剩下跟 Linux 一样: 装 uv → `uv venv` → `uv pip install -r requirements.txt` (见选项 A 第 3 步).

> ⚡ **重要**: 把这个仓库 clone 到 WSL 的 `~/` (Linux 文件系统), **不要 clone 到 `/mnt/c/...`** —
> 跨文件系统访问慢 5-10×, build 会很慢.

### 选项 C — 原生 Windows (不推荐)

技术上可行 (装 MSVC build tools + CUDA Toolkit for Windows), 但 `Makefile` 是 Unix 风格,
要在 Windows 跑得用 Git Bash / MSYS2 / Cygwin, 或者把 Makefile 手翻成 `.bat`.
**强烈建议走 WSL2** (选项 B).

### GPU 架构 / Compute capability

默认 `Makefile` 用 `-arch=sm_80` (A100, 数据中心 Ampere). 不同卡需要改:

| 卡型 | `sm_XX` |
|---|---|
| V100 (Volta) | `sm_70` |
| T4 / RTX 20xx (Turing) | `sm_75` |
| A100, A30 (data-center Ampere) | `sm_80` |
| A40, RTX 30xx (consumer Ampere) | `sm_86` |
| H100 (Hopper) | `sm_90` |
| L40, RTX 40xx (Ada Lovelace) | `sm_89` |
| RTX 50xx (Blackwell, 消费级) | `sm_120` |

直接改 `Makefile` 那一行, 或者命令行覆盖:
```bash
make NVCC_FLAGS="-O2 -arch=sm_86 -std=c++17 -lineinfo --use_fast_math"
```

不确定自己卡的算力, 查 https://developer.nvidia.com/cuda-gpus

---

## 快速上手 / Quick start

clone 完, 用仓库自带的 `data/pop.bin`, 五步看到结果:

```bash
# 1. 编译 — 产出 ./batch_lm 和 ./inspect
make

# 2. 体检种群 (pure C, ~1 秒)
./inspect data/pop.bin
# → 期望: PASS, M_prob=1000, K_max=8, max_stack=11

# 3. 跑 GPU 拟合 (~0.7s on A100, 视卡浮动)
./batch_lm data/pop.bin
# → 写出 data/c_final.bin + data/status.bin

# 4. (可选) 跟 scipy 对账 — 用 16 个 CPU worker 并行 (~100 秒)
# uv 用户:        uv run python verify.py --nproc 16 --gpu-elapsed 0.73
# venv 已 activate: python verify.py --nproc 16 --gpu-elapsed 0.73
# → 写出 data/verify_report.md, 可以打开看跟 scipy 的对账细节
```

### 跑回归测试 / Tests

3 套 fixture + 量级扫描 + parity 闸门, 应全 PASS:

```bash
uv run python tests/test_fixture.py
uv run python tests/test_fixture_op.py
uv run python tests/test_fixture_stress.py
uv run python tests/test_fixture_scale.py    # 常数量级 1e-6…1e6 (相对 FD 步长的验收)
uv run python tests/test_parity_gate.py      # vs scipy-fp64 质量闸门 (~3 分钟, 吃 CPU)
```

**换代规则**: 任何 kernel 改动 (性能优化 / 新变体) 必须过 `test_fixture_scale.py` +
`test_parity_gate.py` 两道闸, 才允许替换 SR 实验在用的版本. 变体二进制用
`BATCH_LM=../batch_lm_fusedfd uv run python tests/test_fixture_scale.py` 指定.

(后面所有 `uv run python ...` 命令, 如果你走的是 venv 路线, 直接 `python ...` 即可.)

如果 fixture binary 没生成, 重新生成:
```bash
uv run python tests/pop_fixture_gen.py
uv run python tests/pop_fixture_op_gen.py
uv run python tests/pop_fixture_stress_gen.py
```

---

## 排错 / Troubleshooting

| 现象 | 多半原因 |
|---|---|
| `nvcc: command not found` | CUDA Toolkit 没装, 或 `nvcc` 不在 `$PATH`. 检查: `which nvcc`; conda 路线要先 `conda activate cuda-toolkit` |
| `no kernel image is available for execution on the device` | GPU 算力跟 `-arch=sm_XX` 不匹配, 改 Makefile 见上面 "GPU 架构" 表 |
| `make: cc: Command not found` | 装 build-essential: `sudo apt install build-essential` |
| `ModuleNotFoundError: No module named 'numpy'` / `'scipy'` | 没装依赖, 或没 activate venv. 见上面 "环境安装" 第 3 步 |
| `error: externally-managed-environment` (pip) | Ubuntu 23.04+/Debian 12+ 的 PEP 668 限制, 必须先 `python3 -m venv .venv && source .venv/bin/activate` 再 pip install |
| WSL 里 `nvidia-smi` 看不到卡 | Win 侧 driver 太老, 升级到 ≥ 525 |
| `./batch_lm` 跑出来全是 `FAIL_NAN` (status=2) | 多半是种群数据本身有问题, 跑 `./inspect data/pop.bin` 看 stats |
| `verify.py` 跑得很慢 (> 5 分钟) | scipy 没并行起来, 用 `--nproc 16` (或你机器的 CPU 核数) |

---

## 数据流 / Pipeline

```
EvoGP 种群 (GPU forest tensor)
        │
        │  dump_evogp.py     ← 唯一依赖 EvoGP 的一步, 只跑一次拿到 pop.bin
        ▼
   data/pop.bin               ← 冻结的输入, 后面所有步骤的契约
        │
        ├──────────────────┐
        ▼                  ▼
   ./inspect (体检)      ./batch_lm (主算法, GPU)
                              │
                              ▼
                  data/c_final.bin + data/status.bin
                              │
                              ▼
                       verify.py (scipy 对账, 多进程并行)
                              │
                              ▼
                     data/verify_report.md
```

四个程序通过 **binary 文件松耦合**, 不互相 import — 任意一步都能单独 rerun.

---

## 算法 / Algorithm

### 数学层 — Levenberg-Marquardt 主循环

每棵树独立解一个 NLS: 找 `c` 使 `loss(c) = (1/2) · Σᵢ (y_pred(xᵢ; c) − yᵢ)²` 最小.
用 Gauss-Newton 配 Marquardt 阻尼:

```
loop until converged or max_iter:
    J  ← Jacobian of residual r(c) = y_pred − y_target, w.r.t. c  (有限差分, ε=1e-3)
    H_damped ← JᵀJ + λ · diag(JᵀJ)                                # 阻尼后的近似 Hessian
    g  ← Jᵀ · r                                                    # 梯度
    解 H_damped · δ = g  (K×K Cholesky)
    c_try ← c + δ
    if loss(c_try) < loss(c):
        c ← c_try;  λ ← λ / 10                                     # 接受 → 步长大胆点
    else:
        λ ← λ × 10                                                  # 拒绝 → 缩到陡降方向, 步小一点
    if ‖δ‖ < xtol · (‖c‖ + xtol):  break                           # 收敛
```

符号:
- `r(c) = y_pred(c) − y_target ∈ ℝᴺ` (residual 向量, N 个数据点)
- `J ∈ ℝᴺˣᴷ`, 第 (i, k) 个元素是 `∂rᵢ / ∂cₖ` (这版用 forward FD 近似)
- `λ` Marquardt 阻尼系数, 1e-3 起跑, 接受步缩 ×10, 拒绝步放 ×10
- `xtol` 收敛阈值 (步长相对常数 magnitude)

### GPU 设计

- **Warp-per-problem**: 32 个 thread (1 个 warp) 处理 1 棵树, 0 warp divergence
- **Column-major Jacobian** `J[m, k, i]` → coalesced memory access
- Host 端跑外层 accept/reject 调度, kernel 跑数值热路径
- **Compile-time bounds**: `MAX_K=32` (每棵树常数数上限) / `MAX_STACK=64` (interpreter 栈深).
  当前 `pop.bin` 实测 `K_max=8` / `max_stack=11`, 余量充足

### Host ↔ GPU 协作 (一轮 LM iteration)

每行末尾 `[L<n>]` = `batch_lm.cu` 行号, 可以直接跳过去看代码.

```
预备 (LM 起跑前):
  - K=0 树预标 finished=3 (跳过, 无常数可优化)              [L341-345]
  - 初始 eval(c_init) + residual + loss                       [L351-358]
  - 初始 loss NaN → 预标 finished=2 (永远救不活)              [L362-365]
  - h_lam[m] = 1e-3, eps_fd = 1e-3, xtol = 1e-5              [L339, 347-348]


────────────── HOST (CPU) ──────────────────  ─────────── GPU (kernel) ────────────

主循环  for it in 0..max_iter:                                                 [L375]
  early-exit: 全员 finished → break                                            [L376-378]

  ─── A. 取 baseline y(c) ──────────────────────────────────────────
  功能: 拿上一轮 G 步算好的 y(c_accepted), FD 要用它做差分起点
  h_yb ← d_y                                                                   [L381]

  ─── B. FD Jacobian (主性能瓶颈, K_max+1 次 launch) ───────────────
  功能: 对每个常数 k 扰动 ε, 算 ∂y/∂c_k ≈ (y(c+ε·e_k) − y(c))/ε
  for k in 0..K_max-1:                                                         [L384]
    host: h_c_pert ← h_c, finished=false 的树 c[k] += ε                       [L385-389]
    H→D:  d_call ← h_c_pert                                                    [L390]
                                                eval_kernel_batched <<<>>>    [L391, def L112]
                                                  算扰动后的 d_yp
    D→H:  h_yp ← d_yp                                                          [L393]
    host: J[m, k, :] = (h_yp − h_yb) / ε                                       [L394-399]
  end
  H→D:  d_J ← h_J 一次性上传                                                   [L401]
  H→D:  d_call ← h_c (恢复未扰动)                                              [L402]

  ─── C. 法方程系数 JᵀJ + Jᵀr ──────────────────────────────────────
  功能: 把 K×N 的 J 缩成 K×K 的 JᵀJ 和 K 的 Jᵀr (warp reduction over N)
                                                build_jtj_jtr_kernel <<<>>>   [L405, def L152]

  ─── D. 解 Levenberg-Marquardt 方程 ────────────────────────────────
  功能: 解 (JᵀJ + λ·diag(JᵀJ))·δ = Jᵀr, Cholesky 分解 (1 thread/problem)
  H→D:  d_lam ← h_lam                                                          [L409]
                                                solve_kernel <<<>>>           [L410, def L183]
  D→H:  h_delta, h_solve_stat ← d_delta, d_stat                                [L412-413]
        (h_solve_stat: 0 = OK, -1 = Cholesky 退化, 必须读)

  ─── E. 算 trial step + loss ────────────────────────────────────────
  功能: c_try = c + δ, 算 y(c_try) → residual → loss_try (准备 accept/reject)
  host: h_c_try ← h_c, finished=false 的树 c[k] += δ[k]                       [L416-421]
  H→D:  d_call ← h_c_try                                                       [L422]
                                                eval_kernel_batched <<<>>>    [L423]
                                                residual_kernel <<<>>>        [L425, def L131]
                                                loss_kernel <<<>>>            [L427, def L137]
  D→H:  h_loss_try ← d_loss                                                    [L429]

  ─── F. Per-tree accept/reject + λ + 收敛判定 ────────────────────
  功能: 每棵树独立决策, 5 个分支:                                              [L439-485]
    for m in 0..M_prob:
      if finished: skip                                                        [L440]
      h_iter[m]++                                                              [L441]

      分支 1 — Cholesky 退化:                                                  [L446-451]
        h_solve_stat[m] != 0 → λ ×= 10, reject, λ > 1e12 标 FAIL_CHOLESKY (4)

      分支 2 — loss_try NaN/Inf:                                               [L456-459]
        c_try 把树推到奇点, 直接放弃 → 标 FAIL_NAN (2)

      算 d_norm = ‖δ‖, c_norm = ‖c_try‖                                       [L461-466]

      分支 3 — 收敛:                                                           [L468-473]
        d_norm < xtol·(c_norm + xtol)
        → c ← c_try, loss ← loss_try, finished=1 (CONVERGED)

      分支 4 — accept (loss 下降但没收敛):                                     [L474-479]
        loss_try < loss
        → c ← c_try, loss ← loss_try, λ ×= 0.1 (下次步长大胆点)

      分支 5 — reject (loss 没下降):                                           [L480-484]
        → λ ×= 10 (下次步长保守点), λ > 1e12 标 fail

  ─── G. 刷新 baseline (给下一轮 A 步用) ───────────────────────────
  功能: 算 y(c_accepted) 落到 d_y, residual 也刷一遍 (build_jtj_jtr 要用)
  H→D:  d_call ← h_c                                                           [L488]
                                                eval_kernel_batched <<<>>>    [L489]
                                                residual_kernel <<<>>>        [L491]

  log: 每 5 轮打印 done=n/M                                                    [L494-498]
end loop

收尾 (循环出来后):
  cudaDeviceSynchronize                                                        [L500]
  h_finished {0,1,2,3,4} → 外部 status_out, 剩余 0 的 → STATUS_MAXITER (1)
  写 status.bin + c_final.bin                                                  [L506-509]
```

**一轮 iter 的 GPU launch + H↔D 流量**:

| 资源 | 次数 | 备注 |
|---|---:|---|
| `eval_kernel_batched` launches | **K_max + 2** | B 步 (K_max 个扰动) + E (1 trial) + G (1 baseline 刷). **fused FD 可把 B 的 K_max 合 1 个** |
| `residual_kernel` launches | 2 | E + G |
| `loss_kernel` / `build_jtj_jtr` / `solve_kernel` | 1 / 1 / 1 | C / D / E |
| H→D copy | K_max + 4 | B 里每轮 K_max 次 + D + E + G |
| D→H copy | K_max + 3 | B 里每轮 K_max 次 + D + E |

Roadmap 里最高优先级的三个改造方向都围着这张表转:

- **device-side LM state** → 干掉 D/E/F 的全部 H↔D 同步 (一轮少 5+ 次)
- **fused FD kernel** → 干掉 B 内的 K_max 个 launch + 全部 H↔D
- **K-bucket dispatch** → 干掉 B 内 `if (k >= K_m) continue` 的空转 (K=2 的树被陪跑 32 次)

---

## 文件结构 / Files

```
batch-lm-sr/
├── README.md
├── LICENSE
├── requirements.txt        ← Python 依赖 (numpy + scipy)
├── Makefile                ← 编译入口
├── pop_format.h            ← binary 格式定义 (Python/C/CUDA 共用)
├── batch_lm.cu             ← GPU 主算法 (CUDA)
├── loader.{h,c}            ← pop.bin → 内存 (host 端 reader)
├── inspect.c               ← pop.bin 统计 dump (pure C, 无 CUDA)
├── m1_interpreter.py       ← Python 端的 tree forward eval (供 verify.py 用)
├── dump_evogp.py           ← 跑 EvoGP + 写 pop.bin (需要 EvoGP, 见"高级")
├── verify.py               ← scipy 并行对账, 写 verify_report.md
├── tests/                  ← 3 套 fixture 回归测试
│   ├── pop_fixture_gen.py            ← 算法基线 (3 棵 archetype)
│   ├── pop_fixture_op_gen.py         ← 算子覆盖 (8 棵, TAN/SINH/COSH/...)
│   ├── pop_fixture_stress_gen.py     ← 边界情况 (K=0 / NaN / 退化)
│   └── test_fixture*.py
└── data/                   ← 大部分内容 gitignored
    ├── pop.bin                       ← 仓库自带, 1000 棵真 EvoGP 种群, 4 MB
    ├── pop.bin.meta.txt              ← dump 元信息 (dataset / gen / seed)
    ├── c_final.bin (gitignored)      ← ./batch_lm 产物
    ├── status.bin  (gitignored)      ← ./batch_lm 产物
    └── verify_report.md              ← verify.py 产物
```

---

## 状态码 / `status.bin` encoding

`./batch_lm` 给每棵树写一个 int32 状态码:

| code | 含义 |
|:---:|---|
| `0` | CONVERGED — `‖δ‖ < xtol·(‖c‖ + xtol)` |
| `1` | MAXITER — 没收敛也没 fail, 用完了 max_iter |
| `2` | FAIL_NAN — loss 或 loss_try 出 NaN/Inf (eval 推到奇点) |
| `3` | K0_SKIP — K=0, 无常数可优化 |
| `4` | FAIL_CHOLESKY — JᵀJ 退化, Cholesky breakdown 反复触发, λ 爆 > 1e12 |

---

## 高级 / Advanced: 重新生成 pop.bin

要在自己的数据集 (或不同 EvoGP 设置) 上跑, 用 `dump_evogp.py`. **额外需要 PyTorch + EvoGP**.

1. 装 PyTorch (匹配你的 CUDA 版本, 见 https://pytorch.org/get-started/locally/):
   ```bash
   # 例如 CUDA 12.4 + Linux + pip:
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

2. 装 EvoGP — 跟 EvoGP 上游仓库的安装指南 (源码: https://github.com/EMI-Group/evogp).
   通常是 clone + `pip install -e .`.

3. 装 sympy (skeleton 表达式解析用):
   ```bash
   uv pip install sympy        # 或者 pip install sympy (venv activated)
   ```

4. 跑:
   ```bash
   uv run python dump_evogp.py \
       --dataset=feynman/I.18.12 --gen=20 --pop=1000 --N=1000 --seed=0 \
       -o data/pop.bin
   ```

加新题目: 在 `dump_evogp.py` 顶部的 `PROBLEMS` dict 里加一条 (skeleton_expr 用 sympy 语法; 见现有条目).

---

## 已知限制 / Known limits

- `MAX_K=32` / `MAX_STACK=64` 是 **compile-time** 上限. 超出会直接报错退出 (不会 silent corrupt)
- 不支持三元节点 (TFUNC, `IF(cond, a, b)`). dump 阶段就过滤掉; 默认 EvoGP 不产
- ~~FD Jacobian `ε=1e-3` 固定~~ → **已修 (2026-06-11)**: 改相对步长 `h = 1e-3·|c|` (c==0 退化为 1e-3),
  除数取 fp32 实际可表示步长, 三个变体同改. 绝对 ε 在常数量级远离 1 时毁掉 Jacobian
  (|c|≫1: 差分被 fp32 舍入吞掉 → J 列全零 → Cholesky 挂; |c|≪1: 割线远离切线 → **假收敛到错误常数**).
  验收: `tests/test_fixture_scale.py` (量级 1e-6…1e6, 修复前 7/11 修复后 11/11).
  eps_rel 取 1e-3 而非教科书 √eps≈3.45e-4: `--use_fast_math` 下 eval 误差 >1 ULP, 3.45e-4 尾部退化, 经 parity 对拍选定
- ~~`xtol=1e-5`~~ → **已收紧 `1e-6` (2026-06-11)**: ≈10 ULP @ fp32, 再紧进噪声. parity 1.05× 91.9→92.4%,
  10× 99.3→99.8%, 代价 +25% 迭代耗时. "只 scipy 收敛" 的剩余差距主要是 `max_iter=50` 上限 + scipy fp64 的余量
- 超参数 `max_iter` / `λ` 起点仍是粗设; `xtol` / FD `ε` 已经 parity 对拍定值 (见上两条)

---

## 路线图 / Roadmap

按预期收益分优先级. **最高优先级三项有协同, 分开做拿不到 10×, 一起做能拿 5-10×**.

### 最高优先级 — 真正的性能瓶颈

| 方向 | 期望影响 |
|---|---|
| **device-side LM state** | accept/reject + λ 调度全部搬上 GPU, 干掉一轮 5+ 次 H↔D 同步 |
| **fused FD kernel** | `K_max+1` 个 `eval_kernel_batched` launch → 1 个, 顺手干掉 `K_max` 次 H↔D |
| **K-bucket dispatch** | 按 K 分桶 launch, 干掉 `K_max` padding 浪费 (K=2 的树被陪跑 32 次) |

### 高优先级 — 拉收敛率 (不动算法, 调超参数)

| 方向 | 期望影响 |
|---|---|
| **xtol 1e-5 → 1e-8/1e-10** | 直接拉高 "只 scipy 收敛" 那 308 棵的回收率 |
| **Adaptive FD ε** | 现在固定 1e-3 偏粗, fp32 理论极限 √eps ≈ 3e-4 |
| **Per-tree adaptive λ 起点** | 现在全 1e-3, 形状差异大的树吃亏 (Cholesky breakdown 高发也跟这有关) |

### 中优先级

| 方向 | 期望影响 |
|---|---|
| **CUDA Graphs** | 最高优先级三项做完后再上, 减 launch overhead |
| **Fuse `residual_kernel` + `eval_kernel`** | eval 完直接减 ym, 省一次 N×M 内存往返 |
| **Multi-restart 抗 local minima** | 3× cost 拉高 1-2% 鲁棒性 |
| **Analytical Jacobian (tree autodiff)** | 终极方案, FD 永远比不过, 但实施成本高 |

### 低优先级 — 工程卫生

| 方向 | 备注 |
|---|---|
| `eval_tree_d` 未知 op 返回 NaN 而不是 0 | silent failure → 显式 fail |
| pop.bin `ym` dedup | 真实场景全树共享 y, 现在复制 M_prob 份白浪费 ~4 MB |
| Dump per-tree `iter` / `rejected` count | debug 信号现在只在 host 内存, 没写出来 |
| SVD / pivoted QR fallback | Cholesky 退化的兜底, 边际收益 |

### 长期 / 战略

| 方向 | 备注 |
|---|---|
| Multi-GPU (8×A100) | 单 GPU 都没跑满, 谈这个浪费 |
| EvoGP in-loop 集成 | 现在是离线 batch; 真做 memetic GP 时再说 |
| fp64 backend / 动态 `MAX_K` | 现在 fail loud 够用, 没明显需求驱动 |

**下一刀**: 如果只动 1 个, 先动 **device-side LM state** — 它是另两个最高优先级项的前置, 不做这个其它两个收益打折扣.

---

## License

MIT. 见 [`LICENSE`](LICENSE).
