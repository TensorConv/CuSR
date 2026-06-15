# `scripts/` —— 仓库级脚本

跨实验的**基础设施脚本**。区别于 `experiments/XXX/scripts/`（实验专属）。

## 文件清单

| 文件 | 用途 |
|------|------|
| `setup_upstream.sh` | 克隆 + pin 所有 `upstream/*` 仓库到指定 SHA（evogp、pse、eml_sr、sr_jl、PSRN 等）。幂等，重跑会跳过已克隆的 |
| `env.sh` | 环境变量注入：`CUDA_HOME`、`LD_LIBRARY_PATH`、`TORCH_CUDA_ARCH_LIST=8.0`（A100）、代理 bypass 等。脚本用 `source scripts/env.sh` 加载 |
| `slurm/` | SLURM 作业脚本模板。详见 `slurm/README.md` |

## 运行约定

1. **从仓库根跑**：所有脚本假定 cwd 是 `/home/weish/sr/`。
2. **先 `source scripts/env.sh`**：任何要用 CUDA / 要绕代理的脚本都得先加载这个。
3. **不自动 commit**：脚本可能改 `upstream/` 的 submodule 指针 / workspace 状态，但不 git commit。

## 新增仓库级脚本

放这里而不是 `experiments/XXX/scripts/` 的判据：**会被多个实验 / 多个 shell 调用**。

例子：
- ✓ "重建 evogp 的 CUDA 扩展"
- ✓ "下载所有 reference paper"（已经有 `papers/download.sh`，那个更贴近 papers/ 所以放 `papers/`）
- ✗ "跑 T-30 的 exp_a" → 放 `experiments/003_nls_bench/scripts/`
