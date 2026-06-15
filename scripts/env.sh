# Source this file to set up CUDA + uv env vars for this project.
# Usage: `source scripts/env.sh`
#
# Why this file exists:
#   - System has no nvcc; conda provides a self-contained CUDA 12.4 toolkit
#     in a dedicated env (see notes/log.md 2026-04-15 entry).
#   - uv still owns Python deps; conda is ONLY used for the toolkit.

# --- Project root (portable regardless of where this is sourced from) ---
_SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export SR_ROOT="$_SR_ROOT"

# --- CUDA toolkit from conda ---
CUDA_TOOLKIT_DIR="$HOME/miniconda3/envs/cuda-toolkit"
if [ -d "$CUDA_TOOLKIT_DIR" ]; then
  export CUDA_HOME="$CUDA_TOOLKIT_DIR"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
  echo "[env.sh] warning: $CUDA_TOOLKIT_DIR not found — run: conda create -n cuda-toolkit -c nvidia cuda-toolkit=12.4" >&2
fi

# --- GPU arch for building CUDA extensions on A100 ---
# sm_80 = A100. Add more if the hardware mix changes (e.g. "8.0;9.0" for H100).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

# --- Bypass local proxy for Chinese pip/conda mirrors (bulk downloads) ---
# These are fast direct; paid proxy traffic is wasted on them.
_SR_NO_PROXY_EXTRA="pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn,mirrors.aliyun.com,mirrors.bfsu.edu.cn,mirrors.ustc.edu.cn"
export no_proxy="${no_proxy:+$no_proxy,}$_SR_NO_PROXY_EXTRA"
export NO_PROXY="$no_proxy"
