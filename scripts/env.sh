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

# --- Julia for PySR's juliacall backend ---
# Point juliapkg at a local Julia so it does NOT try to download one through
# the throttled proxy (that hangs `import pysr`). pysr needs Julia >=1.10.3;
# 1.12.x matches the cached/precompiled packages in ~/.julia (see SETUP.md).
# Grab via: curl --noproxy '*' https://mirrors.tuna.tsinghua.edu.cn/julia-releases/bin/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz
_SR_JULIA="$HOME/julias/julia-1.12.6/bin/julia"
if [ -x "$_SR_JULIA" ]; then
  export PYTHON_JULIAPKG_EXE="$_SR_JULIA"
fi

# --- Bypass local proxy for Chinese pip/conda mirrors (bulk downloads) ---
# These are fast direct; paid proxy traffic is wasted on them.
_SR_NO_PROXY_EXTRA="pypi.tuna.tsinghua.edu.cn,mirrors.tuna.tsinghua.edu.cn,mirrors.aliyun.com,mirrors.bfsu.edu.cn,mirrors.ustc.edu.cn"
export no_proxy="${no_proxy:+$no_proxy,}$_SR_NO_PROXY_EXTRA"
export NO_PROXY="$no_proxy"

# --- Keep `uv run` from auto-syncing the venv ---
# evogp is installed editable (uv pip install -e upstream/evogp) but is NOT in
# uv.lock (by design — it's an optional upstream). A bare `uv sync` / the auto-
# sync that `uv run` does would PRUNE evogp as "not in the lock", silently
# breaking dump_evogp / the demonstrator. The venv is already fully provisioned,
# so disable auto-sync; run an explicit `uv sync` yourself when deps change.
export UV_NO_SYNC=1
