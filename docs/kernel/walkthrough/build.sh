#!/usr/bin/env bash
# 用法: ./build.sh s1     编译并运行 s1_*.cu
#       ./build.sh s4     编译并运行 s4_*.cu
#
# 做的事: source 仓库的 scripts/env.sh 拿 conda 的 nvcc 12.8 →
# nvcc -arch=native(自动识别本机 sm_120)编译该级的 .cu → 运行。
# 产物(无扩展名的可执行文件)被本目录 .gitignore 忽略。
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "用法: ./build.sh <级号>   例: ./build.sh s1" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# walkthrough -> 008_batch_lm_sr -> experiments -> 仓库根
REPO="$(cd "$HERE/../../.." && pwd)"

# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "找不到 nvcc —— 检查 scripts/env.sh 里的 conda cuda-toolkit env" >&2
  exit 1
fi

shopt -s nullglob
matches=("$HERE/$1"*.cu)
if [ ${#matches[@]} -eq 0 ]; then
  echo "找不到 '$1*.cu'(在 $HERE 下)" >&2
  exit 1
fi
if [ ${#matches[@]} -gt 1 ]; then
  echo "'$1*.cu' 匹配到多个,说具体点:" >&2
  printf '  %s\n' "${matches[@]}" >&2
  exit 1
fi
SRC="${matches[0]}"
BIN="${SRC%.cu}"

echo "==> 编译 $SRC"
nvcc -arch=native -O2 -std=c++17 -lineinfo "$SRC" -o "$BIN"
echo "==> 运行 $BIN"
echo "---"
"$BIN"
