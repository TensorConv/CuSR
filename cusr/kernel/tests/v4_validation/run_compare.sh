#!/bin/bash
# AD vs fusedfd(old) vs scipy numerical comparison + AD parity acceptance (rung4).
# Run in background; logs to /tmp/cmp/compare.log.
exec > >(tee /tmp/cmp/compare.log) 2>&1
echo "=== compare run start ==="
cd /home/weish/hao/CuSR || exit 2
source scripts/env.sh >/dev/null 2>&1
FLAGS="-O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math"
cd cusr/kernel || exit 2

echo "=== build old fusedfd (drop-in reference) ==="
nvcc $FLAGS -o batch_lm_fusedfd batch_lm_fusedfd.cu loader.c && echo "fusedfd build OK" || { echo "fusedfd build FAILED"; exit 3; }

POP=../../data/fixtures/pop.bin
mkdir -p /tmp/cmp/ad_out /tmp/cmp/fd_out

echo "=== run AD + fusedfd binaries on pop.bin (keep c_final/status for head-to-head) ==="
CUDA_VISIBLE_DEVICES=0 ./batch_lm_ad       "$POP" /tmp/cmp/ad_out 2>&1 | grep -E "完成|总耗时" | sed 's/^/[AD]  /'
CUDA_VISIBLE_DEVICES=0 ./batch_lm_fusedfd  "$POP" /tmp/cmp/fd_out 2>&1 | grep -E "完成|总耗时" | sed 's/^/[FD]  /'

cd tests || exit 2
echo ""
echo "############ AD vs scipy (parity gate = rung4 acceptance) ############"
BATCH_LM=../batch_lm_ad      GATE_NPROC=64 GATE_REPORT=/tmp/cmp/ad_report.md CUDA_VISIBLE_DEVICES=0 uv run python test_parity_gate.py
echo "AD_GATE_EXIT=$?"
echo ""
echo "############ fusedfd vs scipy (old-version reference) ############"
BATCH_LM=../batch_lm_fusedfd GATE_NPROC=64 GATE_REPORT=/tmp/cmp/fd_report.md CUDA_VISIBLE_DEVICES=0 uv run python test_parity_gate.py
echo "FD_GATE_EXIT=$?"
echo ""
echo "=== compare run DONE ==="
