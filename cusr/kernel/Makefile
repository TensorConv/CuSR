# batch-lm-sr — GPU batched Levenberg-Marquardt for SR populations.
#
# Targets:
#   inspect   — 读 pop.bin 打印统计 (pure C, 无 CUDA 依赖)
#   batch_lm  — GPU batched LM 主程序 (需 nvcc + CUDA toolkit)

NVCC ?= nvcc
CC   ?= cc
NVCC_FLAGS ?= -O2 -arch=sm_80 -std=c++17 -lineinfo --use_fast_math
CC_FLAGS   ?= -O2 -Wall -std=c11

.PHONY: all clean show

all: inspect batch_lm

inspect: inspect.c pop_format.h
	$(CC) $(CC_FLAGS) -o $@ inspect.c

batch_lm: batch_lm.cu loader.c loader.h pop_format.h
	$(NVCC) $(NVCC_FLAGS) -o $@ batch_lm.cu loader.c

show: inspect
	@if [ -f data/pop.bin ]; then ./inspect data/pop.bin; else echo "data/pop.bin 不存在, 先跑 dump_evogp.py"; fi

clean:
	rm -f inspect batch_lm
