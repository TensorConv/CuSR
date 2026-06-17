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

# fused-FD variant (reference / old Jacobian path). Plain build, no PROFILE.
batch_lm_fusedfd: batch_lm_fusedfd.cu loader.c loader.h pop_format.h
	$(NVCC) $(NVCC_FLAGS) -o $@ batch_lm_fusedfd.cu loader.c

# Profiling build of the fused-FD variant (-DPROFILE): per-kernel CUDA-event
# timing + host/transfer residual. Same flags as the real binary so codegen
# matches; emits an extra PROFILE_JSON line. Not committed (see .gitignore).
batch_lm_fusedfd_prof: batch_lm_fusedfd.cu loader.c loader.h pop_format.h
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -o $@ batch_lm_fusedfd.cu loader.c

# v4: forward-mode AD Jacobian (drop-in J producer, replaces fd_jacobian).
# See docs/kernel/AD_JACOBIAN_V4.md. AD_W = chunk width (default 8 in source).
batch_lm_ad: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ batch_lm_ad.cu loader.c

batch_lm_ad_prof: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -o $@ batch_lm_ad.cu loader.c

show: inspect
	@if [ -f data/pop.bin ]; then ./inspect data/pop.bin; else echo "data/pop.bin 不存在, 先跑 dump_evogp.py"; fi

clean:
	rm -f inspect batch_lm
