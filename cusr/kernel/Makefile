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

batch_lm: batch_lm.cu loader.c loader.h pop_format.h lm_core.cuh ad_interp.cuh
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
batch_lm_ad: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh lm_core.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ batch_lm_ad.cu loader.c

batch_lm_ad_prof: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -o $@ batch_lm_ad.cu loader.c

# fp64-honesty guard is ON by default in batch_lm / batch_lm_ad. These -DNO_FP64_GUARD
# builds disable it (byte-identical to the pre-guard kernel) — used by the honesty
# regression tests to demonstrate the gap the guard closes, and for guard A/B timing.
batch_lm_noguard: batch_lm.cu loader.c loader.h pop_format.h lm_core.cuh ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -DNO_FP64_GUARD -o $@ batch_lm.cu loader.c

batch_lm_ad_noguard: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh lm_core.cuh
	$(NVCC) $(NVCC_FLAGS) -DNO_FP64_GUARD -o $@ batch_lm_ad.cu loader.c

# Trust-region (absolute Levenberg damping) variant of the AD Jacobian engine.
batch_lm_ad_tr: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -DTRUST_REGION -o $@ batch_lm_ad.cu loader.c

batch_lm_ad_tr_prof: batch_lm_ad.cu loader.c loader.h pop_format.h ad_interp.cuh
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -DTRUST_REGION -o $@ batch_lm_ad.cu loader.c

# Trust-region (absolute Levenberg damping) variant of the fused-FD Jacobian engine.
batch_lm_fusedfd_tr: batch_lm_fusedfd.cu loader.c loader.h pop_format.h
	$(NVCC) $(NVCC_FLAGS) -DTRUST_REGION -o $@ batch_lm_fusedfd.cu loader.c

batch_lm_fusedfd_tr_prof: batch_lm_fusedfd.cu loader.c loader.h pop_format.h
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -DTRUST_REGION -o $@ batch_lm_fusedfd.cu loader.c

# ---- in-process CO drop-in (P1, docs/kernel/INPROCESS_CO_PLAN.md) ----------
# ONE source (co_lib.cu) compiled to two .so's (MUST-FIX #1): FD (default) and
# AD (-DUSE_AD). Shared device kernels via lm_core.cuh so codegen can't drift.
# Same NVCC_FLAGS as the goldens (byte-parity is FMA-contraction sensitive) plus
# -Xcompiler -fPIC -shared. No loader.c — co_optimize takes raw host pointers.
libcusr_co_fd.so: co_lib.cu co_lib.h lm_core.cuh ad_interp.cuh pop_format.h
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fPIC -shared -o $@ co_lib.cu

libcusr_co_ad.so: co_lib.cu co_lib.h lm_core.cuh ad_interp.cuh pop_format.h
	$(NVCC) $(NVCC_FLAGS) -DUSE_AD -Xcompiler -fPIC -shared -o $@ co_lib.cu

# Profiling builds (-DPROFILE) for the amortization step's setup/loop split only.
libcusr_co_fd_prof.so: co_lib.cu co_lib.h lm_core.cuh ad_interp.cuh pop_format.h
	$(NVCC) $(NVCC_FLAGS) -DPROFILE -Xcompiler -fPIC -shared -o $@ co_lib.cu

libcusr_co_ad_prof.so: co_lib.cu co_lib.h lm_core.cuh ad_interp.cuh pop_format.h
	$(NVCC) $(NVCC_FLAGS) -DUSE_AD -DPROFILE -Xcompiler -fPIC -shared -o $@ co_lib.cu

show: inspect
	@if [ -f data/pop.bin ]; then ./inspect data/pop.bin; else echo "data/pop.bin 不存在, 先跑 dump_evogp.py"; fi

clean:
	rm -f inspect batch_lm
