// co_lib.h — C ABI for the in-process CO drop-in (P1).
//
// See docs/kernel/INPROCESS_CO_PLAN.md. Three extern "C" functions over a
// persistent CUDA context (CoCtx), loaded from Python via ctypes (co_inproc.py).
// The whole point: pay CUDA primary-context init ONCE per process instead of per
// generation; per-call cost is just the LM loop.
//
// Variant (MUST-FIX #1): the SAME co_lib.cu is compiled twice —
//   libcusr_co_fd.so  (plain float, host-side FD Jacobian; mirrors batch_lm.cu)
//   libcusr_co_ad.so  (-DUSE_AD: forward-mode AD Jacobian; mirrors batch_lm_ad.cu)
// Parity is always pinned to the SAME variant on both sides (FD .so vs FD golden,
// AD .so vs AD golden). The end-to-end experiment uses AD.
//
// The .so must NEVER exit() the interpreter mid-evolution: every CUDA error and
// every consistency-check failure RETURNS a negative code (MUST-FIX, plan step 2).

#ifndef CO_LIB_H
#define CO_LIB_H

#include <stdint.h>
#include "pop_format.h"   // PopHeader (64 B) / TreeMeta (16 B)

#ifdef __cplusplus
extern "C" {
#endif

// Return codes (shared by co_init / co_optimize):
//   0          OK
//   2          K_max > MAX_K(32) or max_stack > MAX_STACK(64) — caller pre-filters
//  -1          CUDA error (a cudaError_t was non-success somewhere)
//  -2          sum(K) != header.total_c           (MUST-FIX #6 consistency)
//  -3          sum(n_nodes) != header.total_nodes  (MUST-FIX #6 consistency)
//  -4          null pointer / bad argument
//  -5          out-of-memory growing a device buffer
#define CO_OK                 0
#define CO_ERR_BOUNDS         2
#define CO_ERR_CUDA          -1
#define CO_ERR_SUM_C         -2
#define CO_ERR_SUM_NODES     -3
#define CO_ERR_NULLARG       -4
#define CO_ERR_OOM           -5

// Create/select the device's primary context once and return an opaque handle.
// 0 OK; <0 CUDA error.
int  co_init(int device_id, void **handle_out);

// Run the LM constant-optimization on ONE pop over the persistent context.
// All array sizes derive from `header` (M_prob, total_nodes, total_c, N, n_vars,
// K_max, max_stack). Output buffers are CALLER-allocated:
//   c_final_out   [total_c]   float32
//   status_out    [M_prob]    int32   (STATUS_* from lm_core.cuh)
//   loss_init_out [M_prob]    float32 (may be NULL for FD; AD fills it)
//   loss_final_out[M_prob]    float32 (may be NULL for FD; AD fills it)
// Returns 0 OK; 2 = bounds; <0 = error (see codes above). NEVER exits.
int  co_optimize(void *handle, const PopHeader *header,
                 const int32_t *nt, const float *nv, const int32_t *ci,
                 const TreeMeta *metas,
                 const float *c_init, const float *xs, const float *ym,
                 int max_iter,
                 float *c_final_out, int32_t *status_out,
                 float *loss_init_out, float *loss_final_out);

// Read the setup/loop wall-clock split (ms) of the LAST co_optimize call on this
// process. Returns 0 and fills both when the .so was built with -DPROFILE; returns
// 1 and writes -1.0 sentinels otherwise. NEW symbol — does not touch co_optimize's
// ABI. Used by the amortization suite (docs/kernel/INPROCESS_CO_PLAN.md) to get the
// per-gen setup/loop split from libcusr_co_*_prof.so.
int  co_last_split(double *setup_ms_out, double *loop_ms_out);

// Free all device + host scratch and the handle. Safe on NULL.
void co_teardown(void *handle);

#ifdef __cplusplus
}
#endif

#endif // CO_LIB_H
