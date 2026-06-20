// co_lib.cu — in-process CO drop-in (P1). See docs/kernel/INPROCESS_CO_PLAN.md.
//
// ONE source compiled to two .so's (MUST-FIX #1):
//   libcusr_co_fd.so  (default)   — plain-float solve + host-side FD Jacobian,
//                                    byte-mirrors batch_lm.cu's LM loop.
//   libcusr_co_ad.so  (-DUSE_AD)  — fp32 solve + device forward-AD Jacobian,
//                                    byte-mirrors batch_lm_ad.cu's LM loop.
//
// The FMA-contraction-sensitive __global__ kernels (eval/residual/loss/build_jtj)
// live in lm_core.cuh, single-sourced with the standalones so device codegen
// cannot drift. The variant-specific solve_kernel + Jacobian + LM-loop body are
// COPIED here verbatim from the matching standalone (kept identical so the
// per-variant byte-parity gate in test_inproc_reuse.py passes); any drift makes
// that gate fail with a real byte mismatch.
//
// Persistent context (CoCtx): all device handles + host scratch, each tracked by
// a BYTE capacity and grown (never shrunk) on demand. Products like d_J =
// M*K_max*N and d_JtJ = M*K_max*K_max are sized by total bytes, not per-dim, so
// an M-shrink-then-K-grow can't under-allocate (MUST-FIX #6). Every launch grid
// and every D2H slice uses the CURRENT pop's M/total_c, never capacity
// (MUST-FIX #4). co_optimize re-seeds h_c from THIS pop's c_init, resets
// h_lam/h_finished/h_iter/h_rejected/d_stat/h_solve_stat, re-marks K=0, recomputes
// loss_init — i.e. a fresh start every call (MUST-FIX #4). No exit(): errors RETURN
// a negative code (plan step 2).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <new>
#include <cuda_runtime.h>

#include "lm_core.cuh"   // shared device kernels + STATUS_* (its CUDA_CHECK/exit
                         // is NOT used here; we use CK() which RETURNS instead)
#include "co_lib.h"

// ---- setup/loop SPLIT profiling (-DPROFILE only; amortization step) ----------
// Mirrors the standalone's clock_gettime(CLOCK_MONOTONIC) split (batch_lm_ad.cu:
// setup_ms = load_done..loop_begin, loop_ms = loop_begin..loop_end), but the .so
// has no load phase (host pointers are passed in), so "setup" = co_optimize entry
// up to just before the LM for-loop (grow + H2D + reseed + initial eval/loss) and
// "loop" = the for-loop body. Recorded into a global and read by co_last_split().
// EVERY line here is behind #ifdef PROFILE, so the default (headline) build is
// byte-for-byte the parity-passing kernel — the non-prof .so is what step-3's
// goldens were taken against and must NOT drift.
#ifdef PROFILE
#include <time.h>
static double g_last_setup_ms = -1.0;
static double g_last_loop_ms  = -1.0;
#define SPLIT_MS(a,b) (((b).tv_sec-(a).tv_sec)*1e3 + ((b).tv_nsec-(a).tv_nsec)/1e6)
#endif

// ---- .so error policy: record + RETURN, never exit ------------------------
// (lm_core.cuh's CUDA_CHECK exit()s the interpreter — wrong for a .so.)
#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "[co_lib] CUDA error %s at %s:%d\n", \
            cudaGetErrorString(_e), __FILE__, __LINE__); \
    return CO_ERR_CUDA; } } while (0)

// ===========================================================================
// variant-specific solve_kernel — copied verbatim from the matching standalone.
// FD (default): plain float. AD (-DUSE_AD): identical fp32 math here (no
// SOLVE_FP64/PIVOT_FLOOR/TRUST_REGION — those are ablations not in the goldens).
// ===========================================================================
__global__ void co_solve_kernel(
    const float *JtJ, const float *JtR, const TreeMeta *metas,
    int M_prob, int K_max, const float *lambda_arr,
    float *delta, int *status)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_prob) return;
    int K = metas[m].K;
    float lam = lambda_arr[m];

    float A[MAX_K * MAX_K];
    float b[MAX_K];

    for (int j1 = 0; j1 < K; j1++) {
        b[j1] = -JtR[m*K_max + j1];
        for (int j2 = 0; j2 < K; j2++)
            A[j1*K + j2] = JtJ[m*K_max*K_max + j1*K_max + j2];
        A[j1*K + j1] *= (1.0f + lam);
    }
    for (int j = 0; j < K; j++) {
        float s = A[j*K + j];
        for (int k = 0; k < j; k++) s -= A[j*K+k] * A[j*K+k];
        if (s <= 0.0f) { status[m] = -1; for (int i=0;i<K;i++) delta[m*K_max+i]=0.0f; return; }
        A[j*K + j] = sqrtf(s);
        for (int i = j+1; i < K; i++) {
            float t = A[i*K + j];
            for (int k = 0; k < j; k++) t -= A[i*K+k] * A[j*K+k];
            A[i*K + j] = t / A[j*K + j];
        }
    }
    for (int i = 0; i < K; i++) {
        float s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*K+k] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = K-1; i >= 0; i--) {
        float s = b[i];
        for (int k = i+1; k < K; k++) s -= A[k*K+i] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = 0; i < K; i++) delta[m*K_max + i] = b[i];
    status[m] = 0;
}

// ===========================================================================
// CoCtx — persistent device + host state with per-buffer BYTE capacity.
// ===========================================================================
struct DBuf { void *ptr = nullptr; size_t cap = 0; };

struct CoCtx {
    int device_id = -1;
    // device buffers (grown by byte capacity)
    DBuf d_nt, d_nv, d_ci, d_call, d_metas;
    DBuf d_xs, d_ym, d_y, d_yp, d_r, d_J, d_JtJ, d_JtR, d_delta, d_loss, d_lam, d_stat;
    // host scratch
    DBuf h_c, h_c_try, h_lam, h_loss, h_loss_try, h_loss_init;
    DBuf h_finished, h_iter, h_rejected, h_delta, h_solve_stat;
#ifndef USE_AD
    DBuf h_c_pert, h_yb, h_yp, h_J, h_heps;   // FD host-side Jacobian scratch
#endif
};

// grow-on-demand by bytes; returns 0 OK / CO_ERR_CUDA / CO_ERR_OOM.
static int dbuf_grow(DBuf &b, size_t bytes) {
    if (bytes <= b.cap) return CO_OK;
    if (b.ptr) cudaFree(b.ptr);
    b.ptr = nullptr; b.cap = 0;
    cudaError_t e = cudaMalloc(&b.ptr, bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "[co_lib] cudaMalloc %zu B failed: %s\n",
                bytes, cudaGetErrorString(e));
        return (e == cudaErrorMemoryAllocation) ? CO_ERR_OOM : CO_ERR_CUDA;
    }
    b.cap = bytes;
    return CO_OK;
}

static int hbuf_grow(DBuf &b, size_t bytes) {
    if (bytes <= b.cap) return CO_OK;
    free(b.ptr);
    b.ptr = malloc(bytes);
    if (!b.ptr) { b.cap = 0; return CO_ERR_OOM; }
    b.cap = bytes;
    return CO_OK;
}

#define DGROW(buf, bytes) do { int _r = dbuf_grow(buf, bytes); if (_r) return _r; } while (0)
#define HGROW(buf, bytes) do { int _r = hbuf_grow(buf, bytes); if (_r) return _r; } while (0)

// short device-pointer accessors
#define DI(b) ((int*)(b).ptr)
#define DF(b) ((float*)(b).ptr)
#define DM(b) ((TreeMeta*)(b).ptr)
#define HI(b) ((int*)(b).ptr)
#define HF(b) ((float*)(b).ptr)

// ===========================================================================
// co_init — select device, force the primary context once.
// ===========================================================================
extern "C" int co_init(int device_id, void **handle_out) {
    if (!handle_out) return CO_ERR_NULLARG;
    *handle_out = nullptr;
    cudaError_t e = cudaSetDevice(device_id);
    if (e != cudaSuccess) {
        fprintf(stderr, "[co_lib] cudaSetDevice(%d): %s\n", device_id, cudaGetErrorString(e));
        return CO_ERR_CUDA;
    }
    e = cudaFree(0);   // forces primary-context creation now (one-time)
    if (e != cudaSuccess) {
        fprintf(stderr, "[co_lib] cudaFree(0): %s\n", cudaGetErrorString(e));
        return CO_ERR_CUDA;
    }
    CoCtx *ctx = new (std::nothrow) CoCtx();
    if (!ctx) return CO_ERR_OOM;
    ctx->device_id = device_id;
    *handle_out = ctx;
    return CO_OK;
}

// ===========================================================================
// co_optimize — one LM run on one pop over the persistent context.
// ===========================================================================
extern "C" int co_optimize(void *handle, const PopHeader *h,
                            const int32_t *nt, const float *nv, const int32_t *ci,
                            const TreeMeta *metas,
                            const float *c_init, const float *xs, const float *ym,
                            int max_iter,
                            float *c_final_out, int32_t *status_out,
                            float *loss_init_out, float *loss_final_out) {
    if (!handle || !h || !nt || !nv || !ci || !metas || !c_init || !xs || !ym ||
        !c_final_out || !status_out)
        return CO_ERR_NULLARG;
    CoCtx *ctx = (CoCtx*)handle;
#ifdef PROFILE
    struct timespec t_entry, t_loop_begin, t_loop_end;
    clock_gettime(CLOCK_MONOTONIC, &t_entry);   // SETUP region start
#endif

    cudaError_t se = cudaSetDevice(ctx->device_id);
    if (se != cudaSuccess) { fprintf(stderr, "[co_lib] cudaSetDevice: %s\n",
                                     cudaGetErrorString(se)); return CO_ERR_CUDA; }

    int M_prob = h->M_prob, N = h->N, n_vars = h->n_vars, K_max = h->K_max;
    int total_nodes = h->total_nodes, total_c = h->total_c;

    if (K_max > MAX_K)        return CO_ERR_BOUNDS;
    if (h->max_stack > MAX_STACK) return CO_ERR_BOUNDS;

    // ---- MUST-FIX #6: sum-consistency checks BEFORE any H2D copy ----
    {
        long K_sum = 0, node_sum = 0;
        for (int m = 0; m < M_prob; m++) { K_sum += metas[m].K; node_sum += metas[m].n_nodes; }
        if (K_sum != (long)total_c)       return CO_ERR_SUM_C;
        if (node_sum != (long)total_nodes) return CO_ERR_SUM_NODES;
    }

    // ---- grow device buffers by BYTE capacity (products, MUST-FIX #6) ----
    DGROW(ctx->d_nt,   (size_t)total_nodes * sizeof(int));
    DGROW(ctx->d_nv,   (size_t)total_nodes * sizeof(float));
    DGROW(ctx->d_ci,   (size_t)total_nodes * sizeof(int));
    DGROW(ctx->d_call, (size_t)total_c * sizeof(float));
    DGROW(ctx->d_metas,(size_t)M_prob * sizeof(TreeMeta));
    DGROW(ctx->d_xs,   (size_t)N * n_vars * sizeof(float));
    DGROW(ctx->d_ym,   (size_t)M_prob * N * sizeof(float));
    DGROW(ctx->d_y,    (size_t)M_prob * N * sizeof(float));
    DGROW(ctx->d_yp,   (size_t)M_prob * N * sizeof(float));
    DGROW(ctx->d_r,    (size_t)M_prob * N * sizeof(float));
    DGROW(ctx->d_J,    (size_t)M_prob * K_max * N * sizeof(float));
    DGROW(ctx->d_JtJ,  (size_t)M_prob * K_max * K_max * sizeof(float));
    DGROW(ctx->d_JtR,  (size_t)M_prob * K_max * sizeof(float));
    DGROW(ctx->d_delta,(size_t)M_prob * K_max * sizeof(float));
    DGROW(ctx->d_loss, (size_t)M_prob * sizeof(float));
    DGROW(ctx->d_lam,  (size_t)M_prob * sizeof(float));
    DGROW(ctx->d_stat, (size_t)M_prob * sizeof(int));

    int *d_nt = DI(ctx->d_nt); float *d_nv = DF(ctx->d_nv); int *d_ci = DI(ctx->d_ci);
    float *d_call = DF(ctx->d_call); TreeMeta *d_metas = DM(ctx->d_metas);
    float *d_xs = DF(ctx->d_xs), *d_ym = DF(ctx->d_ym), *d_y = DF(ctx->d_y),
          *d_yp = DF(ctx->d_yp), *d_r = DF(ctx->d_r), *d_J = DF(ctx->d_J),
          *d_JtJ = DF(ctx->d_JtJ), *d_JtR = DF(ctx->d_JtR), *d_delta = DF(ctx->d_delta),
          *d_loss = DF(ctx->d_loss), *d_lam = DF(ctx->d_lam);
    int *d_stat = DI(ctx->d_stat);

    // ---- H2D inputs (current sizes only) ----
    CK(cudaMemcpy(d_nt, nt, (size_t)total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_nv, nv, (size_t)total_nodes*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci, ci, (size_t)total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_metas, metas, (size_t)M_prob*sizeof(TreeMeta), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xs, xs, (size_t)N*n_vars*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ym, ym, (size_t)M_prob*N*sizeof(float), cudaMemcpyHostToDevice));

    // ---- host LM state (grow, then RE-SEED every call: MUST-FIX #4) ----
    HGROW(ctx->h_c,          (size_t)total_c * sizeof(float));
    HGROW(ctx->h_c_try,      (size_t)total_c * sizeof(float));
    HGROW(ctx->h_lam,        (size_t)M_prob * sizeof(float));
    HGROW(ctx->h_loss,       (size_t)M_prob * sizeof(float));
    HGROW(ctx->h_loss_try,   (size_t)M_prob * sizeof(float));
    HGROW(ctx->h_loss_init,  (size_t)M_prob * sizeof(float));
    HGROW(ctx->h_finished,   (size_t)M_prob * sizeof(int));
    HGROW(ctx->h_iter,       (size_t)M_prob * sizeof(int));
    HGROW(ctx->h_rejected,   (size_t)M_prob * sizeof(int));
    HGROW(ctx->h_delta,      (size_t)M_prob * K_max * sizeof(float));
    HGROW(ctx->h_solve_stat, (size_t)M_prob * sizeof(int));
#ifndef USE_AD
    HGROW(ctx->h_c_pert, (size_t)total_c * sizeof(float));
    HGROW(ctx->h_yb,     (size_t)M_prob * N * sizeof(float));
    HGROW(ctx->h_yp,     (size_t)M_prob * N * sizeof(float));
    HGROW(ctx->h_J,      (size_t)M_prob * K_max * N * sizeof(float));
    HGROW(ctx->h_heps,   (size_t)M_prob * sizeof(float));
#endif

    float *h_c = HF(ctx->h_c), *h_c_try = HF(ctx->h_c_try),
          *h_lam = HF(ctx->h_lam), *h_loss = HF(ctx->h_loss),
          *h_loss_try = HF(ctx->h_loss_try), *h_loss_init = HF(ctx->h_loss_init),
          *h_delta = HF(ctx->h_delta);
    int *h_finished = HI(ctx->h_finished), *h_iter = HI(ctx->h_iter),
        *h_rejected = HI(ctx->h_rejected), *h_solve_stat = HI(ctx->h_solve_stat);

    // RE-SEED c from THIS pop's c_init (NOT a prior pop's optimized constants).
    memcpy(h_c, c_init, (size_t)total_c * sizeof(float));
    // RESET per-tree state (zero finished/iter/rejected, lambda baseline).
    memset(h_finished, 0, (size_t)M_prob * sizeof(int));
    memset(h_iter,     0, (size_t)M_prob * sizeof(int));
    memset(h_rejected, 0, (size_t)M_prob * sizeof(int));
    // Also clear the Cholesky solve status, device + host mirror, so the
    // header comment's "resets d_stat/h_solve_stat" is literally true. These
    // are write-before-read every iter (co_solve_kernel writes d_stat, then the
    // D2H copy at ~L388 fills h_solve_stat), so this is output-neutral TODAY;
    // it is defensive against a future solve-kernel edit that ever reads stale
    // status (e.g. a tree that skips the solve on some iter).
    CK(cudaMemset(d_stat, 0, (size_t)M_prob * sizeof(int)));
    memset(h_solve_stat, 0, (size_t)M_prob * sizeof(int));
    for (int m = 0; m < M_prob; m++) h_lam[m] = 1e-3f;   // relative scale baseline

    int block = 256, wpb = block/32;
    int grid = (M_prob + wpb - 1) / wpb;

    // K=0 pre-mark: nothing to optimize -> STATUS_K0_SKIP (internal 3).
    for (int m = 0; m < M_prob; m++)
        if (metas[m].K == 0) h_finished[m] = 3;

    const float xtol = 1e-6f;
#ifndef USE_AD
    const float eps_rel = 1e-3f;
#endif

    // ---- Initial eval + loss ----
    CK(cudaMemcpy(d_call, h_c, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CK(cudaGetLastError());
    residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
    CK(cudaGetLastError());
    loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
    CK(cudaGetLastError());
    CK(cudaMemcpy(h_loss, d_loss, (size_t)M_prob*sizeof(float), cudaMemcpyDeviceToHost));

    // snapshot loss_init (AD diagnostic) BEFORE the loop overwrites h_loss.
    memcpy(h_loss_init, h_loss, (size_t)M_prob * sizeof(float));

    // initial loss NaN/Inf -> unrecoverable, mark FAIL_NAN.
    for (int m = 0; m < M_prob; m++) {
        if (h_finished[m]) continue;
        if (!isfinite(h_loss[m])) h_finished[m] = 2;
    }

#ifndef USE_AD
    float *h_c_pert = HF(ctx->h_c_pert), *h_yb = HF(ctx->h_yb),
          *h_yp = HF(ctx->h_yp), *h_J = HF(ctx->h_J), *h_heps = HF(ctx->h_heps);
#endif

#ifdef PROFILE
    // setup done (grow + H2D + reseed + initial eval/loss); LM loop begins.
    CK(cudaDeviceSynchronize());   // fold the async initial eval/loss into setup
    clock_gettime(CLOCK_MONOTONIC, &t_loop_begin);
#endif

    // ---- LM loop (mirrors the matching standalone byte-for-byte) ----
    for (int it = 0; it < max_iter; it++) {
        int all_done = 1;
        for (int m = 0; m < M_prob; m++) if (!h_finished[m]) { all_done = 0; break; }
        if (all_done) break;

#ifdef USE_AD
        // A. forward-mode AD Jacobian (mirrors batch_lm_ad.cu phase A).
        //    d_call already holds h_c (init / phase G).
        ad_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
            d_xs, n_vars, N, d_call, K_max, d_J);
        CK(cudaGetLastError());
#else
        // A. previous d_y is baseline (init at start; G refreshes after iter).
        CK(cudaMemcpy(h_yb, d_y, (size_t)M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));

        // B. FD Jacobian (K_max launches) — mirrors batch_lm.cu phase B.
        for (int k = 0; k < K_max; k++) {
            memcpy(h_c_pert, h_c, (size_t)total_c*sizeof(float));
            for (int m = 0; m < M_prob; m++) {
                if (h_finished[m]) continue;
                if (k < metas[m].K) {
                    float cv = h_c[metas[m].c_offset+k];
                    float hh = eps_rel * fabsf(cv);
                    if (hh == 0.0f) hh = eps_rel;
                    float cp = cv + hh;
                    h_heps[m] = cp - cv;
                    h_c_pert[metas[m].c_offset+k] = cp;
                }
            }
            CK(cudaMemcpy(d_call, h_c_pert, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice));
            eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
            CK(cudaGetLastError());
            CK(cudaMemcpy(h_yp, d_yp, (size_t)M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
            for (int m = 0; m < M_prob; m++) {
                if (h_finished[m]) continue;
                if (k >= metas[m].K) continue;
                for (int i = 0; i < N; i++)
                    h_J[(size_t)m*K_max*N + k*N + i] = (h_yp[m*N+i] - h_yb[m*N+i]) / h_heps[m];
            }
        }
        CK(cudaMemcpy(d_J, h_J, (size_t)M_prob*K_max*N*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_call, h_c, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice));
#endif

        // C. Build JtJ + JtR
        build_jtj_jtr_kernel<<<grid, block>>>(d_J, d_r, d_metas, M_prob, N, K_max, d_JtJ, d_JtR);
        CK(cudaGetLastError());

        // D. Solve
        CK(cudaMemcpy(d_lam, h_lam, (size_t)M_prob*sizeof(float), cudaMemcpyHostToDevice));
        co_solve_kernel<<<(M_prob+255)/256, 256>>>(d_JtJ, d_JtR, d_metas, M_prob, K_max, d_lam, d_delta, d_stat);
        CK(cudaGetLastError());
        CK(cudaMemcpy(h_delta, d_delta, (size_t)M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_solve_stat, d_stat, (size_t)M_prob*sizeof(int), cudaMemcpyDeviceToHost));

        // E. Apply step, eval, loss_try
        memcpy(h_c_try, h_c, (size_t)total_c*sizeof(float));
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            for (int k = 0; k < metas[m].K; k++)
                h_c_try[metas[m].c_offset+k] += h_delta[m*K_max+k];
        }
        CK(cudaMemcpy(d_call, h_c_try, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
        CK(cudaGetLastError());
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_yp, d_ym, M_prob*N, d_r);
        CK(cudaGetLastError());
        loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
        CK(cudaGetLastError());
        CK(cudaMemcpy(h_loss_try, d_loss, (size_t)M_prob*sizeof(float), cudaMemcpyDeviceToHost));

        // F. Per-tree accept/reject + lambda + convergence.
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            h_iter[m]++;
            int K = metas[m].K;

            if (h_solve_stat[m] != 0) {            // Cholesky breakdown -> reject
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 4;   // FAIL_CHOLESKY
                continue;
            }
            if (!isfinite(h_loss_try[m])) {        // step pushed eval to NaN/Inf
                h_finished[m] = 2;                 // FAIL_NAN
                continue;
            }

            float d_norm_sq = 0, c_norm_sq = 0;
            for (int k = 0; k < K; k++) {
                d_norm_sq += h_delta[m*K_max+k] * h_delta[m*K_max+k];
                c_norm_sq += h_c_try[metas[m].c_offset+k] * h_c_try[metas[m].c_offset+k];
            }
            float d_norm = sqrtf(d_norm_sq), c_norm = sqrtf(c_norm_sq);

#ifdef USE_AD
            // AD ordering (batch_lm_ad.cu): improvement first, then convergence.
            int improved = isfinite(h_loss_try[m]) && (h_loss_try[m] <= h_loss[m]);
            if (improved) {
                for (int k = 0; k < K; k++)
                    h_c[metas[m].c_offset+k] = h_c_try[metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                if (d_norm < xtol * (c_norm + xtol)) {
                    h_finished[m] = 1;             // CONVERGED
                } else {
                    h_lam[m] *= 0.1f;
                    if (h_lam[m] < 1e-12f) h_lam[m] = 1e-12f;
                }
            } else {
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 2;   // FAIL_NAN bucket
            }
#else
            // FD ordering (batch_lm.cu): convergence-on-small-step first.
            if (d_norm < xtol * (c_norm + xtol)) {
                for (int k = 0; k < K; k++)
                    h_c[metas[m].c_offset+k] = h_c_try[metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                h_finished[m] = 1;                 // CONVERGED
            } else if (h_loss_try[m] < h_loss[m]) {
                for (int k = 0; k < K; k++)
                    h_c[metas[m].c_offset+k] = h_c_try[metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                h_lam[m] *= 0.1f;
                if (h_lam[m] < 1e-12f) h_lam[m] = 1e-12f;
            } else {
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 2;   // FAIL_NAN bucket
            }
#endif
        }

        // G. Refresh d_y at h_c (accepted state)
        CK(cudaMemcpy(d_call, h_c, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
        CK(cudaGetLastError());
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());
#ifdef PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_loop_end);
    g_last_setup_ms = SPLIT_MS(t_entry, t_loop_begin);
    g_last_loop_ms  = SPLIT_MS(t_loop_begin, t_loop_end);
#endif

#ifndef NO_FP64_GUARD
    // fp64-honesty boundary audit (see lm_core.cuh). Same device eval as the
    // standalones -> identical revert decisions -> c_final/loss_final stay
    // byte-identical to the matching standalone (the per-variant parity gate).
    // Silent + no ABI change; on CUDA error propagate as CO_ERR_CUDA.
    {
        int n_rev = fp64_boundary_audit(grid, block, d_nt, d_nv, d_ci, d_metas,
            M_prob, d_xs, n_vars, N, d_ym, total_c, metas, c_init,
            h_c, h_finished, h_loss);
        if (n_rev < 0) return CO_ERR_CUDA;
    }
#endif

    // ---- map internal h_finished {0,1,2,3,4} -> caller's status_out ----
    // Written directly to the caller-allocated [M_prob] buffer — sized to the
    // CURRENT pop's M, never capacity (MUST-FIX #4).
    for (int m = 0; m < M_prob; m++) {
        if      (h_finished[m] == 1) status_out[m] = STATUS_CONVERGED;
        else if (h_finished[m] == 2) status_out[m] = STATUS_FAIL_NAN;
        else if (h_finished[m] == 3) status_out[m] = STATUS_K0_SKIP;
        else if (h_finished[m] == 4) status_out[m] = STATUS_FAIL_CHOLESKY;
        else                         status_out[m] = STATUS_MAXITER;
    }

    // ---- copy remaining outputs to CURRENT-sized caller buffers (MUST-FIX #4) ----
    memcpy(c_final_out, h_c, (size_t)total_c * sizeof(float));
    if (loss_init_out)  memcpy(loss_init_out,  h_loss_init, (size_t)M_prob * sizeof(float));
    if (loss_final_out) memcpy(loss_final_out, h_loss,      (size_t)M_prob * sizeof(float));

    return CO_OK;
}

// ===========================================================================
// co_last_split — read the setup/loop wall split of the LAST co_optimize call.
// Returns 0 and writes setup_ms/loop_ms when built with -DPROFILE; returns 1 and
// writes -1.0 sentinels in the non-prof build (the symbol is ALWAYS present so the
// ctypes wrapper can bind it unconditionally, but only the *_prof.so has data).
// This is a NEW symbol — co_optimize's ABI is unchanged (step-3 parity tests link
// the existing signature).
// ===========================================================================
extern "C" int co_last_split(double *setup_ms_out, double *loop_ms_out) {
#ifdef PROFILE
    if (setup_ms_out) *setup_ms_out = g_last_setup_ms;
    if (loop_ms_out)  *loop_ms_out  = g_last_loop_ms;
    return 0;
#else
    if (setup_ms_out) *setup_ms_out = -1.0;
    if (loop_ms_out)  *loop_ms_out  = -1.0;
    return 1;
#endif
}

// ===========================================================================
// co_teardown — free everything; safe on NULL.
// ===========================================================================
static void dbuf_free(DBuf &b) { if (b.ptr) cudaFree(b.ptr); b.ptr = nullptr; b.cap = 0; }
static void hbuf_free(DBuf &b) { free(b.ptr); b.ptr = nullptr; b.cap = 0; }

extern "C" void co_teardown(void *handle) {
    if (!handle) return;
    CoCtx *ctx = (CoCtx*)handle;
    cudaSetDevice(ctx->device_id);
    dbuf_free(ctx->d_nt); dbuf_free(ctx->d_nv); dbuf_free(ctx->d_ci);
    dbuf_free(ctx->d_call); dbuf_free(ctx->d_metas);
    dbuf_free(ctx->d_xs); dbuf_free(ctx->d_ym); dbuf_free(ctx->d_y);
    dbuf_free(ctx->d_yp); dbuf_free(ctx->d_r); dbuf_free(ctx->d_J);
    dbuf_free(ctx->d_JtJ); dbuf_free(ctx->d_JtR); dbuf_free(ctx->d_delta);
    dbuf_free(ctx->d_loss); dbuf_free(ctx->d_lam); dbuf_free(ctx->d_stat);
    hbuf_free(ctx->h_c); hbuf_free(ctx->h_c_try); hbuf_free(ctx->h_lam);
    hbuf_free(ctx->h_loss); hbuf_free(ctx->h_loss_try); hbuf_free(ctx->h_loss_init);
    hbuf_free(ctx->h_finished); hbuf_free(ctx->h_iter); hbuf_free(ctx->h_rejected);
    hbuf_free(ctx->h_delta); hbuf_free(ctx->h_solve_stat);
#ifndef USE_AD
    hbuf_free(ctx->h_c_pert); hbuf_free(ctx->h_yb); hbuf_free(ctx->h_yp);
    hbuf_free(ctx->h_J); hbuf_free(ctx->h_heps);
#endif
    delete ctx;
}
