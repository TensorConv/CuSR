// lm_core.cuh — shared LM core for the batched-LM SR standalones (and, later,
// the in-process co_lib.{cu,h}).
//
// SINGLE SOURCE OF TRUTH for the device kernels that are IDENTICAL across the
// FD (batch_lm.cu) and AD (batch_lm_ad.cu) variants, so device codegen cannot
// drift between them — byte-parity is FMA-contraction sensitive (see
// docs/kernel/INPROCESS_CO_PLAN.md step 1 / MUST-FIX #1).
//
// What lives here (textually identical in both standalones before this refactor):
//   - CUDA_CHECK            host error macro (exit-on-error; the .so step swaps it)
//   - STATUS_* codes        external status enum written to status.bin
//   - eval_tree_d           scalar reverse-order stack interpreter (device)
//   - eval_kernel_batched   per-tree batched forward eval (warp-per-tree)
//   - residual_kernel       r = yp - ym
//   - loss_kernel           0.5 * sum(r^2) per tree (warp reduce)
//   - build_jtj_jtr_kernel  JtJ + JtR (warp-per-tree)
//   - write_blob            host helper to dump a buffer to disk
//
// What does NOT live here (genuinely DIFFERENT between the variants — keeping
// them per-.cu is what preserves each variant's own golden):
//   - solve_kernel          AD adds solve_t / SOLVE_FP64 / PIVOT_FLOOR /
//                           TRUST_REGION machinery the FD path doesn't have
//   - the Jacobian          FD = host-side FD loop; AD = ad_jacobian_kernel
//   - the LM-loop body       different convergence ordering + loss_init + PROFILE
//
// Enums (NTypeE / FuncE), compile-time bounds (MAX_STACK / MAX_K) and TreeMeta
// come from ad_interp.cuh (the v4 single source of truth), so the FD standalone
// no longer keeps its own inline copies. ad_interp.cuh also defines the JVP
// interpreter + ad_jacobian_kernel; those are unused by FD and harmless to
// include (dead-stripped; they do not change the codegen of the kernels below).

#ifndef LM_CORE_CUH
#define LM_CORE_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "pop_format.h"     // PopHeader / TreeMeta
#include "ad_interp.cuh"    // enum NTypeE / F_* / MAX_STACK / MAX_K (+ JVP/AD kernel)

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

// status code:
//   0 = CONVERGED         d_norm < xtol·(c_norm + xtol)
//   1 = MAXITER           没收敛也没 fail, max_iter 用完
//   2 = FAIL_NAN          loss/loss_try 出 NaN/Inf (eval 把树推到奇点)
//   3 = K0_SKIP           K=0, 没常数可优化
//   4 = FAIL_CHOLESKY     Cholesky breakdown 反复, λ 爆 > 1e12 (solve_kernel:s<=0)
#define STATUS_CONVERGED      0
#define STATUS_MAXITER        1
#define STATUS_FAIL_NAN       2
#define STATUS_K0_SKIP        3
#define STATUS_FAIL_CHOLESKY  4

__device__ float eval_tree_d(
    const int *nt, const float *nv, int n_nodes, const int *const_idx,
    const float *x, const float *c)
{
    float stack[MAX_STACK]; int sp = 0;
    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i]; float v = nv[i];
        if (t == N_VAR) stack[sp++] = x[(int)v];
        else if (t == N_CONST) stack[sp++] = c[const_idx[i]];
        else if (t == N_UFUNC) {
            float a = stack[--sp]; int fid = (int)v; float r;
            switch (fid) {
                case F_SIN:  r=sinf(a);  break;
                case F_COS:  r=cosf(a);  break;
                case F_TAN:  r=tanf(a);  break;
                case F_SINH: r=sinhf(a); break;
                case F_COSH: r=coshf(a); break;
                case F_TANH: r=tanhf(a); break;
                case F_LOG:  r=logf(a);  break;
                case F_EXP:  r=expf(a);  break;
                case F_INV:  r=1.0f/a;   break;
                case F_NEG:  r=-a;       break;
                case F_ABS:  r=fabsf(a); break;
                case F_SQRT: r=sqrtf(a); break;
                default:     r=0.0f;     // unknown op -- silent failure 入口
            }
            stack[sp++] = r;
        } else if (t == N_BFUNC) {
            float l = stack[--sp]; float rv = stack[--sp];
            int fid = (int)v; float o;
            switch (fid) {
                case F_ADD: o=l+rv;       break;
                case F_SUB: o=l-rv;       break;
                case F_MUL: o=l*rv;       break;
                case F_DIV: o=l/rv;       break;
                case F_POW: o=powf(l,rv); break;
                case F_MAX: o=fmaxf(l,rv);break;
                case F_MIN: o=fminf(l,rv);break;
                case F_LT:  o=(l < rv)  ? 1.0f : 0.0f; break;
                case F_GT:  o=(l > rv)  ? 1.0f : 0.0f; break;
                case F_LE:  o=(l <= rv) ? 1.0f : 0.0f; break;
                case F_GE:  o=(l >= rv) ? 1.0f : 0.0f; break;
                default:    o=0.0f;       // unknown op -- silent failure 入口
            }
            stack[sp++] = o;
        }
    }
    return stack[0];
}

__global__ void eval_kernel_batched(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N, const float *c_all, float *y_out)
{
    int wpb = blockDim.x / 32;
    int wb = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m = blockIdx.x * wpb + wb;
    if (m >= M_prob) return;
    TreeMeta meta = metas[m];
    const int *nt = nt_all + meta.node_offset;
    const float *nv = nv_all + meta.node_offset;
    const int *ci = ci_all + meta.node_offset;
    const float *c = c_all + meta.c_offset;
    for (int i = lane; i < N; i += 32)
        y_out[m * N + i] = eval_tree_d(nt, nv, meta.n_nodes, ci, xs + i * n_vars, c);
}

__global__ void residual_kernel(const float *yp, const float *ym, int total, float *r) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    r[tid] = yp[tid] - ym[tid];  // ym 是 [M_prob*N] per-tree (real EvoGP dump 复制了)
}

__global__ void loss_kernel(const float *r, int M, int N, float *loss) {
    int wpb = blockDim.x / 32;
    int wb = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m = blockIdx.x * wpb + wb;
    if (m >= M) return;
    float s = 0;
    for (int i = lane; i < N; i += 32) {
        float r_ = r[m*N + i];
        s += r_ * r_;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if (lane == 0) loss[m] = 0.5f * s;
}

__global__ void build_jtj_jtr_kernel(
    const float *J, const float *r, const TreeMeta *metas,
    int M_prob, int N, int K_max, float *JtJ, float *JtR)
{
    int wpb = blockDim.x / 32;
    int wb = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m = blockIdx.x * wpb + wb;
    if (m >= M_prob) return;
    int K_m = metas[m].K;
    const float *Jm = J + (size_t)m * K_max * N;
    const float *rm = r + (size_t)m * N;
    float *JJm = JtJ + (size_t)m * K_max * K_max;
    float *JRm = JtR + (size_t)m * K_max;

    for (int k = 0; k < K_m; k++) {
        float s = 0;
        for (int i = lane; i < N; i += 32) s += Jm[k*N+i] * rm[i];
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
        if (lane == 0) JRm[k] = s;
    }
    for (int j1 = 0; j1 < K_m; j1++) {
        for (int j2 = j1; j2 < K_m; j2++) {
            float s = 0;
            for (int i = lane; i < N; i += 32) s += Jm[j1*N+i] * Jm[j2*N+i];
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            if (lane == 0) { JJm[j1*K_max+j2] = s; JJm[j2*K_max+j1] = s; }
        }
    }
}

// ----------------------------------------------------------------------
// host helper
// ----------------------------------------------------------------------
static void write_blob(const char *path, const void *buf, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "fopen %s failed\n", path); exit(1); }
    if (fwrite(buf, 1, bytes, f) != bytes) {
        fprintf(stderr, "fwrite %s short\n", path); exit(1);
    }
    fclose(f);
}

// ======================================================================
// fp64-honesty boundary audit (default ON; -DNO_FP64_GUARD makes the build
// byte-identical to the un-guarded kernel).
//
// WHY: with --use_fast_math, the fp32 LM accept/reject is deceived near
// singularities — a fast-reciprocal-approx returns a finite-but-wrong small
// residual, so the loop "improves" the loss and COMMITS a step that is
// catastrophically worse in honest arithmetic. The damage is mechanism-
// independent (relative or absolute damping, rescue or not — see the rejected
// trust-region ablation), because the *acceptance test itself* is the liar.
// The cure is to open one honest eye at the boundary: recompute the loss in
// real fp64 (IEEE; --use_fast_math has NO effect on fp64 div/sqrt) at the
// delivered c_final and at c_init, and for any tree that is non-finite or
// strictly worse than its start, DELIVER c_init instead.
//
// This guarantees the monotone invariant "delivered <= init in honest fp64"
// for every tree, which is exactly what an optimizer must never violate.
//
// The DEVICE eval lives here (single source of truth) so the subprocess
// standalones and the in-process .so produce bit-identical fp64 values ->
// identical revert decisions -> byte-identical c_final (the parity the refactor
// protects). The host glue differs per consumer only in error policy
// (standalone exit() vs .so return-code), so fp64_boundary_audit() returns -1
// on a CUDA error and lets the caller decide.
// ======================================================================
#ifndef NO_FP64_GUARD
#include <cmath>      // isfinite (host)
#include <cstring>    // memcpy  (host)

// fp64-honest value interpreter — a double mirror of eval_tree_d. IEEE math
// only (sin/1.0/.../sqrt in double), so --use_fast_math cannot deceive it.
__device__ double eval_tree_d_fp64(
    const int *nt, const float *nv, int n_nodes, const int *const_idx,
    const float *x, const float *c)
{
    double stack[MAX_STACK]; int sp = 0;
    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i]; float v = nv[i];
        if (t == N_VAR) stack[sp++] = (double)x[(int)v];
        else if (t == N_CONST) stack[sp++] = (double)c[const_idx[i]];
        else if (t == N_UFUNC) {
            double a = stack[--sp]; int fid = (int)v; double r;
            switch (fid) {
                case F_SIN:  r=sin(a);  break;
                case F_COS:  r=cos(a);  break;
                case F_TAN:  r=tan(a);  break;
                case F_SINH: r=sinh(a); break;
                case F_COSH: r=cosh(a); break;
                case F_TANH: r=tanh(a); break;
                case F_LOG:  r=log(a);  break;
                case F_EXP:  r=exp(a);  break;
                case F_INV:  r=1.0/a;   break;
                case F_NEG:  r=-a;      break;
                case F_ABS:  r=fabs(a); break;
                case F_SQRT: r=sqrt(a); break;
                default:     r=0.0;
            }
            stack[sp++] = r;
        } else if (t == N_BFUNC) {
            double l = stack[--sp]; double rv = stack[--sp];
            int fid = (int)v; double o;
            switch (fid) {
                case F_ADD: o=l+rv;       break;
                case F_SUB: o=l-rv;       break;
                case F_MUL: o=l*rv;       break;
                case F_DIV: o=l/rv;       break;
                case F_POW: o=pow(l,rv);  break;
                case F_MAX: o=fmax(l,rv); break;
                case F_MIN: o=fmin(l,rv); break;
                case F_LT:  o=(l < rv)  ? 1.0 : 0.0; break;
                case F_GT:  o=(l > rv)  ? 1.0 : 0.0; break;
                case F_LE:  o=(l <= rv) ? 1.0 : 0.0; break;
                case F_GE:  o=(l >= rv) ? 1.0 : 0.0; break;
                default:    o=0.0;
            }
            stack[sp++] = o;
        }
    }
    return stack[0];
}

// per-tree fp64 SSE loss (warp-per-tree; mirror of eval_kernel_batched + loss_kernel)
__global__ void eval_loss_fp64_kernel(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N, const float *c_all, const float *ym_all,
    double *loss_out)
{
    int wpb = blockDim.x / 32;
    int wb = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m = blockIdx.x * wpb + wb;
    if (m >= M_prob) return;
    TreeMeta meta = metas[m];
    const int *nt = nt_all + meta.node_offset;
    const float *nv = nv_all + meta.node_offset;
    const int *ci = ci_all + meta.node_offset;
    const float *c = c_all + meta.c_offset;
    double s = 0.0;
    for (int i = lane; i < N; i += 32) {
        double yp = eval_tree_d_fp64(nt, nv, meta.n_nodes, ci, xs + i * n_vars, c);
        double rr = yp - (double)ym_all[m * N + i];
        s += rr * rr;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if (lane == 0) loss_out[m] = 0.5 * s;
}

// Host boundary audit. Recompute honest fp64 0.5*SSE at c_final_host and
// c_init_host; for every K>0 tree whose fp64(c_final) is non-finite or strictly
// worse than fp64(c_init): revert that tree's constants in c_final_host to
// c_init, demote a fake CONVERGED (internal h_finished==1) to MAXITER (==0), and
// set its reported loss (h_loss, if provided) to the honest fp64 c_init loss so
// the diagnostic loss_final.bin / loss_final_out cannot lie either. Self-
// contained (allocs its own scratch; reads nothing from the engine's live device
// state). Returns #reverted, or -1 on a CUDA/alloc error (caller's error policy).
static int fp64_boundary_audit(
    int grid, int block,
    const int *d_nt, const float *d_nv, const int *d_ci, const TreeMeta *d_metas,
    int M_prob, const float *d_xs, int n_vars, int N, const float *d_ym,
    int total_c, const TreeMeta *metas_host,
    const float *c_init_host, float *c_final_host,
    int *h_finished, float *h_loss)
{
    float *d_c = nullptr; double *d_loss64 = nullptr;
    double *l_init  = (double*)malloc((size_t)M_prob * sizeof(double));
    double *l_final = (double*)malloc((size_t)M_prob * sizeof(double));
    int rc = -1;
    bool ok = (l_init && l_final)
        && cudaMalloc(&d_c,      (size_t)total_c * sizeof(float))  == cudaSuccess
        && cudaMalloc(&d_loss64, (size_t)M_prob  * sizeof(double)) == cudaSuccess;
    // pass 1: c_final
    if (ok) ok = cudaMemcpy(d_c, c_final_host, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) { eval_loss_fp64_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_c, d_ym, d_loss64);
              ok = cudaGetLastError() == cudaSuccess; }
    if (ok) ok = cudaMemcpy(l_final, d_loss64, (size_t)M_prob*sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
    // pass 2: c_init
    if (ok) ok = cudaMemcpy(d_c, c_init_host, (size_t)total_c*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) { eval_loss_fp64_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_c, d_ym, d_loss64);
              ok = cudaGetLastError() == cudaSuccess; }
    if (ok) ok = cudaMemcpy(l_init, d_loss64, (size_t)M_prob*sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok) {
        rc = 0;
        for (int m = 0; m < M_prob; m++) {
            int K = metas_host[m].K;
            if (K == 0) continue;
            double lf = l_final[m], li = l_init[m];
            if (!isfinite(lf) || (isfinite(li) && lf > li)) {
                int co = metas_host[m].c_offset;
                memcpy(c_final_host + co, c_init_host + co, (size_t)K * sizeof(float));
                if (h_finished && h_finished[m] == 1) h_finished[m] = 0;  // CONVERGED -> MAXITER
                if (h_loss) h_loss[m] = (float)li;                        // honest reported loss
                rc++;
            }
        }
    }
    if (d_c) cudaFree(d_c);
    if (d_loss64) cudaFree(d_loss64);
    free(l_init); free(l_final);
    return rc;
}
#endif // NO_FP64_GUARD

#endif // LM_CORE_CUH
