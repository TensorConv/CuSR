// batch_lm.cu — GPU batched Levenberg-Marquardt for SR populations.
//
// 读 pop.bin (一批 GP 候选表达式树 + 各自 c_init + 共享 (xs, ym)),
// 对每棵树并行跑 LM 拟合常数 c, 输出:
//   - status.bin (int32, M_prob 个)
//   - c_final.bin (float32, total_c 个, jagged by per-tree K)
//
// 数值算法:
//   - Gauss-Newton + Marquardt 阻尼 (λ adaptive)
//   - finite-difference Jacobian, ε = 1e-3
//   - K×K Cholesky 解 (warp-per-problem layout, 1 thread/problem 解 K 维方程)
//   - host 端做 accept/reject + λ 调度, kernel 跑数值热路径
//
// Compile-time bounds:
//   - MAX_K=32 (K_max per tree), MAX_STACK=64 (interpreter stack)
//
// usage: ./batch_lm <pop.bin> [<out_dir>] [--max-iter N] [--quiet]
//   out_dir 不传时默认 = dirname(pop_path).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#include "pop_format.h"
#include "ad_interp.cuh"   // v4: enum NTypeE / F_* / MAX_STACK / MAX_K + eval_tree_jvp_d + ad_jacobian_kernel
#include "loader.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

// ---- per-kernel profiling (-DPROFILE only) -----------------------------------
// CUDA-event timing of every launch + H<->D copy in the LM loop, bucketed by
// category, so we can split GPU compute time from host/launch/transfer overhead
// (the analytical roofline conflates them in end-to-end wall). The default build
// (no -DPROFILE) expands TBEG/TEND to nothing => byte-for-byte the frozen,
// parity-passing kernel; only the separate batch_lm_fusedfd_prof binary records.
#ifdef PROFILE
#include <vector>
enum ProfCat { C_FDJAC, C_BUILDJTJ, C_SOLVE, C_EVAL, C_RESID, C_LOSS, C_H2D, C_D2H, NCAT };
static const char *CAT_NAMES[NCAT] =
    {"fd_jacobian","build_jtj","solve","eval","residual","loss","memcpy_H2D","memcpy_D2H"};
struct ProfPair { int cat; cudaEvent_t s, e; };
static std::vector<ProfPair> g_prof;
static inline cudaEvent_t prof_begin() {
    cudaEvent_t s; cudaEventCreate(&s); cudaEventRecord(s, 0); return s;
}
static inline void prof_end(int cat, cudaEvent_t s) {
    cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, 0);
    g_prof.push_back({cat, s, e});
}
#define TBEG(v)      cudaEvent_t v = prof_begin()
#define TEND(cat, v) prof_end(cat, v)
#else
#define TBEG(v)
#define TEND(cat, v)
#endif

// enum NTypeE / FuncE (F_*) 与 MAX_STACK / MAX_K 现由 ad_interp.cuh 提供 (single
// source of truth, 见 docs/kernel/AD_JACOBIAN_V4.md §5.1) — 删此处副本避免重定义。

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

// solve_kernel 数值旋钮 (exp012, 见 012/MIXED_PRECISION_PROBE.md). 起因: inner-const-heavy
// tier-B 64.8% 软肋, 520 个 miss = 高 K 树 fp32 Cholesky s≤0. 根因 = 死/冗余常数(结构秩亏)
// 使求解器一遇退化方向就整步放弃, 不是精度问题. 默认 = 安全基线 (整步放弃, 过 W0 parity gate).
//   (1) -DPIVOT_FLOOR —— 实验: 退化主元抬到正地板 ε (=PIVOT_FLOOR_EPS) 放行. 在调过的 preset 上
//       救回 tier-B/选择保真度, 但 **W0 parity gate 不过**: 固定小 ε → 退化方向 δ~b/√ε 巨步,
//       把 MAXITER/marginal 树推飞 (data/pop.bin 123 棵中位 59× 变差). 固定地板是错招, 正解是
//       绝对 Levenberg 阻尼 (λ trust-region 约束步长). 仅留作 ablation. -DPIVOT_FLOOR_EPS 调 ε.
//   (2) -DSOLVE_FP64 —— K×K 解抬 fp64 (ablation, **已证伪**: 失败是结构秩亏非精度, +0 tier-B).
//       per-point eval/Jacobian 仍 fp32.
#ifdef SOLVE_FP64
typedef double solve_t;
#define SOLVE_SQRT(x) sqrt(x)
#else
typedef float solve_t;
#define SOLVE_SQRT(x) sqrtf(x)
#endif
#ifndef PIVOT_FLOOR_EPS
#define PIVOT_FLOOR_EPS 1e-9f
#endif

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

// Fused FD Jacobian: one warp owns one tree and does ALL of that tree's K[m]
// perturbed evaluations + the differencing, straight into d_J. Replaces the
// per-column eval launches, the device differencing kernel, AND the per-column
// host c-perturbation copies (the kernel bumps the constant locally).
//   J[m,j,i] = (eval(c with c[j]+eps, x_i) - y_base[m,i]) / eps,  for j < K[m]
// __fdiv_rn forces IEEE division (bit-identical to host). Columns j >= K[m] are
// never written here and never read by build_jtj, so the result matches exactly.
// c_all is the current (accepted) constants — d_call already holds them here.
__global__ void fd_jacobian_fused_kernel(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N,
    const float *c_all, const float *y_base, float eps, int K_max, float *J)
{
    int wpb  = blockDim.x / 32;
    int lane = threadIdx.x & 31;
    int m    = blockIdx.x * wpb + (threadIdx.x / 32);
    if (m >= M_prob) return;
    TreeMeta meta = metas[m];
    int K = meta.K;
    if (K == 0) return;                               // nothing to differentiate
    const int   *nt = nt_all + meta.node_offset;
    const float *nv = nv_all + meta.node_offset;
    const int   *ci = ci_all + meta.node_offset;
    const float *c  = c_all  + meta.c_offset;
    const float *yb = y_base + (size_t)m * N;
    float       *Jm = J      + (size_t)m * K_max * N;

    float cloc[MAX_K];                                // this tree's constants (local copy)
    for (int t = 0; t < K; t++) cloc[t] = c[t];

    for (int j = 0; j < K; j++) {
        float saved = cloc[j];
        float h = eps * fabsf(saved);                 // 相对步长 (eps 即 eps_rel), 见 batch_lm.cu 注释
        if (h == 0.0f) h = eps;                       // c==0 fallback
        float cp = saved + h;
        h = cp - saved;                               // fp32 实际步长 (差分除数)
        if (h == 0.0f) h = eps;                       // --use_fast_math 的 FTZ 可把上行 flush 成 0 (|c|≈1e-35 窗口), 防 ÷0
        cloc[j] = cp;                                 // bump column j (所有 lane 同值, 无分歧)
        for (int i = lane; i < N; i += 32) {
            float yj = eval_tree_d(nt, nv, meta.n_nodes, ci, xs + i * n_vars, cloc);
            Jm[(size_t)j * N + i] = __fdiv_rn(yj - yb[i], h);
        }
        cloc[j] = saved;                             // restore for next column
    }
}

__global__ void solve_kernel(
    const float *JtJ, const float *JtR, const TreeMeta *metas,
    int M_prob, int K_max, const float *lambda_arr,
    float *delta, int *status)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_prob) return;
    int K = metas[m].K;
    float lam = lambda_arr[m];

    // solve_t = float (baseline) 或 double (-DSOLVE_FP64). fp32 路径下所有 cast
    // 都是恒等, codegen 与冻结基线一致.
    solve_t A[MAX_K * MAX_K];
    solve_t b[MAX_K];

    for (int j1 = 0; j1 < K; j1++) {
        b[j1] = -(solve_t)JtR[m*K_max + j1];
        for (int j2 = 0; j2 < K; j2++)
            A[j1*K + j2] = (solve_t)JtJ[m*K_max*K_max + j1*K_max + j2];
        A[j1*K + j1] *= (solve_t)(1.0f + lam);   // damping 形式不动 (relative), 隔离纯精度效应
    }
    for (int j = 0; j < K; j++) {
        solve_t s = A[j*K + j];
        for (int k = 0; k < j; k++) s -= A[j*K+k] * A[j*K+k];
#ifdef PIVOT_FLOOR
        // ablation (W0 gate 不过, 见顶部注释): 退化主元抬到正地板 ε → δ~b/√ε 不受控巨步.
        if (s < (solve_t)PIVOT_FLOOR_EPS) s = (solve_t)PIVOT_FLOOR_EPS;
#else
        // 默认 (安全, 过 W0 parity gate): 退化主元 s≤0 整步放弃, best-finite 兜底.
        if (s <= (solve_t)0) { status[m] = -1; for (int i=0;i<K;i++) delta[m*K_max+i]=0.0f; return; }
#endif
        A[j*K + j] = SOLVE_SQRT(s);
        for (int i = j+1; i < K; i++) {
            solve_t t = A[i*K + j];
            for (int k = 0; k < j; k++) t -= A[i*K+k] * A[j*K+k];
            A[i*K + j] = t / A[j*K + j];
        }
    }
    for (int i = 0; i < K; i++) {
        solve_t s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*K+k] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = K-1; i >= 0; i--) {
        solve_t s = b[i];
        for (int k = i+1; k < K; k++) s -= A[k*K+i] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = 0; i < K; i++) delta[m*K_max + i] = (float)b[i];
    status[m] = 0;
}

// ----------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------
static void write_blob(const char *path, const void *buf, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "fopen %s failed\n", path); exit(1); }
    if (fwrite(buf, 1, bytes, f) != bytes) {
        fprintf(stderr, "fwrite %s short\n", path); exit(1);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pop.bin> [out_dir] [--max-iter N] [--quiet]\n", argv[0]);
        return 1;
    }
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    const char *pop_path = argv[1];
    const char *out_dir = NULL;
    int max_iter = 50;
    int quiet = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--max-iter") == 0 && i+1 < argc) { max_iter = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--quiet") == 0) { quiet = 1; }
        else if (argv[i][0] != '-') { out_dir = argv[i]; }
    }
    // 没显式传 out_dir → 从 pop_path 提 dirname. ./batch_lm data/pop.bin → data/, ./batch_lm pop.bin → ./
    static char dir_buf[1024];
    if (out_dir == NULL) {
        const char *slash = strrchr(pop_path, '/');
        if (slash) {
            size_t len = (size_t)(slash - pop_path);
            if (len >= sizeof(dir_buf)) len = sizeof(dir_buf) - 1;
            memcpy(dir_buf, pop_path, len);
            dir_buf[len] = '\0';
            out_dir = dir_buf;
        } else {
            out_dir = ".";
        }
    }

    PopData pop;
    if (load_pop_bin(pop_path, &pop)) {
        fprintf(stderr, "load failed.\n"); return 1;
    }
#ifdef PROFILE
    struct timespec t_load_done, t_loop_begin, t_loop_end;
    clock_gettime(CLOCK_MONOTONIC, &t_load_done);
    int n_loop_iters = 0;
#endif
    const PopHeader *h = &pop.header;
    if (!quiet) {
        printf("[batch-lm] loaded %s: M_prob=%d total_nodes=%d total_c=%d N=%d n_vars=%d K_max=%d max_stack=%d\n",
               pop_path, h->M_prob, h->total_nodes, h->total_c, h->N, h->n_vars, h->K_max, h->max_stack);
    }

    if (h->K_max > MAX_K) {
        fprintf(stderr, "[batch-lm] K_max=%d > MAX_K=%d (compile-time). 增 MAX_K 重编.\n",
                h->K_max, MAX_K);
        free_pop_data(&pop); return 2;
    }
    if (h->max_stack > MAX_STACK) {
        fprintf(stderr, "[batch-lm] max_stack=%d > MAX_STACK=%d (compile-time). 增 MAX_STACK 重编.\n",
                h->max_stack, MAX_STACK);
        free_pop_data(&pop); return 2;
    }

    int M_prob = h->M_prob, N = h->N, n_vars = h->n_vars, K_max = h->K_max;
    int total_nodes = h->total_nodes, total_c = h->total_c;

    // ---- host bufs ----
    float *h_c = (float*)malloc(total_c * sizeof(float));        // c_curr
    float *h_c_try = (float*)malloc(total_c * sizeof(float));
    float *h_c_pert = (float*)malloc(total_c * sizeof(float));
    memcpy(h_c, pop.c_init, total_c * sizeof(float));            // 从 pop.bin 来

    // ---- D allocs ----
    int *d_nt; float *d_nv; int *d_ci; float *d_call; TreeMeta *d_metas;
    float *d_xs, *d_ym, *d_y, *d_yp, *d_r, *d_J, *d_JtJ, *d_JtR, *d_delta, *d_loss, *d_lam;
    int *d_stat;
    CUDA_CHECK(cudaMalloc(&d_nt, total_nodes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nv, total_nodes*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ci, total_nodes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_call, total_c*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_metas, M_prob*sizeof(TreeMeta)));
    CUDA_CHECK(cudaMalloc(&d_xs, (size_t)N*n_vars*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ym, (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yp, (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r,  (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_J,  (size_t)M_prob*K_max*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_JtJ,(size_t)M_prob*K_max*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_JtR,(size_t)M_prob*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta,(size_t)M_prob*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, M_prob*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lam, M_prob*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stat, M_prob*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nt, pop.nt, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv, pop.nv, total_nodes*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, pop.ci, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, pop.metas, M_prob*sizeof(TreeMeta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, pop.xs, (size_t)N*n_vars*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ym, pop.ym, (size_t)M_prob*N*sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, wpb = block/32;
    int grid = (M_prob + wpb - 1) / wpb;

    // ---- Host LM state ----
    float *h_lam = (float*)malloc(M_prob * sizeof(float));
    float *h_loss = (float*)malloc(M_prob * sizeof(float));
    float *h_loss_try = (float*)malloc(M_prob * sizeof(float));
    int   *h_finished = (int*)calloc(M_prob, sizeof(int));
    int   *h_iter = (int*)calloc(M_prob, sizeof(int));
    int   *h_rejected = (int*)calloc(M_prob, sizeof(int));
    for (int m = 0; m < M_prob; m++) h_lam[m] = 1e-3f;

    // K=0 跳过: 没常数可优化, 立即标 finished + status=3
    int n_k0_pre = 0;
    for (int m = 0; m < M_prob; m++) {
        if (pop.metas[m].K == 0) { h_finished[m] = 3; n_k0_pre++; }
    }

    // v4: 前向 AD 精确算 Jacobian, 不再需要 FD 相对步长 eps (随 fd_jacobian_fused_kernel
    // 一起退役; 该 kernel 仍在文件中定义但不再被调用, 见 §2 drop-in 范围)。
    const float xtol = 1e-6f;   // 选值理由见 batch_lm.cu 同位置注释

    // ---- Initial eval + loss ----
    CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());
    residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
    CUDA_CHECK(cudaGetLastError());
    loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, M_prob*sizeof(float), cudaMemcpyDeviceToHost));

    // 收敛诚实性诊断 (additive, 不改任何 numerics / accept-reject): 抓初始 per-tree loss
    // 副本, 在 LM loop 覆写 h_loss 之前. 写到 loss_init.bin, 对比 loss_final.bin 检验
    // "假收敛" — status==CONVERGED 的树 loss_final 不该 > loss_init.
    float *h_loss_init = (float*)malloc(M_prob * sizeof(float));
    memcpy(h_loss_init, h_loss, M_prob * sizeof(float));

    // 初始 loss NaN/Inf → 整棵树没救, 直接标 fail. c_init 都 produce NaN 说明这棵树
    // 没法被 LM 拯救.
    for (int m = 0; m < M_prob; m++) {
        if (h_finished[m]) continue;
        if (!isfinite(h_loss[m])) h_finished[m] = 2;
    }

    // (option A: no h_yb / h_yp / h_J — the Jacobian is built on the device)
    float *h_delta = (float*)malloc(M_prob*K_max*sizeof(float));
    int   *h_solve_stat = (int*)malloc(M_prob * sizeof(int));  // 必须读 solve_kernel 的 status (避免假阳性 converge, 见 F 分支 1)

    // ---- LM loop ----
    int last_print = -10;
#ifdef PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_loop_begin);
#endif
    for (int it = 0; it < max_iter; it++) {
        int all_done = 1;
        for (int m = 0; m < M_prob; m++) if (!h_finished[m]) { all_done = 0; break; }
        if (all_done) break;
#ifdef PROFILE
        n_loop_iters++;
#endif

        // A. Forward-mode AD Jacobian (v4 drop-in, 替 fd_jacobian_fused_kernel)。one warp
        // owns a tree, 32 lane 跨 N; 外层按 AD_W 分桶遍历 K 列, 一趟前向同时算值+切向量,
        // 写精确 ∂y_i/∂c_j 直入 d_J (同一 layout)。无 eps, 无 y_base (AD 精确)。
        // d_call already holds h_c (init / phase G), read directly.
        TBEG(e_fd);
        ad_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
            d_xs, n_vars, N, d_call, K_max, d_J);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_FDJAC, e_fd);

        // C. Build JtJ + JtR
        TBEG(e_bj);
        build_jtj_jtr_kernel<<<grid, block>>>(d_J, d_r, d_metas, M_prob, N, K_max, d_JtJ, d_JtR);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_BUILDJTJ, e_bj);

        // D. Solve
        TBEG(e_lam);
        CUDA_CHECK(cudaMemcpy(d_lam, h_lam, M_prob*sizeof(float), cudaMemcpyHostToDevice));
        TEND(C_H2D, e_lam);
        TBEG(e_sol);
        solve_kernel<<<(M_prob+255)/256, 256>>>(d_JtJ, d_JtR, d_metas, M_prob, K_max, d_lam, d_delta, d_stat);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_SOLVE, e_sol);
        TBEG(e_del);
        CUDA_CHECK(cudaMemcpy(h_delta, d_delta, M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));
        TEND(C_D2H, e_del);
        TBEG(e_sst);
        CUDA_CHECK(cudaMemcpy(h_solve_stat, d_stat, M_prob*sizeof(int), cudaMemcpyDeviceToHost));
        TEND(C_D2H, e_sst);

        // E. Apply step, eval, loss_try
        memcpy(h_c_try, h_c, total_c*sizeof(float));
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            for (int k = 0; k < pop.metas[m].K; k++)
                h_c_try[pop.metas[m].c_offset+k] += h_delta[m*K_max+k];
        }
        TBEG(e_dc1);
        CUDA_CHECK(cudaMemcpy(d_call, h_c_try, total_c*sizeof(float), cudaMemcpyHostToDevice));
        TEND(C_H2D, e_dc1);
        TBEG(e_ev1);
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_EVAL, e_ev1);
        TBEG(e_rs1);
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_yp, d_ym, M_prob*N, d_r);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_RESID, e_rs1);
        TBEG(e_ls1);
        loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_LOSS, e_ls1);
        TBEG(e_lt);
        CUDA_CHECK(cudaMemcpy(h_loss_try, d_loss, M_prob*sizeof(float), cudaMemcpyDeviceToHost));
        TEND(C_D2H, e_lt);

        // F. Per-tree accept/reject + λ + convergence
        // h_finished 编码 (internal, == status_out 的 enum):
        //   0 = 还在跑
        //   1 = converged
        //   2 = FAIL_NAN          (loss_try NaN/Inf)
        //   3 = K=0 skip          (pre-loop set)
        //   4 = FAIL_CHOLESKY     (Cholesky breakdown 反复, λ 爆)
        // 主循环看 == 0 决定是否继续. 末尾 maxiter 用还在 0 的项填.
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            h_iter[m]++;
            int K = pop.metas[m].K;

            // Cholesky breakdown: solve_kernel 设 status[m]=-1, delta=0.
            // 必须显式读 d_stat → 否则 δ=0 让 d_norm=0 假阳性 "converge". Treat as reject.
            if (h_solve_stat[m] != 0) {
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 4;  // FAIL_CHOLESKY
                continue;
            }

            // NaN/Inf: loss_try 出错说明 c_try 把 tree eval 推到 Inf/NaN 了.
            // 不要 reject 后重试 (c 没被污染, 但下一步 lam *= 10 后还会同一种 c_try 再算
            // — 因为 δ 由 J / r 决定, 都跟 c 有关. 直接放弃这棵).
            if (!isfinite(h_loss_try[m])) {
                h_finished[m] = 2;  // FAIL_NAN
                continue;
            }

            float d_norm_sq = 0, c_norm_sq = 0;
            for (int k = 0; k < K; k++) {
                d_norm_sq += h_delta[m*K_max+k] * h_delta[m*K_max+k];
                c_norm_sq += h_c_try[pop.metas[m].c_offset+k] * h_c_try[pop.metas[m].c_offset+k];
            }
            float d_norm = sqrtf(d_norm_sq), c_norm = sqrtf(c_norm_sq);

            // 收敛诚实性修复 (canonical LM ordering): 先判 improvement, 再判收敛.
            // 旧 bug: d_norm 小就标 CONVERGED 并 commit c_try, 没查 loss_try<=loss —
            // 一个微小的 uphill step 被当成收敛 (loss_final>loss_init). 见
            // tests/test_convergence_honesty.py.
            int improved = isfinite(h_loss_try[m]) && (h_loss_try[m] <= h_loss[m]);
            if (improved) {
                // accept (loss 单调非增). K=0 不走这条 — 已经 pre-skip.
                for (int k = 0; k < K; k++)
                    h_c[pop.metas[m].c_offset+k] = h_c_try[pop.metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                if (d_norm < xtol * (c_norm + xtol)) {
                    h_finished[m] = 1;  // CONVERGED (step 已被接受, loss 不增)
                } else {
                    h_lam[m] *= 0.1f;
                    if (h_lam[m] < 1e-12f) h_lam[m] = 1e-12f;
                }
            } else {
                // worsening (或 NaN/Inf): REJECT. 绝不标 converged, 绝不 commit c_try,
                // 即便 d_norm 极小. λ 加大重试; 反复爆 λ → 同旧 reject 路径的 FAIL_NAN.
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 2;  // FAIL_NAN bucket: lam-blow-up without Cholesky 是真的 NaN-ish 路径
            }
        }

        // G. Refresh d_y at h_c (accepted state)
        TBEG(e_dc2);
        CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));
        TEND(C_H2D, e_dc2);
        TBEG(e_ev2);
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_EVAL, e_ev2);
        TBEG(e_rs2);
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
        CUDA_CHECK(cudaGetLastError());
        TEND(C_RESID, e_rs2);

        if (!quiet && (it - last_print >= 5 || all_done)) {
            int n_done = 0; for (int m = 0; m < M_prob; m++) if (h_finished[m]) n_done++;
            printf("  it %2d: done=%d/%d\n", it, n_done, M_prob);
            last_print = it;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
#ifdef PROFILE
    clock_gettime(CLOCK_MONOTONIC, &t_loop_end);
#endif

    // ---- Final: 把内部 h_finished {0,1,2,3,4} 映射成外部 status_out ----
    int n_conv = 0, n_maxiter = 0, n_fail_nan = 0, n_k0 = 0, n_fail_chol = 0;
    int *status_out = (int*)malloc(M_prob * sizeof(int));
    for (int m = 0; m < M_prob; m++) {
        if      (h_finished[m] == 1) { status_out[m] = STATUS_CONVERGED;     n_conv++; }
        else if (h_finished[m] == 2) { status_out[m] = STATUS_FAIL_NAN;      n_fail_nan++; }
        else if (h_finished[m] == 3) { status_out[m] = STATUS_K0_SKIP;       n_k0++; }
        else if (h_finished[m] == 4) { status_out[m] = STATUS_FAIL_CHOLESKY; n_fail_chol++; }
        else                         { status_out[m] = STATUS_MAXITER;       n_maxiter++; }
    }

    char buf[2048];  // > dir_buf[1024] + "/c_final.bin" 留余地, 避 snprintf fortify 警告
    snprintf(buf, sizeof(buf), "%s/status.bin", out_dir);
    write_blob(buf, status_out, (size_t)M_prob * sizeof(int));
    snprintf(buf, sizeof(buf), "%s/c_final.bin", out_dir);
    write_blob(buf, h_c, (size_t)total_c * sizeof(float));
    // diagnostic (W0 MAXITER trace): kernel 自报 fp32 loss; 对比 interp fp64 判别 fp32-撒谎 vs 真更差盆地. additive, 不影响 gate.
    snprintf(buf, sizeof(buf), "%s/loss_final.bin", out_dir);
    write_blob(buf, h_loss, (size_t)M_prob * sizeof(float));
    // diagnostic (收敛诚实性): 初始 per-tree loss (LM loop 前). additive, 不影响 gate.
    snprintf(buf, sizeof(buf), "%s/loss_init.bin", out_dir);
    write_blob(buf, h_loss_init, (size_t)M_prob * sizeof(float));

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_s = (t_end.tv_sec - t_start.tv_sec)
                     + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    if (!quiet) {
        printf("[batch-lm] 完成: converged=%d (%.1f%%)  maxiter=%d  fail_nan=%d  fail_cholesky=%d  k0_skip=%d  (M_prob=%d)\n",
               n_conv, 100.0 * n_conv / M_prob, n_maxiter, n_fail_nan, n_fail_chol, n_k0, M_prob);
        printf("[batch-lm] 写出 %s/status.bin (%zu B) + %s/c_final.bin (%zu B)\n",
               out_dir, (size_t)M_prob * sizeof(int),
               out_dir, (size_t)total_c * sizeof(float));
        printf("[batch-lm] 总耗时 %.3f 秒 (%.0f trees/s, M_prob=%d)\n",
               elapsed_s, (double)M_prob / elapsed_s, M_prob);
    }

#ifdef PROFILE
    // Sum per-category GPU time from the recorded event pairs (one final sync
    // already happened), then emit a single machine-parseable line. host_residual
    // = loop wall not covered by any GPU op = launch latency + host accept/reject
    // + memcpy-wait beyond transfer. Printed regardless of --quiet.
    {
        double cat_ms[NCAT] = {0}; long cat_n[NCAT] = {0};
        for (size_t i = 0; i < g_prof.size(); i++) {
            float ms = 0; cudaEventElapsedTime(&ms, g_prof[i].s, g_prof[i].e);
            cat_ms[g_prof[i].cat] += ms; cat_n[g_prof[i].cat] += 1;
            cudaEventDestroy(g_prof[i].s); cudaEventDestroy(g_prof[i].e);
        }
        double gpu_sum = 0; for (int c = 0; c < NCAT; c++) gpu_sum += cat_ms[c];
#define TS_MS(a,b) (((b).tv_sec-(a).tv_sec)*1e3 + ((b).tv_nsec-(a).tv_nsec)/1e6)
        double load_ms  = TS_MS(t_start, t_load_done);
        double setup_ms = TS_MS(t_load_done, t_loop_begin);
        double loop_ms  = TS_MS(t_loop_begin, t_loop_end);
        double total_ms = TS_MS(t_start, t_end);
#undef TS_MS
        printf("PROFILE_JSON {\"M\":%d,\"N\":%d,\"iters_run\":%d,"
               "\"load_ms\":%.4f,\"setup_ms\":%.4f,\"loop_ms\":%.4f,\"total_ms\":%.4f,"
               "\"gpu_sum_ms\":%.4f,\"host_residual_ms\":%.4f,\"cats\":{",
               M_prob, N, n_loop_iters, load_ms, setup_ms, loop_ms, total_ms,
               gpu_sum, loop_ms - gpu_sum);
        for (int c = 0; c < NCAT; c++)
            printf("%s\"%s\":{\"ms\":%.4f,\"launches\":%ld}",
                   c ? "," : "", CAT_NAMES[c], cat_ms[c], cat_n[c]);
        printf("}}\n");
    }
#endif

    // ---- cleanup ----
    free(h_c); free(h_c_try); free(h_c_pert);
    free(h_lam); free(h_loss); free(h_loss_try);
    free(h_loss_init);
    free(h_finished); free(h_iter); free(h_rejected);
    free(h_delta);
    free(h_solve_stat);
    free(status_out);
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_ym); cudaFree(d_y); cudaFree(d_yp); cudaFree(d_r);
    cudaFree(d_J); cudaFree(d_JtJ); cudaFree(d_JtR); cudaFree(d_delta);
    cudaFree(d_loss); cudaFree(d_lam); cudaFree(d_stat);
    free_pop_data(&pop);
    return 0;
}
