// s9_lm_loop.cu — host 端 LM 主循环 A–G,把 s4–s8 的 kernel 串成一台能用的求解器。
//
// 这一级新增的【唯一概念】:
//   host 端的 LM 外层迭代循环 —— 每轮按 A→G 顺序调用 s4–s8 那几个 kernel,
//   配上 per-tree 的 accept/reject + λ 调度 + 收敛判定,把常数 c 从 c_init
//   一步步推到 c_true。到这一级,你手上就是一台 toy 版 batch_lm 了。
//
// 假设你已看过 s1..s8,已经懂:
//   - s1/s2: 树怎么存、解释器怎么求值(逆前缀 + 栈);
//   - s3:    线程索引 i = blockIdx.x*blockDim.x+threadIdx.x —— 1 线程 1 数据点(单棵树);
//   - s4:    eval_kernel_batched —— warp-per-problem 映射(一个 warp 算一棵树)+ 完整
//            解释器(UFUNC: EXP/NEG 等),三棵 archetype 一起批量 eval;
//   - s5:    residual_kernel (r = y_pred - y_meas) + loss_kernel (0.5·Σr²,shfl 归约);
//   - s6:    FD Jacobian —— 数值差分 J[k] = (y(c+ε·e_k) - y(c))/ε(用 eval kernel 拼);
//   - s7:    build_jtj_jtr_kernel —— 从 J、r 攒出每棵树的 JᵀJ (K×K)、JᵀR (K);
//   - s8:    solve_kernel —— per-tree 解 (JᵀJ + λ·diag)·δ = -JᵀR(Marquardt 阻尼 + Cholesky)。
// 这一级【只加】把它们串起来的那个 host driver:外层 for + per-tree LM 状态机。
//
// LM 一句话回顾(数学符号先用大白话铺):
//   我们要给一棵固定结构的树挑一组常数 c,让它在 64 个数据点上尽量贴合 y_meas。
//   "贴合程度" 用 loss = 0.5·Σ(y_pred(c) - y_meas)² 衡量,越小越好。
//   LM = Gauss-Newton + 阻尼:每轮在当前 c 处线性化,解一个 K×K 小方程得到
//   "该往哪挪、挪多少" 的步长 δ。λ(阻尼)大 → 步子小而稳(像梯度下降),
//   λ 小 → 步子大而准(像 Newton)。走对了(loss 降)就把 λ 调小、胆子放大;
//   走错了(loss 没降)就把 λ 调大、退回去重试。δ 缩到几乎不动 → 收敛。
//
// 数值【全抄 m3_4c_lm.cu】,一个都没自创:
//   λ 初值 1e-3;accept → λ*=0.1(下限 1e-12);reject → λ*=10(λ>1e12 判失败);
//   eps_fd=1e-3;xtol=1e-5;收敛判据 d_norm < xtol·(c_norm + xtol);max_iter=50。
//
// 对拍用【独立 CPU oracle】(不是 device kernel 的转写):
//   把每棵树的 c_final 代进【解析公式】(powerlaw/quadratic/expsat 三式),
//   在 CPU 上独立重算 loss,并和 c_true 比 c_rel_err。判据(抄 _SPEC §4):
//     c_rel_err = ‖c_final - c_true‖/‖c_true‖ < 1e-2,且 loss → ~0(每棵树)。
//
// 跑: ./build.sh s9     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - A–G 七步里,哪几步是 launch kernel、哪几步是纯 host 逻辑?为什么 accept/reject
//     必须放在 host?
//   - reject 时为什么【不更新 c】、只把 λ 调大?accept 时为什么把 λ 调小?
//   - 收敛判据 d_norm < xtol·(c_norm + xtol) 为什么用 c_norm 做尺度,而不是固定阈值?

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ── 共享代码(逐字来自 _SPEC §2,不改名/不改值)──────────────────────────────
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

enum NTypeE { N_VAR=0, N_CONST=1, N_UFUNC=2, N_BFUNC=3, N_TFUNC=4 };
enum FuncE {
    F_ADD=1, F_SUB=2, F_MUL=3, F_DIV=4, F_POW=6,
    F_SIN=14, F_COS=15, F_LOG=20, F_EXP=22, F_NEG=25, F_SQRT=27,
};
#define MAX_STACK 64    // 和生产 batch_lm.cu 一致
#define MAX_K     32

struct TreeMeta { int node_offset; int n_nodes; int c_offset; int K; };

// canonical 解释器(s4 起逐字用这一份)。逻辑 s1/s2 已讲透,这里不再重复。
__device__ float eval_tree_d(
    const int *nt, const float *nv, int n_nodes,
    const int *const_idx, const float *x, const float *c)
{
    float stack[MAX_STACK];
    int sp = 0;
    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i]; float v = nv[i];
        if (t == N_VAR) stack[sp++] = x[(int)v];
        else if (t == N_CONST) stack[sp++] = c[const_idx[i]];
        else if (t == N_UFUNC) {
            float a = stack[--sp]; int fid = (int)v; float r;
            switch (fid) {
                case F_SIN: r=sinf(a); break;  case F_COS: r=cosf(a); break;
                case F_LOG: r=logf(a); break;  case F_EXP: r=expf(a); break;
                case F_NEG: r=-a;      break;  case F_SQRT:r=sqrtf(a);break;
                default:    r=0.0f;
            }
            stack[sp++] = r;
        } else if (t == N_BFUNC) {
            float l = stack[--sp]; float r = stack[--sp];
            int fid = (int)v; float o;
            switch (fid) {
                case F_ADD: o=l+r; break; case F_SUB: o=l-r; break;
                case F_MUL: o=l*r; break; case F_DIV: o=l/r; break;
                case F_POW: o=powf(l,r); break; default: o=0.0f;
            }
            stack[sp++] = o;
        }
    }
    return stack[0];
}

// ── s4–s8 的 kernel(逐字沿用前几级 / 生产命名,这一级不改它们)──────────────
// 这些你 s4–s8 已经分别吃透了。s9 把它们当现成积木用,顺序串起来就行。

// s4: warp-per-problem,一个 warp 算一棵树在全部 N 个点上的 y_pred。
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

// s5: r = y_pred - y_meas,一个线程一个 (tree,point)。
__global__ void residual_kernel(const float *yp, const float *ym, int total, float *r) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    r[tid] = yp[tid] - ym[tid];
}

// s5: loss[m] = 0.5·Σ_i r[m*N+i]²,warp 内 shfl 蝶形归约。
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

// s7: 从 J、r 攒出每棵树的 JᵀR (K) 和 JᵀJ (K×K,对称只算上三角再镜像)。
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

// s8: per-tree 1 个线程解 (JᵀJ + λ·diag)·δ = -JᵀR。Cholesky;s<=0(分解失败)→ status=-1、δ=0。
__global__ void solve_kernel(
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
        A[j1*K + j1] *= (1.0f + lam);          // Marquardt 阻尼: 对角 ×(1+λ)
    }
    for (int j = 0; j < K; j++) {              // Cholesky 分解 A = L·Lᵀ
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
    for (int i = 0; i < K; i++) {              // 前代 L·y = b
        float s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*K+k] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = K-1; i >= 0; i--) {           // 回代 Lᵀ·δ = y
        float s = b[i];
        for (int k = i+1; k < K; k++) s -= A[k*K+i] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = 0; i < K; i++) delta[m*K_max + i] = b[i];
    status[m] = 0;
}

// ── 三个 fixture(逐字来自 _SPEC §3)─────────────────────────────────────────
// 每棵树带两组常数: c_true(造数据用的真值)和 c_init(LM 的起点)。
struct TreeSpec {
    const char *name; int n_nodes;
    const int *nt; const float *nv; const int *ci;
    int K; const float *c_true; const float *c_init;
};
// powerlaw_K2: c0 * pow(x0, c1)  →  prefix MUL c0 POW x0 c1
static const int   tree0_nt[5] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
static const float tree0_nv[5] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f };
static const int   tree0_ci[5] = { -1, 0, -1, -1, 1 };
static const float tree0_ct[2] = { 2.5f, 1.3f };   // c_true
static const float tree0_c0[2] = { 1.0f, 1.0f };   // c_init (LM 起点)

// quadratic_K3: c0 + c1*x0 + c2*x0*x0  →  prefix ADD c0 ADD MUL c1 x0 MUL c2 MUL x0 x0
static const int   tree1_nt[11] = { N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR,
                                    N_BFUNC, N_CONST, N_BFUNC, N_VAR,   N_VAR };
static const float tree1_nv[11] = { F_ADD,   0.0f,    F_ADD,   F_MUL,   0.0f,    0.0f,
                                    F_MUL,   0.0f,    F_MUL,   0.0f,    0.0f };
static const int   tree1_ci[11] = { -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1 };
static const float tree1_ct[3]  = { 0.5f, -1.2f, 0.3f };   // c_true
static const float tree1_c0[3]  = { 0.0f, 0.0f, 0.0f };    // c_init

// expsat_K3: c0 * exp(neg(c1 * x0)) + c2  →  prefix ADD MUL c0 EXP NEG MUL c1 x0 c2
static const int   tree2_nt[9] = { N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC,
                                   N_CONST, N_VAR,   N_CONST };
static const float tree2_nv[9] = { F_ADD,   F_MUL,   0.0f,    F_EXP,   F_NEG,   F_MUL,
                                   0.0f,    0.0f,    0.0f };
static const int   tree2_ci[9] = { -1, -1, 0, -1, -1, -1, 1, -1, 2 };
static const float tree2_ct[3] = { 3.0f, 0.4f, 0.7f };     // c_true
static const float tree2_c0[3] = { 1.0f, 1.0f, 0.0f };     // c_init

static const TreeSpec FIXTURES[] = {
    {"powerlaw_K2", 5,  tree0_nt, tree0_nv, tree0_ci, 2, tree0_ct, tree0_c0},
    {"quadratic_K3",11, tree1_nt, tree1_nv, tree1_ci, 3, tree1_ct, tree1_c0},
    {"expsat_K3",   9,  tree2_nt, tree2_nv, tree2_ci, 3, tree2_ct, tree2_c0},
};

// ── 独立 CPU oracle:解析公式(造 y_meas + 末尾独立重算 loss 都用它)──────────
// 这是【不依赖任何 device kernel】的第二套实现 —— 对拍才不是自欺。
static float y_analytic(int fid, float x, const float *c) {
    if (fid == 0) return c[0] * powf(x, c[1]);            // powerlaw
    if (fid == 1) return c[0] + c[1]*x + c[2]*x*x;        // quadratic
    if (fid == 2) return c[0] * expf(-c[1]*x) + c[2];     // expsat
    return 0.0f;
}

int main() {
    int M_prob = (int)(sizeof(FIXTURES)/sizeof(FIXTURES[0]));
    int N = 64, n_vars = 1;
    int K_max = 0;
    for (int m = 0; m < M_prob; m++) if (FIXTURES[m].K > K_max) K_max = FIXTURES[m].K;
    if (K_max > MAX_K) {
        fprintf(stderr, "K_max=%d 超过 MAX_K=%d, solve_kernel 会 OOB.\n", K_max, MAX_K);
        return 2;
    }

    // ---- host 端打平存树(node_offset / c_offset 拼成一长条)----
    TreeMeta h_metas[16];
    int total_nodes = 0, total_c = 0;
    for (int m = 0; m < M_prob; m++) {
        h_metas[m] = {total_nodes, FIXTURES[m].n_nodes, total_c, FIXTURES[m].K};
        total_nodes += FIXTURES[m].n_nodes;
        total_c     += FIXTURES[m].K;
    }
    int   *h_nt    = (int*)  malloc(total_nodes*sizeof(int));
    float *h_nv    = (float*)malloc(total_nodes*sizeof(float));
    int   *h_ci    = (int*)  malloc(total_nodes*sizeof(int));
    float *h_c     = (float*)malloc(total_c*sizeof(float));   // c_curr(当前接受的常数)
    float *h_c_try = (float*)malloc(total_c*sizeof(float));   // c_curr + δ(试探点)
    float *h_c_pert= (float*)malloc(total_c*sizeof(float));   // FD 扰动用
    for (int m = 0; m < M_prob; m++) {
        memcpy(h_nt+h_metas[m].node_offset, FIXTURES[m].nt, FIXTURES[m].n_nodes*sizeof(int));
        memcpy(h_nv+h_metas[m].node_offset, FIXTURES[m].nv, FIXTURES[m].n_nodes*sizeof(float));
        memcpy(h_ci+h_metas[m].node_offset, FIXTURES[m].ci, FIXTURES[m].n_nodes*sizeof(int));
        for (int k = 0; k < FIXTURES[m].K; k++)
            h_c[h_metas[m].c_offset+k] = FIXTURES[m].c_init[k];   // LM 从 c_init 起步
    }
    // xs ∈ [0.1, 5.0] 均匀采 N=64
    float *h_xs = (float*)malloc(N*n_vars*sizeof(float));
    for (int i = 0; i < N; i++) h_xs[i] = 0.1f + (5.0f-0.1f)*i/(N-1);
    // y_meas = 用 c_true 代进【解析公式】造的"实验数据"(独立 oracle 的一半)
    float *h_ym = (float*)malloc(M_prob*N*sizeof(float));
    for (int m = 0; m < M_prob; m++)
        for (int i = 0; i < N; i++) h_ym[m*N+i] = y_analytic(m, h_xs[i], FIXTURES[m].c_true);

    // ---- device 缓冲 ----
    int *d_nt; float *d_nv; int *d_ci; float *d_call; TreeMeta *d_metas;
    float *d_xs, *d_ym, *d_y, *d_yp, *d_r, *d_J, *d_JtJ, *d_JtR, *d_delta, *d_loss, *d_lam;
    int *d_stat;
    CUDA_CHECK(cudaMalloc(&d_nt, total_nodes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nv, total_nodes*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ci, total_nodes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_call, total_c*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_metas, M_prob*sizeof(TreeMeta)));
    CUDA_CHECK(cudaMalloc(&d_xs, N*n_vars*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ym, M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yp, M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r,  M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_J,  (size_t)M_prob*K_max*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_JtJ,(size_t)M_prob*K_max*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_JtR,(size_t)M_prob*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta,(size_t)M_prob*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, M_prob*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lam, M_prob*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stat, M_prob*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nt, h_nt, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv, h_nv, total_nodes*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_ci, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, M_prob*sizeof(TreeMeta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, N*n_vars*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ym, h_ym, M_prob*N*sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, wpb = block/32;            // BLOCK=256 → 8 warp/block
    int grid = (M_prob + wpb - 1) / wpb;

    // ---- per-tree 的 LM 状态(都在 host 上)----
    //   h_lam[m]      当前阻尼 λ
    //   h_loss[m]     当前已接受的 loss(基准)
    //   h_loss_try[m] 试探点 c+δ 的 loss
    //   h_finished[m] 0=还在跑, 1=收敛成功, 2=λ 爆掉判失败
    float h_lam[16], h_loss[16], h_loss_try[16];
    int   h_finished[16] = {0};
    int   h_iter[16] = {0};
    int   h_rejected[16] = {0};
    int   h_solve_stat[16] = {0};   // solve_kernel 的退化标志(0=正常, -1=Cholesky 分解失败)
    for (int m = 0; m < M_prob; m++) h_lam[m] = 1e-3f;   // λ 初值(抄 m3_4c)

    // ---- 数值常数(全抄 m3_4c_lm.cu)----
    const int   max_iter = 50;
    const float eps_fd   = 1e-3f;
    const float xtol     = 1e-5f;

    float *h_yb    = (float*)malloc(M_prob*N*sizeof(float));   // baseline y(c_curr)
    float *h_yp    = (float*)malloc(M_prob*N*sizeof(float));   // 扰动 / 试探 y
    float *h_J     = (float*)malloc((size_t)M_prob*K_max*N*sizeof(float));
    float *h_delta = (float*)malloc(M_prob*K_max*sizeof(float));

    // ---- 初始 eval + residual + loss(进循环前先把基准算出来)----
    CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());
    residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
    CUDA_CHECK(cudaGetLastError());
    loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, M_prob*sizeof(float), cudaMemcpyDeviceToHost));

    printf("s9: host LM 主循环 A–G,c_init → 收敛到 c_true\n");
    printf("  M_prob=%d, N=%d, K_max=%d, max_iter=%d, xtol=%.0e, eps_fd=%.0e\n",
           M_prob, N, K_max, max_iter, xtol, eps_fd);
    printf("  iter:  loss[0]      loss[1]      loss[2]     | λ[0]     λ[1]     λ[2]    | fin\n");
    printf("  init: %.4e   %.4e   %.4e | %.1e %.1e %.1e | %d %d %d\n",
           h_loss[0], h_loss[1], h_loss[2], h_lam[0], h_lam[1], h_lam[2],
           h_finished[0], h_finished[1], h_finished[2]);

    // ═══════════════ LM 主循环:每轮 A→G ═══════════════════════════════════
    for (int it = 0; it < max_iter; it++) {
        // 全部 finished 就停。(每棵树独立收敛;早收敛的 lane 这版仍空跑,m4 才早退)
        int all_done = 1;
        for (int m = 0; m < M_prob; m++) if (!h_finished[m]) { all_done = 0; break; }
        if (all_done) break;

        // ── A. baseline y_pred(c_curr):已在 d_y 里(init 时算过,或上一轮 G 步刷新过)。
        //       拷回 host 当 FD 的"未扰动基准 y(c)"。
        CUDA_CHECK(cudaMemcpy(h_yb, d_y, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));

        // ── B. FD Jacobian:对第 k 个常数 +eps_fd,eval 一次,(y_pert - y_base)/eps。
        //       K_max 次 perturb-launch,每次同时给所有树的第 k 个常数加扰动。
        for (int k = 0; k < K_max; k++) {
            memcpy(h_c_pert, h_c, total_c*sizeof(float));
            for (int m = 0; m < M_prob; m++) {
                if (h_finished[m]) continue;
                if (k < h_metas[m].K) h_c_pert[h_metas[m].c_offset+k] += eps_fd;
            }
            CUDA_CHECK(cudaMemcpy(d_call, h_c_pert, total_c*sizeof(float), cudaMemcpyHostToDevice));
            eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_yp, d_yp, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
            for (int m = 0; m < M_prob; m++) {
                if (h_finished[m] || k >= h_metas[m].K) continue;
                for (int i = 0; i < N; i++)
                    h_J[(size_t)m*K_max*N + k*N + i] = (h_yp[m*N+i] - h_yb[m*N+i]) / eps_fd;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_J, h_J, (size_t)M_prob*K_max*N*sizeof(float), cudaMemcpyHostToDevice));

        // ── C. build JᵀJ + JᵀR。注意 d_r 此刻仍是 c_curr 处的 residual(init / G 步留下的)。
        build_jtj_jtr_kernel<<<grid, block>>>(d_J, d_r, d_metas, M_prob, N, K_max, d_JtJ, d_JtR);
        CUDA_CHECK(cudaGetLastError());

        // ── D. solve δ(per-tree λ)。
        CUDA_CHECK(cudaMemcpy(d_lam, h_lam, M_prob*sizeof(float), cudaMemcpyHostToDevice));
        solve_kernel<<<(M_prob+255)/256, 256>>>(d_JtJ, d_JtR, d_metas, M_prob, K_max, d_lam, d_delta, d_stat);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_delta, d_delta, M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));
        // status 也必须拷回:Cholesky 分解失败时 δ=0,若不读 status,下面 d_norm=0 会
        // 抢先命中收敛判据 → 假阳性"收敛"。读回来在 F 步当"拒绝"处理(对齐生产 batch_lm.cu)。
        CUDA_CHECK(cudaMemcpy(h_solve_stat, d_stat, M_prob*sizeof(int), cudaMemcpyDeviceToHost));

        // ── E. 试探:c_try = c_curr + δ,eval → residual → loss_try。
        memcpy(h_c_try, h_c, total_c*sizeof(float));
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            for (int k = 0; k < h_metas[m].K; k++)
                h_c_try[h_metas[m].c_offset+k] += h_delta[m*K_max+k];
        }
        CUDA_CHECK(cudaMemcpy(d_call, h_c_try, total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
        CUDA_CHECK(cudaGetLastError());
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_yp, d_ym, M_prob*N, d_r);
        CUDA_CHECK(cudaGetLastError());
        loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_loss_try, d_loss, M_prob*sizeof(float), cudaMemcpyDeviceToHost));

        // ── F. host 端 per-tree accept/reject + λ 调度 + 收敛判定(全抄 m3_4c)。
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            h_iter[m]++;
            int K = h_metas[m].K;

            // Cholesky 失败:solve_kernel 设了 status=-1、δ=0。必须在收敛判据【之前】拦掉,
            // 否则 δ=0 → d_norm=0 会假装"收敛"。当成拒绝:调大 λ 重来;λ 爆掉才判失败。
            if (h_solve_stat[m] != 0) {
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 2;   // λ 爆掉,判失败
                continue;
            }
            // loss_try 出 NaN/Inf:c_try 把树 eval 推爆了。这棵 c 已经污染,直接放弃
            // (对齐生产 batch_lm.cu;本级三个 fixture 都不会触发,纯为和生产保持一致)。
            if (!isfinite(h_loss_try[m])) {
                h_finished[m] = 2;
                continue;
            }

            float d_norm_sq = 0, c_norm_sq = 0;
            for (int k = 0; k < K; k++) {
                d_norm_sq += h_delta[m*K_max+k] * h_delta[m*K_max+k];
                c_norm_sq += h_c_try[h_metas[m].c_offset+k] * h_c_try[h_metas[m].c_offset+k];
            }
            float d_norm = sqrtf(d_norm_sq), c_norm = sqrtf(c_norm_sq);

            // 收敛: δ 已经缩到和 c 的尺度相比可忽略 → 接受 c_try、标记完成。
            // 用 c_norm 当尺度: c 大的问题,δ 也得相应更大才算"动了";固定阈值对不同
            // 量级的常数会一刀切错。
            if (d_norm < xtol * (c_norm + xtol)) {
                for (int k = 0; k < K; k++)
                    h_c[h_metas[m].c_offset+k] = h_c_try[h_metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                h_finished[m] = 1;
            } else if (h_loss_try[m] < h_loss[m]) {
                // accept: loss 降了 → 收下 c_try,λ 调小(下次胆子放大,更像 Newton)。
                for (int k = 0; k < K; k++)
                    h_c[h_metas[m].c_offset+k] = h_c_try[h_metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                h_lam[m] *= 0.1f;
                if (h_lam[m] < 1e-12f) h_lam[m] = 1e-12f;
            } else {
                // reject: loss 没降 → 【不更新 c】,λ 调大(下次步子变小更稳)。
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 2;   // λ 爆掉,判失败
            }
        }

        // ── G. 用当前已接受的 c_curr 刷新 d_y 和 d_r,给下一轮 A 步当基准。
        //       (reject 的树 c 没变,这一步把它们的 y/r 重算成 c_curr 处的,和
        //        accept 的树统一。)
        CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
        CUDA_CHECK(cudaGetLastError());
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
        CUDA_CHECK(cudaGetLastError());

        if (it < 5 || (it % 5) == 0) {
            printf("  it %2d: %.4e   %.4e   %.4e | %.1e %.1e %.1e | %d %d %d\n",
                   it, h_loss[0], h_loss[1], h_loss[2], h_lam[0], h_lam[1], h_lam[2],
                   h_finished[0], h_finished[1], h_finished[2]);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ═══════════════ 对拍:独立 CPU oracle ═══════════════════════════════════
    // 拿 device 跑出来的 c_final,在 CPU 上用【解析公式】独立重算 loss(不碰任何
    // device kernel),并和 c_true 比 c_rel_err。判据(_SPEC §4):
    //   c_rel_err < 1e-2  且  loss → ~0(每棵树)。
    printf("\n  ─── Final(对拍 CPU oracle)───\n");
    int all_pass = 1;
    for (int m = 0; m < M_prob; m++) {
        int K = h_metas[m].K;
        const float *cf = h_c + h_metas[m].c_offset;

        // c_rel_err = ‖c_final - c_true‖ / ‖c_true‖
        float num = 0, den = 0;
        for (int k = 0; k < K; k++) {
            float diff = cf[k] - FIXTURES[m].c_true[k];
            num += diff * diff;
            den += FIXTURES[m].c_true[k] * FIXTURES[m].c_true[k];
        }
        float c_rel = sqrtf(num) / sqrtf(den);

        // 独立重算 loss = 0.5·Σ(y_analytic(c_final) - y_meas)²(CPU,纯解析)
        float oracle_loss = 0;
        for (int i = 0; i < N; i++) {
            float yp = y_analytic(m, h_xs[i], cf);
            float r  = yp - h_ym[m*N+i];
            oracle_loss += r * r;
        }
        oracle_loss *= 0.5f;

        // 三重判据:有限 + 真收敛(fin==1,不是 2 失败) + c_rel 在容差内。
        // (只看 c_rel 对 NaN 失效——NaN 和任何数比都 false,会让 harness 撒谎。)
        int ok = isfinite(c_rel) && isfinite(oracle_loss) && h_finished[m] == 1 && c_rel < 1e-2f;
        if (!ok) all_pass = 0;

        printf("  [%d] %-13s iter=%2d rej=%d fin=%d  c_rel_err=%.3e  oracle_loss=%.3e  %s\n",
               m, FIXTURES[m].name, h_iter[m], h_rejected[m], h_finished[m],
               c_rel, oracle_loss, ok ? "ok" : "MISMATCH");
        printf("        c_final=[");
        for (int k = 0; k < K; k++) printf("%s%.5f", k==0?"":", ", cf[k]);
        printf("]  vs c_true=[");
        for (int k = 0; k < K; k++) printf("%s%.5f", k==0?"":", ", FIXTURES[m].c_true[k]);
        printf("]\n");
    }

    free(h_nt); free(h_nv); free(h_ci); free(h_c); free(h_c_try); free(h_c_pert);
    free(h_xs); free(h_ym); free(h_yb); free(h_yp); free(h_J); free(h_delta);
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_ym); cudaFree(d_y); cudaFree(d_yp); cudaFree(d_r);
    cudaFree(d_J); cudaFree(d_JtJ); cudaFree(d_JtR); cudaFree(d_delta);
    cudaFree(d_loss); cudaFree(d_lam); cudaFree(d_stat);

    printf("\n%s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. A–G 七步里,哪几步在 launch kernel、哪几步是纯 host 逻辑?为什么
//     accept/reject 这一步(F)非得放 host 不可?
//  2. reject 时为什么【不更新 c】、只把 λ 调大?accept 时为什么反而把 λ 调小?
//     如果反过来(reject 调小 / accept 调大)会怎样?
//  3. 收敛判据写成 d_norm < xtol·(c_norm + xtol),为什么乘上 c_norm 而不是
//     直接 d_norm < xtol?那个 "+ xtol" 是防什么的?
//  4. 这一版每轮所有树都跑满 A–G,早收敛的树在 finished 后还占着 warp 空转。
//     B/E 里的 if(h_finished[m])continue 跳过了它们的【数据准备】,但 kernel 仍按
//     M_prob 全量 launch。要让那些 warp 真正不被调度,得改哪里?生产 batch_lm 在这点
//     上和本级有区别吗?
//
// ## 参考答案
//
//  1. launch kernel 的:B(K_max 次 eval 拼 Jacobian)、C(build_jtj_jtr)、D(solve)、
//     E(eval+residual+loss 算 loss_try)、G(eval+residual 刷新基准)。纯 host 逻辑:
//     F(accept/reject + λ 调度 + 收敛判定)。A 既不是 launch 也不是纯逻辑——它只是
//     一次 D2H cudaMemcpy(把 init/G 步已经算好的 baseline d_y 拷回 host 当 FD 基准)。
//     F 放 host,是因为它要做【带分支的串行决策】(每棵树看 loss 降没降、λ 该乘几、
//     收没收敛),数据量极小(每棵树几个标量),搬上 GPU 既无并行收益、又让控制流
//     变复杂;这一版索性让 host 统筹,kernel 只跑数值热路径(eval / 线代)。这正是
//     "host 端 LM driver" 的本意。
//
//  2. LM 的逻辑:δ 是在【当前 c 处线性化】解出来的,只在小邻域可信。loss 降了 →
//     说明这步走对、线性近似在这一带靠谱 → 收下新 c,并把 λ 调小让下一步更大更准
//     (趋近 Newton)。loss 没降 → 这步走过头了、近似不可信 → 必须退回原 c(不更新),
//     把 λ 调大让步子变小、更靠近"稳但慢"的梯度下降方向,再用同一个 c 重解一次。
//     反过来的话:reject 还把步子调更大 → 越走越偏、发散;accept 把步子调小 → 明明
//     走对了却越走越保守,收敛奇慢甚至卡住。方向反了,LM 就废了。
//
//  3. 收敛要的是"δ 相对 c 的尺度可忽略",不是"δ 绝对小"。c 量级大(比如几千)时,
//     一个绝对值 0.01 的 δ 其实微不足道;c 量级小(比如 0.001)时,同样 0.01 的 δ
//     却是大改动。所以拿 c_norm 当尺子:d_norm < xtol·c_norm 才是"相对没怎么动"。
//     "+ xtol" 是兜底:当 c_norm 趋近 0(常数都接近 0)时,纯相对判据 xtol·c_norm→0
//     永远满足不了、会卡死;加一个 xtol 量级的绝对地板,c 接近 0 时退化成
//     d_norm < xtol·xtol 这种绝对判据,避免除零式的死循环。
//
//  4. B 步(FD perturb)和 E 步(apply δ)里的 if(h_finished[m]) continue 已经在
//     host 侧跳过 finished 的树的【数据准备】,但 kernel 仍按 M_prob 全量 launch,
//     finished 树对应的 warp 照样被调度、空算一遍(它的 c 没变,结果被丢弃)。
//     生产 batch_lm 在这点上和本级【一样】:它的 finished 也全是 host 侧的(grep
//     batch_lm.cu 里每处 finished 都在 host 上,eval / loss / build / solve 全都按
//     M_prob 全量 launch、留同样的空转 lane)。要真正省掉,得在 device 端加一个
//     finished[] mask、kernel 开头 if(finished[m]) return 让 warp 直接退出——这是
//     生产 batch_lm 也没做的进一步优化,不是本级欠的债。
// ===========================================================================
