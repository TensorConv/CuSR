// s7_jtj_jtr.cu — 把 J 和 r 装配成正规方程的两块: JᵀJ 和 JᵀR。
//
// 假设你已看过 s1..s6, 已经懂 J[M][K][N](每棵树的雅可比, s6 用有限差分 FD 算的)
// 和 r[M][N](残差, s5 算的, r = y_pred - y_meas)。这一级只加一件事:
//   把 J 和 r 装配成 JᵀJ(一个 K×K 矩阵)和 JᵀR(一个长 K 的向量),
//   每个矩阵/向量元素用【一次 warp-shfl 归约】算出来。
//
// 先用大白话说清这两块是什么、为什么要它:
//   LM(Levenberg-Marquardt)每一步要解一个小线性方程组求 δ(常数该往哪挪):
//        (JᵀJ + λI) · δ = −JᵀR
//   等号左边的系数矩阵 JᵀJ 是 K×K(K = 这棵树有几个常数, 这里 2 或 3),
//   右边 JᵀR 是长 K 的向量。s8 才真去解这个方程;s7 只负责【把这两块算出来】。
//
//   JᵀJ 和 JᵀR 各自的元素是什么(把 J 看成 K 行、每行 N 个数的表):
//        JᵀR[k]      = Σ_i  J[k, i] * r[i]              (k 从 0 到 K-1)
//        JᵀJ[j1,j2]  = Σ_i  J[j1, i] * J[j2, i]         (j1,j2 各从 0 到 K-1)
//   每个元素都是【对 N 个数据点求和】—— 正好是一个 warp(32 lane)能干的归约活,
//   和 s5 算 loss = 0.5·Σrᵢ² 用的【完全同一个 shfl 蝶形归约】套路。
//
//   两个省事的点(都顺手讲, 不是新概念):
//     1. JᵀJ 是【对称】矩阵: JᵀJ[j1,j2] == JᵀJ[j2,j1]。所以只算上三角(j2 ≥ j1)
//        一次, 写进 [j1,j2] 和 [j2,j1] 两个位置(镜像填), 省一半归约。
//     2. 一个 warp 包一棵树(s4 起就是这个映射), 树与树之间互不干扰。
//
// 跑: ./build.sh s7     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - JᵀJ 为什么是 K×K 而不是 N×N?它和数据点数 N 是什么关系?
//   - "只算上三角再镜像填两边" 省掉了哪一半的活?靠的是矩阵的什么性质?
//   - 这里每个 shfl 归约求和的是【N 个数据点】, 和 s5 算 loss 那次归约求和的是什么 —— 同一类活吗?

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// 全程约定: 每个 CUDA API 调用都包它(出错立刻打印 + 退出)。
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

enum NType { N_VAR = 0, N_CONST = 1, N_UFUNC = 2, N_BFUNC = 3, N_TFUNC = 4 };
enum Func {
    F_ADD = 1, F_SUB = 2, F_MUL = 3, F_DIV = 4, F_POW = 6,
    F_SIN = 14, F_COS = 15, F_LOG = 20, F_EXP = 22, F_NEG = 25, F_SQRT = 27,
};
#define MAX_STACK 64    // 和生产 batch_lm.cu 一致
#define MAX_K     32

struct TreeMeta { int node_offset; int n_nodes; int c_offset; int K; };

// ---------------------------------------------------------------------------
// canonical 解释器(s4 起逐字这一份)。s7 不改它一个字 —— 这里只是为了
// 给 s6 的 FD 雅可比和 s5 的残差当底层求值器用。
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 下面三个 kernel 是 s4/s5 的旧货, s7 原样拿来用(不是这一级的新概念):
//   eval_kernel_batched : warp-per-problem 求值 y_pred(s4)
//   residual_kernel     : r = y_pred - y_meas(s5)
// J 这一级不在 GPU 上算导数, 沿用 s6 的写法(host 端从两次 eval 的差 / eps 出 FD), 见 main。
// ---------------------------------------------------------------------------
__global__ void eval_kernel_batched(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N,
    const float *c_all, float *y_out)
{
    int wpb  = blockDim.x / 32;     // BLOCK=256 → 8 warp = 8 棵树/block
    int wb   = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m    = blockIdx.x * wpb + wb;
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
    r[tid] = yp[tid] - ym[tid];
}

// ===========================================================================
// 这一级的【唯一新 kernel】: 从 J 和 r 装配 JᵀJ + JᵀR, 每个元素一次 shfl 归约。
//
//   数据布局(s5/s6 定死的, 和生产 batch_lm.cu 一致):
//     J   : [M × K_max × N]  —— 第 m 棵树的第 k 行第 i 点是 J[m*K_max*N + k*N + i]
//                               (每棵树占 K_max 行, 但只有前 K_m 行有效)
//     r   : [M × N]          —— 第 m 棵树第 i 点的残差是 r[m*N + i]
//     JtJ : [M × K_max × K_max]  行主序, 第 m 棵树的 (j1,j2) 在 JJm[j1*K_max + j2]
//     JtR : [M × K_max]
//
//   一个 warp 包一棵树(m), warp 内 32 个 lane 对 N 个数据点做 strided 求和,
//   再用蝶形 shfl 把 32 个 lane 的部分和汇成一个总和(s5 教的那套)。
// ===========================================================================
__global__ void build_jtj_jtr_kernel(
    const float *J, const float *r, const TreeMeta *metas,
    int M_prob, int N, int K_max, float *JtJ, float *JtR)
{
    // warp-per-problem 映射(s4 起每个 warp-per-problem kernel 开头都这一段)
    int wpb  = blockDim.x / 32;
    int wb   = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m    = blockIdx.x * wpb + wb;
    if (m >= M_prob) return;

    int K_m = metas[m].K;                          // 这棵树有几个常数
    const float *Jm  = J   + (size_t)m * K_max * N;  // 第 m 棵树那块 J
    const float *rm  = r   + (size_t)m * N;          // 第 m 棵树那段 r
    float       *JJm = JtJ + (size_t)m * K_max * K_max;
    float       *JRm = JtR + (size_t)m * K_max;

    // ---- JᵀR[k] = Σ_i J[k,i] * r[i]   (K_m 个独立归约) ----
    for (int k = 0; k < K_m; k++) {
        float s = 0.0f;
        for (int i = lane; i < N; i += 32)         // 32 个 lane 分摊 N 个点
            s += Jm[k * N + i] * rm[i];
        // 蝶形 shfl: 把 32 个 lane 的部分和合成一个(和 s5 的 loss 归约一字不差)
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
        if (lane == 0) JRm[k] = s;                 // 归约后每个 lane 都持有总和, lane0 落盘
    }

    // ---- JᵀJ[j1,j2] = Σ_i J[j1,i] * J[j2,i]   (对称, 只算上三角再镜像填) ----
    for (int j1 = 0; j1 < K_m; j1++) {
        for (int j2 = j1; j2 < K_m; j2++) {        // j2 从 j1 起 = 只走上三角
            float s = 0.0f;
            for (int i = lane; i < N; i += 32)
                s += Jm[j1 * N + i] * Jm[j2 * N + i];
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            if (lane == 0) {
                JJm[j1 * K_max + j2] = s;           // 上三角
                JJm[j2 * K_max + j1] = s;           // 镜像到下三角(对称性省一半归约)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 三个 fixture(逐字来自 _SPEC §3 / m3_2)。
// ---------------------------------------------------------------------------
struct TreeSpec { const char *name; int n_nodes; const int *nt; const float *nv; const int *ci; int K; const float *c_true; const float *c_init; };

// powerlaw_K2: c0 * pow(x0, c1)
static const int   tree0_nt[5] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
static const float tree0_nv[5] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f };
static const int   tree0_ci[5] = { -1, 0, -1, -1, 1 };
static const float tree0_ct[2] = { 2.5f, 1.3f };   // c_true
static const float tree0_c0[2] = { 1.0f, 1.0f };   // c_init (LM 起点; s7 在这处求 J/r, 不优化常数)

// quadratic_K3: c0 + c1*x0 + c2*x0*x0
static const int   tree1_nt[11] = { N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR,
                                    N_BFUNC, N_CONST, N_BFUNC, N_VAR,   N_VAR };
static const float tree1_nv[11] = { F_ADD,   0.0f,    F_ADD,   F_MUL,   0.0f,    0.0f,
                                    F_MUL,   0.0f,    F_MUL,   0.0f,    0.0f };
static const int   tree1_ci[11] = { -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1 };
static const float tree1_ct[3]  = { 0.5f, -1.2f, 0.3f };   // c_true
static const float tree1_c0[3]  = { 0.0f, 0.0f, 0.0f };    // c_init

// expsat_K3: c0 * exp(neg(c1 * x0)) + c2
static const int   tree2_nt[9] = { N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC,
                                   N_CONST, N_VAR,   N_CONST };
static const float tree2_nv[9] = { F_ADD,   F_MUL,   0.0f,    F_EXP,   F_NEG,   F_MUL,
                                   0.0f,    0.0f,    0.0f };
static const int   tree2_ci[9] = { -1, -1, 0, -1, -1, -1, 1, -1, 2 };
static const float tree2_ct[3] = { 3.0f, 0.4f, 0.7f };     // c_true
static const float tree2_c0[3] = { 1.0f, 1.0f, 0.0f };     // c_init
// 注: 和 s5/s6 一致, 这一级也在 c_init(tree*_c0, 即 s9 的 LM 起点)处求 J/r 算 JᵀJ/JᵀR。
//     c_init ≠ c_true, 所以起点残差 r 已经非零, JᵀR 自然非平凡 —— 不用人为偏离真值。
//     s7 不优化常数(s9 才迭代), c_true 这里只用来生成干净观测 y_meas。

static const TreeSpec FIXTURES[] = {
    {"powerlaw_K2", 5,tree0_nt,tree0_nv,tree0_ci,2,tree0_ct,tree0_c0},
    {"quadratic_K3",11,tree1_nt,tree1_nv,tree1_ci,3,tree1_ct,tree1_c0},
    {"expsat_K3",   9,tree2_nt,tree2_nv,tree2_ci,3,tree2_ct,tree2_c0},
};

// 干净观测 y_meas = y(c_true): 用三式解析值(独立 oracle 的求值入口)。
static float y_ref_eval(int fid, float x) {
    if (fid == 0) { float c0=tree0_ct[0], c1=tree0_ct[1]; return c0 * powf(x, c1); }
    if (fid == 1) { float c0=tree1_ct[0], c1=tree1_ct[1], c2=tree1_ct[2]; return c0 + c1*x + c2*x*x; }
    if (fid == 2) { float c0=tree2_ct[0], c1=tree2_ct[1], c2=tree2_ct[2]; return c0 * expf(-c1*x) + c2; }
    return 0.0f;
}

int main(void)
{
    int M_prob = (int)(sizeof(FIXTURES)/sizeof(FIXTURES[0]));
    int N = 64, n_vars = 1;
    int K_max = 0;
    for (int m = 0; m < M_prob; m++) if (FIXTURES[m].K > K_max) K_max = FIXTURES[m].K;

    // ---- 打平 fixture 到连续数组 + 各树的 meta(offset / n_nodes / c_offset / K)----
    TreeMeta h_metas[16];
    int total_nodes = 0, total_c = 0;
    for (int m = 0; m < M_prob; m++) {
        h_metas[m].node_offset = total_nodes;
        h_metas[m].n_nodes     = FIXTURES[m].n_nodes;
        h_metas[m].c_offset    = total_c;
        h_metas[m].K           = FIXTURES[m].K;
        total_nodes += FIXTURES[m].n_nodes;
        total_c     += FIXTURES[m].K;
    }

    int   *h_nt = (int*)malloc(total_nodes * sizeof(int));
    float *h_nv = (float*)malloc(total_nodes * sizeof(float));
    int   *h_ci = (int*)malloc(total_nodes * sizeof(int));
    float *h_call_init = (float*)malloc(total_c * sizeof(float));
    float *h_call_pert = (float*)malloc(total_c * sizeof(float));
    for (int m = 0; m < M_prob; m++) {
        memcpy(h_nt + h_metas[m].node_offset, FIXTURES[m].nt, FIXTURES[m].n_nodes*sizeof(int));
        memcpy(h_nv + h_metas[m].node_offset, FIXTURES[m].nv, FIXTURES[m].n_nodes*sizeof(float));
        memcpy(h_ci + h_metas[m].node_offset, FIXTURES[m].ci, FIXTURES[m].n_nodes*sizeof(int));
        // 在 c_init 处求值(和 s5/s6 同一个点)。c_init ≠ c_true, 起点残差 r 已非零。
        memcpy(h_call_init + h_metas[m].c_offset, FIXTURES[m].c_init, FIXTURES[m].K*sizeof(float));
    }

    float *h_xs = (float*)malloc(N * n_vars * sizeof(float));
    for (int i = 0; i < N; i++) h_xs[i] = 0.1f + (5.0f - 0.1f) * (float)i / (float)(N - 1);

    float *h_ymeas = (float*)malloc(M_prob * N * sizeof(float));
    for (int m = 0; m < M_prob; m++)
        for (int i = 0; i < N; i++)
            h_ymeas[m * N + i] = y_ref_eval(m, h_xs[i]);

    // ---- device 分配 ----
    int *d_nt; CUDA_CHECK(cudaMalloc(&d_nt, total_nodes*sizeof(int)));
    float *d_nv; CUDA_CHECK(cudaMalloc(&d_nv, total_nodes*sizeof(float)));
    int *d_ci; CUDA_CHECK(cudaMalloc(&d_ci, total_nodes*sizeof(int)));
    float *d_call; CUDA_CHECK(cudaMalloc(&d_call, total_c*sizeof(float)));
    TreeMeta *d_metas; CUDA_CHECK(cudaMalloc(&d_metas, M_prob*sizeof(TreeMeta)));
    float *d_xs; CUDA_CHECK(cudaMalloc(&d_xs, N*n_vars*sizeof(float)));
    float *d_ymeas; CUDA_CHECK(cudaMalloc(&d_ymeas, M_prob*N*sizeof(float)));
    float *d_y; CUDA_CHECK(cudaMalloc(&d_y, M_prob*N*sizeof(float)));
    float *d_yp; CUDA_CHECK(cudaMalloc(&d_yp, M_prob*N*sizeof(float)));
    float *d_r; CUDA_CHECK(cudaMalloc(&d_r, M_prob*N*sizeof(float)));
    float *d_J; CUDA_CHECK(cudaMalloc(&d_J, (size_t)M_prob*K_max*N*sizeof(float)));
    float *d_JtJ; CUDA_CHECK(cudaMalloc(&d_JtJ, (size_t)M_prob*K_max*K_max*sizeof(float)));
    float *d_JtR; CUDA_CHECK(cudaMalloc(&d_JtR, (size_t)M_prob*K_max*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_nt, h_nt, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv, h_nv, total_nodes*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_ci, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, M_prob*sizeof(TreeMeta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, N*n_vars*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ymeas, h_ymeas, M_prob*N*sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int wpb = block / 32;
    int grid = (M_prob + wpb - 1) / wpb;

    printf("s7: 装配 JᵀJ + JᵀR(warp-shfl 归约)\n");
    printf("  M_prob=%d, N=%d, K_max=%d, 在 c_init 求值(c_init ≠ c_true, 起点 r 已非零)\n\n", M_prob, N, K_max);

    // ---- 准备 J 和 r(s5 + s6 的旧活, 这里只是把上游凑齐)----
    // base: y_pred at c_init
    CUDA_CHECK(cudaMemcpy(d_call, h_call_init, total_c*sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());

    // r = y_pred - y_meas  (s5)
    int rblock = 128, rgrid = (M_prob * N + rblock - 1) / rblock;
    residual_kernel<<<rgrid, rblock>>>(d_y, d_ymeas, M_prob*N, d_r);
    CUDA_CHECK(cudaGetLastError());

    // J via FD (s6): J[m,k,:] = (y(c_init + eps·e_k) − y(c_init)) / eps, eps=1e-3。
    // host 端凑 J(和 s6/生产一致): K_max 次扰动 launch, 每次把差除以 eps 落到 h_J 对应行。
    const float eps_fd = 1e-3f;
    float *h_y_base = (float*)malloc(M_prob*N*sizeof(float));
    float *h_y_pert = (float*)malloc(M_prob*N*sizeof(float));
    float *h_J = (float*)calloc((size_t)M_prob*K_max*N, sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_y_base, d_y, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
    for (int k = 0; k < K_max; k++) {
        memcpy(h_call_pert, h_call_init, total_c*sizeof(float));
        for (int m = 0; m < M_prob; m++)
            if (k < h_metas[m].K) h_call_pert[h_metas[m].c_offset + k] += eps_fd;
        CUDA_CHECK(cudaMemcpy(d_call, h_call_pert, total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_y_pert, d_yp, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
        for (int m = 0; m < M_prob; m++) {
            if (k >= h_metas[m].K) continue;
            for (int i = 0; i < N; i++)
                h_J[(size_t)m*K_max*N + k*N + i] = (h_y_pert[m*N+i] - h_y_base[m*N+i]) / eps_fd;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_J, h_J, (size_t)M_prob*K_max*N*sizeof(float), cudaMemcpyHostToDevice));

    // ---- 这一级的主角: GPU 上装配 JᵀJ + JᵀR ----
    build_jtj_jtr_kernel<<<grid, block>>>(d_J, d_r, d_metas, M_prob, N, K_max, d_JtJ, d_JtR);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_JtJ = (float*)malloc(M_prob*K_max*K_max*sizeof(float));
    float *h_JtR = (float*)malloc(M_prob*K_max*sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_JtJ, d_JtJ, M_prob*K_max*K_max*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_JtR, d_JtR, M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));

    // 把 device 上的 r 拷回来, 让 oracle 用【同一份 J 和 r】(对拍只验装配那一步对不对)。
    float *h_r = (float*)malloc(M_prob*N*sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_r, d_r, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));

    // =======================================================================
    // 独立 CPU oracle: 直接 double-loop 从同一份 h_J、h_r 重算 JᵀJ/JᵀR。
    // 【刻意不照抄上面 device kernel 的形状】—— 不用对称性、不分上下三角, 老老实实
    // 把每个 (j1,j2) 都从头加一遍 N 项; JᵀR 也独立单独一层循环。这样若 device 端
    // 把对称镜像写错、或归约漏 lane, oracle 不会跟着错, 对拍才有意义。
    // =======================================================================
    int all_pass = 1;
    for (int m = 0; m < M_prob; m++) {
        int K_m = h_metas[m].K;

        float jtj_ref[MAX_K*MAX_K];
        float jtr_ref[MAX_K];
        // JᵀR[k] = Σ_i J[k,i]·r[i]
        for (int k = 0; k < K_m; k++) {
            float s = 0.0f;
            for (int i = 0; i < N; i++)
                s += h_J[(size_t)m*K_max*N + k*N + i] * h_r[m*N + i];
            jtr_ref[k] = s;
        }
        // JᵀJ[j1,j2] = Σ_i J[j1,i]·J[j2,i]  —— 整张矩阵都算(不抄对称那招)
        for (int j1 = 0; j1 < K_m; j1++)
            for (int j2 = 0; j2 < K_m; j2++) {
                float s = 0.0f;
                for (int i = 0; i < N; i++)
                    s += h_J[(size_t)m*K_max*N + j1*N + i] * h_J[(size_t)m*K_max*N + j2*N + i];
                jtj_ref[j1*K_max + j2] = s;
            }

        // ---- 对拍: device vs oracle, 相对误差 < 1e-4(_SPEC §4) ----
        float max_jtj_re = 0.0f, max_jtr_re = 0.0f;
        int mism = 0;
        for (int j1 = 0; j1 < K_m; j1++) {
            for (int j2 = 0; j2 < K_m; j2++) {
                float ref = jtj_ref[j1*K_max + j2];
                float got = h_JtJ[m*K_max*K_max + j1*K_max + j2];
                float re = fabsf(got - ref) / (fabsf(ref) + 1e-30f);
                if (re > max_jtj_re) max_jtj_re = re;
                if (re > 1e-4f) mism++;
            }
            float refr = jtr_ref[j1];
            float gotr = h_JtR[m*K_max + j1];
            float rer = fabsf(gotr - refr) / (fabsf(refr) + 1e-30f);
            if (rer > max_jtr_re) max_jtr_re = rer;
            if (rer > 1e-4f) mism++;
        }
        int total = K_m*K_m + K_m;
        printf("  [%d] %-13s K=%d  JtJ max_rel=%.2e  JtR max_rel=%.2e  mismatch=%d/%d  %s\n",
               m, FIXTURES[m].name, K_m, max_jtj_re, max_jtr_re, mism, total,
               mism == 0 ? "ok" : "MISMATCH");
        if (mism != 0) all_pass = 0;
    }

    // 抽样打印 m=0 (powerlaw_K2) 的 2×2 JᵀJ 和 JᵀR, 直观看一眼。
    printf("\n  sample [m=0 powerlaw_K2]:\n");
    printf("    JᵀJ = [%9.4f %9.4f]   JᵀR = [%9.4f]\n",
           h_JtJ[0*K_max + 0], h_JtJ[0*K_max + 1], h_JtR[0]);
    printf("          [%9.4f %9.4f]         [%9.4f]\n",
           h_JtJ[1*K_max + 0], h_JtJ[1*K_max + 1], h_JtR[1]);

    printf("\n%s\n", all_pass ? "PASS" : "FAIL");

    free(h_nt); free(h_nv); free(h_ci); free(h_call_init); free(h_call_pert);
    free(h_xs); free(h_ymeas); free(h_y_base); free(h_y_pert); free(h_J);
    free(h_JtJ); free(h_JtR); free(h_r);
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_ymeas); cudaFree(d_y); cudaFree(d_yp); cudaFree(d_r);
    cudaFree(d_J); cudaFree(d_JtJ); cudaFree(d_JtR);
    return all_pass ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. JᵀJ 为什么是 K×K(这里 2×2 或 3×3)而不是 N×N(64×64)?N 跑到哪去了?
//  2. "JᵀJ 对称, 只算上三角再镜像填两边" —— 省掉了哪一半的归约?为什么能这么省?
//     如果把内层循环写成 `for j2 = 0` 而不是 `for j2 = j1`, 结果会变吗?会变慢多少?
//  3. 这里 JᵀR[k] 的那次 shfl 归约, 求和的是【N 个数据点上 J[k,i]·r[i] 的乘积】;
//     s5 算 loss 那次 shfl 归约, 求和的是什么?两者是不是同一类活?
//  4. oracle 为什么【不照抄】device kernel 的对称那招, 偏要把整张 JᵀJ 都重算一遍?
//     如果 oracle 也只算上三角再镜像, 对拍还能查出"镜像那行写错列"这种 bug 吗?
//
// ## 参考答案
//
//  1. JᵀJ[j1,j2] = Σ_i J[j1,i]·J[j2,i] —— 下标 j1,j2 各跑遍 K 个常数, 所以矩阵是
//     K×K。N(数据点)是被【求和吃掉】的那个维度: 每个矩阵元素都是对 i=0..N-1
//     求一个和, N 进了求和号、不进矩阵尺寸。直觉: JᵀJ 是"K 个常数方向两两的内积",
//     和你有多少观测点无关, 只和有几个待定常数有关。
//
//  2. 省掉了【下三角】那一半的归约(j2 < j1 的那些 (j1,j2))。能省是因为 JᵀJ 对称:
//     JᵀJ[j1,j2] = Σ J[j1,i]·J[j2,i] = Σ J[j2,i]·J[j1,i] = JᵀJ[j2,j1], 上下三角值相同,
//     算一次写两个位置即可。改成 `for j2 = 0` 结果不变(值一样), 但每棵树多做了约一半的
//     归约(K=3 时: 上三角含对角是 6 个, 全矩阵是 9 个 —— 多做 3 个, 慢约 1.5×)。
//
//  3. s5 那次 shfl 归约求和的是【N 个数据点上 rᵢ²】(再乘 0.5 得 loss)。两者确实是同
//     一类活: 都是"warp 内 32 lane 各扛一摞数据点的部分和, 再用蝶形 shfl 把 32 个部分
//     和合成一个总和"。区别只在每个 lane 往部分和里加的那一项是什么(rᵢ² 还是 J·r 还是
//     J·J), 归约骨架一模一样 —— 所以 s5 学会的 shfl, s7 直接复用。
//
//  4. 因为 oracle 的职责是【独立地】给出正确答案, 好把 device 的 bug 照出来。如果 oracle
//     也照抄"上三角算一次、镜像填两边", 那 device 把镜像写成 `JJm[j2*K_max+j2]=s`(列写错)
//     这种 bug, oracle 会犯【同样】的错, 两边一致 -> 假 PASS, bug 漏网。oracle 老实把整张
//     矩阵每个元素都从 N 项独立加出来, 才不会和 device 共享同一个错误假设, 对拍才有效。
// ===========================================================================
