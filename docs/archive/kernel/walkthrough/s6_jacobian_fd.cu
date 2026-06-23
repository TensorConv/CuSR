// s6_jacobian_fd.cu — 有限差分 Jacobian + column-major-per-problem 的 J 布局。
//
// 这一级新增的【唯一概念】:Jacobian(雅可比)—— 也就是
//   "输出 y 对每个常数 c_k 的偏导" J[k][i] = ∂y_i / ∂c_k。
// 怎么算?不解析求导,而是用【有限差分 / finite-difference(FD)】:
//   把 c_k 挪一丁点 eps_fd,看 y 变了多少,除以 eps_fd —— 这就是导数的定义。
// 怎么存?把 J 排成 "column-major per problem":同一棵树 m、同一个常数 k 的
//   N 个点 J[k][0..N-1] 在内存里连成一段(下面解释为啥这么排)。
//
// 假设你已看过 s1..s5, 已经懂 residual / loss; 这一级只加 finite-difference
// Jacobian + column-major-per-problem J 布局。
//
// 先用大白话把名词铺一遍(define-before-use),后面才放符号:
//   - 常数 c_k:每棵树里待拟合的参数(powerlaw 的 c0/c1 那种),LM 要调它。
//   - 残差 residual r_i = y_pred_i - y_meas_i:第 i 个点预测和真值的差(s5 讲过)。
//   - Jacobian J:一张表,J[k][i] = "把 c_k 推一点点,第 i 个点的预测 y 会怎么变"。
//     LM 全靠它判断 "往哪个方向调 c 能让残差变小",是下一级 JᵀJ / JᵀR 的原料。
//   - 有限差分:导数 ∂y/∂c_k ≈ ( y(c_k+eps) − y(c) ) / eps,eps 取很小的数。
//     这是一阶前向差分(forward difference),抄 m3_3 用 eps_fd = 1e-3。
//
// 【为什么不写新 kernel】:算 y(c) 和 y(c+eps) 都只是 "在某组 c 下 eval 整批树",
//   正是 s4 的 eval_kernel_batched 干的事。所以 FD 的做法 = 多调几次同一个
//   eval_kernel_batched(1 次基准 + K_max 次扰动),J 的组装(减、除 eps)放在
//   host 上做。这一级不碰 GPU 上的新代码,delta 干净成一句:"多 launch 几次 + host 端减"。
//
// 【J 的内存布局:column-major per problem】:存成
//      J[m*K_max*N + k*N + i]    (m=哪棵树, k=哪个常数, i=哪个点)
//   含义是 "同一棵树、同一个 c_k 的 N 个点是连续的一整列"。为什么这么排?
//   下一级 s7 要算 JᵀJ_{k1,k2} = Σ_i J[k1][i]·J[k2][i],由一个 warp 的 32 个 lane
//   分摊 i:lane i 同时读 J[k1][i] 和 J[k2][i]。这个布局下 J[k*N + i] 里相邻 lane
//   (i, i+1) 读的是相邻地址(stride-1)—— 32 个 lane 的访存能合并成一次
//   (coalesced),带宽用满。若反过来按 row-major(J[i*K+k]),相邻 lane 隔着 K 个
//   float(stride-K),访存散开、合并不了。s7 选这个布局,根子就在这里。
//
// 跑: ./build.sh s6     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - 有限差分为啥用 "相对 1e-2" 这种宽容差,而不是 s2 那种 1e-5?
//   - J 存成 J[m*K_max*N + k*N + i] 而不是 J[m*N*K_max + i*K_max + k],
//     对 s7 warp 里 lane i 的访存有什么区别?
//   - 这一级在 c_init(LM 起点)处求 J,而不是 c_true,为什么?

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// 每个 CUDA API 调用都包它(全程约定 1)。
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

enum NType { N_VAR = 0, N_CONST = 1, N_UFUNC = 2, N_BFUNC = 3, N_TFUNC = 4 };
enum Func {
    F_ADD = 1, F_SUB = 2, F_MUL = 3, F_DIV = 4, F_POW = 6,
    F_SIN = 14, F_COS = 15, F_LOG = 20, F_EXP = 22, F_NEG = 25, F_SQRT = 27,
};
#define MAX_STACK 64    // 和生产 batch_lm.cu 一致(m3_2 用的是 32,这里统一 64)
#define MAX_K     32

// 每棵树在 "拼平了的大数组" 里的位置:节点从哪开始、几个节点、常数从哪开始、几个常数。
struct TreeMeta { int node_offset; int n_nodes; int c_offset; int K; };

// ---------------------------------------------------------------------------
// canonical 解释器 —— 和 s4/s5 一字不差(逆前缀 + 栈,s1 已讲透,这里不重复)。
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
// eval_kernel_batched —— 和 s4/s5 一字不差(warp-per-problem:一个 warp 一棵树,
// warp 内 32 个 lane 沿 i 分摊 N 个点)。FD 这一级【复用它】,不写新 kernel。
// ---------------------------------------------------------------------------
__global__ void eval_kernel_batched(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N, const float *c_all, float *y_out)
{
    int wpb  = blockDim.x / 32;     // BLOCK=256 → 8 warp = 8 棵树/block
    int wb   = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m    = blockIdx.x * wpb + wb;
    if (m >= M_prob) return;

    TreeMeta meta = metas[m];
    const int   *nt = nt_all + meta.node_offset;
    const float *nv = nv_all + meta.node_offset;
    const int   *ci = ci_all + meta.node_offset;
    const float *c  = c_all  + meta.c_offset;

    for (int i = lane; i < N; i += 32)
        y_out[m * N + i] = eval_tree_d(nt, nv, meta.n_nodes, ci, xs + i * n_vars, c);
}

// ---------------------------------------------------------------------------
// 三个 fixture(逐字复制自 _SPEC §3)。注意这一级用的是 c_init(c0),不是 c_true:
// LM 的第一张 Jacobian 就是在起点 c_init 处求的,所以这里也在 c_init 求,oracle
// 也用 c_init 代入解析导数(见下面 dy_dc_analytical)。
// ---------------------------------------------------------------------------
struct TreeSpec {
    const char *name;
    int n_nodes;
    const int   *nt;
    const float *nv;
    const int   *ci;
    int K;
    const float *c_init;
};

// powerlaw_K2: c0 * pow(x0, c1)
static const int   tree0_nt[5] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
static const float tree0_nv[5] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f };
static const int   tree0_ci[5] = { -1, 0, -1, -1, 1 };
static const float tree0_c0[2] = { 1.0f, 1.0f };   // c_init (LM 起点) —— 这一级在这求 J
// (c_true = {2.5, 1.3};这一级在 c_init 求 J,所以 c_true 这里用不上,略去免 unused 警告。)

// quadratic_K3: c0 + c1*x0 + c2*x0*x0
static const int   tree1_nt[11] = { N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR,
                                    N_BFUNC, N_CONST, N_BFUNC, N_VAR,   N_VAR };
static const float tree1_nv[11] = { F_ADD,   0.0f,    F_ADD,   F_MUL,   0.0f,    0.0f,
                                    F_MUL,   0.0f,    F_MUL,   0.0f,    0.0f };
static const int   tree1_ci[11] = { -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1 };
static const float tree1_c0[3]  = { 0.0f, 0.0f, 0.0f };    // c_init  (c_true = {0.5, -1.2, 0.3})

// expsat_K3: c0 * exp(neg(c1 * x0)) + c2
static const int   tree2_nt[9] = { N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC,
                                   N_CONST, N_VAR,   N_CONST };
static const float tree2_nv[9] = { F_ADD,   F_MUL,   0.0f,    F_EXP,   F_NEG,   F_MUL,
                                   0.0f,    0.0f,    0.0f };
static const int   tree2_ci[9] = { -1, -1, 0, -1, -1, -1, 1, -1, 2 };
static const float tree2_c0[3] = { 1.0f, 1.0f, 0.0f };     // c_init  (c_true = {3.0, 0.4, 0.7})

static const TreeSpec FIXTURES[] = {
    { "powerlaw_K2",  5,  tree0_nt, tree0_nv, tree0_ci, 2, tree0_c0 },
    { "quadratic_K3", 11, tree1_nt, tree1_nv, tree1_ci, 3, tree1_c0 },
    { "expsat_K3",    9,  tree2_nt, tree2_nv, tree2_ci, 3, tree2_c0 },
};

// ---------------------------------------------------------------------------
// 独立 oracle —— 故意【不】转写上面那段 FD device kernel,而是用【解析导数】:
// 手推每棵树 ∂y/∂c_k 的公式,代入 c_init 求值。两套方法(数值 FD vs 解析)都算
// 同一个量,对得上才说明 FD 真算对了(TDD 的核心:对照独立实现,不是自己抄自己)。
//
//   powerlaw  y = c0 * x^c1     ∂c0 = x^c1               ∂c1 = c0 * x^c1 * ln(x)
//   quadratic y = c0+c1·x+c2·x² ∂c0 = 1   ∂c1 = x        ∂c2 = x²
//   expsat    y = c0·e^(-c1·x)+c2  ∂c0 = e^(-c1·x)
//                                   ∂c1 = -c0·x·e^(-c1·x)  ∂c2 = 1
//   —— 这些导数代入的是 c_init(c0),和 FD 求 J 的点一致。
// ---------------------------------------------------------------------------
static float dy_dc_analytical(int fixture_idx, int k, float x) {
    if (fixture_idx == 0) {
        float c0 = tree0_c0[0], c1 = tree0_c0[1];   // c_init
        if (k == 0) return powf(x, c1);
        if (k == 1) return c0 * powf(x, c1) * logf(x);
    } else if (fixture_idx == 1) {
        if (k == 0) return 1.0f;
        if (k == 1) return x;
        if (k == 2) return x * x;
    } else if (fixture_idx == 2) {
        float c0 = tree2_c0[0], c1 = tree2_c0[1];   // c_init
        float e  = expf(-c1 * x);
        if (k == 0) return e;
        if (k == 1) return -c0 * x * e;
        if (k == 2) return 1.0f;
    }
    return 0.0f;
}

int main(void)
{
    int M_prob = (int)(sizeof(FIXTURES) / sizeof(FIXTURES[0]));
    int N      = 64;
    int n_vars = 1;

    // K_max = 各树 K 的最大值;FD 要按 k=0..K_max-1 逐方向扰动,K_max+1 次 launch。
    int K_max = 0;
    for (int m = 0; m < M_prob; m++) if (FIXTURES[m].K > K_max) K_max = FIXTURES[m].K;

    // ---- 0. 把三棵树拼平成大数组(节点、常数都首尾相接;metas 记每棵的偏移)----
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

    int   *h_nt = (int*)  malloc(total_nodes * sizeof(int));
    float *h_nv = (float*)malloc(total_nodes * sizeof(float));
    int   *h_ci = (int*)  malloc(total_nodes * sizeof(int));
    float *h_c_base    = (float*)malloc(total_c * sizeof(float));   // c_init,FD 的基准点
    float *h_c_perturb = (float*)malloc(total_c * sizeof(float));   // c_init + eps(某个方向)
    for (int m = 0; m < M_prob; m++) {
        memcpy(h_nt + h_metas[m].node_offset, FIXTURES[m].nt, FIXTURES[m].n_nodes * sizeof(int));
        memcpy(h_nv + h_metas[m].node_offset, FIXTURES[m].nv, FIXTURES[m].n_nodes * sizeof(float));
        memcpy(h_ci + h_metas[m].node_offset, FIXTURES[m].ci, FIXTURES[m].n_nodes * sizeof(int));
        memcpy(h_c_base + h_metas[m].c_offset, FIXTURES[m].c_init, FIXTURES[m].K * sizeof(float));
    }

    // 数据点 xs ∈ [0.1, 5.0] 均匀采 N=64(_SPEC §3 的公式)。
    float *h_xs = (float*)malloc(N * n_vars * sizeof(float));
    for (int i = 0; i < N; i++) h_xs[i] = 0.1f + (5.0f - 0.1f) * (float)i / (float)(N - 1);

    // ---- 1. device 缓冲 ----
    int      *d_nt;    CUDA_CHECK(cudaMalloc(&d_nt,    total_nodes * sizeof(int)));
    float    *d_nv;    CUDA_CHECK(cudaMalloc(&d_nv,    total_nodes * sizeof(float)));
    int      *d_ci;    CUDA_CHECK(cudaMalloc(&d_ci,    total_nodes * sizeof(int)));
    float    *d_call;  CUDA_CHECK(cudaMalloc(&d_call,  total_c * sizeof(float)));
    TreeMeta *d_metas; CUDA_CHECK(cudaMalloc(&d_metas, M_prob * sizeof(TreeMeta)));
    float    *d_xs;    CUDA_CHECK(cudaMalloc(&d_xs,    N * n_vars * sizeof(float)));
    float    *d_y;     CUDA_CHECK(cudaMalloc(&d_y,     M_prob * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_nt,    h_nt,    total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv,    h_nv,    total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,    h_ci,    total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, M_prob * sizeof(TreeMeta),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs,    h_xs,    N * n_vars * sizeof(float),  cudaMemcpyHostToDevice));

    int block = 256;
    int wpb   = block / 32;
    int grid  = (M_prob + wpb - 1) / wpb;

    const float eps_fd = 1e-3f;   // 抄 m3_3,不许改

    // ---- 2. baseline:在 c(= c_init)处 eval,得 yb ----
    CUDA_CHECK(cudaMemcpy(d_call, h_c_base, total_c * sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());
    float *h_yb = (float*)malloc(M_prob * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_yb, d_y, M_prob * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. 每个方向 k:c[k]+=eps_fd → 复用 eval 得 yp → J[m,k,i]=(yp-yb)/eps_fd ----
    // J 存 column-major per problem:J[m*K_max*N + k*N + i]。
    float *h_J = (float*)calloc((size_t)M_prob * K_max * N, sizeof(float));
    float *h_yp = (float*)malloc(M_prob * N * sizeof(float));

    for (int k = 0; k < K_max; k++) {
        // 这一轮只动 "第 k 个常数"。每棵树各自的第 k 个常数 +eps_fd(没有第 k 个的树跳过)。
        memcpy(h_c_perturb, h_c_base, total_c * sizeof(float));
        for (int m = 0; m < M_prob; m++) {
            if (k < h_metas[m].K) h_c_perturb[h_metas[m].c_offset + k] += eps_fd;
        }
        CUDA_CHECK(cudaMemcpy(d_call, h_c_perturb, total_c * sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_yp, d_y, M_prob * N * sizeof(float), cudaMemcpyDeviceToHost));

        for (int m = 0; m < M_prob; m++) {
            if (k >= h_metas[m].K) continue;     // 这棵树没有第 k 个常数,那一列留 0
            for (int i = 0; i < N; i++) {
                float dy = h_yp[m * N + i] - h_yb[m * N + i];
                h_J[(size_t)m * K_max * N + k * N + i] = dy / eps_fd;
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 4. 对拍:FD 的 J vs 独立解析导数(代入 c_init)----
    // 容差抄 m3_3:绝对 ae > 1e-2*|j_ref| + 1e-3 才算 mismatch(相对 1e-2 + 1e-3 绝对地板)。
    // 为啥这么宽?FD 是一阶差分,截断误差 ~ O(eps) 叠加 float32 舍入 ~ √eps ≈ 3e-4,
    // 天生就到不了 s2 那种 1e-5;m3_3 实测 max_rel ≈ 2.5e-3,1e-2 是留了余量的真容差,
    // 不是放松到恒过。绝对地板 1e-3 护着 |导数|≈0 的点(相对误差会爆,但绝对差很小)。
    printf("s6: FD Jacobian (forward, eps_fd=%.0e), 在 c_init 处求 J\n", eps_fd);
    printf("  M_prob=%d  N=%d  K_max=%d  →  K_max+1=%d 次 eval launch\n", M_prob, N, K_max, K_max + 1);
    printf("  J 布局: column-major per problem  J[m*K_max*N + k*N + i]\n");
    printf("  对拍容差: 相对 1e-2 + 绝对 1e-3(oracle = 解析导数,独立于 FD)\n\n");

    int all_pass = 1;
    for (int m = 0; m < M_prob; m++) {
        int K_m = h_metas[m].K;
        float max_ae = 0, max_re = 0;
        int mismatches = 0, total = 0;
        for (int k = 0; k < K_m; k++) {
            for (int i = 0; i < N; i++) {
                float j_gpu = h_J[(size_t)m * K_max * N + k * N + i];
                float j_ref = dy_dc_analytical(m, k, h_xs[i]);
                float ae = fabsf(j_gpu - j_ref);
                float re = ae / (fabsf(j_ref) + 1e-30f);
                if (ae > max_ae) max_ae = ae;
                if (re > max_re) max_re = re;
                if (ae > 1e-2f * fabsf(j_ref) + 1e-3f) mismatches++;
                total++;
            }
        }
        printf("  [%d] %-13s K=%d  max_abs=%.2e max_rel=%.2e  mismatches=%d/%d  %s\n",
               m, FIXTURES[m].name, K_m, max_ae, max_re, mismatches, total,
               mismatches == 0 ? "ok" : "MISMATCH");
        if (mismatches != 0) all_pass = 0;
    }

    // 抽一列给眼睛看:powerlaw 在中间那个点上,FD 的 ∂y/∂c_k vs 解析值。
    printf("\n  抽样 J [m=0 powerlaw, x=xs[N/2]=%.3f]:\n", h_xs[N/2]);
    for (int k = 0; k < h_metas[0].K; k++) {
        float gpu = h_J[(size_t)0 * K_max * N + k * N + N/2];
        float ref = dy_dc_analytical(0, k, h_xs[N/2]);
        printf("    ∂y/∂c%d: fd=%.6f  analytic=%.6f  diff=%.2e\n", k, gpu, ref, gpu - ref);
    }

    // ---- 5. 释放 ----
    CUDA_CHECK(cudaFree(d_nt)); CUDA_CHECK(cudaFree(d_nv)); CUDA_CHECK(cudaFree(d_ci));
    CUDA_CHECK(cudaFree(d_call)); CUDA_CHECK(cudaFree(d_metas));
    CUDA_CHECK(cudaFree(d_xs)); CUDA_CHECK(cudaFree(d_y));
    free(h_nt); free(h_nv); free(h_ci); free(h_c_base); free(h_c_perturb);
    free(h_xs); free(h_yb); free(h_yp); free(h_J);

    printf("\n%s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. 有限差分 ∂y/∂c_k ≈ (y(c_k+eps) − y(c)) / eps,这里 eps_fd=1e-3。为什么
//     对拍用 "相对 1e-2 + 绝对 1e-3" 这种宽容差,而不是 s2 那种 1e-5?
//  2. J 存成 J[m*K_max*N + k*N + i](同一个 c_k 的 N 个点连续)。若改存成
//     J[m*N*K_max + i*K_max + k](同一个点 i 的 K 个常数连续),s7 里一个 warp
//     的 lane i 沿 i 求和 Σ_i J[k1][i]·J[k2][i] 时,访存会变好还是变坏?
//  3. 这一级在 c_init(LM 起点,powerlaw 是 c0=1,c1=1)处求 J,而不是 c_true
//     (2.5, 1.3)。为什么?换成 c_true 会怎样?
//  4. K_max=3,但 powerlaw 只有 K=2。第 k=2 轮扰动时,powerlaw 那一列 J 是什么?
//     对拍会不会因为这一列出问题?
//
// ## 参考答案
//
//  1. 因为 FD 本身就有误差,不可能到 1e-5:一阶前向差分有 O(eps) 的截断误差
//     (用差商近似导数,丢了泰勒展开的高阶项),再叠上 float32 的舍入(y(c+eps) 和
//     y(c) 很接近,相减是 "大数减大数" 的灾难性抵消,有效位数掉一截)。两者折中的
//     最优 eps 大约在 √machine_eps ≈ 3e-4 量级,达到的相对精度也就这个量级。
//     m3_3 实测 max_rel ≈ 2.5e-3,所以容差定 1e-2(留了约 4 倍余量,既不会因为
//     FD 的固有误差误报 FAIL,也没松到 "随便什么都过")。绝对地板 1e-3 是给 |导数|
//     接近 0 的点兜底:那里相对误差会因为分母趋零而炸,但绝对差其实很小,不该算错。
//
//  2. 变坏。s7 让一个 warp 的 32 个 lane 按 i 分工(lane 0 拿 i=0、lane 1 拿
//     i=1…),同一时刻 lane i 要读 J[k1][i]。当前布局 J[k*N+i] 里,相邻 lane(i,
//     i+1)读的是相邻地址(stride-1),32 个 lane 的读能合并成一两次访存
//     (coalesced),带宽用满。换成 J[i*K_max+k] 后,相邻 lane 读的地址隔着 K_max
//     个 float(stride-K),32 个 lane 的请求散在不同 cache line 上,合并不了
//     (uncoalesced),有效带宽掉到 1/K。所以 s7 选 column-major-per-problem,
//     根子就是这里的 J 布局。
//
//  3. 因为 LM 是迭代法:它从起点 c_init 出发,在【当前 c】处求 Jacobian,据此走
//     一步,再在新 c 处重新求 J…… 第一张 Jacobian 就是在 c_init 处算的。这一级
//     是 "LM 第一步" 的零件,所以在 c_init 求,oracle 也必须代入 c_init(否则
//     FD 在 c_init 算、解析在 c_true 算,两边根本不是同一个点,对拍无意义)。
//     换成 c_true:powerlaw/expsat 的导数依赖 c,J 的数值会变(quadratic 的导数
//     1/x/x² 不含 c,换不换都一样)—— 只要 FD 和 oracle 用【同一个】c,仍会 PASS,
//     但那就不是 LM 实际要的第一张 J 了。
//
//  4. powerlaw 那一列(k=2)全是 0:第 3 步里 `if (k < h_metas[m].K)` 这一关让
//     powerlaw(K=2)在 k=2 轮【不被扰动】,h_c_perturb 等于基准,yp==yb,本可
//     算出 0;但更直接的是组装时 `if (k >= h_metas[m].K) continue` 直接跳过,那
//     一列保持 calloc 的 0。对拍不碰它:第 4 步循环只到 `k < K_m`(powerlaw 是
//     2),压根不看 k=2 那列。所以多出来的列既不影响别的树,也不进对拍 —— 这正是
//     "拼平成大数组 + 每棵树各看自己的 K" 这套 ragged batch 布局要处理的事。
// ===========================================================================
