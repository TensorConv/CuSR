// s8_cholesky_solve.cu — 每个问题解一个 K×K 线性方程组,求 LM 的一步 δ。
//
// 这一级新增的唯一概念:
//   对每棵树,把 s7 算好的 JᵀJ、JᵀR 组成一个 K×K 的线性方程组
//       (JᵀJ + λ·diag(JᵀJ)) · δ = -JᵀR
//   然后【在 GPU 上、1 个线程负责 1 个问题】用 Cholesky 分解把它解出来,
//   得到这一步要往哪挪常数:δ(下一级 s9 才真把 c += δ 迭代起来)。
//
// 假设你已看过 s1..s7,已经懂 s7(从 J 和 r 攒出每棵树的 K×K 矩阵 JᵀJ 和
// K 维向量 JᵀR)。这一级只加:把那个 K×K 方程组解掉,拿到 δ。
//
// 先用大白话把符号铺一遍(define-before-use,别假设是常识):
//   - K       这棵树有几个待拟合常数(powerlaw 2 个,quadratic/expsat 3 个)。
//   - J       Jacobian,N×K:第 i 行第 k 列 = "第 i 个数据点的残差,对第 k 个
//             常数的偏导"。s6 用有限差分(FD)算出来,s7 用它攒矩阵。
//   - r       残差向量,N 维:r[i] = y_pred[i] - y_measured[i]。
//   - JᵀJ     K×K 矩阵(s7 算的)。把它当"二阶信息"(海森的近似)就行。
//   - JᵀR     K 维向量(s7 算的)。它指向"loss 上升最快的方向",所以负号那侧
//             (-JᵀR)就是下降方向。
//   - λ       Marquardt 阻尼(damping)。λ 越大,这一步越保守(越像小步长的
//             梯度下降);λ 越小,越像大步的高斯-牛顿。这一级固定 λ=1e-3,
//             下一级 s9 才让 λ 随接受/拒绝自适应。
//   - δ       这一步的常数增量,K 维。解出来后(s9 里)做 c_new = c + δ。
//
// 为什么要加 λ·diag、为什么解的是 (JᵀJ+λ·diag)δ=-JᵀR 这个式子:
//   高斯-牛顿想解 JᵀJ·δ = -JᵀR(把 loss 局部当二次型,一步跳到谷底)。
//   但 JᵀJ 可能接近奇异(数值上解不动 / 步子乱飞)。Levenberg-Marquardt 的
//   办法是把对角线放大一点:A[j][j] *= (1+λ)。λ→0 退回高斯-牛顿;λ 很大时
//   A 近似对角占优、保证可解、且步子变小变稳。这一招既稳住了数值,又能在
//   "激进/保守"之间连续调节。这一级只解一步,体会"解出来的 δ 确实满足方程"。
//
// 为什么用 Cholesky(而不是一般的 LU / 高斯消元):
//   A = JᵀJ + λ·diag 是【对称正定】(SPD)的(JᵀJ 半正定,加正的对角更稳)。
//   对 SPD 矩阵,Cholesky 把 A 分解成 A = L·Lᵀ(L 是下三角),计算量约是
//   一般 LU 的一半,而且不用选主元。解 A·δ=b 就拆成两个三角方程:
//     先前代(forward)解 L·y = b,再回代(back)解 Lᵀ·δ = y。
//   退化情形:若分解中某个对角的平方项 s<=0,说明 A 在浮点下没保持正定
//   (λ 还不够大压住病态),这一步判失败:status=-1、δ=0。s9 的 LM 主循环
//   会把这种情况当"拒绝",调大 λ 再来。
//
// 为什么这一级换成【1 线程 1 problem】,而不是沿用 s4..s7 的 warp-per-problem:
//   s4..s7 那几步,每个问题的活儿是"在 N=64 个数据点上做归约求和"——N 远大于
//   一个常数维度,32 个 lane 并行扫 N 个点、再用 shfl 蝶形归约,很划算。
//   但解 K×K 方程是另一种活儿:K<=32(实际 2~3),而且 Cholesky 的前代/回代
//   是【强顺序依赖】的——解第 i 个未知数要先有第 0..i-1 个,拆不成 32 个独立
//   lane 同时干。硬塞进一个 warp 只会让 1 个 lane 干活、其余 31 个空转(warp
//   divergence + 浪费)。所以这里回到最朴素的映射:1 个线程独立解 1 个问题,
//   A[K*K]、b[K] 都放线程的局部数组里。M_prob 个问题之间本来就互相独立,线程
//   之间天然并行,这才是这步的正确粒度。
//
// 跑: ./build.sh s8     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - 为什么解的是 (JᵀJ+λ·diag)δ=-JᵀR,而不是直接 JᵀJ·δ=-JᵀR?λ 起什么作用?
//   - Cholesky 把 A 拆成 L·Lᵀ 后,怎么用两次三角回代解出 δ?
//   - 为什么这一步用 1 线程 1 problem,而 s5/s7 用 warp-per-problem?

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// 错误检查宏(全程约定:每个 CUDA API 调用都包它)
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

struct TreeMeta { int node_offset; int n_nodes; int c_offset; int K; };

// canonical 解释器(s4 起逐字用这一份)——算法 s1 已讲透,这里不再重复。
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
// 下面这几个 kernel 是 s4..s7 已经讲过的"前序流水线",这一级原样拿来用,只为
// 喂给新 kernel 一份真实的 JᵀJ / JᵀR。你已经懂它们了,这里只贴不再细讲:
//   eval_kernel_batched : 每个问题在 N 个点上求值 y_pred(warp-per-problem)
//   residual_kernel     : r = y_pred - y_measured
//   build_jtj_jtr_kernel: 从 J、r 攒出每棵树的 JᵀJ(K×K)、JᵀR(K)  ← s7
// ---------------------------------------------------------------------------
__global__ void eval_kernel_batched(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N,
    const float *c_all, float *y_out)
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
    r[tid] = yp[tid] - ym[tid];
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

// ===========================================================================
// 【本级新增 kernel】solve_kernel:每个问题解一个 K×K 方程组,1 线程 1 problem。
//
//   A = JᵀJ,且把对角放大成 Marquardt 阻尼:A[j][j] *= (1+λ)   ← (JᵀJ+λ·diag)
//   b = -JᵀR
//   解 A·δ = b,把 δ 写到 delta[m*K_max + 0..K-1]。
//
//   线程把整块 A[K*K]、b[K] 读进自己的局部数组(K<=32,放得下),原地做
//   Cholesky 分解 + 前代 + 回代。和生产 batch_lm.cu 的 solve_kernel 同算法,
//   也和 CPU 参考实现(lm_cpu chol_solve)同算法——只是这一级 λ 是个标量。
//
//   status[m]:0 = 解成功;-1 = 分解中对角 s<=0,A 在浮点下没保持正定(退化)。
// ===========================================================================
__global__ void solve_kernel(
    const float *JtJ, const float *JtR,
    const TreeMeta *metas, int M_prob, int K_max, float lambda,
    float *delta, int *status)
{
    // 1 线程 1 problem:不再分 warp / lane(K 太小,顺序回代也分不动,见 header)。
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_prob) return;
    int K = metas[m].K;

    // A、b 放线程的局部数组(MAX_K=32 → 最多 32*32 + 32 floats,够小,放得下)。
    float A[MAX_K * MAX_K];
    float b[MAX_K];

    // 装载:b = -JᵀR;A = JᵀJ,并把对角乘 (1+λ) 做 Marquardt 阻尼。
    //   注意这里把 A 紧密排成 K×K(行 j1 列 j2 → A[j1*K+j2]),而源 JᵀJ 是按
    //   K_max 跨距存的(JtJ[m*K_max*K_max + j1*K_max + j2])——别把 K 和 K_max 搞混。
    for (int j1 = 0; j1 < K; j1++) {
        b[j1] = -JtR[m*K_max + j1];
        for (int j2 = 0; j2 < K; j2++)
            A[j1*K + j2] = JtJ[m*K_max*K_max + j1*K_max + j2];
        A[j1*K + j1] *= (1.0f + lambda);   // Marquardt:放大对角
    }

    // Cholesky 原地分解:把下三角写成 L(A = L·Lᵀ)。
    //   对角 L[j][j] = sqrt( A[j][j] - Σ_{k<j} L[j][k]² )
    //   非对角 L[i][j] = ( A[i][j] - Σ_{k<j} L[i][k]·L[j][k] ) / L[j][j]
    for (int j = 0; j < K; j++) {
        float s = A[j*K + j];
        for (int k = 0; k < j; k++) s -= A[j*K+k] * A[j*K+k];
        if (s <= 0.0f) {                   // 没保持正定:这一步判失败
            status[m] = -1;
            for (int i = 0; i < K; i++) delta[m*K_max+i] = 0.0f;
            return;
        }
        A[j*K + j] = sqrtf(s);
        for (int i = j+1; i < K; i++) {
            float t = A[i*K + j];
            for (int k = 0; k < j; k++) t -= A[i*K+k] * A[j*K+k];
            A[i*K + j] = t / A[j*K + j];
        }
    }
    // 前代(forward):解 L·y = b,结果原地覆盖回 b。
    for (int i = 0; i < K; i++) {
        float s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*K+k] * b[k];
        b[i] = s / A[i*K + i];
    }
    // 回代(back):解 Lᵀ·δ = y,结果原地覆盖回 b。
    for (int i = K-1; i >= 0; i--) {
        float s = b[i];
        for (int k = i+1; k < K; k++) s -= A[k*K+i] * b[k];
        b[i] = s / A[i*K + i];
    }
    // 写出 δ
    for (int i = 0; i < K; i++) delta[m*K_max + i] = b[i];
    status[m] = 0;
}

// ====== 三个 fixture(逐字来自 _SPEC,和生产对齐)======
struct TreeSpec { const char *name; int n_nodes; const int *nt; const float *nv; const int *ci; int K; const float *c_true; };

// powerlaw_K2: c0 * pow(x0, c1)  →  prefix MUL c0 POW x0 c1
static const int   tree0_nt[5] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
static const float tree0_nv[5] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f };
static const int   tree0_ci[5] = { -1, 0, -1, -1, 1 };
static const float tree0_ct[2] = { 2.5f, 1.3f };   // c_true

// quadratic_K3: c0 + c1*x0 + c2*x0*x0  →  prefix ADD c0 ADD MUL c1 x0 MUL c2 MUL x0 x0
static const int   tree1_nt[11] = { N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR,
                                    N_BFUNC, N_CONST, N_BFUNC, N_VAR,   N_VAR };
static const float tree1_nv[11] = { F_ADD,   0.0f,    F_ADD,   F_MUL,   0.0f,    0.0f,
                                    F_MUL,   0.0f,    F_MUL,   0.0f,    0.0f };
static const int   tree1_ci[11] = { -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1 };
static const float tree1_ct[3]  = { 0.5f, -1.2f, 0.3f };   // c_true

// expsat_K3: c0 * exp(neg(c1 * x0)) + c2  →  prefix ADD MUL c0 EXP NEG MUL c1 x0 c2
static const int   tree2_nt[9] = { N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC,
                                   N_CONST, N_VAR,   N_CONST };
static const float tree2_nv[9] = { F_ADD,   F_MUL,   0.0f,    F_EXP,   F_NEG,   F_MUL,
                                   0.0f,    0.0f,    0.0f };
static const int   tree2_ci[9] = { -1, -1, 0, -1, -1, -1, 1, -1, 2 };
static const float tree2_ct[3] = { 3.0f, 0.4f, 0.7f };     // c_true

static const TreeSpec FIXTURES[] = {
    {"powerlaw_K2", 5,  tree0_nt, tree0_nv, tree0_ci, 2, tree0_ct},
    {"quadratic_K3",11, tree1_nt, tree1_nv, tree1_ci, 3, tree1_ct},
    {"expsat_K3",   9,  tree2_nt, tree2_nv, tree2_ci, 3, tree2_ct},
};

// 每棵树的解析值(独立 oracle,和 _SPEC 三式一致)
static float y_ref_eval(int fid, float x) {
    if (fid == 0) return tree0_ct[0] * powf(x, tree0_ct[1]);
    if (fid == 1) return tree1_ct[0] + tree1_ct[1]*x + tree1_ct[2]*x*x;
    if (fid == 2) return tree2_ct[0] * expf(-tree2_ct[1]*x) + tree2_ct[2];
    return 0.0f;
}

// ---------------------------------------------------------------------------
// 独立 CPU oracle:高斯消元(带部分主元),【独立于 Cholesky】。
//   解同一个 K×K 系统 A·δ = b(A 已经含 (1+λ) 阻尼),拿到 δ_gauss。
//   故意用和 device 不同的算法:device 走 Cholesky,oracle 走高斯消元。
//   返回 0=解出,-1=主元几乎为 0(奇异)。
// ---------------------------------------------------------------------------
static int gauss_solve(const double *A_in, const double *b_in, int K, double *x_out) {
    double A[MAX_K*MAX_K], b[MAX_K];
    for (int i = 0; i < K*K; i++) A[i] = A_in[i];
    for (int i = 0; i < K; i++)   b[i] = b_in[i];

    for (int col = 0; col < K; col++) {
        // 部分主元:在 col..K-1 行里找该列绝对值最大的当主元行,换上来
        int piv = col;
        double best = fabs(A[col*K+col]);
        for (int r = col+1; r < K; r++) {
            double v = fabs(A[r*K+col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-30) return -1;       // 主元太小,视作奇异
        if (piv != col) {
            for (int c = 0; c < K; c++) { double t=A[col*K+c]; A[col*K+c]=A[piv*K+c]; A[piv*K+c]=t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        // 消元:用主元行把下面各行该列清零
        for (int r = col+1; r < K; r++) {
            double f = A[r*K+col] / A[col*K+col];
            for (int c = col; c < K; c++) A[r*K+c] -= f * A[col*K+c];
            b[r] -= f * b[col];
        }
    }
    // 回代
    for (int i = K-1; i >= 0; i--) {
        double s = b[i];
        for (int c = i+1; c < K; c++) s -= A[i*K+c] * x_out[c];
        x_out[i] = s / A[i*K+i];
    }
    return 0;
}

int main() {
    int M_prob = (int)(sizeof(FIXTURES)/sizeof(FIXTURES[0]));
    int N = 64, n_vars = 1;
    int K_max = 0;
    for (int m = 0; m < M_prob; m++) if (FIXTURES[m].K > K_max) K_max = FIXTURES[m].K;

    // ---- meta + 打平的树数组(和 s4..s7 一样的批量布局)----
    TreeMeta h_metas[16];
    int total_nodes = 0, total_c = 0;
    for (int m = 0; m < M_prob; m++) {
        h_metas[m] = {total_nodes, FIXTURES[m].n_nodes, total_c, FIXTURES[m].K};
        total_nodes += FIXTURES[m].n_nodes;
        total_c += FIXTURES[m].K;
    }
    int *h_nt = (int*)malloc(total_nodes*sizeof(int));
    float *h_nv = (float*)malloc(total_nodes*sizeof(float));
    int *h_ci = (int*)malloc(total_nodes*sizeof(int));
    float *h_ct = (float*)malloc(total_c*sizeof(float));   // c_test(LM 当前点)
    float *h_pt = (float*)malloc(total_c*sizeof(float));   // c 扰动副本(FD 用)
    for (int m = 0; m < M_prob; m++) {
        memcpy(h_nt+h_metas[m].node_offset, FIXTURES[m].nt, FIXTURES[m].n_nodes*sizeof(int));
        memcpy(h_nv+h_metas[m].node_offset, FIXTURES[m].nv, FIXTURES[m].n_nodes*sizeof(float));
        memcpy(h_ci+h_metas[m].node_offset, FIXTURES[m].ci, FIXTURES[m].n_nodes*sizeof(int));
        // c_test = c_true*0.9:故意从真值偏一点,这样残差非零、δ 才有意义
        for (int k = 0; k < FIXTURES[m].K; k++)
            h_ct[h_metas[m].c_offset+k] = FIXTURES[m].c_true[k] * 0.9f;
    }

    // 数据点 xs ∈ [0.1, 5.0] 均匀 N=64;y_measured = 各树在 c_true 下的解析值
    float *h_xs = (float*)malloc(N*n_vars*sizeof(float));
    for (int i = 0; i < N; i++) h_xs[i] = 0.1f + (5.0f-0.1f)*i/(N-1);
    float *h_ym = (float*)malloc(M_prob*N*sizeof(float));
    for (int m = 0; m < M_prob; m++)
        for (int i = 0; i < N; i++) h_ym[m*N+i] = y_ref_eval(m, h_xs[i]);

    // ---- device 缓冲 ----
    int *d_nt; float *d_nv; int *d_ci; float *d_call; TreeMeta *d_metas;
    float *d_xs, *d_ym, *d_y, *d_yp, *d_r, *d_J, *d_JtJ, *d_JtR, *d_delta;
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
    CUDA_CHECK(cudaMalloc(&d_stat, M_prob*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nt, h_nt, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv, h_nv, total_nodes*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_ci, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, M_prob*sizeof(TreeMeta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, N*n_vars*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ym, h_ym, M_prob*N*sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, wpb = block/32;
    int grid = (M_prob + wpb - 1) / wpb;

    // ===== 前序流水线(s4..s7),把 JᵀJ / JᵀR 攒出来喂给新 kernel =====
    // (1) y_pred @ c_test
    CUDA_CHECK(cudaMemcpy(d_call, h_ct, total_c*sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());
    // (2) r = y_pred - y_measured
    residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
    CUDA_CHECK(cudaGetLastError());

    // (3) J via FD(ε=1e-3,host 端定稿——这是 s6 已讲的有限差分)
    float *h_yb = (float*)malloc(M_prob*N*sizeof(float));
    float *h_yp = (float*)malloc(M_prob*N*sizeof(float));
    float *h_J  = (float*)calloc((size_t)M_prob*K_max*N, sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_yb, d_y, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
    const float eps = 1e-3f;
    for (int k = 0; k < K_max; k++) {
        memcpy(h_pt, h_ct, total_c*sizeof(float));
        for (int m = 0; m < M_prob; m++)
            if (k < h_metas[m].K) h_pt[h_metas[m].c_offset+k] += eps;
        CUDA_CHECK(cudaMemcpy(d_call, h_pt, total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_yp, d_yp, M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
        for (int m = 0; m < M_prob; m++) {
            if (k >= h_metas[m].K) continue;
            for (int i = 0; i < N; i++)
                h_J[(size_t)m*K_max*N + k*N + i] = (h_yp[m*N+i] - h_yb[m*N+i]) / eps;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_J, h_J, (size_t)M_prob*K_max*N*sizeof(float), cudaMemcpyHostToDevice));

    // (4) JᵀJ + JᵀR  ← 这是 s7
    build_jtj_jtr_kernel<<<grid, block>>>(d_J, d_r, d_metas, M_prob, N, K_max, d_JtJ, d_JtR);
    CUDA_CHECK(cudaGetLastError());

    // ===== 本级新增:K×K Cholesky solve,λ=1e-3 =====
    const float lambda = 1e-3f;
    int sblock = 256;
    int sgrid = (M_prob + sblock - 1) / sblock;   // 1 线程 1 problem
    solve_kernel<<<sgrid, sblock>>>(d_JtJ, d_JtR, d_metas, M_prob, K_max, lambda, d_delta, d_stat);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 把 JᵀJ / JᵀR / δ / status 拿回来对拍
    float *h_JtJ = (float*)malloc(M_prob*K_max*K_max*sizeof(float));
    float *h_JtR = (float*)malloc(M_prob*K_max*sizeof(float));
    float *h_delta = (float*)malloc(M_prob*K_max*sizeof(float));
    int *h_stat = (int*)malloc(M_prob*sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_JtJ, d_JtJ, M_prob*K_max*K_max*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_JtR, d_JtR, M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta, d_delta, M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stat, d_stat, M_prob*sizeof(int), cudaMemcpyDeviceToHost));

    // ===== 对拍 =====
    // 容差(_SPEC §4):线性方程组残差  ‖A·δ - b‖ / ‖b‖ < 1e-4。
    // 并同时和独立的高斯消元解 δ_gauss 比一下方向是否一致(双保险,但 PASS 门
    // 卡的是残差这一项——它直接验"GPU 解出来的 δ 真满足方程")。
    printf("s8: 每问题 K×K Cholesky solve (Marquardt 阻尼, λ=%.0e), 1 线程 1 problem\n", lambda);
    printf("    对拍: GPU(Cholesky) δ  vs  独立 CPU 高斯消元; 容差 ‖Aδ-b‖/‖b‖ < 1e-4\n\n");

    const float TOL = 1e-4f;
    int all_pass = 1;
    for (int m = 0; m < M_prob; m++) {
        int K = h_metas[m].K;

        // 在 host 上重建这道题的 A(含 (1+λ) 阻尼)和 b——用 double 提精度
        double A[MAX_K*MAX_K], b[MAX_K];
        for (int i = 0; i < K; i++) {
            b[i] = -(double)h_JtR[m*K_max + i];
            for (int j = 0; j < K; j++) {
                double Aij = (double)h_JtJ[m*K_max*K_max + i*K_max + j];
                if (i == j) Aij *= (1.0 + (double)lambda);
                A[i*K + j] = Aij;
            }
        }

        if (h_stat[m] != 0) {
            printf("  [%d] %-13s K=%d  GPU SOLVE FAIL (status=%d, non-SPD)\n",
                   m, FIXTURES[m].name, K, h_stat[m]);
            all_pass = 0;
            continue;
        }

        // (a) 主验:GPU 的 δ 代回方程,看残差 ‖A·δ - b‖_∞ / ‖b‖₂
        double max_lin_res = 0.0, b_norm_sq = 0.0;
        for (int i = 0; i < K; i++) {
            double Ad = 0.0;
            for (int j = 0; j < K; j++) Ad += A[i*K+j] * (double)h_delta[m*K_max+j];
            double ri = Ad - b[i];
            if (fabs(ri) > max_lin_res) max_lin_res = fabs(ri);
            b_norm_sq += b[i]*b[i];
        }
        double b_norm = sqrt(b_norm_sq);
        double rel = max_lin_res / (b_norm + 1e-30);

        // (b) 旁验:独立高斯消元解 δ_gauss,和 GPU 的 δ 逐元素比
        double xg[MAX_K];
        int g_ok = gauss_solve(A, b, K, xg);
        double max_dd = 0.0, dnorm_sq = 0.0;
        for (int i = 0; i < K; i++) {
            double dd = fabs((double)h_delta[m*K_max+i] - xg[i]);
            if (dd > max_dd) max_dd = dd;
            dnorm_sq += xg[i]*xg[i];
        }
        double dnorm = sqrt(dnorm_sq);
        double delta_rel = max_dd / (dnorm + 1e-30);

        int pass_m = (g_ok == 0) && (rel < TOL);
        if (!pass_m) all_pass = 0;

        printf("  [%d] %-13s K=%d  ‖Aδ-b‖/‖b‖ = %.2e   ‖δ_gpu-δ_gauss‖/‖δ‖ = %.2e   %s\n",
               m, FIXTURES[m].name, K, rel, delta_rel, pass_m ? "PASS" : "FAIL");
        printf("        δ_gpu   = [");
        for (int i = 0; i < K; i++) printf("%s%.6f", i?", ":"", h_delta[m*K_max+i]);
        printf("]\n        δ_gauss = [");
        for (int i = 0; i < K; i++) printf("%s%.6f", i?", ":"", xg[i]);
        printf("]\n");
    }

    printf("\n%s\n", all_pass ? "PASS" : "FAIL");

    free(h_nt); free(h_nv); free(h_ci); free(h_ct); free(h_pt);
    free(h_xs); free(h_ym); free(h_yb); free(h_yp); free(h_J);
    free(h_JtJ); free(h_JtR); free(h_delta); free(h_stat);
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_ym); cudaFree(d_y); cudaFree(d_yp); cudaFree(d_r);
    cudaFree(d_J); cudaFree(d_JtJ); cudaFree(d_JtR); cudaFree(d_delta); cudaFree(d_stat);
    return all_pass ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. 这一级解的是 (JᵀJ + λ·diag)·δ = -JᵀR。如果把 λ 设成 0,解的是什么?
//     λ 很大(比如 1e6)时,δ 会变长还是变短、方向更像梯度下降还是高斯-牛顿?
//  2. Cholesky 把 A 拆成 L·Lᵀ。拿到 L 后,为什么解 A·δ=b 要做两次三角回代
//     (先 L·y=b 再 Lᵀ·δ=y)?如果分解时某个对角 s<=0,代码怎么处理、为什么?
//  3. s5/s7 用 warp-per-problem(32 个 lane 协作 + shfl 归约),这一级却用
//     1 线程 1 problem。换粒度的理由是什么?(提示:这步的活儿规模 + 顺序依赖)
//  4. 对拍里既有"‖Aδ-b‖/‖b‖"又有"和高斯消元解的逐元素差",为什么要两个?
//     哪一个才是 PASS 的判据,另一个起什么作用?
//
// ## 参考答案
//
//  1. λ=0 时解的就是高斯-牛顿方程 JᵀJ·δ=-JᵀR(纯二阶,一步跳到局部二次近似的
//     谷底,但 JᵀJ 病态时会乱飞)。λ 很大时,A≈(1+λ)·diag(JᵀJ)、近似对角阵,
//     δ ≈ -JᵀR / ((1+λ)·对角) —— 步子变【短】,方向更像【梯度下降】(沿 -JᵀR
//     按对角缩放)。所以 λ 是在"激进的高斯-牛顿"和"保守的梯度下降小步"之间
//     连续调节的旋钮;s9 会让它随接受/拒绝自适应。
//
//  2. 因为 A=L·Lᵀ,A·δ=b 就是 L·(Lᵀ·δ)=b。先把 Lᵀ·δ 当作未知向量 y:解
//     L·y=b(L 是下三角,从上往下【前代】,一个未知数一行顺次解出);再解
//     Lᵀ·δ=y(Lᵀ 是上三角,从下往上【回代】)。两次三角解都是 O(K²),比直接
//     求逆稳又快。若某对角 s<=0,说明 A 在浮点下没保持对称正定(λ 不够大压不住
//     病态),sqrt 会出 NaN —— 代码立刻判退化:status[m]=-1、δ=0 全置零,交给
//     s9 的 LM 主循环当"拒绝",调大 λ 重来。
//
//  3. 两个理由叠加。规模:这步每个问题只是个 K×K(K=2~3,上限 32)的小方程,
//     不像 s5/s7 那样要在 N=64 个数据点上做归约——没有"几十个点"可以分给 32 个
//     lane。顺序依赖:Cholesky 的前代/回代,解第 i 个未知数必须先有第 0..i-1 个,
//     是强顺序的,拆不成 32 个独立 lane 并行。硬塞进一个 warp 只会 1 个 lane 干活、
//     其余 31 个空转(divergence + 浪费占用)。问题之间本就独立,所以正确粒度是
//     1 线程独立解 1 个问题,靠 M_prob 个线程之间并行拿满 GPU。
//
//  4. "‖Aδ-b‖/‖b‖"验的是【GPU 解出来的 δ 是否真满足这个线性方程组】——把 δ 代回
//     原方程看残差,这是 PASS 的判据(_SPEC §4 容差 1e-4)。"和高斯消元的逐元素
//     差"是【独立交叉验证】:高斯消元是另一套算法(不是 Cholesky 的转写),若两条
//     完全独立的路径解出同一个 δ,就更确信不是两边写错了同一个 bug。残差小但解错
//     的情况理论上能被这条旁验抓出来(虽然对良态 SPD 系统两者本就该一致)。
// ===========================================================================
