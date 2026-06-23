// s5_residual_loss.cu — 从 y_pred 算 residual 和 loss,落地 warp 内 shfl 归约 (L09)。
//
// 这一级新增的【唯一概念】:
//   warp 内用 __shfl_xor_sync 做"蝶形归约"(butterfly reduction)把 32 个 lane
//   各自的部分和汇成一个总和 —— 不碰 shared memory、不 __syncthreads。
//
// 开场白:假设你已看过 s1..s4,已经懂 warp-per-problem 算出 y_pred[M][N]
//        (一个 warp 32 个 lane 协同把一棵树的 N 个点算完);这一级只加
//        "拿 y_pred 算 residual 和 loss,并用 shfl 在 warp 内求和"。
//
// 先用大白话把三个词铺一遍(define-before-use):
//   - 实测值 ym (measured):你手上的"真实数据点"。这里没有真传感器,我们拿
//     每棵树在【真常数 c_true】处的解析值当作模拟实测 —— 就当这是观测到的 y。
//   - 预测值 yp (predicted):树在【当前常数 c_init】处算出来的 y。LM 的起点
//     c_init 故意离 c_true 很远,所以 yp 和 ym 对不上。
//   - residual(残差) r = yp - ym:每个数据点上"预测差了多少"。N 个点 → N 个 r。
//   - loss(损失) = 0.5 * Σ r²:把 N 个残差平方加起来再乘 0.5,一棵树一个标量。
//     乘 0.5 是惯例(对 c 求导时那个 2 正好被约掉,后面 s7 会用到)。LM 干的事
//     就是反复调 c 把这个 loss 往 0 推。
//
// 这一级只加两个 kernel,eval_kernel_batched 原样沿用 s4:
//   (a) residual_kernel:1 线程 1 元素,r[t] = yp[t] - ym[t]。最朴素,没归约。
//   (b) loss_kernel:warp-per-problem,每个 lane 先把自己负责的那几个 r² 累成
//       局部和 s,再用 shfl 蝶形归约把 32 个 s 汇成一个,lane0 写 loss=0.5*s。
//
// 为什么 warp 内求和要用 shfl,而不是 shared memory(本级重点):
//   传统 shared-memory 归约:32 个 lane 先把局部和写进一块 __shared__ 数组,
//   __syncthreads() 等所有人写完,再树状两两相加,每一轮都要再 __syncthreads()。
//   代价 = 一次 shared 写 + 一次读 + 多次全 block 同步。
//   shfl 归约:__shfl_xor_sync 让一个 lane 直接读【同 warp 另一个 lane 寄存器里
//   的值】,不经过任何内存。也用不到 __syncthreads —— 那是【整个 block】的栅栏,
//   而 warp 归约只需要 warp 范围内协调,作用域对不上(根本不该拿 block 栅栏来同步
//   一个 warp)。同步从哪来?就藏在 __shfl_xor_sync 名字里的 _sync 后缀:它带的
//   mask=0xFFFFFFFF 表示"这 32 个 lane 要先到齐再 shuffle",同步由这条指令自己保证
//   (注:从 Volta/sm_70 起有 Independent Thread Scheduling,warp 不再保证 lockstep,
//   所以必须用带 mask 的 _sync 版,不能靠"隐式同步"——老的无 mask __shfl_xor 已弃用)。
//   省下的就是那块 shared memory + block 级同步开销。
//   (这是 L09 warp primitives 的核心:warp 内通信走寄存器,别绕内存。)
//
// 蝶形(xor)归约怎么走(off = 16,8,4,2,1):
//   每一轮,lane i 和 lane (i xor off) 交换各自的 s 并相加。
//   off=16: lane0<->16, lane1<->17, ... 16 对同时换,每个 lane 的 s 变成两人之和;
//   off=8 : 再把相邻 8 隔的两人合并 ... 5 轮后,【所有 32 个 lane 的 s 都等于总和】。
//   我们只要 lane0 那一份去写 loss(其它 lane 的总和一样,丢掉即可)。
//
// 怎么跑: ./build.sh s5     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - residual 和 loss 各是什么?loss 前面那个 0.5 是干嘛的?
//   - __shfl_xor_sync 一行做了什么?为什么 warp 内归约用它比 shared memory 省?
//   - loss_kernel 里 5 轮 shfl 跑完,32 个 lane 的 s 都是总和,为什么只让 lane0 写?

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

// ---------------------------------------------------------------------------
// canonical eval_tree_d —— s4 起逐字这一份,算法 s1 已懂(逆前缀 + 栈),不再赘述。
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
// eval_kernel_batched —— 和 s4 完全一样(warp-per-problem),这一级原样沿用。
//   一个 warp 协同算一棵树:warp 内 32 个 lane 在 N 个点上 strided loop,
//   y_out[m*N + i] = eval(...)。命名/布局都和生产 batch_lm.cu 对齐。
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
// (a) residual_kernel —— 最朴素的逐元素 kernel:1 线程 1 个数据点,没有归约。
//   总共 M_prob*N 个元素拉成一条直线,tid 从 0 到 total-1 各管一个。
//   r[tid] = yp[tid] - ym[tid]。
//   ym 这里是 per-tree 排成 [M_prob*N](生产里真 EvoGP dump 把同一份 ym 复制
//   到每棵树名下,布局就是这样,我们对齐它)。
// ---------------------------------------------------------------------------
__global__ void residual_kernel(const float *yp, const float *ym, int total, float *r) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    r[tid] = yp[tid] - ym[tid];
}

// ---------------------------------------------------------------------------
// (b) loss_kernel —— warp-per-problem + shfl 归约,本级新概念落地处。
//   每棵树一个 warp。先让每个 lane 把自己 strided 负责的那几个 r² 累成局部和 s;
//   再用 5 轮 __shfl_xor_sync 把 32 个 lane 的 s 汇成总和;lane0 写 0.5*总和。
// ---------------------------------------------------------------------------
__global__ void loss_kernel(const float *r, int M, int N, float *loss) {
    int wpb  = blockDim.x / 32;
    int wb   = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int m    = blockIdx.x * wpb + wb;
    if (m >= M) return;

    // 1) 每个 lane 先算自己那一份:i = lane, lane+32, lane+64 ... 把 r² 累进 s。
    float s = 0;
    for (int i = lane; i < N; i += 32) {
        float r_ = r[m*N + i];
        s += r_ * r_;
    }

    // 2) warp 内蝶形归约:这一行就是本级的重点。
    //    __shfl_xor_sync(mask, s, off) = "把【lane (myid xor off)】寄存器里的 s
    //    取过来"。mask=0xFFFFFFFF 表示 warp 里 32 个 lane 全参与。off 从 16 减半
    //    到 1,5 轮后每个 lane 的 s 都等于这棵树 N 个 r² 的总和 —— 全程不碰内存、
    //    不 __syncthreads —— warp 范围的同步由 __shfl_xor_sync 的 _sync 语义 + mask
    //    自带(注意 Volta+ 起 warp 已不保证 lockstep,正因如此才必须用 _sync 版)。
    //    这就是它比 shared 归约省的地方。
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);

    // 3) 32 个 lane 的 s 现在都是总和,挑 lane0 写一次就够(其它写也对、纯浪费)。
    if (lane == 0) loss[m] = 0.5f * s;
}

// ============================================================
// Host:3 个 archetype fixture(逐字来自 m3_2 / _SPEC)。
//   tree*_ct = c_true(算 ym:模拟实测);tree*_c0 = c_init(LM 起点,算 yp)。
// ============================================================

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

struct TreeSpec {
    const char *name; int n_nodes;
    const int *nt; const float *nv; const int *ci;
    int K; const float *c_true; const float *c_init;
};
static const TreeSpec FIXTURES[] = {
    { "powerlaw_K2",  5,  tree0_nt, tree0_nv, tree0_ci, 2, tree0_ct, tree0_c0 },
    { "quadratic_K3", 11, tree1_nt, tree1_nv, tree1_ci, 3, tree1_ct, tree1_c0 },
    { "expsat_K3",    9,  tree2_nt, tree2_nv, tree2_ci, 3, tree2_ct, tree2_c0 },
};

// ---------------------------------------------------------------------------
// 独立 oracle:解析公式算 y(完全不经过 device 解释器)。
//   ym = y(c_true)(模拟实测);yp_oracle = y(c_init)(预测)。
//   用哪组常数由调用方传进来 → 同一个函数既能算 ym 也能算 yp。
// ---------------------------------------------------------------------------
static float y_analytic(int fixture_idx, float x, const float *c) {
    switch (fixture_idx) {
        case 0: return c[0] * powf(x, c[1]);              // powerlaw
        case 1: return c[0] + c[1] * x + c[2] * x * x;    // quadratic
        case 2: return c[0] * expf(-c[1] * x) + c[2];     // expsat
        default: return 0.0f;
    }
}

int main(void)
{
    int M_prob = (int)(sizeof(FIXTURES) / sizeof(FIXTURES[0]));
    int N      = 64;
    int n_vars = 1;

    // ---- pack metas + 拼接 node 数组 + 拼接 c_init(yp 在 c_init 处算)----
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

    int   *h_nt   = (int*)  malloc((size_t)total_nodes * sizeof(int));
    float *h_nv   = (float*)malloc((size_t)total_nodes * sizeof(float));
    int   *h_ci   = (int*)  malloc((size_t)total_nodes * sizeof(int));
    float *h_call = (float*)malloc((size_t)total_c     * sizeof(float));  // = c_init
    for (int m = 0; m < M_prob; m++) {
        memcpy(h_nt   + h_metas[m].node_offset, FIXTURES[m].nt, FIXTURES[m].n_nodes * sizeof(int));
        memcpy(h_nv   + h_metas[m].node_offset, FIXTURES[m].nv, FIXTURES[m].n_nodes * sizeof(float));
        memcpy(h_ci   + h_metas[m].node_offset, FIXTURES[m].ci, FIXTURES[m].n_nodes * sizeof(int));
        memcpy(h_call + h_metas[m].c_offset,    FIXTURES[m].c_init, FIXTURES[m].K * sizeof(float));
    }

    // ---- xs ∈ [0.1, 5.0] 均匀采 N 点 ----
    float *h_xs = (float*)malloc((size_t)N * n_vars * sizeof(float));
    for (int i = 0; i < N; i++)
        h_xs[i] = 0.1f + (5.0f - 0.1f) * (float)i / (float)(N - 1);

    // ---- ym = 解析@c_true,排成 per-tree [M_prob*N](生产布局)----
    float *h_ym = (float*)malloc((size_t)M_prob * N * sizeof(float));
    for (int m = 0; m < M_prob; m++)
        for (int i = 0; i < N; i++)
            h_ym[m * N + i] = y_analytic(m, h_xs[i], FIXTURES[m].c_true);

    // ---- D allocs + H2D ----
    int      *d_nt;   CUDA_CHECK(cudaMalloc(&d_nt,    total_nodes * sizeof(int)));
    float    *d_nv;   CUDA_CHECK(cudaMalloc(&d_nv,    total_nodes * sizeof(float)));
    int      *d_ci;   CUDA_CHECK(cudaMalloc(&d_ci,    total_nodes * sizeof(int)));
    float    *d_call; CUDA_CHECK(cudaMalloc(&d_call,  total_c * sizeof(float)));
    TreeMeta *d_metas;CUDA_CHECK(cudaMalloc(&d_metas, M_prob * sizeof(TreeMeta)));
    float    *d_xs;   CUDA_CHECK(cudaMalloc(&d_xs,    (size_t)N * n_vars * sizeof(float)));
    float    *d_ym;   CUDA_CHECK(cudaMalloc(&d_ym,    (size_t)M_prob * N * sizeof(float)));
    float    *d_y;    CUDA_CHECK(cudaMalloc(&d_y,     (size_t)M_prob * N * sizeof(float)));  // = yp
    float    *d_r;    CUDA_CHECK(cudaMalloc(&d_r,     (size_t)M_prob * N * sizeof(float)));
    float    *d_loss; CUDA_CHECK(cudaMalloc(&d_loss,  M_prob * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_nt,    h_nt,    total_nodes * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv,    h_nv,    total_nodes * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,    h_ci,    total_nodes * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_call,  h_call,  total_c * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, M_prob * sizeof(TreeMeta),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs,    h_xs,    (size_t)N * n_vars * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ym,    h_ym,    (size_t)M_prob * N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- device 三连:eval@c_init → residual → loss ----
    int block = 256, wpb = block / 32;
    int grid  = (M_prob + wpb - 1) / wpb;

    // 1) yp = 解释器在 c_init 处 eval(s4 的 kernel,原样)
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());
    // 2) r = yp - ym(逐元素,M_prob*N 个线程)
    residual_kernel<<<(M_prob*N + 127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
    CUDA_CHECK(cudaGetLastError());
    // 3) loss = 0.5 * Σr²(warp-per-problem + shfl 归约)
    loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- D2H ----
    float *h_r    = (float*)malloc((size_t)M_prob * N * sizeof(float));
    float *h_loss = (float*)malloc((size_t)M_prob * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_r,    d_r,    (size_t)M_prob * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, (size_t)M_prob * sizeof(float),     cudaMemcpyDeviceToHost));

    // ---- 对拍:独立 CPU oracle 重算 residual 和 0.5*Σr² ----
    //   oracle 全程走解析公式,从不调 device 解释器,也不抄上面两个 kernel:
    //     yp_oracle = y_analytic(c_init);  r_oracle = yp_oracle - ym;
    //     loss_oracle = 0.5 * Σ r_oracle²。
    printf("s5: residual + loss (shfl 归约)\n");
    printf("  M_prob=%d, N=%d, BLOCK=%d (%d warps), grid=%d\n", M_prob, N, block, wpb, grid);
    printf("  ym = 解析@c_true(模拟实测), yp = eval@c_init(预测)\n\n");

    int bad = 0;
    for (int m = 0; m < M_prob; m++) {
        // residual:逐元素比 device 的 h_r 和 oracle
        float max_re_r = 0.0f;
        double loss_oracle = 0.0;
        for (int i = 0; i < N; i++) {
            float ym       = y_analytic(m, h_xs[i], FIXTURES[m].c_true);
            float yp       = y_analytic(m, h_xs[i], FIXTURES[m].c_init);
            float r_oracle = yp - ym;
            loss_oracle += (double)r_oracle * (double)r_oracle;

            float r_dev = h_r[m*N + i];
            float re = fabsf(r_dev - r_oracle) / (fabsf(r_oracle) + 1e-30f);
            if (re > max_re_r) max_re_r = re;
            if (re >= 1e-4f) bad++;
        }
        loss_oracle *= 0.5;

        // loss:per-tree 比 device 的 h_loss 和 oracle
        float loss_dev = h_loss[m];
        float re_loss  = fabsf(loss_dev - (float)loss_oracle) / (fabsf((float)loss_oracle) + 1e-30f);
        bool ok_loss = re_loss < 1e-4f;
        if (!ok_loss) bad++;

        printf("  [%d] %-13s  loss_dev=%.6f  loss_oracle=%.6f  rel=%.1e   r_max_rel=%.1e   %s\n",
               m, FIXTURES[m].name, loss_dev, (float)loss_oracle, re_loss, max_re_r,
               (ok_loss && max_re_r < 1e-4f) ? "ok" : "MISMATCH");
    }

    // ---- cleanup ----
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_ym); cudaFree(d_y); cudaFree(d_r); cudaFree(d_loss);
    free(h_nt); free(h_nv); free(h_ci); free(h_call); free(h_xs); free(h_ym);
    free(h_r); free(h_loss);

    printf("\n%s (%d bad)\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. residual r 和 loss 各是什么?loss = 0.5*Σr² 里那个 0.5 有什么用?
//  2. __shfl_xor_sync(0xFFFFFFFF, s, off) 一行做了什么?为什么 warp 内求和用它
//     比写 __shared__ 数组 + __syncthreads() 那套省?
//  3. loss_kernel 跑完 5 轮 shfl 后,32 个 lane 的 s 都等于总和。为什么只让
//     lane==0 写 loss[m],别的 lane 不写?如果 32 个 lane 全写同一个 loss[m] 会怎样?
//  4. 如果把 fixture 里的 c_init 改成等于 c_true(yp==ym),loss 该是多少?对拍还
//     PASS 吗?这能说明对拍有没有"恒过"的水分?
//
// ## 参考答案
//
//  1. r = yp - ym,每个数据点上"预测比实测差了多少",N 个点 N 个残差。
//     loss = 0.5*Σr² 把 N 个残差平方求和,一棵树压成一个标量,衡量整体拟合差多少。
//     那个 0.5 纯是数学惯例:对 c 求 loss 的梯度时,平方求导带出来的 2 正好和 0.5
//     约掉,梯度写出来干净(= Jᵀr,不带 2)。后面 s7 build JᵀR 就吃这个便利。
//
//  2. 它把【同 warp 里 lane (myid xor off) 寄存器中的 s】直接读过来,不经过任何
//     内存。配合 off=16,8,4,2,1 的 5 轮,每轮把两两 lane 的部分和合并,跑完每个
//     lane 的 s 都是 warp 总和。比 shared 版省在:(i) 不占 shared memory,(ii) 不用
//     __syncthreads —— 那是 block 级栅栏,作用域比 warp 归约大,本来就不该用它来同步
//     一个 warp。warp 范围的同步由 __shfl_xor_sync 的 _sync 后缀 + mask=0xFFFFFFFF 自带
//     (从 Volta 起 warp 不保证 lockstep,所以靠这条指令显式同步,而不是靠隐式 lockstep)。
//     少了内存往返 + block 级同步,这就是 L09 "warp 内通信走寄存器" 的好处。
//
//  3. 因为 5 轮后所有 lane 的 s 都一样(都是总和),只需要把结果落盘一次。挑 lane0
//     是约定。32 个 lane 全写 loss[m]:它们写的值相同,结果不会错,但这是 32 路线程
//     同时写同一个地址 —— 多余的内存事务,纯浪费带宽(本例值一样不会冲突出错,但
//     一般"多线程写同一地址"是该避免的模式)。所以只留 lane0 写。
//
//  4. yp==ym → 每个 r=0 → loss = 0.5*Σ0 = 0,device 和 oracle 都得 0。对拍仍 PASS。
//     但这恰恰说明【不能用 c_init==c_true 当 fixture】:残差全 0 时,哪怕 residual_kernel
//     写错符号、loss_kernel 漏乘 0.5,结果都还是 0,对拍照样过 —— 那就是"恒过"的水分。
//     本级故意让 c_init 远离 c_true,逼出非零 residual 和非零 loss,对拍才真有鉴别力。
// ===========================================================================
