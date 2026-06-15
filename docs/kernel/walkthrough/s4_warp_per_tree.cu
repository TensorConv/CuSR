// s4_warp_per_tree.cu — 一批树:M 棵树并行,一个 warp 管一棵树。
//
// 这一级新增的【唯一概念】= warp-per-problem 布局:
//   把 GPU 的线程按 32 个一组(一个 warp)切开,【一个 warp 负责一棵树】,
//   warp 里的 32 条 lane 再用跨步循环 for(i=lane; i<N; i+=32) 分担这棵树的 N 个点。
//
// 假设你已看过 s1..s3,已经懂 s3(1 棵树、N 个点、1 线程 1 点)。这一级只加
// 把「一棵树」扩成「M 棵树的种群」,并按 warp-per-problem 把它们铺到 GPU 上。
//
// ⚠ 注意本级有【两个耦合的 delta】,不是一个 —— 必须说清楚,别被「只加一个概念」
//    的口号骗了:
//   (delta-1, 主角)warp-per-problem 映射 + 多棵树的种群(拼接 nt/nv/ci 数组、
//                   一张 TreeMeta offset 表、拼接的 c_all)。
//   (delta-2, 被迫带上)解释器从 s3 的「BFUNC-only 子集」补全成 canonical 版本
//                   (加回 N_UFUNC 分支:EXP/NEG/SIN/...)。为什么被迫?因为本级
//                   新加的第 3 个 fixture expsat = c0*exp(-c1*x)+c2 用到了 EXP 和
//                   NEG 这两个【单目算子(UFUNC)】,s3 的子集解释器不认识它们,
//                   会走 default 返回 0 —— 不补全就根本算不对 expsat。
//   这两件事是同时发生的(新 fixture 逼着补解释器),所以这一级老实承认「1.x 个
//   delta」,而不是假装干净的「1 个」。
//
// 跑: ./build.sh s4     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - warp / lane 是什么?为什么「一个 warp 一棵树」能让解释器 0 divergence?
//   - 拼接数组 + TreeMeta 的 node_offset / c_offset 在解决什么问题?
//   - for(i=lane; i<N; i+=32) 这个跨步(strided)循环,和 s3 的 1 线程 1 点比,
//     32 条 lane 各自做了哪些 i?

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// 每个 CUDA API 调用都包它:出错立刻打印 + 退出(全程约定 1)。
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

// 节点种类 / 算子编号 —— 和生产 batch_lm.cu 严格一致。
// 这一级起把 N_UFUNC(单目算子)正式用上了(expsat 要 EXP/NEG)。
enum NType { N_VAR = 0, N_CONST = 1, N_UFUNC = 2, N_BFUNC = 3, N_TFUNC = 4 };
enum Func {
    F_ADD = 1, F_SUB = 2, F_MUL = 3, F_DIV = 4, F_POW = 6,
    F_SIN = 14, F_COS = 15, F_LOG = 20, F_EXP = 22, F_NEG = 25, F_SQRT = 27,
};
#define MAX_STACK 64    // 和生产 batch_lm.cu 一致
#define MAX_K     32

// 一棵树在「拼接大数组」里的元信息(metadata)。多棵树长度不一(ragged),
// 没法整齐地排成二维表,所以把所有树的节点数组首尾拼成一根长数组,再用这张
// 表记下「第 m 棵树从哪开始 / 有几个节点 / 它的常数从 c_all 哪开始 / 有几个常数」。
struct TreeMeta {
    int node_offset;  // 这棵树的节点在 nt_all/nv_all/ci_all 里的起始下标
    int n_nodes;      // 这棵树的节点数
    int c_offset;     // 这棵树的常数在 c_all 里的起始下标
    int K;            // 常数个数
};

// ---------------------------------------------------------------------------
// canonical 解释器:和 s1/s2/s3 同一套「逆前缀 + 栈」算法,这里补全到完整版
// —— 相对 s3 的 BFUNC-only 子集,多了 N_UFUNC(单目算子)这一整个分支。
//
//   单目算子(UFUNC)只吃 1 个操作数:弹 1 个、算完压回 1 个。
//   双目算子(BFUNC)吃 2 个:弹 2 个、算完压回 1 个(s1 已讲过弹出顺序)。
//
// expsat 树里 EXP 和 NEG 就是 UFUNC:NEG(a) = -a,EXP(a) = e^a。
// 没有这个分支,EXP/NEG 会走不到、当成未知节点 —— expsat 必算错。
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
// kernel:warp-per-problem。和生产 batch_lm.cu 的 eval_kernel_batched 同名同结构。
//
// 先把硬件名词用大白话铺一遍:
//   - warp = GPU 调度的最小单位,固定 32 条 lane(线程)。同一个 warp 的 32 条
//     lane 在硬件上【锁步(lockstep)】执行同一条指令。
//   - 我们让「一个 warp 跑一棵树」:这 32 条 lane 跑的是同一棵树、同一份解释器
//     代码,走的 if/switch 分支【完全一样】—— 所以 warp 内 0 divergence(分支
//     分歧),硬件不用串行化不同分支,效率最高。
//   - 不同 warp 跑不同的树,各管各的,互不干扰。
//
// 一个 block 我们开 256 线程 = 256/32 = 8 个 warp = 一个 block 管 8 棵树。
// 下面这 4 行就是从 (blockIdx, threadIdx) 反推出「我是第几号 warp(= 第几棵树 m)、
// 我是 warp 里的第几条 lane」—— s4 起凡 warp-per-problem 的 kernel 开头都用这段。
// ---------------------------------------------------------------------------
__global__ void eval_kernel_batched(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N, const float *c_all, float *y_out)
{
    int wpb  = blockDim.x / 32;     // warps per block:BLOCK=256 → 8 warp = 8 棵树/block
    int wb   = threadIdx.x / 32;    // warp in block:我在本 block 里是第几个 warp
    int lane = threadIdx.x & 31;    // lane id:我是 warp 里的第几条(0..31)
    int m    = blockIdx.x * wpb + wb;  // 全局树编号:我这个 warp 负责第 m 棵树
    if (m >= M_prob) return;        // 树数不是 8 的整数倍时,多出来的 warp 直接退出

    // 用 TreeMeta 把「第 m 棵树」在大数组里的那一段切出来(指针 + offset)。
    // metas[m] 这一次读:warp 内 32 条 lane 读同一个地址 → broadcast(广播)load。
    TreeMeta meta = metas[m];
    const int   *nt = nt_all + meta.node_offset;
    const float *nv = nv_all + meta.node_offset;
    const int   *ci = ci_all + meta.node_offset;
    const float *c  = c_all  + meta.c_offset;

    // 跨步(strided)循环把 N 个点分给 32 条 lane:
    //   lane 0 做 i=0,32,64,...;lane 1 做 i=1,33,...;以此类推。
    //   N=64 时每条 lane 正好做 2 个点(i=lane 和 i=lane+32)。
    // 这一步 xs+i*n_vars 相邻 lane 读相邻地址 → coalesced(合并访存);
    // y_out 写也是相邻 lane 写相邻地址 → 同样 coalesced。
    for (int i = lane; i < N; i += 32) {
        float y = eval_tree_d(nt, nv, meta.n_nodes, ci, xs + i * n_vars, c);
        y_out[m * N + i] = y;       // 输出按 [M_prob × N] 行优先:第 m 棵树第 i 个点
    }
}

// ============================================================
// Host 端:3 个 archetype fixture(逐字来自 m3_2_multi_tree.cu)
// ============================================================

struct TreeSpec {
    const char *name;
    int n_nodes;
    const int   *nt;
    const float *nv;
    const int   *ci;
    int K;
    const float *c_true;
};

// powerlaw_K2: c0 * pow(x0, c1)  →  prefix MUL c0 POW x0 c1
static const int   tree0_nt[5] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
static const float tree0_nv[5] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f };
static const int   tree0_ci[5] = { -1, 0, -1, -1, 1 };
static const float tree0_ct[2] = { 2.5f, 1.3f };   // c_true
static const float tree0_c0[2] = { 1.0f, 1.0f };   // c_init (LM 起点,本级未用)

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
    { "powerlaw_K2",  5,  tree0_nt, tree0_nv, tree0_ci, 2, tree0_ct },
    { "quadratic_K3", 11, tree1_nt, tree1_nv, tree1_ci, 3, tree1_ct },
    { "expsat_K3",    9,  tree2_nt, tree2_nv, tree2_ci, 3, tree2_ct },
};

// ---------------------------------------------------------------------------
// 独立 oracle:每棵树的【解析公式】,直接照数学式子算 —— 完全不碰上面的栈机
// 解释器。这是对拍的关键:device 是「栈机解释器」,oracle 是「闭式公式」,两条
// 不同的路算出来要一致,才证明解释器没写错(否则就是拿自己验自己的 tautology)。
// ---------------------------------------------------------------------------
static float y_ref_eval(int fixture_idx, float x) {
    const float *c;
    switch (fixture_idx) {
        case 0: c = tree0_ct; return c[0] * powf(x, c[1]);          // powerlaw
        case 1: c = tree1_ct; return c[0] + c[1] * x + c[2] * x * x; // quadratic
        case 2: c = tree2_ct; return c[0] * expf(-c[1] * x) + c[2];  // expsat
        default: return 0.0f;
    }
}

int main(void)
{
    int M_prob = (int)(sizeof(FIXTURES) / sizeof(FIXTURES[0]));   // = 3
    int N      = 64;
    int n_vars = 1;

    // c_init(LM 迭代起点)本级用不到 —— 它们是给 s9 的 LM 主循环留的。这里点一下
    // 名,免得编译器对「声明了没用」报 warning(数值本身和生产/spec 一字不差)。
    (void)tree0_c0; (void)tree1_c0; (void)tree2_c0;

    // ---- pack metas + 拼接 node 数组 + 拼接 c_all ----
    // 一遍扫过去:记录每棵树的起始 offset,顺手累加总节点数 / 总常数数。
    TreeMeta h_metas[16];
    int total_nodes = 0;
    int total_c     = 0;
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
    float *h_call = (float*)malloc((size_t)total_c     * sizeof(float));
    for (int m = 0; m < M_prob; m++) {
        memcpy(h_nt   + h_metas[m].node_offset, FIXTURES[m].nt, FIXTURES[m].n_nodes * sizeof(int));
        memcpy(h_nv   + h_metas[m].node_offset, FIXTURES[m].nv, FIXTURES[m].n_nodes * sizeof(float));
        memcpy(h_ci   + h_metas[m].node_offset, FIXTURES[m].ci, FIXTURES[m].n_nodes * sizeof(int));
        memcpy(h_call + h_metas[m].c_offset,    FIXTURES[m].c_true, FIXTURES[m].K * sizeof(float));
    }

    // ---- xs ∈ [0.1, 5.0] 均匀 N=64 个 + 每棵树的 analytical y_ref(oracle)----
    float *h_xs  = (float*)malloc((size_t)N * n_vars * sizeof(float));
    float *h_ref = (float*)malloc((size_t)M_prob * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_xs[i] = 0.1f + (5.0f - 0.1f) * (float)i / (float)(N - 1);
    }
    for (int m = 0; m < M_prob; m++)
        for (int i = 0; i < N; i++)
            h_ref[m * N + i] = y_ref_eval(m, h_xs[i]);

    // ---- device 开内存 + 拷上去 ----
    int      *d_nt;     CUDA_CHECK(cudaMalloc(&d_nt,    total_nodes * sizeof(int)));
    float    *d_nv;     CUDA_CHECK(cudaMalloc(&d_nv,    total_nodes * sizeof(float)));
    int      *d_ci;     CUDA_CHECK(cudaMalloc(&d_ci,    total_nodes * sizeof(int)));
    float    *d_call;   CUDA_CHECK(cudaMalloc(&d_call,  total_c * sizeof(float)));
    TreeMeta *d_metas;  CUDA_CHECK(cudaMalloc(&d_metas, M_prob * sizeof(TreeMeta)));
    float    *d_xs;     CUDA_CHECK(cudaMalloc(&d_xs,    N * n_vars * sizeof(float)));
    float    *d_y;      CUDA_CHECK(cudaMalloc(&d_y,     M_prob * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_nt,    h_nt,    total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv,    h_nv,    total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci,    h_ci,    total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_call,  h_call,  total_c * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, M_prob * sizeof(TreeMeta),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs,    h_xs,    N * n_vars * sizeof(float),  cudaMemcpyHostToDevice));

    // ---- 启动:BLOCK=256 = 8 warp = 8 棵树/block;grid 向上取整 ----
    int block = 256;
    int wpb   = block / 32;
    int grid  = (M_prob + wpb - 1) / wpb;     // M_prob=3 → grid=1
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
                                         d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());           // 抓启动配置错(异步)
    CUDA_CHECK(cudaDeviceSynchronize());      // 阻塞到跑完,抓运行期错

    float *h_y = (float*)malloc((size_t)M_prob * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, M_prob * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 对拍:每棵树 GPU 算的 vs 解析 oracle ----
    printf("s4: warp-per-problem batched 解释器\n");
    printf("  M_prob=%d, N=%d, n_vars=%d, BLOCK=%d (%d warps), grid=%d\n",
           M_prob, N, n_vars, block, wpb, grid);
    printf("  total_nodes=%d, total_c=%d\n\n", total_nodes, total_c);

    int all_pass = 1;
    for (int m = 0; m < M_prob; m++) {
        float max_ae = 0, max_re = 0;
        int mismatches = 0;
        for (int i = 0; i < N; i++) {
            float ref = h_ref[m * N + i];
            float ae  = fabsf(h_y[m * N + i] - ref);
            float re  = ae / (fabsf(ref) + 1e-30f);
            if (ae > max_ae) max_ae = ae;
            if (re > max_re) max_re = re;
            // m3_2 判据:绝对误差超过 1e-5*|ref| + 1e-7 才算一处 mismatch。
            if (ae > 1e-5f * fabsf(ref) + 1e-7f) mismatches++;
        }
        printf("  [%d] %-16s n_nodes=%2d K=%d  max_abs=%.2e max_rel=%.2e  mismatches=%d/%d  %s\n",
               m, FIXTURES[m].name, FIXTURES[m].n_nodes, FIXTURES[m].K,
               max_ae, max_re, mismatches, N,
               mismatches == 0 ? "ok" : "MISMATCH");
        if (mismatches != 0) all_pass = 0;
    }

    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call);
    cudaFree(d_metas); cudaFree(d_xs); cudaFree(d_y);
    free(h_nt); free(h_nv); free(h_ci); free(h_call);
    free(h_xs); free(h_ref); free(h_y);

    printf("\n%s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. warp 是什么?「一个 warp 一棵树」为什么能保证解释器分支 0 divergence?
//     如果反过来「一条 lane 一棵树、32 条 lane 跑 32 棵不同的树」会怎样?
//  2. 三棵树长度不一样(5 / 11 / 9 个节点),为什么不直接排成二维数组,而要拼成
//     一根长数组 + TreeMeta 的 node_offset?c_offset 又是干嘛的?
//  3. 这一级说有「两个耦合的 delta」,主角是 warp-per-problem,另一个是什么?
//     是哪个 fixture 逼出来的、逼着补了解释器的哪个分支?
//  4. for(i=lane; i<N; i+=32) 里,N=64 时 lane=5 这条线程具体算了哪几个 i?
//     如果某棵树只有 N=20 个点,lane=25 这条线程会进循环体吗?
//
// ## 参考答案
//
//  1. warp = GPU 调度的最小单位,固定 32 条 lane,硬件上锁步执行同一条指令。
//     「一个 warp 一棵树」时,32 条 lane 跑同一棵树、同一份解释器代码,每个 if/
//     switch 分支走向完全相同 → 没有分歧 → 0 divergence,硬件不必串行化不同分支。
//     反过来「一条 lane 一棵树」:32 条 lane 跑 32 棵【结构不同】的树,解释器在同
//     一步可能有的走 BFUNC 有的走 UFUNC,warp 必须把每条分支轮流串行跑一遍(其余
//     lane 闲等)→ 严重 divergence,这正是 warp-per-problem 要避开的坑。
//
//  2. 三棵树节点数不同(ragged / 参差),排二维表得按最长的 11 补齐、浪费且要记
//     padding。改成把所有树的节点首尾拼成一根长数组,再用 TreeMeta.node_offset
//     记「第 m 棵树从长数组的哪个下标开始」、n_nodes 记「有几个」,就能精确切片、
//     零浪费。c_offset 同理,是这棵树的常数在拼接后的 c_all 里的起始下标(因为各
//     树常数个数 K 也不同:2 / 3 / 3)。
//
//  3. 另一个 delta 是「把解释器从 s3 的 BFUNC-only 子集补全成 canonical 完整版」,
//     具体是加回 N_UFUNC(单目算子)那一整个分支。是第 3 个 fixture expsat =
//     c0*exp(-c1*x)+c2 逼出来的:它用了 EXP 和 NEG,这俩是 UFUNC;不补这个分支,
//     EXP/NEG 会走 default 返回 0,expsat 必算错。所以本级老实承认「1.x 个 delta」。
//
//  4. N=64、步长 32:lane=5 算 i=5 和 i=5+32=37(再 +32=69 已 ≥64,停)。两个点。
//     N=20 时 lane=25:循环初值 i=25 已经 ≥ N=20,条件 i<N 一开始就不成立 → 这条
//     lane【一次都不进循环体】,直接空跑退出。strided 循环天然处理「线程比点多」
//     的情形,不会越界(这也是为什么不写 if(i<N) 也安全)。
// ===========================================================================
