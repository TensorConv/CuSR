// s3_tree_gpu_Npoints.cu — 1 个线程算 1 个数据点(线程索引登场)。
//
// 这一级的【唯一新概念】:线程索引 i = blockIdx.x*blockDim.x + threadIdx.x。
//   s2 只开 1 个线程,它用一个 for 循环把 N 个点全做完。这一级把那个 for 删掉,
//   改成"开 N 个线程,第 i 个线程只算第 i 个点",这才是 GPU 真正的用法 ——
//   N 个点同时算,而不是一个线程串着算。
//
// 假设你已看过 s1..s2,已经懂 "1 棵 powerlaw 树,用 1 个线程的 for 循环把 N 个点
// 全做了";这一级只加 "把那个 for 换成线程索引 i,1 线程 1 数据点"。
//
// 跟 s2 比,代码改动【只有三处】,算法和解释器一个字没动:
//   (1) kernel 里删掉 for,换成 i = blockIdx.x*blockDim.x+threadIdx.x; if(i<N) ...;
//   (2) 启动从 <<<1,1>>> 改成 <<<grid, block>>>,grid 按点数算出来;
//   (3) fixture 的 N 从 5 变 64(点多了,1 线程 1 点才有意义;不是新概念,
//       就是让"每个线程一个点"这件事看得出来)。
//
// 跑: ./build.sh s3     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - i = blockIdx.x*blockDim.x + threadIdx.x 这一行在干嘛?为什么它就是"全局点号"?
//   - block=128 而 N=64,启动了几个线程?if(i<N) 这道 guard 到底挡的是谁?
//   - 把 if(i<N) 删掉会发生什么?(不是"算错一个点"那么轻)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// 每个 CUDA API 调用都包它:出错立刻打印 + 退出,别让错误悄悄溜过(全程约定 1)。
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

enum NTypeE { N_VAR = 0, N_CONST = 1, N_UFUNC = 2, N_BFUNC = 3 };
enum FuncE  { F_ADD = 1, F_SUB = 2, F_MUL = 3, F_DIV = 4, F_POW = 6 };
#define MAX_STACK 64

// ---------------------------------------------------------------------------
// 解释器:和 s2 的 eval_tree_d 函数体【一字不差】(逆前缀 + 栈,powerlaw 只用
// MUL/POW,所以这里只留 BFUNC 分支)。s2 已经讲透,不再重复。
// ---------------------------------------------------------------------------
__device__ float eval_tree_d(
    const int *nt, const float *nv, int n_nodes,
    const int *const_idx, const float *x, const float *c)
{
    float stack[MAX_STACK];
    int sp = 0;
    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i];
        float v = nv[i];
        if (t == N_VAR) {
            stack[sp++] = x[(int)v];
        } else if (t == N_CONST) {
            stack[sp++] = c[const_idx[i]];
        } else if (t == N_BFUNC) {
            float l  = stack[--sp];
            float rv = stack[--sp];
            int fid = (int)v;
            float o;
            switch (fid) {
                case F_ADD: o = l + rv;      break;
                case F_SUB: o = l - rv;      break;
                case F_MUL: o = l * rv;      break;
                case F_DIV: o = l / rv;      break;
                case F_POW: o = powf(l, rv); break;
                default:    o = 0.0f;        break;
            }
            stack[sp++] = o;
        }
    }
    return stack[0];
}

// ---------------------------------------------------------------------------
// kernel:【这一级的全部新意都在这 3 行里】。
//
//   i = blockIdx.x * blockDim.x + threadIdx.x;
//     启动时我们开了一堆线程,它们排成 grid(若干 block)× block(每 block 若干
//     线程)。每个线程想知道"我是全局第几个",就靠这条公式:
//        blockDim.x  = 每个 block 有多少线程(这里 128);
//        blockIdx.x  = 我在第几个 block(0,1,2,...);
//        threadIdx.x = 我在自己 block 里排第几(0..127);
//     拼起来 i 就是【全局唯一的线程号】。我们让线程号 = 数据点号:第 i 个线程
//     专门算第 i 个点,N 个点就被 N 个线程【同时】算了(不再像 s2 串着循环)。
//
//   if (i >= N) return;
//     这道 guard 挡的是【多出来的线程】。线程数得按 block 整批开(见下面 grid 怎么
//     算),通常会比 N 多 —— 比如 N=64、block=128,就会开出 128 个线程,可点只有
//     64 个。第 64..127 号线程没有对应的点,必须让它们【啥也不干直接 return】;
//     否则它们会去写 y_out[64..127],而 y_out 只开了 N=64 个 float —— 越界写,
//     踩到不属于这块缓冲的显存。所以 guard 不是"算错一个点"这种小事,是防越界。
// ---------------------------------------------------------------------------
__global__ void eval_kernel(
    const int *nt, const float *nv, int n_nodes, const int *ci,
    const float *xs, int N, const float *c, float *y_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // L01 落到真代码:全局线程号 = 点号
    if (i >= N) return;                              // 多开的线程(i>=N)直接退,别越界写
    y_out[i] = eval_tree_d(nt, nv, n_nodes, ci, &xs[i], c);
}

int main(void)
{
    // ---- 0. fixture 在 host 上:powerlaw_K2  y = c0 * x^c1,c0=2.5 c1=1.3 ----
    //    树结构和 s1/s2 完全一样,只是数据点从 5 个加到 64 个。
    const int   h_nt[] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
    const float h_nv[] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f    };
    const int   h_ci[] = { -1,      0,       -1,      -1,    1       };
    const int   n_nodes = 5;
    float       h_c[]  = { 2.5f, 1.3f };          // c_true
    const int   K = 2;

    // xs ∈ [0.1, 5.0] 均匀采 N=64 个点(n_vars=1)。
    const int   N = 64;
    float h_xs[64];
    for (int i = 0; i < N; i++)
        h_xs[i] = 0.1f + (5.0f - 0.1f) * (float)i / (float)(N - 1);

    printf("fixture: y = c0 * x^c1,  c0=%.1f c1=%.1f   (GPU, 1 线程 1 数据点, N=%d)\n\n",
           h_c[0], h_c[1], N);

    // ---- 1. 在 GPU 上开内存(device 缓冲,指针名 d_*)----
    int   *d_nt, *d_ci;
    float *d_nv, *d_xs, *d_c, *d_y;
    CUDA_CHECK(cudaMalloc(&d_nt, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nv, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ci, n_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_xs, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c,  K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  N * sizeof(float)));

    // ---- 2. 输入搬上 GPU(H2D = HostToDevice)----
    CUDA_CHECK(cudaMemcpy(d_nt, h_nt, n_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv, h_nv, n_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, h_ci, n_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, N * sizeof(float),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c,  h_c,  K * sizeof(float),       cudaMemcpyHostToDevice));

    // ---- 3. 启动 kernel:这次不是 <<<1,1>>>,而是【按点数把线程开够】----
    //    block = 每个 block 的线程数(常取 128/256 这类 32 的倍数)。
    //    grid  = 需要几个 block,= ceil(N / block) = (N + block - 1) / block。
    //            这个"上取整"保证线程数 >= N(每个点都有线程管);通常会比 N 多
    //            出一截,多出来的那些靠 kernel 里的 if(i<N) 挡掉。
    //    这里 N=64、block=128 -> grid=1 -> 共开 128 个线程,真正干活的只有前 64 个。
    int block = 128;
    int grid  = (N + block - 1) / block;
    eval_kernel<<<grid, block>>>(d_nt, d_nv, n_nodes, d_ci, d_xs, N, d_c, d_y);
    // 两步查错:kernel 启动是异步的 ——
    CUDA_CHECK(cudaGetLastError());       //   同步:抓"启动配置错"
    CUDA_CHECK(cudaDeviceSynchronize());  //   阻塞到真跑完:抓"运行期错"(如越界访问)

    // ---- 4. 结果搬回 CPU(D2H = DeviceToHost)----
    float h_y[64];
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 5. 对拍:GPU 算的 h_y[i] vs 独立解析公式 2.5 * x^1.3 ----
    //    oracle 是直接套公式逐点算,跟 device 解释器完全两条路 —— 对得上才算数。
    //    64 个点不逐行打,汇总 max_rel + mismatch 数,再抽 2 个样本看具体值。
    int   bad = 0;
    float max_rel = 0.0f;
    for (int i = 0; i < N; i++) {
        float ref = 2.5f * powf(h_xs[i], 1.3f);                 // 独立 oracle
        float rel = fabsf(h_y[i] - ref) / (fabsf(ref) + 1e-30f);
        if (rel > max_rel) max_rel = rel;
        if (rel >= 1e-5f) bad++;                                // float ULP 容差,不放松不收紧
    }
    printf("  N=%d  max_rel=%.1e  mismatches=%d/%d  (容差 rel<1e-5)\n", N, max_rel, bad, N);
    printf("  sample i=%d : x=%.4f  gpu=%.6f  ref=%.6f\n",
           N/2, h_xs[N/2], h_y[N/2], 2.5f * powf(h_xs[N/2], 1.3f));
    printf("  sample i=%d : x=%.4f  gpu=%.6f  ref=%.6f\n",
           N-1, h_xs[N-1], h_y[N-1], 2.5f * powf(h_xs[N-1], 1.3f));

    // ---- 6. 释放 device 内存 ----
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci);
    cudaFree(d_xs); cudaFree(d_c);  cudaFree(d_y);

    printf("\n%s (%d/%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad, N);
    return bad == 0 ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. i = blockIdx.x*blockDim.x + threadIdx.x 这一行在算什么?为什么算出来的 i
//     正好是"该线程负责的那个数据点的下标"?
//  2. 本例 N=64、block=128,<<<grid,block>>> 一共启动了多少个线程?其中真正
//     往 y_out 写值的有几个?if(i<N) 把哪些线程挡在外面?
//  3. 如果把 kernel 里的 if(i>=N) return; 删掉,会 PASS 还是出问题?是"多算几个
//     无所谓的点",还是更严重?为什么?
//  4. 把 block 改成 32(N 仍 64),grid 会变成几?对拍结果会变吗?为什么?
//
// ## 参考答案
//
//  1. 线程是按 grid(若干 block)× block(每 block 若干线程)排的。blockDim.x 是
//     每 block 的线程数(128),blockIdx.x 是当前 block 的序号,threadIdx.x 是线程
//     在自己 block 内的序号(0..127)。blockIdx.x*blockDim.x 跳过前面所有 block 的
//     线程,再加 threadIdx.x 得到【全局唯一线程号】。我们约定"线程号 = 点号",
//     所以第 i 个线程就用 xs[i] 这一个点去 eval,N 个点被 N 个线程同时算完。
//
//  2. grid = ceil(64/128) = 1,block=128,共 1*128 = 128 个线程。但点只有 64 个,
//     所以只有 i=0..63 这 64 个线程真正写 y_out;i=64..127 这 64 个线程被
//     if(i<N) 挡住、直接 return,什么都不做。线程总是整 block 开,128 不能"只开
//     一半",多出来的就靠 guard 收拾。
//
//  3. 出问题,而且不是小问题。i=64..127 的线程会去执行 y_out[i] = ...,可 d_y 只
//     cudaMalloc 了 N=64 个 float,y_out[64..127] 是【越界写】,踩到不属于这块
//     缓冲的显存。轻则结果不可预测,重则 cudaDeviceSynchronize() 直接报 illegal
//     memory access 退出。guard 防的是越界,不是"算错一个点"。
//
//  4. grid = ceil(64/32) = 2。现在开 2 个 block × 32 = 64 个线程,正好一人一个点,
//     一个都不浪费,i 不会超过 63,guard 一次也用不上。对拍结果【完全不变】——
//     block/grid 只决定"线程怎么分组开",不改每个点算出来的值。换句话说:多大的
//     block 都对,只要线程数 >= N 且有 guard 兜底。
// ===========================================================================
