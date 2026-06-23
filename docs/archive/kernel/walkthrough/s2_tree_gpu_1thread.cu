// s2_tree_gpu_1thread.cu — 把 s1 的解释器原样搬上 GPU,用 1 个线程跑。
//
// 和 s1 的差别【只有三件事】,算法逻辑一个字没改:
//   (1) eval_tree 前面加了 __device__  -> 变成"能在 GPU 上跑"的函数;
//   (2) 多了一个 __global__ kernel      -> GPU 的入口,从 CPU 用 <<<>>> 启动;
//   (3) 数据要显式搬:cudaMalloc 在 GPU 开内存 + cudaMemcpy 来回拷(L02 那六步)。
//
// 这一级【还不碰线程索引】—— 只开 1 个线程,它自己把 N 个点循环做完。
// 索引(N 个线程各做 1 个点)是 s3 的事。这样 s2->s3 的差别就干净成一句话:
// "把这个 for 循环拆给 N 个线程"。
//
// 跑: ./build.sh s2     预期最后一行 PASS。
//
// 学完应能回答(答案在文件末尾):
//   - __device__ / __global__ 各是什么,谁能调谁?
//   - 为什么 kernel 不能直接用 host 上的 h_nt,必须先 cudaMemcpy 到 d_nt?
//   - 启动后为什么要 cudaGetLastError() 和 cudaDeviceSynchronize() 两步查错?

#include <cstdio>
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
// 解释器:和 s1 的 eval_tree 函数体【一字不差】,只是前面多了 __device__。
//
//   __device__ = "这个函数在 GPU 上执行,只能被 GPU 上的代码(kernel 或其它
//                 __device__ 函数)调用"。CPU 不能直接调它。
//
// 算法你 s1 已经懂了(逆前缀 + 栈),这里不再重复讲。
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
// kernel:GPU 的入口函数,标 __global__。
//
//   __global__ = "从 CPU 用 <<<...>>> 启动,在 GPU 上跑"。它是 host 和 device
//                的边界:CPU 这边写 eval_kernel<<<1,1>>>(...),GPU 那边就开始跑。
//
// 这一级我们只开 1 个线程(下面 <<<1,1>>>),所以这唯一的线程把 N 个点全做了。
// 它能碰的指针(nt/nv/ci/xs/c/y_out)【必须都是 device 内存】—— kernel 跑在
// GPU 上,够不到 CPU 的内存。所以 main 里要先把数据 cudaMemcpy 上来。
// ---------------------------------------------------------------------------
__global__ void eval_kernel_1thread(
    const int *nt, const float *nv, int n_nodes, const int *ci,
    const float *xs, int N, const float *c, float *y_out)
{
    for (int i = 0; i < N; i++)               // 1 个线程,N 个点全包(s3 拆这循环)
        y_out[i] = eval_tree_d(nt, nv, n_nodes, ci, &xs[i], c);
}

int main(void)
{
    // ---- 0. fixture 在 host 上(和 s1 完全一样)----
    const int   h_nt[] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
    const float h_nv[] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f    };
    const int   h_ci[] = { -1,      0,       -1,      -1,    1       };
    const int   n_nodes = 5;
    float       h_c[]  = { 2.5f, 1.3f };
    const int   K = 2;
    const float h_xs[] = { 0.1f, 0.5f, 1.0f, 2.0f, 5.0f };
    const int   N = 5;

    printf("fixture: y = c0 * x^c1,  c0=%.1f c1=%.1f   (GPU, 1 线程)\n\n", h_c[0], h_c[1]);

    // ---- 1. 在 GPU 上开内存(device 缓冲,指针名 d_*)----
    //    d_* 的值是"GPU 那边的地址",在 CPU 上解引用会崩(L02)。
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

    // ---- 3. 启动 kernel:1 个 block,每 block 1 个线程 ----
    eval_kernel_1thread<<<1, 1>>>(d_nt, d_nv, n_nodes, d_ci, d_xs, N, d_c, d_y);
    // 两步查错(L01):kernel 启动是异步的 ——
    CUDA_CHECK(cudaGetLastError());       //   同步:抓"启动配置错"(如线程数超 1024)
    CUDA_CHECK(cudaDeviceSynchronize());  //   阻塞到真跑完:抓"运行期错"(如非法访问)

    // ---- 4. 结果搬回 CPU(D2H = DeviceToHost),否则你看不到 ----
    float h_y[5];
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 5. 对拍:GPU 算的 h_y[i] vs 解析公式 2.5 * x^1.3 ----
    int bad = 0;
    for (int i = 0; i < N; i++) {
        float ref = 2.5f * powf(h_xs[i], 1.3f);
        float rel = fabsf(h_y[i] - ref) / (fabsf(ref) + 1e-30f);
        bool ok = rel < 1e-5f;
        if (!ok) bad++;
        printf("  x=%.2f   gpu=%.6f   ref=%.6f   rel=%.1e   %s\n",
               h_xs[i], h_y[i], ref, rel, ok ? "ok" : "MISMATCH");
    }

    // ---- 6. 释放 device 内存 ----
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci);
    cudaFree(d_xs); cudaFree(d_c);  cudaFree(d_y);

    printf("\n%s (%d/%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad, N);
    return bad == 0 ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. __device__ 和 __global__ 各是什么?谁从 CPU 启动、谁只能在 GPU 内部被调?
//  2. kernel 收到的 d_nt 为什么不能换成 host 上的 h_nt?(回忆 L02:两块物理内存)
//  3. 把第 3 步改成 <<<1, 1>>> 不动,但把 kernel 里的 for 删了、直接
//     y_out[0]=... 只算第 1 个点 —— 对拍会怎样?为什么?
//  4. 注释掉第 4 步那次 D2H cudaMemcpy,结果会 PASS 还是 FAIL?h_y 里是什么?
//
// ## 参考答案
//
//  1. __global__ = kernel,从 CPU 用 <<<G,T>>> 启动、在 GPU 上跑,是 host/device
//     的边界。__device__ = 只在 GPU 上跑、只能被 kernel 或别的 __device__ 函数
//     调用的普通函数(这里的 eval_tree_d)。CPU 不能直接调 __device__ 函数。
//
//  2. d_nt 在 GPU 显存(HBM),kernel 跑在 GPU 上,只能碰 GPU 内存。h_nt 在 CPU
//     内存,GPU 的执行单元物理上够不到 -> 必须先 cudaMemcpy 把内容拷成 d_nt。
//     (类型都是指针、长得一样,但分属两块内存,不能混用 —— L02 的核心。)
//
//  3. 只有 x=0.10 那行 ok,其余 4 行是 MISMATCH(其实是没算、h_y[1..4] 是
//     cudaMalloc 给的垃圾值)。说明"1 个线程只做 1 件事"时,N 个点得靠循环
//     (s2)或多线程(s3)覆盖,不能只算一个。
//
//  4. FAIL。GPU 把结果写在了 d_y(显存),你没搬回来,h_y 还是没初始化的栈垃圾。
//     "GPU 算成功 ≠ 你拿到结果",少了 D2H 就看不到 —— 和 L02 那题同一个教训。
// ===========================================================================
