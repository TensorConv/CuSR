# _SPEC.md — walkthrough s3–s10 的单一事实源(给写手/审查 agent 用)

这份文件是**权威规格**。s3–s10 的每个文件都必须严格遵守这里的共享不变量、
文风、数值、oracle 规则。**凡是这里定死的,写手不得自创。** 审查 agent 逐条核对。

目标:让学习者不啃 13 课教程,直接读懂 `../batch_lm.cu`。办法 = 从 s1/s2 出发,
每级一个自包含文件,**只加一个新概念**,本机 GPU 编译跑通 + 和**独立 CPU oracle**
对拍 PASS,enum/fixture/命名全程和生产 `../batch_lm.cu` 对齐,搭到 s9 就是 toy 版
batch_lm,s10 通读真文件。

正确性锚:`../../../learn/cuda/project_hetero_lm/m3_*.cu`(已验证 PASS 的参考实现)。
每级镜像其中对应的一个,**抄它的算法和数值,不要自己发明**。

---

## 1. 文风 & 每个文件的形态(全部 8 个统一)

- 中文注释,**大白话**,数学符号/术语**先用大白话铺一遍再用**(define-before-use)。
- 硬件概念**不打比方**,直接讲真硬件(warp / SM / lane / shfl),但**只在加分处讲**
  (s3 的索引讲轻一点,s5 的 shfl 才正式落 L09)。
- 不写没在本机验证过的经验数字;注释里每条技术论断都要对(审查会挑错的延迟数字 /
  讲错的 divergence)。
- 每个 `.cu` 顶部 header 注释必须含:
  1. **这一级新增的唯一概念**(一句话);
  2. **开场白**:"假设你已看过 s1..s{N-1},已经懂 {prev}。这一级只加 {delta}。";
  3. 怎么跑:`./build.sh sN`,预期最后一行 PASS;
  4. "学完应能回答" 的 2–3 个问题(答案放文件末尾)。
- 文件**末尾**附 `## 自检(先自己答,再看下面)` + `## 参考答案`。
- `main` 末尾**对拍打印 PASS/FAIL**,PASS 返回 0、FAIL 返回 1。
- 自包含:不依赖共享头(除 `<cuda_runtime.h>` 等系统头),每个文件能单独 `./build.sh sN`。

## 2. 共享代码(逐字复制,不要改名/改值)

```c
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
```

**canonical `eval_tree_d`**(s4 起逐字用这一份;s3 用下方的"BFUNC-only 子集"):

```c
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
```

s3 的 BFUNC-only 子集 = 上面去掉 `N_UFUNC` 整个分支(powerlaw 只用 MUL/POW)。
s3 这样保证"相对 s2 只加索引这一个 delta";完整解释器留到 s4(因为 expsat 要 EXP/NEG)。

**warp-per-problem 映射**(s4 起,凡 warp-per-problem 的 kernel 开头都用这段):

```c
int wpb  = blockDim.x / 32;     // BLOCK=256 → 8 warp = 8 棵树/block
int wb   = threadIdx.x / 32;
int lane = threadIdx.x & 31;
int m    = blockIdx.x * wpb + wb;
if (m >= M_prob) return;
```

**shfl 蝶形归约**(s5 起,warp 内求和都用这段):

```c
for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
```

**kernel 命名必须和生产一致**:`eval_kernel_batched`、`residual_kernel`、
`loss_kernel`、`build_jtj_jtr_kernel`、`solve_kernel`。device 指针 `d_*`、host `h_*`。

## 3. 三个 fixture(逐字复制,来自 m3_2_multi_tree.cu)

```c
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
```

数据点:`xs ∈ [0.1, 5.0]` 均匀采 `N=64`,`n_vars=1`:
`xs[i] = 0.1f + (5.0f-0.1f) * i/(N-1)`。

每棵树的解析值(独立 oracle,**导数也由此独立求**):
- powerlaw : `y = c0 * powf(x, c1)`
- quadratic: `y = c0 + c1*x + c2*x*x`
- expsat   : `y = c0 * expf(-c1*x) + c2`

## 4. 数值(从 m3 参考**抄**,写手不得自创,审查核对)

- **s6 FD**:`eps_fd = 1e-3`。FD 截断误差上限 ≈ √machine_eps ≈ 3e-4,所以验证容差
  用**相对 1e-2**(m3_3 实测 max_rel ≈ 2.5e-3)。oracle = 上面三式的**解析导数**。
- **s9 LM**(全部从 `m3_4c_lm.cu` 抄):`lambda` 初值 `1e-3`;accept → `λ*=0.1`
  (下限 `1e-12`);reject → `λ*=10`(`λ>1e12` 判失败);`xtol=1e-5`;收敛判据
  `d_norm < xtol*(c_norm + xtol)`;`max_iter` 用 m3_4c 的值。成功标准:
  `c_rel_err = ‖c_final−c_true‖/‖c_true‖ < 1e-2` 且 loss→~0。
- 其余对拍容差:s3 `rel<1e-5`(float ULP);s4 沿用 m3_2 判据
  (`ae > 1e-5*|ref| + 1e-7` 算 mismatch);s5/s7 `rel<1e-4`;s8 `‖Aδ−b‖/‖b‖<1e-4`。
- **容差不许放松到恒过,也不许紧到误报。** 审查确认。

## 5. oracle 独立性(TDD 的核心,审查死守)

每个 `.cu` 的对拍必须对照**独立实现的 CPU oracle**,不能是 device kernel 的转写,
也不能用松到恒过的容差(否则就是自欺的 tautology):
- s3/s4:对照**解析公式**(上面三式)。
- s5:CPU **独立**重算 residual 和 `0.5 Σr²`。
- s6:CPU 用**解析导数**当 oracle 比 FD 的 J(故意用不同方法)。
- s7:CPU **独立** double-loop 从同一份 J、r 算 JᵀJ/JᵀR。
- s8:CPU 用**高斯消元**(独立于 Cholesky)解同一 K×K 系统,或直接验 `A·δ ≈ b` 残差。
- s9:`c_final ≈ c_true`(c_rel_err)+ loss→~0。

## 6. 每级清单(file / 概念 / 镜像 / prev / delta / oracle)见 workflow STEPS;
   本文件负责"共享 + 数值 + oracle + 文风",STEPS 负责"每级具体增量"。
