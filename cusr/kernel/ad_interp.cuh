// ad_interp.cuh — Forward-mode AD (JVP) 解释器 + chunked Jacobian kernel.
//
// v4 的单一真源 (single source of truth), 见 docs/kernel/AD_JACOBIAN_V4.md §5.1:
//   - enum NTypeE / F_*           (节点类型 + 算子枚举, 与 batch_lm_fusedfd.cu 一致)
//   - MAX_STACK / MAX_K / AD_W    (编译期上界; AD_W 默认 8, -DAD_W=N 可调)
//   - eval_tree_jvp_d(...)        (前向 JVP 解释器, __host__ __device__ 以便 host 单测)
//   - ad_jacobian_kernel(...)     (Rung 2 加入; chunked forward J 生产, drop-in d_J)
//   - eval_tree_val_host(...)     (value-only oracle, **仅** host 单测中心差分参考; 改名以避与
//                                  batch_lm_ad.cu 里的 eval_tree_d 冲突)
//
// 把 eval_tree_d 的标量栈机升级成带切向量的栈机: 每个栈槽 (value, tangent[W]),
// tangent[w] = ∂value/∂c_{col_lo+w}. 全部用 host+device 安全 math (sinf/cosf/.../powf),
// 无 device-only intrinsic, 故同一份代码 host (nvcc CPU) / device 都能跑。

#ifndef AD_INTERP_CUH
#define AD_INTERP_CUH

#include "pop_format.h"   // TreeMeta

// ---- 节点类型 + 算子枚举 (与 batch_lm_fusedfd.cu 严格一致) -----------------
enum NTypeE { N_VAR=0, N_CONST=1, N_UFUNC=2, N_BFUNC=3, N_TFUNC=4 };
// Func enum 跟 EvoGP utils.py 严格对齐. LOOSE_* 在 dump 端退化, 解算端不识别
// (走 default 0.0 — 视作未知 op silent failure 入口).
enum FuncE {
    F_ADD=1, F_SUB=2, F_MUL=3, F_DIV=4, F_POW=6,
    F_MAX=8, F_MIN=9, F_LT=10, F_GT=11, F_LE=12, F_GE=13,
    F_SIN=14, F_COS=15, F_TAN=16,
    F_SINH=17, F_COSH=18, F_TANH=19,
    F_LOG=20, F_EXP=22, F_INV=23,
    F_NEG=25, F_ABS=26, F_SQRT=27,
};

// ---- 编译期上界 (用 #ifndef 守卫, 便于覆写) -------------------------------
#ifndef MAX_STACK
#define MAX_STACK 64
#endif
#ifndef MAX_K
#define MAX_K     32
#endif
#ifndef AD_W
#define AD_W      8       // chunk 宽 (一趟前向同时算的常数列数). 见 §5 lmem 预算.
#endif

#include <cmath>          // host fp32 math (sinf, cosf, logf, powf, ...)

// ============================================================================
// Forward-mode JVP 解释器 — 单点 (value + 对 c[col_lo, col_lo+W) 的切向量)
// ============================================================================
//
// 镜像 eval_tree_d (reverse-order 栈机), 但每槽携带 (value, tangent[W])。
//   叶子 seed:
//     N_VAR:           push (x[v], 0)
//     N_CONST (k=ci):  push (c[k], e),  e[w] = (k == col_lo+w) ? 1 : 0
//   BFUNC: 先弹 l 再弹 rv, o=f(l,rv), 切向量按 §4 规则表更新。
//
// W 是运行期有效宽 (<= AD_W); 切向量在栈上以固定步长 AD_W 存储, 只用前 W 列。
// out_val 写值, out_t[0..W) 写切向量。
__host__ __device__ inline void eval_tree_jvp_d(
    const int *nt, const float *nv, int n_nodes, const int *ci,
    const float *x, const float *c, int col_lo, int W,
    float *out_val, float *out_t)
{
    float sv[MAX_STACK];                 // 栈: 值
    float st[MAX_STACK * AD_W];          // 栈: 切向量 (槽 s 的第 w 列 = st[s*AD_W + w])
    int sp = 0;

    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i];
        float v = nv[i];
        if (t == N_VAR) {
            sv[sp] = x[(int)v];
            float *ts = st + sp * AD_W;
            for (int w = 0; w < W; w++) ts[w] = 0.0f;
            sp++;
        } else if (t == N_CONST) {
            int k = ci[i];
            sv[sp] = c[k];
            float *ts = st + sp * AD_W;
            for (int w = 0; w < W; w++) ts[w] = (k == col_lo + w) ? 1.0f : 0.0f;
            sp++;
        } else if (t == N_UFUNC) {
            int fid = (int)v;
            float a = sv[--sp];
            float *ta = st + sp * AD_W;   // da (in place; 结果写回同一槽)
            float r;
            // 每个 case 先算 r, 再原地把 da -> dr。
            switch (fid) {
                case F_SIN: {
                    r = sinf(a);
                    float d = cosf(a);
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_COS: {
                    r = cosf(a);
                    float d = -sinf(a);
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_TAN: {
                    r = tanf(a);
                    float d = 1.0f + r * r;            // sec^2 = 1 + tan^2
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_SINH: {
                    r = sinhf(a);
                    float d = coshf(a);
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_COSH: {
                    r = coshf(a);
                    float d = sinhf(a);
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_TANH: {
                    r = tanhf(a);
                    float d = 1.0f - r * r;
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_LOG: {
                    r = logf(a);                      // a<=0 → inf/nan, 与 eval 一致
                    for (int w = 0; w < W; w++) ta[w] = ta[w] / a;
                    break;
                }
                case F_EXP: {
                    r = expf(a);
                    for (int w = 0; w < W; w++) ta[w] = r * ta[w];
                    break;
                }
                case F_INV: {
                    r = 1.0f / a;                     // a==0 → inf, 与 eval 一致
                    float d = -r * r;
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                case F_NEG: {
                    r = -a;
                    for (int w = 0; w < W; w++) ta[w] = -ta[w];
                    break;
                }
                case F_ABS: {
                    r = fabsf(a);
                    float s = (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f); // sign(0):=0
                    for (int w = 0; w < W; w++) ta[w] = s * ta[w];
                    break;
                }
                case F_SQRT: {
                    r = sqrtf(a);                     // a<0 → nan; a==0 → r=0 → da/(2r)=inf, 与 eval 一致
                    float d = 1.0f / (2.0f * r);
                    for (int w = 0; w < W; w++) ta[w] = d * ta[w];
                    break;
                }
                default:
                    r = 0.0f;                         // unknown op — silent failure 入口
                    for (int w = 0; w < W; w++) ta[w] = 0.0f;
            }
            sv[sp] = r;                               // 写回同一槽 (sp 已是弹出后的位置)
            sp++;
        } else if (t == N_BFUNC) {
            int fid = (int)v;
            // 先弹 l 再弹 rv (与 eval_tree_d 一致)。l 在栈顶 (高地址), rv 在其下。
            float l  = sv[--sp]; float *tl = st + sp * AD_W;
            float rv = sv[--sp]; float *tr = st + sp * AD_W;
            float *to = tr;                           // 结果落到下层槽 (rv 的位置)
            float o;
            switch (fid) {
                case F_ADD: {
                    o = l + rv;
                    for (int w = 0; w < W; w++) to[w] = tl[w] + tr[w];
                    break;
                }
                case F_SUB: {
                    o = l - rv;
                    for (int w = 0; w < W; w++) to[w] = tl[w] - tr[w];
                    break;
                }
                case F_MUL: {
                    o = l * rv;
                    for (int w = 0; w < W; w++) to[w] = l * tr[w] + rv * tl[w];
                    break;
                }
                case F_DIV: {
                    o = l / rv;                       // rv==0 → inf/nan, 与 eval 一致
                    for (int w = 0; w < W; w++) to[w] = (tl[w] - o * tr[w]) / rv;
                    break;
                }
                case F_POW: {
                    o = powf(l, rv);
                    // 指数项守卫: 指数常为定常数 (dr==0), 此时跳过 logf(l) (l<=0 假 NaN)。
                    int exp_has_tan = 0;
                    for (int w = 0; w < W; w++) if (tr[w] != 0.0f) { exp_has_tan = 1; break; }
                    float base_coef = rv * powf(l, rv - 1.0f);   // ∂/∂l
                    if (exp_has_tan) {
                        float lg = logf(l);                       // ∂/∂rv = o*log(l)
                        for (int w = 0; w < W; w++)
                            to[w] = base_coef * tl[w] + o * lg * tr[w];
                    } else {
                        for (int w = 0; w < W; w++)
                            to[w] = base_coef * tl[w];
                    }
                    break;
                }
                case F_MAX: {
                    o = fmaxf(l, rv);
                    for (int w = 0; w < W; w++) to[w] = (l >= rv) ? tl[w] : tr[w];
                    break;
                }
                case F_MIN: {
                    o = fminf(l, rv);
                    for (int w = 0; w < W; w++) to[w] = (l <= rv) ? tl[w] : tr[w];
                    break;
                }
                case F_LT: o = (l <  rv) ? 1.0f : 0.0f; for (int w=0;w<W;w++) to[w]=0.0f; break;
                case F_GT: o = (l >  rv) ? 1.0f : 0.0f; for (int w=0;w<W;w++) to[w]=0.0f; break;
                case F_LE: o = (l <= rv) ? 1.0f : 0.0f; for (int w=0;w<W;w++) to[w]=0.0f; break;
                case F_GE: o = (l >= rv) ? 1.0f : 0.0f; for (int w=0;w<W;w++) to[w]=0.0f; break;
                default:
                    o = 0.0f;                         // unknown op — silent failure 入口
                    for (int w = 0; w < W; w++) to[w] = 0.0f;
            }
            sv[sp] = o;                               // sp 现指向下层槽 (= to 所在)
            sp++;
        }
    }

    *out_val = sv[0];
    float *t0 = st + 0 * AD_W;
    for (int w = 0; w < W; w++) out_t[w] = t0[w];
}

// ============================================================================
// ad_jacobian_kernel — chunked forward-mode Jacobian (drop-in d_J producer)
// ============================================================================
//
// 逐字节 drop-in 替换 fd_jacobian_fused_kernel: warp-per-tree, 32 lane 跨 N。
// 输出契约 (见 §3, 必须与 fd_jacobian 逐字节一致):
//   Jm = J + (size_t)m * K_max * N
//   Jm[(size_t)j * N + i] = ∂y_i / ∂c_j     for j < K[m], i < N
//   列 j >= K[m] 不写 (build_jtj 也不读); K == 0 的树直接 return。
// 与 FD 不同: 无 eps, 无 y_base (AD 精确), 一趟前向同时算 AD_W 列。
// 外层按 AD_W 分桶遍历 K, 每桶对 N 个点跑 eval_tree_jvp_d, 写 W 列切向量。
__global__ void ad_jacobian_kernel(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N,
    const float *c_all, int K_max, float *J)   // 注意: 无 eps, 无 y_base
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
    float       *Jm = J      + (size_t)m * K_max * N;

    for (int col_lo = 0; col_lo < K; col_lo += AD_W) {
        int W = min(AD_W, K - col_lo);                // 尾桶 W < AD_W
        for (int i = lane; i < N; i += 32) {
            float val, t[AD_W];
            eval_tree_jvp_d(nt, nv, meta.n_nodes, ci, xs + (size_t)i * n_vars,
                            c, col_lo, W, &val, t);
            for (int w = 0; w < W; w++)
                Jm[(size_t)(col_lo + w) * N + i] = t[w];   // 精确 ∂y_i/∂c_{col_lo+w}
        }
    }
}

// ============================================================================
// value-only oracle — **仅** host 单测的中心差分参考 (不要命名 eval_tree_d)
// ============================================================================
// 与 eval_tree_d 等价的纯值栈机, 用于在 host 上对 JVP 切向量做中心差分对拍。
__host__ __device__ inline float eval_tree_val_host(
    const int *nt, const float *nv, int n_nodes, const int *ci,
    const float *x, const float *c)
{
    float stack[MAX_STACK]; int sp = 0;
    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i]; float v = nv[i];
        if (t == N_VAR) stack[sp++] = x[(int)v];
        else if (t == N_CONST) stack[sp++] = c[ci[i]];
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
                default:     r=0.0f;
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
                default:    o=0.0f;
            }
            stack[sp++] = o;
        }
    }
    return stack[0];
}

#endif // AD_INTERP_CUH
