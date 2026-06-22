// revad_interp.cuh — Reverse-mode AD (VJP) 解释器 + 单次 point-pass Jacobian kernel (v5)。
//
// 与 ad_interp.cuh (forward-mode JVP, v4) 的关系:
//   forward 每次前向推 AD_W 列切向量, 跑 ceil(K/8) 趟 → 成本 ∝ K。
//   reverse 一次前向 (记 tape) + 一次反向扫描 → 一次 point-pass 产出**全部 K 列**, K-flat。
//   y 是标量 (单输出), 常数 K 个 (多输入) ⟹ reverse 是对的模式 (= 反向传播)。
//
// 本文件 **#include "ad_interp.cuh"** 复用其:
//   - enum NTypeE / FuncE        (节点/算子枚举)
//   - MAX_STACK / MAX_K / AD_W   (编译期上界)
//   - eval_tree_jvp_d / ad_jacobian_kernel  (forward 参照, 给单测对拍用)
//   - eval_tree_val_host         (value-only 中心差分 oracle)
// **绝不修改 ad_interp.cuh** —— 仅 include 复用。算子规则逐条镜像 ad_interp.cuh (同 math、
// 同奇点行为), 使 reverse 的值与偏导和 forward 在有限处逐元素一致。
// 例外 (NaN-safety, 2026-06-22): 反向乘法用 revad_safe_mul 做 0-湮灭, 故在**值有限**的奇点
// (真梯度有限, 但朴素 AD 出 0*inf/inf*0=NaN) 处, reverse 给正确有限值而 forward 仍 NaN —— 这是
// 蓄意的、scipy 证实更正确的发散 (parity 测试归入已接受的 fwd-nan/rev-finite 桶, rev-worse 仍 0)。

#ifndef REVAD_INTERP_CUH
#define REVAD_INTERP_CUH

#include "ad_interp.cuh"   // 复用枚举/上界/forward 参照/eval_tree_val_host (含 <cmath>)

// 每点 reverse tape 的逐节点上界。观测 (synth early-gen / inner-const) max n_nodes = 32;
// 128 给 4× 余量。tape = d1[MAX_NODES]+d2[MAX_NODES] floats/lane, 加 sv/sa[MAX_STACK],
// 每 lane ~1.5KB local mem (仍 < forward-AD 的 ~2.3KB st[MAX_STACK*AD_W])。
// batch_lm_revad.cu 在 load 期断言 max(n_nodes) <= MAX_NODES (镜像 max_stack 检查)。
#ifndef MAX_NODES
#define MAX_NODES 128
#endif

// adjoint × local-partial, 带 0-湮灭 (NaN-safety)。见 experiments/revad_v5/
// FINDINGS_codex_review_shared_nan.md: 若任一因子**恰为 0.0f**, 该边对梯度的真实贡献为 0
// (0 伴随 ⟹ 该节点不影响输出; 0 偏导 ⟹ 该操作数不影响其父节点), 故返回 0 —— 不让
// 0*inf / inf*0 / 0*nan 把该列毒化成 NaN。数学上 true-zero 因子湮灭乘积, 与另一 (可能非有限)
// 因子的取值/极限无关。仅"恰为 0.0f"时湮灭: denormal/微小非零仍走 a*b (此时积也微小, 在容差内),
// 真正 inf 偏导配非零 adjoint (真奇点, 真梯度确为 inf) 仍给 inf。
// **已知局限 (Codex P2)**: 可去奇点 —— 若 0·∞ 实际收敛到非零有限极限 (如 pow(sqrt(c0),2) 在 c0=0,
// 真右导=1), 本 mask 会**沉默地给 0** 而非真值。这是一阶 AD 的固有局限: forward AD、未加 mask 的
// reverse、中心差分 在此点同样失败 (全给 NaN); 只有符号化简/单侧差分能得真值。实测我们的语料
// (inner-const 27.4M + early-gen 45.4M) **0 例** (tests/_probe_corpus_silent.cu): 全部被 mask 影响的
// 9000+3000 个元素真梯度均为 0, mask 给 0 是对的。保留 mask: 对真实语料全对, 且 0 比 NaN 对 LM 更安全
// (NaN 毒化整列; 0 仅少计该点)。纯局部无法区分"真零"与"可去非零"(局部都是 0·∞)。
// scipy fp64 oracle + Python fp32 mirror 证实: 修掉样本全部 36 个 shared-NaN 残留, broke=0。
__host__ __device__ inline float revad_safe_mul(float a, float b) {
    return (a == 0.0f || b == 0.0f) ? 0.0f : a * b;
}

// ============================================================================
// Reverse-mode VJP 解释器 — 单点, 一次 pass 产出全部 K 列 ∂y/∂c_j
// ============================================================================
//
// FORWARD pass (i = n_nodes-1 .. 0, 与 eval_tree_d 同序): 值栈 sv 算每节点值;
//   逐节点把**本地偏导**记到 tape:
//     UFUNC: d1[i] = df/da
//     BFUNC: d1[i] = ∂o/∂l,  d2[i] = ∂o/∂rv
//   规则逐条镜像 ad_interp.cuh 的 eval_tree_jvp_d (含 POW base_coef、DIV、MAX/MIN 选择、比较恒 0)。
// BACKWARD pass (i = 0 .. n_nodes-1, 前向序 = forward 的逆序): 伴随栈 sa 镜像值栈的 push/pop:
//   起始 sa=[1.0] (= ∂y/∂y)。每节点先弹出该节点输出的 adjoint, 再:
//     VAR  : 丢弃
//     CONST: out_grad[ci[i]] += adj
//     UFUNC: push (adj * d1[i])                      // operand a 的 adjoint
//     BFUNC: push (adj * d2[i]) 然后 push (adj * d1[i]) // 先 rv-adj (深), 再 l-adj (顶),
//            镜像 forward 弹序 (forward 先弹顶=l, 再弹下=rv)
// out_val 不输出 (需要值用 eval_tree_val_host)。
__host__ __device__ inline void eval_tree_vjp_d(
    const int *nt, const float *nv, int n_nodes, const int *ci,
    const float *x, const float *c, int K,
    float *out_grad)
{
    float sv[MAX_STACK];          // 值栈
    float d1[MAX_NODES];          // tape: UFUNC df/da 或 BFUNC ∂o/∂l
    float d2[MAX_NODES];          // tape: BFUNC ∂o/∂rv
    int sp = 0;

    // ---- FORWARD: 算值 + 记 tape -------------------------------------------
    for (int i = n_nodes - 1; i >= 0; i--) {
        int t = nt[i];
        float v = nv[i];
        if (t == N_VAR) {
            sv[sp++] = x[(int)v];
        } else if (t == N_CONST) {
            sv[sp++] = c[ci[i]];
        } else if (t == N_UFUNC) {
            int fid = (int)v;
            float a = sv[--sp];
            float r, d;
            switch (fid) {
                case F_SIN:  r = sinf(a);  d = cosf(a);            break;
                case F_COS:  r = cosf(a);  d = -sinf(a);           break;
                case F_TAN:  r = tanf(a);  d = 1.0f + r * r;       break; // sec^2 = 1+tan^2
                case F_SINH: r = sinhf(a); d = coshf(a);           break;
                case F_COSH: r = coshf(a); d = sinhf(a);           break;
                case F_TANH: r = tanhf(a); d = 1.0f - r * r;       break;
                case F_LOG:  r = logf(a);  d = 1.0f / a;           break; // a<=0 → r NaN/inf (与 eval 一致)
                case F_EXP:  r = expf(a);  d = r;                  break;
                case F_INV:  r = 1.0f / a; d = -r * r;             break; // a==0 → r inf
                case F_NEG:  r = -a;       d = -1.0f;              break;
                case F_ABS:  r = fabsf(a); d = (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f); break; // sign(0):=0
                case F_SQRT: r = sqrtf(a); d = 1.0f / (2.0f * r);  break; // a==0 → d inf (与 eval 一致)
                default:     r = 0.0f;     d = 0.0f;                       // unknown op — silent failure 入口
            }
            d1[i] = d;
            sv[sp++] = r;
        } else if (t == N_BFUNC) {
            int fid = (int)v;
            float l  = sv[--sp];   // 顶 = l
            float rv = sv[--sp];   // 下 = rv
            float o, dl, dr;
            switch (fid) {
                case F_ADD: o = l + rv; dl = 1.0f;       dr = 1.0f;       break;
                case F_SUB: o = l - rv; dl = 1.0f;       dr = -1.0f;      break;
                case F_MUL: o = l * rv; dl = rv;         dr = l;          break;
                case F_DIV: o = l / rv; dl = 1.0f / rv;  dr = -o / rv;    break; // ∂/∂l=1/rv, ∂/∂rv=-o/rv (= -l/rv^2)
                case F_POW: {
                    o = powf(l, rv);
                    dl = rv * powf(l, rv - 1.0f);   // ∂/∂l = base_coef (与 forward 逐字一致)
                    dr = o * logf(l);               // ∂/∂rv = o*log(l); l<=0 → NaN (与 forward exp_has_tan 路径一致)
                    break;
                }
                case F_MAX: o = fmaxf(l, rv); dl = (l >= rv) ? 1.0f : 0.0f; dr = (l >= rv) ? 0.0f : 1.0f; break;
                case F_MIN: o = fminf(l, rv); dl = (l <= rv) ? 1.0f : 0.0f; dr = (l <= rv) ? 0.0f : 1.0f; break;
                case F_LT:  o = (l <  rv) ? 1.0f : 0.0f; dl = 0.0f; dr = 0.0f; break;
                case F_GT:  o = (l >  rv) ? 1.0f : 0.0f; dl = 0.0f; dr = 0.0f; break;
                case F_LE:  o = (l <= rv) ? 1.0f : 0.0f; dl = 0.0f; dr = 0.0f; break;
                case F_GE:  o = (l >= rv) ? 1.0f : 0.0f; dl = 0.0f; dr = 0.0f; break;
                default:    o = 0.0f; dl = 0.0f; dr = 0.0f;                     // unknown op — silent failure 入口
            }
            d1[i] = dl;
            d2[i] = dr;
            sv[sp++] = o;
        }
    }

    // ---- BACKWARD: 伴随栈反扫, CONST 处累加梯度 -----------------------------
    for (int k = 0; k < K; k++) out_grad[k] = 0.0f;
    float sa[MAX_STACK];
    int asp = 0;
    sa[asp++] = 1.0f;                       // ∂y/∂y (forward 末态 sv 仅根值一项)

    for (int i = 0; i < n_nodes; i++) {
        int t = nt[i];
        if (t == N_VAR) {
            --asp;                          // 丢弃 (VAR 无可拟合常数)
        } else if (t == N_CONST) {
            float adj = sa[--asp];
            out_grad[ci[i]] += adj;
        } else if (t == N_UFUNC) {
            float adj = sa[--asp];                      // 该节点输出的 adjoint
            sa[asp++] = revad_safe_mul(adj, d1[i]);     // operand a 的 adjoint (0-湮灭防 0*inf)
        } else if (t == N_BFUNC) {
            float adj = sa[--asp];                      // 输出 o 的 adjoint
            sa[asp++] = revad_safe_mul(adj, d2[i]);     // rv-adjoint (深)
            sa[asp++] = revad_safe_mul(adj, d1[i]);     // l-adjoint  (顶); 镜像 forward 弹序 (l 在顶)
        }
    }
}

// ============================================================================
// rev_jacobian_kernel — 逐字节 drop-in 替换 ad_jacobian_kernel / fd_jacobian。
// ============================================================================
// warp-per-tree, 32 lane 跨 N。输出契约与 ad_jacobian_kernel 完全一致:
//   Jm = J + (size_t)m * K_max * N;  Jm[(size_t)j*N + i] = ∂y_i/∂c_j  (j < K[m], i < N)。
//   列 j >= K[m] 不写; K == 0 的树直接 return。
// 与 forward 不同: 一次 point-pass (eval_tree_vjp_d) 即得全部 K 列 (无 col_lo+=AD_W 外层)。
__global__ void rev_jacobian_kernel(
    const int *nt_all, const float *nv_all, const int *ci_all,
    const TreeMeta *metas, int M_prob,
    const float *xs, int n_vars, int N,
    const float *c_all, int K_max, float *J)
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

    for (int i = lane; i < N; i += 32) {
        float g[MAX_K];                               // 全部 K 列 ∂y_i/∂c_j
        eval_tree_vjp_d(nt, nv, meta.n_nodes, ci, xs + (size_t)i * n_vars, c, K, g);
        for (int j = 0; j < K; j++)
            Jm[(size_t)j * N + i] = g[j];             // 精确 ∂y_i/∂c_j
    }
}

#endif // REVAD_INTERP_CUH
