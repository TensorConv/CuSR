// test_revad_host.cu — Reverse-AD T1 host 单测: reverse-mode VJP 解释器 vs 中心差分。
//
// 镜像 tests/test_jvp_host.cu (forward JVP 版), 但被测对象是 reverse-mode
// eval_tree_vjp_d(nt, nv, n_nodes, ci, x, c, K, out_grad[K]) —— 一次 point-pass
// 产出**全部 K 列** ∂y/∂c_j。**只** include revad_interp.cuh (它 transitively
// 拉 ad_interp.cuh, 给 eval_tree_val_host 当中心差分 oracle)。nvcc 编成 host 程序
// (CPU 跑), 不需要 GPU。
//
// 关键: eval_tree_vjp_d **没有 out_val** 输出 (不像 forward 的 eval_tree_jvp_d)。
// 需要值时一律调 eval_tree_val_host。
//
//   每个 op (12 UFUNC + 11 BFUNC) + VAR/CONST seed:
//     建最小 token 树, eval_tree_vjp_d 反向梯度 vs 中心差分, max rel err < 1e-3 → PASS。
//   额外硬覆盖 (镜像 test_jvp_host §7 风险点):
//     c0^2 (定常指数 POW), x^c0 (含参指数 POW), 双侧含参 POW,
//     log(c0*x) 在 c0*x<=0 (value 必须 NaN/inf; 但梯度 1/c0 有限 → 也对拍),
//     一个 division, 嵌套 sin(c0*x)+c1*exp(c2*x) K=3, K=1 极端量级 1e-6 / 1e6。
//
// 容差: rel err < 1e-3 (abs floor 1e-4); 任何 NaN/Inf 出现在应有限的梯度 = FAIL。
// 最后一行 RESULT: PASS / RESULT: FAIL, exit code 反映之。
//
// build/gate:
//   source /home/weish/hao/CuSR/scripts/env.sh && cd /home/weish/hao/CuSR/cusr/kernel && \
//   nvcc -O2 -std=c++17 -o tests/test_revad_host tests/test_revad_host.cu && \
//   ./tests/test_revad_host ; echo EXIT=$?

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../revad_interp.cuh"   // reverse-AD impl (transitively pulls ad_interp.cuh)

// ---- 树容器 (host 端小工具) -------------------------------------------------
struct Tree {
    std::vector<int>   nt;
    std::vector<float> nv;
    std::vector<int>   ci;   // const_idx, -1 if not CONST
};

static int g_fail = 0;
static int g_pass = 0;

// 中心差分参考: ∂value/∂c[k] at (x, c)。step ~ 1e-3*|c[k]|, abs floor 1e-5。
// (与 test_jvp_host.cu 完全一致的 oracle。)
static float central_diff(const Tree &T, const float *x, const float *c, int K, int k) {
    std::vector<float> cp(c, c + K);
    float ck = c[k];
    float h = 1e-3f * fabsf(ck);
    if (h < 1e-5f) h = 1e-5f;
    cp[k] = ck + h;
    float fp = eval_tree_val_host(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x, cp.data());
    cp[k] = ck - h;
    float fm = eval_tree_val_host(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x, cp.data());
    return (fp - fm) / (2.0f * h);
}

// reverse-mode 一次 pass 拿 K 列梯度, 对每列 vs 中心差分。
// 返回 max_{k in [k_lo, k_hi)} |g[k] - fd_k| / (|fd_k| + abs_floor)。
// k_lo/k_hi 允许只检子集列 (POW 定常指数那种只想检 base 列的 case)。
// g[] 里任何 NaN/Inf (在我们要检的列) 也强制 FAIL (rel=HUGE_VALF)。
static float check_cols(const Tree &T, const float *x, const float *c,
                        int K, int k_lo, int k_hi) {
    std::vector<float> g(K, 0.0f);
    eval_tree_vjp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(),
                    x, c, K, g.data());
    float worst = 0.0f;
    for (int k = k_lo; k < k_hi; k++) {
        float fd = central_diff(T, x, c, K, k);
        float denom = fabsf(fd) + 1e-4f;
        float rel = fabsf(g[k] - fd) / denom;
        bool bad = !isfinite(g[k]);   // 应有限的梯度若 NaN/Inf → 强制 FAIL
        if (bad || rel > worst) worst = bad ? HUGE_VALF : rel;
    }
    return worst;
}

// 单 op / 单点 PASS/FAIL 报告。默认检全部列 [0, K)。
static void report(const char *name, const Tree &T, const float *x, const float *c, int K) {
    float worst = check_cols(T, x, c, K, 0, K);
    bool ok = (worst < 1e-3f);
    if (!ok) g_fail++; else g_pass++;
    printf("  %-26s K=%d  max_rel_err=%.3e  %s\n", name, K, worst, ok ? "PASS" : "FAIL");
}

// 解析式断言 (不用中心差分 oracle): 给奇点 fixture 用 —— 中心差分在 sqrt/log 定义域边界
// 自己会变 NaN (步长跨进负域), 故这些 case 直接比手算的已知真梯度。要求 value 有限 (确认
// 这是"有限值奇点", 真梯度有意义); 任一被检列出现 NaN/Inf = FAIL; |g[k]-expect[k]| 超 tol = FAIL。
static void report_analytic(const char *name, const Tree &T, const float *x,
                            const float *c, int K, const float *expect, int k_lo, int k_hi) {
    std::vector<float> g(K, 0.0f);
    eval_tree_vjp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x, c, K, g.data());
    float val = eval_tree_val_host(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x, c);
    float worst = 0.0f; bool any_bad = false;
    for (int k = k_lo; k < k_hi; k++) {
        bool bad = !isfinite(g[k]);
        float rel = fabsf(g[k] - expect[k]) / (fabsf(expect[k]) + 1e-4f);
        if (bad) any_bad = true;
        if (bad || rel > worst) worst = bad ? HUGE_VALF : rel;
    }
    bool ok = isfinite(val) && !any_bad && (worst < 1e-3f);
    if (!ok) g_fail++; else g_pass++;
    printf("  %-26s K=%d val=%.4g max_rel_err=%.3e  %s\n", name, K, val, worst, ok ? "PASS" : "FAIL");
}

// ---- 树构造助手 (镜像 test_jvp_host.cu) --------------------------------------
// op(c0*x0), K=1, var x0。 pre-order: [UFUNC(op), BFUNC(MUL), CONST(c0), VAR(x0)]
static Tree ufunc_of_c0x(int op) {
    Tree T;
    T.nt = {N_UFUNC, N_BFUNC, N_CONST, N_VAR};
    T.nv = {(float)op, (float)F_MUL, 0.0f, 0.0f};
    T.ci = {-1, -1, 0, -1};
    return T;
}
// (c0*x0) <op> (c1*x0), K=2。
//   pre-order: [BFUNC(op), BFUNC(MUL,c0,x0), BFUNC(MUL,c1,x0)]
static Tree bin_two_terms(int op) {
    Tree T;
    T.nt = {N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_BFUNC, N_CONST, N_VAR};
    T.nv = {(float)op, (float)F_MUL, 0.0f, 0.0f, (float)F_MUL, 0.0f, 0.0f};
    T.ci = {-1, -1, 0, -1, -1, 1, -1};
    return T;
}

int main() {
    printf("[test_revad_host] reverse-mode VJP vs central difference (tol rel < 1e-3)\n");

    // 单变量点。各 op 选安全 x/c 避奇点 (奇点单独测)。与 test_jvp_host.cu 同。
    float x1[1] = {0.7f};

    // ---------------- VAR / CONST seeding ----------------
    // CONST seed: y = c0 (恒等), ∂/∂c0 = 1。  [CONST(c0)]
    {
        Tree T; T.nt = {N_CONST}; T.nv = {0.0f}; T.ci = {0};
        float c[1] = {1.3f};
        report("seed_CONST", T, x1, c, 1);
    }
    // VAR seed: y = c0*x0 → ∂/∂c0 = x0 (验 VAR 叶子丢弃 adjoint + CONST 收集 adjoint)。
    {
        Tree T; T.nt = {N_BFUNC, N_CONST, N_VAR}; T.nv = {(float)F_MUL, 0.0f, 0.0f}; T.ci = {-1, 0, -1};
        float c[1] = {0.9f};
        report("seed_VAR(c0*x0)", T, x1, c, 1);
    }

    // ---------------- 12 UFUNC: op(c0*x0), K=1 ----------------
    {
        struct { const char *nm; int op; float c0; } U[] = {
            {"SIN",  F_SIN,  1.1f},
            {"COS",  F_COS,  1.1f},
            {"TAN",  F_TAN,  0.8f},   // c0*x0 = 0.56, 远离 π/2
            {"SINH", F_SINH, 0.9f},
            {"COSH", F_COSH, 0.9f},
            {"TANH", F_TANH, 1.3f},
            {"LOG",  F_LOG,  1.4f},   // c0*x0 = 0.98 > 0
            {"EXP",  F_EXP,  1.0f},
            {"INV",  F_INV,  1.2f},   // c0*x0 = 0.84 != 0
            {"NEG",  F_NEG,  1.7f},
            {"ABS",  F_ABS,  1.5f},   // c0*x0 = 1.05 > 0 (smooth)
            {"SQRT", F_SQRT, 1.3f},   // c0*x0 = 0.91 > 0
        };
        for (auto &u : U) {
            Tree T = ufunc_of_c0x(u.op);
            float c[1] = {u.c0};
            char nm[40]; snprintf(nm, sizeof(nm), "UFUNC_%s", u.nm);
            report(nm, T, x1, c, 1);
        }
    }

    // ---------------- 11 BFUNC ----------------
    // ADD/SUB/MUL/DIV: (c0*x0) <op> (c1*x0), K=2。
    {
        struct { const char *nm; int op; float c0, c1; } B[] = {
            {"ADD", F_ADD, 1.2f, 0.8f},
            {"SUB", F_SUB, 1.2f, 0.8f},
            {"MUL", F_MUL, 1.1f, 0.9f},
            {"DIV", F_DIV, 1.3f, 0.7f},   // rv = c1*x0 = 0.49 != 0
        };
        for (auto &b : B) {
            Tree T = bin_two_terms(b.op);
            float c[2] = {b.c0, b.c1};
            char nm[40]; snprintf(nm, sizeof(nm), "BFUNC_%s", b.nm);
            report(nm, T, x1, c, 2);
        }
    }
    // MAX/MIN: 两支 (c0*x0) vs (c1*x0), 选 c0,c1 使一支严格胜 (避免平局处不可导)。
    {
        struct { const char *nm; int op; float c0, c1; } B[] = {
            {"MAX", F_MAX, 1.5f, 0.5f},   // c0*x0 > c1*x0 → 选 l 支 (∂/∂c0=x0, ∂/∂c1=0)
            {"MIN", F_MIN, 1.5f, 0.5f},   // c0*x0 > c1*x0 → MIN 选 rv 支 (∂/∂c0=0, ∂/∂c1=x0)
        };
        for (auto &b : B) {
            Tree T = bin_two_terms(b.op);
            float c[2] = {b.c0, b.c1};
            char nm[40]; snprintf(nm, sizeof(nm), "BFUNC_%s", b.nm);
            report(nm, T, x1, c, 2);
        }
    }
    // LT/GT/LE/GE: 导数恒 0。中心差分也给 0 (除非恰跨阈值)。选远离相等点。
    {
        struct { const char *nm; int op; } B[] = {
            {"LT", F_LT}, {"GT", F_GT}, {"LE", F_LE}, {"GE", F_GE},
        };
        for (auto &b : B) {
            Tree T = bin_two_terms(b.op);
            float c[2] = {1.5f, 0.5f};   // 两支差距大, 小扰动不跨阈值 → fd≈0, grad=0
            char nm[40]; snprintf(nm, sizeof(nm), "BFUNC_%s", b.nm);
            report(nm, T, x1, c, 2);
        }
    }

    // ---------------- POW: 必测 c0^2 (定常指数) ----------------
    // [BFUNC(POW), CONST(c0=base), CONST(exp=c1)]; 只检 col 0 (base)。
    //   exp 列 (c1) 是定常指数, reverse 对它也会算梯度 (∂/∂c1 = c0^c1 * log(c0)),
    //   c0=1.4>0 → log 有定义, 该列也对拍 → 这里检全 K=2 列 (base + exp 都该对)。
    {
        Tree T; T.nt = {N_BFUNC, N_CONST, N_CONST}; T.nv = {(float)F_POW, 0.0f, 0.0f}; T.ci = {-1, 0, 1};
        float c[2] = {1.4f, 2.0f};
        report("POW_c0sq(base+exp)", T, x1, c, 2);
        // 另检: 只对 base 列 (∂/∂c0 = 2*c0), 卡 reverse 在 exp_has_tan 路径外的 base 偏导。
        float w = check_cols(T, x1, c, 2, 0, 1);
        bool ok = (w < 1e-3f); if (!ok) g_fail++; else g_pass++;
        printf("  %-26s K=%d  max_rel_err=%.3e  %s\n", "POW_c0sq(d/dc0 only)", 2, w, ok ? "PASS" : "FAIL");
    }

    // ---------------- POW: 含参指数 x^c0 ----------------
    // y = x0 ^ c0, ∂/∂c0 = x0^c0 * log(x0)。base=VAR(x0), exp=CONST(c0)。
    //   [BFUNC(POW), VAR(x0), CONST(c0)]  x0=0.7>0 → log 有定义。
    {
        Tree T; T.nt = {N_BFUNC, N_VAR, N_CONST}; T.nv = {(float)F_POW, 0.0f, 0.0f}; T.ci = {-1, -1, 0};
        float c[1] = {1.8f};
        report("POW_x_pow_c0", T, x1, c, 1);
    }
    // POW: 两侧含参 (base+exp 都带梯度) — base=c0*x0, exp=c1。
    //   y = (c0*x0)^c1, ∂/∂c0 = c1*(c0*x0)^(c1-1)*x0, ∂/∂c1 = (c0*x0)^c1 * log(c0*x0)。
    //   [BFUNC(POW), BFUNC(MUL,c0,x0), CONST(c1)]  c0*x0>0 → log 有定义。
    {
        Tree T;
        T.nt = {N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST};
        T.nv = {(float)F_POW, (float)F_MUL, 0.0f, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1, 1};
        float c[2] = {1.3f, 1.6f};   // c0*x0 = 0.91 > 0
        report("POW_base+exp_param", T, x1, c, 2);
    }

    // ---------------- LOG 奇点: log(c0*x0) at c0*x0 <= 0 ----------------
    // value 必须 NaN/inf (非错的有限值)。但**梯度** ∂/∂c0 = 1/c0 是有限的, reverse
    // 应当干净给出 (此处没有 0*NaN: 唯一 NaN 在 value, 不在任何 local partial)。
    // → 两条都查: (1) value 非有限 (经 eval_tree_val_host), (2) 梯度 == 1/c0 (rel<1e-3)。
    {
        Tree T = ufunc_of_c0x(F_LOG);
        float c[1] = {-1.0f};   // c0*x0 = -0.7 < 0 → log → NaN
        // (1) value 必须 NaN/inf
        float val = eval_tree_val_host(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x1, c);
        bool ok_val = !isfinite(val);
        if (!ok_val) g_fail++; else g_pass++;
        printf("  %-26s val=%g (isfinite=%d)  %s\n",
               "LOG_sing(c0x<0)val", val, (int)isfinite(val), ok_val ? "PASS(NaN/inf)" : "FAIL(finite!)");
        // (2) 梯度 ∂/∂c0 = 1/c0 = -1.0 必须有限且正确。
        float g[1] = {0.0f};
        eval_tree_vjp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x1, c, 1, g);
        float want = 1.0f / c[0];   // d/dc0 log(c0*x0) = (1/(c0*x0))*x0 = 1/c0
        float rel = fabsf(g[0] - want) / (fabsf(want) + 1e-4f);
        bool ok_g = isfinite(g[0]) && (rel < 1e-3f);
        if (!ok_g) g_fail++; else g_pass++;
        printf("  %-26s grad=%g want=%g rel=%.3e  %s\n",
               "LOG_sing(c0x<0)grad", g[0], want, rel, ok_g ? "PASS" : "FAIL");

        // c0*x0 == 0 边界 (x0=0 → log(0) = -inf)。value 必须非有限。
        float x0z[1] = {0.0f};
        float c2[1] = {1.0f};
        float val2 = eval_tree_val_host(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x0z, c2);
        bool ok_val2 = !isfinite(val2);
        if (!ok_val2) g_fail++; else g_pass++;
        printf("  %-26s val=%g (isfinite=%d)  %s\n",
               "LOG_sing(c0x==0)val", val2, (int)isfinite(val2), ok_val2 ? "PASS(NaN/inf)" : "FAIL(finite!)");
    }

    // ---------------- 显式 division (c0 / (c1 + x0)) ----------------
    // y = c0 / (c1 + x0). ∂/∂c0 = 1/(c1+x0), ∂/∂c1 = -c0/(c1+x0)^2。
    //   [BFUNC(DIV), CONST(c0), BFUNC(ADD, CONST(c1), VAR(x0))]
    {
        Tree T;
        T.nt = {N_BFUNC, N_CONST, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)F_DIV, 0.0f, (float)F_ADD, 0.0f, 0.0f};
        T.ci = {-1, 0, -1, 1, -1};
        float c[2] = {2.5f, 1.1f};   // c1+x0 = 1.8 != 0
        report("DIV_c0_over_c1x", T, x1, c, 2);
    }

    // ---------------- 嵌套: sin(c0*x0) + c1*exp(c2*x0), K=3 ----------------
    // pre-order: [ADD, SIN, MUL(c0,x0), MUL(c1, EXP(MUL(c2,x0)))]
    {
        Tree T;
        T.nt = {N_BFUNC,
                N_UFUNC, N_BFUNC, N_CONST, N_VAR,
                N_BFUNC, N_CONST, N_UFUNC, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)F_ADD,
                (float)F_SIN, (float)F_MUL, 0.0f, 0.0f,
                (float)F_MUL, 0.0f, (float)F_EXP, (float)F_MUL, 0.0f, 0.0f};
        T.ci = {-1,
                -1, -1, 0, -1,
                -1, 1, -1, -1, 2, -1};
        float c[3] = {1.3f, 0.6f, 0.8f};   // c2*x0 = 0.56 (exp 不爆)
        report("nested_sin+exp_K3", T, x1, c, 3);
    }

    // ---------------- 更深嵌套 K=4 (再卡一次 reverse adjoint 栈纪律) ----------------
    // y = (c0*x0 + c1) * sin(c2*x0 + c3), 全 4 列各自非线性耦合。
    //   pre-order: [MUL, ADD(MUL(c0,x0), c1), SIN(ADD(MUL(c2,x0), c3))]
    {
        Tree T;
        T.nt = {N_BFUNC,
                N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST,
                N_UFUNC, N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST};
        T.nv = {(float)F_MUL,
                (float)F_ADD, (float)F_MUL, 0.0f, 0.0f, 0.0f,
                (float)F_SIN, (float)F_ADD, (float)F_MUL, 0.0f, 0.0f, 0.0f};
        T.ci = {-1,
                -1, -1, 0, -1, 1,
                -1, -1, -1, 2, -1, 3};
        float c[4] = {1.1f, 0.4f, 0.9f, 0.3f};   // c2*x0+c3 = 0.93 (远离 sin 拐点巧合)
        report("nested_mul_sin_K4", T, x1, c, 4);
    }

    // ---------------- K=1 极端量级 (1e-6 / 1e6) — 验梯度规则在 fp32 极端常数下成立 ----------------
    // 镜像 test_jvp_host.cu 的诚实说明: reverse 梯度是**精确**量, CD 是脆弱 oracle。
    // 极端量级下若失配, 失配方多半是 CD 的 fp32 精度而非梯度规则。故只用 CD 截断恒 0 的
    // 线性 op (c0*x, c0/x, MUL 以 c0 为因子) 上大/小量级; EXP-of-small 配大 x 让参数动起来。
    {
        const float TOLm = 1e-3f;
        // 线性 c0*x0: [MUL, CONST(c0), VAR(x0)]
        Tree TL; TL.nt = {N_BFUNC, N_CONST, N_VAR}; TL.nv = {(float)F_MUL, 0.0f, 0.0f}; TL.ci = {-1, 0, -1};
        // MUL 两支带参 (c0*x0)*(c1*x0), 微分 c0
        Tree TM = bin_two_terms(F_MUL);
        // DIV c0/x0: [DIV, CONST(c0), VAR(x0)]  (c0 在分子, 线性)
        Tree TD; TD.nt = {N_BFUNC, N_CONST, N_VAR}; TD.nv = {(float)F_DIV, 0.0f, 0.0f}; TD.ci = {-1, 0, -1};
        // EXP(c0*x0): [EXP, MUL(c0,x0)]
        Tree TE; TE.nt = {N_UFUNC, N_BFUNC, N_CONST, N_VAR}; TE.nv = {(float)F_EXP, (float)F_MUL, 0.0f, 0.0f}; TE.ci = {-1, -1, 0, -1};

        struct Case { const char *nm; Tree *T; float x; float c0; float c1; int K; };
        float x0   = 0.7f;     // O(1) 点 (线性 op 量级无关)
        float xbig = 1e3f;     // 让 EXP 参数 c0*x 动起来 (CD 有意义)
        Case cases[] = {
            {"MAG_lin_c0x  c0=1e6",  &TL, x0,   1e6f,  0.0f, 1},
            {"MAG_lin_c0x  c0=1e-6", &TL, x0,   1e-6f, 0.0f, 1},
            {"MAG_MUL_d/dc0 c0=1e6", &TM, x0,   1e6f,  0.9f, 2},
            {"MAG_MUL_d/dc0 c0=1e-6",&TM, x0,   1e-6f, 0.9f, 2},
            {"MAG_DIV_c0/x c0=1e6",  &TD, x0,   1e6f,  0.0f, 1},
            {"MAG_DIV_c0/x c0=1e-6", &TD, x0,   1e-6f, 0.0f, 1},
            {"MAG_EXP_small c0=1e-6 x=1e3", &TE, xbig, 1e-6f, 0.0f, 1},
        };
        for (auto &cs : cases) {
            float xv[1] = { cs.x };
            float cv[2] = { cs.c0, cs.c1 };
            // 只检 col 0 (大/小量级常数所在列)。
            float worst = check_cols(*cs.T, xv, cv, cs.K, 0, 1);
            bool ok = (worst < TOLm); if (!ok) g_fail++; else g_pass++;
            printf("  %-30s max_rel_err=%.3e  %s\n", cs.nm, worst, ok ? "PASS" : "FAIL");
        }
    }

    // ============================================================================
    // NaN-safety (mask 修复): 有限值奇点处真梯度有限, 但朴素 reverse 出 0*inf / inf*0 = NaN。
    // 真梯度 = 0 经 scipy fp64 oracle + Python fp32 mirror 证实 (FINDINGS_codex_review_shared_nan.md
    // + reverse_vjp_py.py)。这两条在加 safe-multiply 之前 RED (NaN), 之后 GREEN (0)。
    // ============================================================================
    {
        // Fixture A (m=1247 类: adj==0 × inf-partial): y = (x0-x0)*sqrt(c0), c0=0。
        //   value = 0*0 = 0 (有限); 真 ∂/∂c0 = (x0-x0)*sqrt'(c0) = 0 ∀c0。
        //   朴素: SQRT 子树 adjoint=0, ×sqrt'(0)=inf → 0*inf = NaN。
        Tree A;
        A.nt = {N_BFUNC, N_BFUNC, N_VAR, N_VAR, N_UFUNC, N_CONST};
        A.nv = {(float)F_MUL, (float)F_SUB, 0.0f, 0.0f, (float)F_SQRT, 0.0f};
        A.ci = {-1, -1, -1, -1, -1, 0};
        float cA[1] = {0.0f}; float xA[1] = {0.7f}; float eA[1] = {0.0f};
        report_analytic("NaNsafe_A_zeroXinf", A, xA, cA, 1, eA, 0, 1);

        // Fixture B (inf-adj × 0-partial; 卡 "只 mask adj==0" 不够, 必须 safe-multiply 双侧):
        //   y = sqrt(c0*(x0-x0)), c0=2。 value = sqrt(2*0)=0 (有限); 真 ∂/∂c0 = 0。
        //   朴素: sqrt'(0)=inf 带 adj=1 → inf; 到 MUL 的 c0 支 ×0 → inf*0 = NaN。
        Tree B;
        B.nt = {N_UFUNC, N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_VAR};
        B.nv = {(float)F_SQRT, (float)F_MUL, 0.0f, (float)F_SUB, 0.0f, 0.0f};
        B.ci = {-1, -1, 0, -1, -1, -1};
        float cB[1] = {2.0f}; float xB[1] = {0.7f}; float eB[1] = {0.0f};
        report_analytic("NaNsafe_B_infXzero", B, xB, cB, 1, eB, 0, 1);
    }

    // ============================================================================
    // 闸门补洞 #5a: 重复 ci (同一常数用两次) — 卡 CONST 累加必须 += 而非 = 。
    //   y = c0*x0 + c0, ∂/∂c0 = x0 + 1。 += → x0+1 (对); = → 1 (错, 后写覆盖前写)。
    //   real-pop 无重复 ci (survivorB 等价变异), 此条让 +=/= 可区分 (mutation 见证)。
    // ============================================================================
    {
        Tree T;
        T.nt = {N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST};
        T.nv = {(float)F_ADD, (float)F_MUL, 0.0f, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1, 0};   // ci[2]=ci[4]=0: 同一个 c0 用两次
        float c[1] = {1.3f}; float xv[1] = {0.7f};
        float e[1] = {xv[0] + 1.0f};   // ∂/∂c0 = x0 + 1
        report_analytic("dupci_c0x_plus_c0", T, xv, c, 1, e, 0, 1);
    }

    // ============================================================================
    // 闸门补洞 #5b: 投毒 out_grad — 卡 eval_tree_vjp_d 必须内部清零 out_grad。
    //   host 调用点此前都预清零, 删 zero-init 是 host 盲点 (survivorA); 这条传非零垃圾进去,
    //   若无内部清零 → 结果 = 垃圾 + 真值 (错)。 y = c0*x0, ∂/∂c0 = x0 (= 0.7, 非 999.7)。
    // ============================================================================
    {
        Tree T; T.nt = {N_BFUNC, N_CONST, N_VAR}; T.nv = {(float)F_MUL, 0.0f, 0.0f}; T.ci = {-1, 0, -1};
        float c[1] = {0.9f}; float xv[1] = {0.7f};
        float g[1] = {999.0f};   // 投毒: 非零垃圾入
        eval_tree_vjp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), xv, c, 1, g);
        float want = xv[0];
        float rel = fabsf(g[0] - want) / (fabsf(want) + 1e-4f);
        bool ok = isfinite(g[0]) && (rel < 1e-3f);
        if (!ok) g_fail++; else g_pass++;
        printf("  %-26s g=%.4g want=%.4g rel=%.3e  %s\n",
               "poisonbuf_zeroinit", g[0], want, rel, ok ? "PASS" : "FAIL");
    }

    // ---------------- 总结 ----------------
    printf("\n[test_revad_host] %d PASS, %d FAIL\n", g_pass, g_fail);
    if (g_fail) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
