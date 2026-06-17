// test_jvp_host.cu — Rung 1 host 单测: forward-mode JVP 解释器 vs 中心差分。
//
// 见 docs/kernel/AD_JACOBIAN_V4.md §6 Rung 1。**只** include ad_interp.cuh
// (完全不碰 batch_lm_ad.cu)。nvcc 编成 host 程序 (CPU 跑), 不需要 GPU。
//
//   每个 op (12 UFUNC + 11 BFUNC) + VAR/CONST seed:
//     建最小 token 树, eval_tree_jvp_d 切向量 vs 中心差分 (eval_tree_val_host),
//     max rel err < 1e-3 → PASS。
//   额外硬覆盖 (§7 风险点):
//     c0^2 (定常指数 POW), x^c0 (含参指数 POW), log(c0*x) 在 c0*x<=0 (断言 NaN/inf),
//     一个 division, 嵌套 sin(c0*x)+c1*exp(c2*x), K=3 树跑 W=2/col_lo=0 + W=1/col_lo=2
//     (卡 chunk 边界)。
//
// build/gate:
//   source scripts/env.sh && cd cusr/kernel && \
//   nvcc -O2 -std=c++17 -o test_jvp_host tests/test_jvp_host.cu && ./test_jvp_host ; echo EXIT=$?

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../ad_interp.cuh"

// ---- 树容器 (host 端小工具) -------------------------------------------------
struct Tree {
    std::vector<int>   nt;
    std::vector<float> nv;
    std::vector<int>   ci;   // const_idx, -1 if not CONST
};

static int g_fail = 0;
static int g_pass = 0;

// 中心差分参考: ∂value/∂c[k] at (x, c)。step ~ 1e-3*|c[k]|, abs floor 1e-5。
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

// max rel err over a set of (col, c-index) checks at one point, 单 chunk [col_lo, col_lo+W)。
// 返回 max_{w} |jvp_t[w] - fd| / (|fd| + abs_floor)。
static float check_chunk(const char *name, const Tree &T, const float *x, const float *c,
                         int K, int col_lo, int W) {
    float val, t[AD_W];
    eval_tree_jvp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(),
                    x, c, col_lo, W, &val, t);
    float worst = 0.0f;
    for (int w = 0; w < W; w++) {
        int k = col_lo + w;
        if (k >= K) break;
        float fd = central_diff(T, x, c, K, k);
        float denom = fabsf(fd) + 1e-4f;
        float rel = fabsf(t[w] - fd) / denom;
        if (rel > worst) worst = rel;
    }
    (void)name;
    return worst;
}

// 单 op / 单点 PASS/FAIL 报告。所有列默认 chunk [0, K)。
static void report(const char *name, const Tree &T, const float *x, const float *c, int K) {
    float worst = check_chunk(name, T, x, c, K, 0, K);
    bool ok = (worst < 1e-3f);
    if (!ok) g_fail++; else g_pass++;
    printf("  %-22s K=%d  max_rel_err=%.3e  %s\n", name, K, worst, ok ? "PASS" : "FAIL");
}

// ---- 树构造助手 --------------------------------------------------------------
// f(c0_or_x):  c0 <ufunc> 应用在 (c0*x) 上, 即 op(c0*x), K=1, var x0。
//   树 (pre-order): [UFUNC(op), BFUNC(MUL), CONST(c0), VAR(x0)]
static Tree ufunc_of_c0x(int op) {
    Tree T;
    T.nt = {N_UFUNC, N_BFUNC, N_CONST, N_VAR};
    T.nv = {(float)op, (float)F_MUL, 0.0f, 0.0f};
    T.ci = {-1, -1, 0, -1};
    return T;
}
int main() {
    printf("[test_jvp_host] forward-mode JVP vs central difference (tol rel < 1e-3)\n");

    // 单变量点。各 op 选安全 x/c 避奇点 (奇点单独测)。
    float x1[1] = {0.7f};

    // ---------------- VAR / CONST seeding ----------------
    // CONST seed: y = c0 (恒等), ∂/∂c0 = 1。  [CONST(c0)]
    {
        Tree T; T.nt = {N_CONST}; T.nv = {0.0f}; T.ci = {0};
        float c[1] = {1.3f};
        report("seed_CONST", T, x1, c, 1);
    }
    // VAR seed: y = x0 (无常数依赖) — 用 c0*x0 验 VAR 的 0 切向量 + CONST 的 x 切向量。
    //   [BFUNC(MUL), CONST(c0), VAR(x0)] → ∂/∂c0 = x0。
    {
        Tree T; T.nt = {N_BFUNC, N_CONST, N_VAR}; T.nv = {(float)F_MUL, 0.0f, 0.0f}; T.ci = {-1, 0, -1};
        float c[1] = {0.9f};
        report("seed_VAR(c0*x0)", T, x1, c, 1);
    }

    // ---------------- 12 UFUNC: op(c0*x0), K=1 ----------------
    // 选 c0 使 c0*x0 落在安全区。x0=0.7。
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
            char nm[32]; snprintf(nm, sizeof(nm), "UFUNC_%s", u.nm);
            report(nm, T, x1, c, 1);
        }
    }

    // ---------------- 11 BFUNC ----------------
    // ADD/SUB/MUL/DIV: (c0*x0) <op> (c1*x0), K=2。
    //   pre-order: [BFUNC(op), BFUNC(MUL,c0,x0), BFUNC(MUL,c1,x0)]
    //   l = first child (c0*x0), rv = second child (c1*x0)。
    auto bin_two_terms = [](int op) {
        Tree T;
        T.nt = {N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)op, (float)F_MUL, 0.0f, 0.0f, (float)F_MUL, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1, -1, 1, -1};
        return T;
    };
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
            char nm[32]; snprintf(nm, sizeof(nm), "BFUNC_%s", b.nm);
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
            char nm[32]; snprintf(nm, sizeof(nm), "BFUNC_%s", b.nm);
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
            float c[2] = {1.5f, 0.5f};   // 两支差距大, 小扰动不跨阈值 → fd≈0, jvp=0
            char nm[32]; snprintf(nm, sizeof(nm), "BFUNC_%s", b.nm);
            report(nm, T, x1, c, 2);
        }
    }

    // ---------------- POW: 必测 c0^2 (定常指数) ----------------
    // [BFUNC(POW), CONST(c0=base), CONST(exp)]; exp 是 c[1]=2.0 但我们只微分 col 0。
    //   exp_has_tan=false (∂exp/∂c0 = 0) → POW 守卫跳过 logf。 ∂/∂c0 (c0^2) = 2 c0。
    {
        Tree T; T.nt = {N_BFUNC, N_CONST, N_CONST}; T.nv = {(float)F_POW, 0.0f, 0.0f}; T.ci = {-1, 0, 1};
        float c[2] = {1.4f, 2.0f};
        // 只微分 col 0 (base); exp (c[1]) 视作定常 → chunk [0,1)。
        float worst = check_chunk("POW_c0sq", T, x1, c, 2, 0, 1);
        bool ok = (worst < 1e-3f); if (!ok) g_fail++; else g_pass++;
        printf("  %-22s K=%d  max_rel_err=%.3e  %s\n", "POW_c0sq(d/dc0)", 2, worst, ok ? "PASS" : "FAIL");
    }

    // ---------------- POW: 含参指数 x^c0 ----------------
    // y = x0 ^ c0, ∂/∂c0 = x0^c0 * log(x0)。base=VAR(x0), exp=CONST(c0)。
    //   [BFUNC(POW), VAR(x0), CONST(c0)]  exp_has_tan=true → 走完整公式 (logf(x0), x0>0)。
    {
        Tree T; T.nt = {N_BFUNC, N_VAR, N_CONST}; T.nv = {(float)F_POW, 0.0f, 0.0f}; T.ci = {-1, -1, 0};
        float c[1] = {1.8f};
        report("POW_x_pow_c0", T, x1, c, 1);
    }
    // POW: 两侧含参 (base+exp 都带切向量) — base=c0*x0, exp=c1。
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

    // ---------------- LOG 奇点: log(c0*x0) at c0*x0 <= 0 (断言 NaN/inf, 非错的有限值) ----------------
    {
        Tree T = ufunc_of_c0x(F_LOG);
        float c[1] = {-1.0f};   // c0*x0 = -0.7 < 0 → log → NaN
        float val, t[AD_W];
        eval_tree_jvp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(),
                        x1, c, 0, 1, &val, t);
        // ∂/∂c0 = 1/c0 = -1.0 是有限的, 但 VALUE 必须是 NaN (log(neg))。
        // 规范要求: 在 c0*x<=0 处断言产生 NaN/inf, 而不是错的有限值 (针对 value)。
        bool val_bad = !isfinite(val);
        bool ok = val_bad;
        if (!ok) g_fail++; else g_pass++;
        printf("  %-22s val=%g (isfinite=%d) tan=%g  %s\n",
               "LOG_sing(c0x<=0)", val, (int)isfinite(val), t[0], ok ? "PASS(NaN/inf)" : "FAIL(finite!)");

        // 再测 c0*x0 == 0 边界 (x0=0 → log(0) = -inf)。
        float x0z[1] = {0.0f};
        float c2[1] = {1.0f};
        eval_tree_jvp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(),
                        x0z, c2, 0, 1, &val, t);
        bool ok2 = !isfinite(val);   // log(0) = -inf
        if (!ok2) g_fail++; else g_pass++;
        printf("  %-22s val=%g (isfinite=%d)  %s\n",
               "LOG_sing(c0x==0)", val, (int)isfinite(val), ok2 ? "PASS(NaN/inf)" : "FAIL(finite!)");
    }

    // ---------------- 显式 division (再来一个独立的, c0 / (c1 + x0)) ----------------
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
    //   = [BFUNC(ADD),
    //        UFUNC(SIN), BFUNC(MUL), CONST(c0), VAR(x0),
    //        BFUNC(MUL), CONST(c1), UFUNC(EXP), BFUNC(MUL), CONST(c2), VAR(x0)]
    auto nested_tree = []() {
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
        return T;
    };
    {
        Tree T = nested_tree();
        float c[3] = {1.3f, 0.6f, 0.8f};   // c2*x0 = 0.56 (exp 不爆)
        report("nested_sin+exp_K3", T, x1, c, 3);
    }

    // ---------------- chunk 边界: K=3 树, W=2/col_lo=0 然后 W=1/col_lo=2 ----------------
    // 用同一棵 nested_sin+exp K=3 树。两个 chunk 合起来覆盖全 3 列, 卡 AD_W 分桶尾桶。
    {
        Tree T = nested_tree();
        float c[3] = {1.3f, 0.6f, 0.8f};
        // chunk 0: [0,2) → 微分 c0, c1
        float w0 = check_chunk("chunk[0,2)", T, x1, c, 3, 0, 2);
        bool ok0 = (w0 < 1e-3f); if (!ok0) g_fail++; else g_pass++;
        printf("  %-22s W=2 col_lo=0  max_rel_err=%.3e  %s\n", "CHUNK_lo0_W2", w0, ok0 ? "PASS" : "FAIL");
        // chunk 1: [2,3) → 微分 c2 (尾桶 W=1)
        float w1 = check_chunk("chunk[2,3)", T, x1, c, 3, 2, 1);
        bool ok1 = (w1 < 1e-3f); if (!ok1) g_fail++; else g_pass++;
        printf("  %-22s W=1 col_lo=2  max_rel_err=%.3e  %s\n", "CHUNK_lo2_W1", w1, ok1 ? "PASS" : "FAIL");
    }

    // ---------------- 大/小量级 (extreme magnitude) — 验导数规则在 fp32 极端常数下成立 ----------------
    // 动机: 退役 eps_fd 的相对步长机制依赖 "精确导数在任意量级都对"。这里喂 c=1e6 / c=1e-6
    // 给若干 op, 断言 JVP 切向量仍与中心差分 (CD) 吻合 (rel < 1e-3, 同上面所有 case 的容差)。
    //
    // 诚实说明 (这是一个 finding, 不靠放松 1e-3 来掩盖):
    //   JVP 切向量是**精确**量; CD 是脆弱的 oracle。极端量级下若失配, 失配方是 CD 的 fp32 精度,
    //   不是 JVP 规则。两类 CD 失效 (均源于步长 h=max(1e-3|c|,1e-5)):
    //     (a) 线性-in-c 的 op (c0*x, MUL 以 c0 为因子, c0/x): CD 截断误差恒为 0 →
    //         1e6 与 1e-6 双向都干净 (实测 rel<=6e-6)。极端 case 以这些打头。
    //     (b) EXP(c0*x) 在 c0=1e-6, x~O(1): 参数 c0*x 几乎不动, h 触底 1e-5, CD 抵消误差
    //         ~1e-7/(x*h)~1e-2 > 1e-3 —— CD 解不出, **非** JVP 错 (JVP=x 精确)。要让 CD 有意义,
    //         必须让参数动起来 (配大 x): c0=1e-6, x=1e3 时 rel~1.5e-5 (实测) → 用这个量级。
    //         EXP 大 c0=1e6 直接溢出 inf (CD=nan), 物理预期, 不测大量级 EXP。
    {
        const float TOLm = 1e-3f;
        // 线性 c0*x0 (CD 截断恒 0 → 双向干净): [MUL, CONST(c0), VAR(x0)]
        Tree TL; TL.nt = {N_BFUNC, N_CONST, N_VAR}; TL.nv = {(float)F_MUL, 0.0f, 0.0f}; TL.ci = {-1, 0, -1};
        // MUL 两支带参 (c0*x0)*(c1*x0), 微分 c0: 复用 bin_two_terms 结构。
        Tree TM = bin_two_terms(F_MUL);
        // DIV c0/x0: [DIV, CONST(c0), VAR(x0)]  (c0 在分子, 线性 → CD 干净)
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
            // EXP-of-small: 配大 x 使参数移动 (见上方诚实说明)。x=1e3 → c0*x=1e-3。
            {"MAG_EXP_small c0=1e-6 x=1e3", &TE, xbig, 1e-6f, 0.0f, 1},
        };
        for (auto &cs : cases) {
            float xv[1] = { cs.x };
            float cv[2] = { cs.c0, cs.c1 };
            // 只检 col 0 (大/小量级常数所在列); chunk [0,1) 覆盖该列。
            float worst = check_chunk(cs.nm, *cs.T, xv, cv, cs.K, 0, 1);
            bool ok = (worst < TOLm); if (!ok) g_fail++; else g_pass++;
            printf("  %-30s max_rel_err=%.3e  %s\n", cs.nm, worst, ok ? "PASS" : "FAIL");
        }
    }

    // ---------------- 总结 ----------------
    printf("\n[test_jvp_host] %d PASS, %d FAIL\n", g_pass, g_fail);
    if (g_fail) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
