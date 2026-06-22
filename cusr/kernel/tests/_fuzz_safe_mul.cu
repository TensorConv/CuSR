// _fuzz_safe_mul.cu — adversarial property-test of revad_safe_mul soundness.
// Tries to falsify "revad_safe_mul only removes spurious NaN, never produces a wrong gradient":
//   - SINGULAR trees (finite value, known analytic grad=0 via 0*inf / inf*0 / 0*nan): assert AD finite & ~0.
//   - NORMAL trees: AD (which uses safe_mul) vs fp32 central difference, assert match.
// If safe_mul ever wrongly zeroed a genuinely-nonzero gradient, a NORMAL case would diverge from CD,
// or a SINGULAR analytic expectation would be violated. Host-only.
// build: source ../../scripts/env.sh && nvcc -O2 -std=c++17 -o tests/_fuzz_safe_mul tests/_fuzz_safe_mul.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../revad_interp.cuh"

struct Tree { std::vector<int> nt; std::vector<float> nv; std::vector<int> ci; };
static int fails = 0, tested = 0;

static void adgrad(const Tree &T, const float *x, const float *c, int K, float *g) {
    eval_tree_vjp_d(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x, c, K, g);
}
static float val(const Tree &T, const float *x, const float *c) {
    return eval_tree_val_host(T.nt.data(), T.nv.data(), (int)T.nt.size(), T.ci.data(), x, c);
}
static float cd(const Tree &T, const float *x, const float *c, int K, int k) {
    std::vector<float> cp(c, c + K); float ck = c[k]; float h = 1e-3f * fabsf(ck); if (h < 1e-5f) h = 1e-5f;
    cp[k] = ck + h; float fp = val(T, x, cp.data());
    cp[k] = ck - h; float fm = val(T, x, cp.data());
    return (fp - fm) / (2.0f * h);
}
// SINGULAR: finite value, known analytic grad. Assert AD finite & |AD-expect|<tol (NO blind NaN, NO wrong 0).
static void singular(const char *nm, const Tree &T, const float *x, const float *c, int K, const float *expect) {
    tested++;
    std::vector<float> g(K, 0.0f); adgrad(T, x, c, K, g.data());
    float v = val(T, x, c);
    bool ok = isfinite(v);
    for (int k = 0; k < K; k++) if (!isfinite(g[k]) || fabsf(g[k] - expect[k]) > 1e-3f) ok = false;
    if (!ok) fails++;
    printf("  [SING] %-26s val=%-9.4g ad=[", nm, v);
    for (int k = 0; k < K; k++) printf("%s%.4g", k ? "," : "", g[k]);
    printf("] want=["); for (int k = 0; k < K; k++) printf("%s%.4g", k ? "," : "", expect[k]);
    printf("]  %s\n", ok ? "OK" : "*** WRONG ***");
}
// NORMAL: AD vs central diff. A safe_mul that wrongly zeroed a nonzero grad would FAIL here.
static void normal(const char *nm, const Tree &T, const float *x, const float *c, int K) {
    tested++;
    std::vector<float> g(K, 0.0f); adgrad(T, x, c, K, g.data());
    float worst = 0.0f;
    for (int k = 0; k < K; k++) {
        float d = cd(T, x, c, K, k);
        float rel = fabsf(g[k] - d) / (fabsf(d) + 1e-4f);
        if (!isfinite(g[k])) rel = 1e9f;
        if (rel > worst) worst = rel;
    }
    bool ok = worst < 2e-2f; if (!ok) fails++;
    printf("  [NORM] %-26s max_rel=%.3e  %s\n", nm, worst, ok ? "OK" : "*** WRONG ***");
}

int main() {
    printf("[fuzz_safe_mul] adversarial soundness property-test\n");
    float x[2] = {0.7f, 1.3f};

    // ---- SINGULAR (finite value, true grad 0; each hits a different 0*non-finite path) ----
    { Tree T; T.nt={N_BFUNC,N_BFUNC,N_VAR,N_VAR,N_UFUNC,N_CONST}; T.nv={(float)F_MUL,(float)F_SUB,0,0,(float)F_SQRT,0}; T.ci={-1,-1,-1,-1,-1,0};
      float c[1]={0.0f}, e[1]={0.0f}; singular("MUL(x-x,sqrt(c0))@0  0*inf", T, x, c, 1, e); }       // adj==0 x inf
    { Tree T; T.nt={N_UFUNC,N_BFUNC,N_CONST,N_BFUNC,N_VAR,N_VAR}; T.nv={(float)F_SQRT,(float)F_MUL,0,(float)F_SUB,0,0}; T.ci={-1,-1,0,-1,-1,-1};
      float c[1]={2.0f}, e[1]={0.0f}; singular("sqrt(c0*(x-x))     inf*0", T, x, c, 1, e); }          // inf-adj x 0-partial
    { Tree T; T.nt={N_BFUNC,N_BFUNC,N_VAR,N_VAR,N_BFUNC,N_CONST,N_CONST}; T.nv={(float)F_MUL,(float)F_SUB,0,0,(float)F_POW,0,0}; T.ci={-1,-1,-1,-1,-1,0,1};
      float c[2]={0.0f,0.5f}, e[2]={0.0f,0.0f}; singular("MUL(x-x,pow(c0,c1))@0  0*inf", T, x, c, 2, e); }
    { Tree T; T.nt={N_BFUNC,N_BFUNC,N_VAR,N_VAR,N_BFUNC,N_CONST,N_CONST}; T.nv={(float)F_MUL,(float)F_SUB,0,0,(float)F_POW,0,0}; T.ci={-1,-1,-1,-1,-1,0,1};
      float c[2]={-1.0f,2.0f}, e[2]={0.0f,0.0f}; singular("MUL(x-x,pow(-1,2))  0*nan", T, x, c, 2, e); }  // 0 x nan (log(-1))
    { Tree T; T.nt={N_BFUNC,N_VAR,N_UFUNC,N_CONST}; T.nv={(float)F_MAX,0,(float)F_SQRT,0}; T.ci={-1,-1,-1,0};
      float c[1]={0.0f}, e[1]={0.0f}; singular("MAX(x0,sqrt(c0))@0 deadbr", T, x, c, 1, e); }          // dead branch selector=0 x inf

    // ---- NORMAL controls (true grad nonzero; safe_mul must NOT zero them) ----
    { Tree T; T.nt={N_BFUNC,N_CONST,N_VAR}; T.nv={(float)F_MUL,0,0}; T.ci={-1,0,-1};
      float c[1]={1.3f}; normal("c0*x0", T, x, c, 1); }
    { Tree T; T.nt={N_BFUNC,N_CONST,N_BFUNC,N_CONST,N_VAR}; T.nv={(float)F_DIV,0,(float)F_ADD,0,0}; T.ci={-1,0,-1,1,-1};
      float c[2]={2.5f,1.1f}; normal("c0/(c1+x0)", T, x, c, 2); }
    { Tree T; T.nt={N_UFUNC,N_BFUNC,N_CONST,N_VAR}; T.nv={(float)F_SIN,(float)F_MUL,0,0}; T.ci={-1,-1,0,-1};
      float c[1]={1.1f}; normal("sin(c0*x0)", T, x, c, 1); }
    { Tree T; T.nt={N_UFUNC,N_BFUNC,N_CONST,N_VAR}; T.nv={(float)F_TANH,(float)F_MUL,0,0}; T.ci={-1,-1,0,-1};
      float c[1]={1.1f}; normal("tanh(c0*x0)", T, x, c, 1); }
    { Tree T; T.nt={N_BFUNC,N_BFUNC,N_CONST,N_VAR,N_CONST}; T.nv={(float)F_POW,(float)F_MUL,0,0,0}; T.ci={-1,-1,0,-1,1};
      float c[2]={1.3f,1.6f}; normal("pow(c0*x0,c1)", T, x, c, 2); }
    { Tree T; T.nt={N_BFUNC,N_BFUNC,N_CONST,N_VAR,N_BFUNC,N_CONST,N_VAR}; T.nv={(float)F_DIV,(float)F_MUL,0,0,(float)F_MUL,0,0}; T.ci={-1,-1,0,-1,-1,1,-1};
      float c[2]={1.3f,0.7f}; normal("(c0*x0)/(c1*x0)", T, x, c, 2); }
    { Tree T;
      T.nt={N_BFUNC, N_BFUNC,N_BFUNC,N_CONST,N_VAR,N_CONST, N_UFUNC,N_BFUNC,N_BFUNC,N_CONST,N_VAR,N_CONST};
      T.nv={(float)F_MUL, (float)F_ADD,(float)F_MUL,0,0,0, (float)F_SIN,(float)F_ADD,(float)F_MUL,0,0,0};
      T.ci={-1, -1,-1,0,-1,1, -1,-1,-1,2,-1,3};
      float c[4]={1.1f,0.4f,0.9f,0.3f}; normal("(c0x+c1)*sin(c2x+c3) K4", T, x, c, 4); }

    printf("\n[fuzz_safe_mul] %d tested, %d WRONG\n", tested, fails);
    if (fails) { printf("RESULT: COUNTEREXAMPLE FOUND\n"); return 1; }
    printf("RESULT: SOUND (no counterexample across battery)\n");
    return 0;
}
