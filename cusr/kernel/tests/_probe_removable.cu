// _probe_removable.cu — test Codex's "removable singularity" counterexample.
// pow(sqrt(c0), x0): on c0>=0 this is c0^(x0/2), so d/dc0 = (x0/2)*c0^(x0/2-1).
//   x0=1 -> 0.5*c0^-0.5 -> +inf  (genuine singularity)
//   x0=2 -> 1            (REMOVABLE: 0*inf cancels to 1) <-- Codex's case
//   x0=3 -> 1.5*c0^0.5   -> 0 at c0=0
//   x0=4 -> 2*c0         -> 0 at c0=0
// Compare: tree VALUE, reverse-AD grad (CURRENT = safe_mul), forward-AD grad (chunked),
// one-sided FD (in-domain), and the analytic right-derivative.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../revad_interp.cuh"   // eval_tree_vjp_d (safe_mul) + eval_tree_jvp_d (fwd) + eval_tree_val_host

static void fwd_grad(const int *nt, const float *nv, int n, const int *ci,
                     const float *x, const float *c, int K, float *g) {
    for (int lo = 0; lo < K; lo += AD_W) {
        int W = (AD_W < K - lo) ? AD_W : (K - lo);
        float val, t[AD_W];
        eval_tree_jvp_d(nt, nv, n, ci, x, c, lo, W, &val, t);
        for (int w = 0; w < W; w++) g[lo + w] = t[w];
    }
}

int main() {
    // pow(sqrt(c0), x0):  pre-order [POW, SQRT, c0, x0]
    int   nt[4] = {N_BFUNC, N_UFUNC, N_CONST, N_VAR};
    float nv[4] = {(float)F_POW, (float)F_SQRT, 0.0f, 0.0f};
    int   ci[4] = {-1, -1, 0, -1};
    float c[1]  = {0.0f};   // c0 = 0 (the singularity)

    printf("pow(sqrt(c0), x0) at c0=0  —  rev=CURRENT safe_mul kernel\n");
    printf("  x0 | value | rev d/dc0 | fwd d/dc0 | 1-sided FD | analytic right-deriv\n");
    for (float x0 : {1.0f, 2.0f, 3.0f, 4.0f}) {
        float x[1] = {x0};
        float gr[1] = {0}, gf[1] = {0};
        eval_tree_vjp_d(nt, nv, 4, ci, x, c, 1, gr);   // reverse (safe_mul)
        fwd_grad(nt, nv, 4, ci, x, c, 1, gf);          // forward
        float val = eval_tree_val_host(nt, nv, 4, ci, x, c);
        // one-sided FD, in-domain (c0+h > 0)
        float h = 1e-4f; float cp[1] = {0.0f + h};
        float fph = eval_tree_val_host(nt, nv, 4, ci, x, cp);
        float f0  = eval_tree_val_host(nt, nv, 4, ci, x, c);
        float fd  = (fph - f0) / h;
        // analytic: (x0/2)*c0^(x0/2-1) at c0->0+
        double e = (x0/2.0)*pow(0.0, x0/2.0 - 1.0);  // 0^neg=inf, 0^0=1, 0^pos=0
        printf("  %.0f  | %6.4g | %9.4g | %9.4g | %10.4g | %g\n", x0, val, gr[0], gf[0], fd, e);
    }
    return 0;
}
