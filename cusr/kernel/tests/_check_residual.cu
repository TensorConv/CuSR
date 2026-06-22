// _check_residual.cu — corpus-wide classification of reverse-AD non-finite gradients.
// After the NaN-safety fix, classify EVERY (tree,point,column) where reverse gives a
// non-finite gradient by whether the tree VALUE is finite:
//   - rev=NaN  & value finite  -> SUSPECTED CONTAMINATION (0*inf/inf*0 residual; should be 0)
//   - rev=Inf  & value finite  -> true infinite-slope singularity (defensible; sqrt'/log'/inv')
//   - !value finite            -> genuine undefined point (both AD legitimately non-finite)
// Host-only (eval_tree_vjp_d + eval_tree_val_host run on CPU), covers ALL elements.
// build: source ../../scripts/env.sh && cd cusr/kernel &&
//   nvcc -O2 -std=c++17 -o tests/_check_residual tests/_check_residual.cu loader.c
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../revad_interp.cuh"   // eval_tree_vjp_d + eval_tree_val_host (via ad_interp.cuh)
#include "../loader.h"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/home/weish/hao/CuSR/data/workload/synth/synth_inner-const-heavy_M4000_N1000_seed0.bin";
    PopData pop;
    if (load_pop_bin(path, &pop)) { fprintf(stderr, "load failed %s\n", path); return 2; }
    const PopHeader *h = &pop.header;
    int N = h->N, n_vars = h->n_vars;
    fprintf(stderr, "[check] pop=%s M=%d N=%d K_max=%d\n", path, h->M_prob, N, h->K_max);

    long long n_cmp = 0, rev_nonfinite = 0;
    long long contam_nan_valfin = 0;   // rev NaN & value finite  -> should be 0
    long long inf_valfin = 0;          // rev Inf & value finite  -> true inf slope (OK)
    long long genuine_valnan = 0;      // value non-finite        -> genuine undefined
    int ex = 0;

    for (int m = 0; m < h->M_prob; m++) {
        TreeMeta meta = pop.metas[m];
        int K = meta.K; if (K == 0) continue;
        const int   *nt = pop.nt + meta.node_offset;
        const float *nv = pop.nv + meta.node_offset;
        const int   *ci = pop.ci + meta.node_offset;
        const float *c  = pop.c_init + meta.c_offset;
        for (int i = 0; i < N; i++) {
            const float *x = pop.xs + (size_t)i * n_vars;
            float g[MAX_K];
            eval_tree_vjp_d(nt, nv, meta.n_nodes, ci, x, c, K, g);
            float val = eval_tree_val_host(nt, nv, meta.n_nodes, ci, x, c);
            bool vfin = isfinite(val);
            for (int k = 0; k < K; k++) {
                n_cmp++;
                if (isfinite(g[k])) continue;
                rev_nonfinite++;
                if (vfin && isnan(g[k]))      { contam_nan_valfin++;
                    if (ex < 8) { fprintf(stderr, "  CONTAM m=%d i=%d k=%d val=%g rev=%g\n", m, i, k, val, g[k]); ex++; } }
                else if (vfin && isinf(g[k])) inf_valfin++;
                else                          genuine_valnan++;
            }
        }
    }
    printf("[check_residual] pop=%s\n", path);
    printf("  elements compared            = %lld\n", n_cmp);
    printf("  reverse non-finite (any)     = %lld\n", rev_nonfinite);
    printf("  rev NaN & value FINITE       = %lld   <-- contamination (MUST be 0)\n", contam_nan_valfin);
    printf("  rev Inf & value finite       = %lld   (true infinite-slope singularity; OK)\n", inf_valfin);
    printf("  value non-finite (genuine)   = %lld   (function undefined; both AD non-finite OK)\n", genuine_valnan);
    free_pop_data(&pop);
    if (contam_nan_valfin == 0) { printf("RESULT: PASS (no finite-value NaN contamination)\n"); return 0; }
    printf("RESULT: FAIL (%lld residual contamination elements)\n", contam_nan_valfin);
    return 1;
}
