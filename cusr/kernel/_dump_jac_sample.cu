// _dump_jac_sample.cu — TEMP PROBE (not a frozen test): dump forward-AD & reverse-AD
// gradients for a sample of REAL-pop trees so an independent scipy fp64 oracle (Python)
// can decide which one is correct at the singular points where they diverge.
//
// Uses the ACTUAL impl under test (eval_tree_jvp_d forward, eval_tree_vjp_d reverse — both
// __host__ __device__, run here on host) + the real loader. Emits JSONL to stdout:
//   {"m","K","n_nodes","n_vars","nt":[],"nv":[],"ci":[],"c":[],
//    "pts":[{"i","x":[],"fwd":[],"rev":[]}]}
// NaN/Inf are printed as JSON-Python tokens (NaN / Infinity / -Infinity) — Python json.loads
// accepts them. Dumps up to N_SING singular trees (forward emits any non-finite) + N_CLEAN
// clean trees, each with a few points (singular trees: points where forward is non-finite first).
//
// build: source ../../scripts/env.sh && cd cusr/kernel &&
//   nvcc -O2 -std=c++17 -o _dump_jac_sample _dump_jac_sample.cu loader.c

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "revad_interp.cuh"   // eval_tree_vjp_d (rev) + eval_tree_jvp_d (fwd) + enums/bounds
#include "loader.h"

// forward-AD gradient for ONE point (chunked, mirrors ad_jacobian_kernel).
static void fwd_grad(const int *nt, const float *nv, int n_nodes, const int *ci,
                     const float *x, const float *c, int K, float *g) {
    for (int col_lo = 0; col_lo < K; col_lo += AD_W) {
        int W = (AD_W < K - col_lo) ? AD_W : (K - col_lo);
        float val, t[AD_W];
        eval_tree_jvp_d(nt, nv, n_nodes, ci, x, c, col_lo, W, &val, t);
        for (int w = 0; w < W; w++) g[col_lo + w] = t[w];
    }
}

static void jprint(float v) {
    if (isnan(v)) printf("NaN");
    else if (isinf(v)) printf(v < 0 ? "-Infinity" : "Infinity");
    else printf("%.9g", v);
}
static void jarr_f(const float *a, int n) {
    printf("[");
    for (int k = 0; k < n; k++) { if (k) printf(","); jprint(a[k]); }
    printf("]");
}
static void jarr_i(const int *a, int n) {
    printf("[");
    for (int k = 0; k < n; k++) printf("%s%d", k ? "," : "", a[k]);
    printf("]");
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/home/weish/hao/CuSR/data/workload/synth/synth_inner-const-heavy_M4000_N1000_seed0.bin";
    const int N_SING = 8, N_CLEAN = 8, PTS = 6;

    PopData pop;
    if (load_pop_bin(path, &pop)) { fprintf(stderr, "load failed %s\n", path); return 2; }
    const PopHeader *h = &pop.header;
    int N = h->N, n_vars = h->n_vars;
    fprintf(stderr, "[dump] pop=%s M=%d N=%d n_vars=%d K_max=%d\n", path, h->M_prob, N, n_vars, h->K_max);

    int n_sing = 0, n_clean = 0;
    for (int m = 0; m < h->M_prob && (n_sing < N_SING || n_clean < N_CLEAN); m++) {
        TreeMeta meta = pop.metas[m];
        int K = meta.K;
        if (K == 0) continue;
        const int   *nt = pop.nt + meta.node_offset;
        const float *nv = pop.nv + meta.node_offset;
        const int   *ci = pop.ci + meta.node_offset;
        const float *c  = pop.c_init + meta.c_offset;

        // classify: singular if forward produces any non-finite over all points.
        // collect point indices: for singular trees prefer points where fwd is non-finite.
        std::vector<int> sing_pts, fin_pts;
        for (int i = 0; i < N; i++) {
            float gf[MAX_K];
            fwd_grad(nt, nv, meta.n_nodes, ci, pop.xs + (size_t)i * n_vars, c, K, gf);
            bool nf = false;
            for (int k = 0; k < K; k++) if (!isfinite(gf[k])) { nf = true; break; }
            if (nf) sing_pts.push_back(i); else fin_pts.push_back(i);
        }
        bool singular = !sing_pts.empty();
        if (singular && n_sing >= N_SING) continue;
        if (!singular && n_clean >= N_CLEAN) continue;
        if (singular) n_sing++; else n_clean++;

        // choose up to PTS points: for singular, mix non-finite + finite to test sibling cols.
        std::vector<int> chosen;
        for (size_t a = 0; a < sing_pts.size() && (int)chosen.size() < PTS/2 + 1; a++) chosen.push_back(sing_pts[a]);
        for (size_t a = 0; a < fin_pts.size() && (int)chosen.size() < PTS; a++) chosen.push_back(fin_pts[a]);

        printf("{\"m\":%d,\"K\":%d,\"n_nodes\":%d,\"n_vars\":%d,\"singular\":%s,",
               m, K, meta.n_nodes, n_vars, singular ? "true" : "false");
        printf("\"nt\":"); jarr_i(nt, meta.n_nodes);
        printf(",\"nv\":"); jarr_f(nv, meta.n_nodes);
        printf(",\"ci\":"); jarr_i(ci, meta.n_nodes);
        printf(",\"c\":"); jarr_f(c, K);
        printf(",\"pts\":[");
        for (size_t a = 0; a < chosen.size(); a++) {
            int i = chosen[a];
            const float *x = pop.xs + (size_t)i * n_vars;
            float gf[MAX_K], gr[MAX_K];
            fwd_grad(nt, nv, meta.n_nodes, ci, x, c, K, gf);
            eval_tree_vjp_d(nt, nv, meta.n_nodes, ci, x, c, K, gr);
            if (a) printf(",");
            printf("{\"i\":%d,\"x\":", i); jarr_f(x, n_vars);
            printf(",\"fwd\":"); jarr_f(gf, K);
            printf(",\"rev\":"); jarr_f(gr, K);
            printf("}");
        }
        printf("]}\n");
    }
    fprintf(stderr, "[dump] dumped %d singular + %d clean trees\n", n_sing, n_clean);
    free_pop_data(&pop);
    return 0;
}
