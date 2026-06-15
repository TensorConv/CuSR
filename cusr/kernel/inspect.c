// inspect.c — 读 pop.bin, 打印统计 (不跑 LM, 不依赖 CUDA).
//
// 用途: 验证 dump 通路 + 看 K 分布决定 MAX_K 是否够.
//
// 报: header values, K 分布 (p50/p90/p99/max + CDF), n_nodes 分布, op 频次,
// K=0 棵数, K>32 棵数 (假设 MAX_K=32, batch_lm 的 compile-time bound), max_stack.
//
// 不读 c_init / X / y, 那些等 batch_lm 处理. 只验 nt_all / nv_all / ci_all / metas.
//
// build: cc -O2 -o inspect inspect.c
// run:   ./inspect data/pop.bin

#include "pop_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 跟 batch_lm 的 compile-time bound 一致. 改这里要同步改 batch_lm.cu.
#define INSPECT_MAX_K 32

// EvoGP Func enum (mirrors EvoGP's tree/utils.py)
enum {
    F_IF=0, F_ADD=1, F_SUB=2, F_MUL=3, F_DIV=4,
    F_LOOSE_DIV=5, F_POW=6, F_LOOSE_POW=7,
    F_MAX=8, F_MIN=9, F_LT=10, F_GT=11, F_LE=12, F_GE=13,
    F_SIN=14, F_COS=15, F_TAN=16, F_SINH=17, F_COSH=18, F_TANH=19,
    F_LOG=20, F_LOOSE_LOG=21, F_EXP=22, F_INV=23, F_LOOSE_INV=24,
    F_NEG=25, F_ABS=26, F_SQRT=27, F_LOOSE_SQRT=28, F_NOP=29
};
enum { N_VAR=0, N_CONST=1, N_UFUNC=2, N_BFUNC=3, N_TFUNC=4 };

static const char *FUNC_NAME[30] = {
    "IF","ADD","SUB","MUL","DIV","LDIV","POW","LPOW","MAX","MIN",
    "LT","GT","LE","GE","SIN","COS","TAN","SINH","COSH","TANH",
    "LOG","LLOG","EXP","INV","LINV","NEG","ABS","SQRT","LSQRT","NOP",
};

static int cmp_int(const void *a, const void *b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x>y) - (x<y);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pop.bin>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    PopHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "header read short\n"); return 1;
    }
    if (hdr.magic != POP_MAGIC) {
        fprintf(stderr, "bad magic 0x%08x (want 0x%08x)\n", hdr.magic, POP_MAGIC);
        return 1;
    }
    if (hdr.version != POP_VERSION) {
        fprintf(stderr, "bad version %d (want %d)\n", hdr.version, POP_VERSION);
        return 1;
    }

    printf("== %s ==\n", argv[1]);
    printf("  magic=0x%08x version=%d\n", hdr.magic, hdr.version);
    printf("  M_prob=%d total_nodes=%d total_c=%d N=%d n_vars=%d K_max=%d max_stack=%d\n",
           hdr.M_prob, hdr.total_nodes, hdr.total_c, hdr.N, hdr.n_vars,
           hdr.K_max, hdr.max_stack);

    if (hdr.K_max > INSPECT_MAX_K) {
        printf("  ** WARNING K_max=%d > INSPECT_MAX_K=%d **\n",
               hdr.K_max, INSPECT_MAX_K);
    }

    // ---- 读 nt / nv / ci ----
    int *nt = malloc((size_t)hdr.total_nodes * sizeof(int));
    float *nv = malloc((size_t)hdr.total_nodes * sizeof(float));
    int *ci = malloc((size_t)hdr.total_nodes * sizeof(int));
    if (fread(nt, sizeof(int), hdr.total_nodes, f) != (size_t)hdr.total_nodes) goto eof;
    if (fread(nv, sizeof(float), hdr.total_nodes, f) != (size_t)hdr.total_nodes) goto eof;
    if (fread(ci, sizeof(int), hdr.total_nodes, f) != (size_t)hdr.total_nodes) goto eof;

    // ---- 读 metas ----
    TreeMeta *metas = malloc((size_t)hdr.M_prob * sizeof(TreeMeta));
    if (fread(metas, sizeof(TreeMeta), hdr.M_prob, f) != (size_t)hdr.M_prob) goto eof;

    // ---- K 分布 ----
    int *Karr = malloc((size_t)hdr.M_prob * sizeof(int));
    int *nlen = malloc((size_t)hdr.M_prob * sizeof(int));
    int K0_count = 0, Kover_count = 0;
    int K_max_actual = 0, n_max_actual = 0;
    for (int i = 0; i < hdr.M_prob; i++) {
        Karr[i] = metas[i].K;
        nlen[i] = metas[i].n_nodes;
        if (metas[i].K == 0) K0_count++;
        if (metas[i].K > INSPECT_MAX_K) Kover_count++;
        if (metas[i].K > K_max_actual) K_max_actual = metas[i].K;
        if (metas[i].n_nodes > n_max_actual) n_max_actual = metas[i].n_nodes;
    }
    qsort(Karr, hdr.M_prob, sizeof(int), cmp_int);
    qsort(nlen, hdr.M_prob, sizeof(int), cmp_int);
    int p50 = Karr[hdr.M_prob/2];
    int p90 = Karr[(hdr.M_prob*90)/100];
    int p99 = Karr[(hdr.M_prob*99)/100];
    int nlen_p50 = nlen[hdr.M_prob/2];
    int nlen_p90 = nlen[(hdr.M_prob*90)/100];
    int nlen_p99 = nlen[(hdr.M_prob*99)/100];

    printf("  K distribution: p50=%d p90=%d p99=%d max=%d\n", p50, p90, p99, K_max_actual);
    printf("  n_nodes distribution: p50=%d p90=%d p99=%d max=%d\n",
           nlen_p50, nlen_p90, nlen_p99, n_max_actual);
    printf("  K=0 trees: %d  (跳过 LM, 无常数可优化)\n", K0_count);
    printf("  K>%d trees: %d  (超过 batch_lm compile-time MAX_K, dump 应过滤)\n",
           INSPECT_MAX_K, Kover_count);

    // ---- op 频次 ----
    int ufunc_count[30] = {0}, bfunc_count[30] = {0};
    int tfunc_count = 0, var_count = 0, const_count = 0, ufunc_total = 0, bfunc_total = 0;
    for (int i = 0; i < hdr.total_nodes; i++) {
        int t = nt[i];
        int fid = (int)nv[i];
        if (t == N_VAR) var_count++;
        else if (t == N_CONST) const_count++;
        else if (t == N_UFUNC) { ufunc_total++; if (fid >= 0 && fid < 30) ufunc_count[fid]++; }
        else if (t == N_BFUNC) { bfunc_total++; if (fid >= 0 && fid < 30) bfunc_count[fid]++; }
        else if (t == N_TFUNC) tfunc_count++;
    }
    printf("  nodes: var=%d const=%d ufunc=%d bfunc=%d tfunc=%d\n",
           var_count, const_count, ufunc_total, bfunc_total, tfunc_count);
    if (tfunc_count > 0) {
        printf("  ** WARNING tfunc=%d in dumped pop — batch_lm doesn't handle TFUNC, "
               "should be filtered at dump.\n", tfunc_count);
    }
    printf("  unary op freq (only non-zero):");
    for (int i = 14; i <= 28; i++) if (ufunc_count[i] > 0) printf(" %s=%d", FUNC_NAME[i], ufunc_count[i]);
    printf("\n  binary op freq (only non-zero):");
    for (int i = 1; i <= 13; i++) if (bfunc_count[i] > 0) printf(" %s=%d", FUNC_NAME[i], bfunc_count[i]);
    printf("\n");

    // LOOSE_* 检测 — dump 应已退化, 这里如果还有就是 bug
    int loose_total = ufunc_count[F_LOOSE_LOG] + ufunc_count[F_LOOSE_INV]
                    + ufunc_count[F_LOOSE_SQRT] + bfunc_count[F_LOOSE_DIV]
                    + bfunc_count[F_LOOSE_POW];
    if (loose_total > 0) {
        printf("  ** WARNING loose_total=%d — dumper should degrade LOOSE_* to standard ops.\n",
               loose_total);
    }

    // ---- consistency: K sum, ci range ----
    int K_sum = 0;
    for (int i = 0; i < hdr.M_prob; i++) K_sum += metas[i].K;
    if (K_sum != hdr.total_c) {
        printf("  ** sanity: sum(K)=%d != header.total_c=%d **\n", K_sum, hdr.total_c);
    }
    int node_sum = 0;
    for (int i = 0; i < hdr.M_prob; i++) node_sum += metas[i].n_nodes;
    if (node_sum != hdr.total_nodes) {
        printf("  ** sanity: sum(n_nodes)=%d != header.total_nodes=%d **\n",
               node_sum, hdr.total_nodes);
    }

    // ---- ci 一致性: 每棵树 CONST 节点的 ci 应是 [0, K) prefix 序 ----
    int ci_bad = 0;
    for (int m = 0; m < hdr.M_prob; m++) {
        int off = metas[m].node_offset;
        int n = metas[m].n_nodes;
        int K = metas[m].K;
        int seen_k = 0;
        for (int i = 0; i < n; i++) {
            int t = nt[off + i];
            int cidx = ci[off + i];
            if (t == N_CONST) {
                if (cidx != seen_k) { ci_bad++; break; }
                seen_k++;
            } else {
                if (cidx != -1) { ci_bad++; break; }
            }
        }
        if (seen_k != K) ci_bad++;
    }
    if (ci_bad > 0) printf("  ** ci 不一致 trees: %d **\n", ci_bad);

    printf("\n[inspect] PASS — header valid, %d trees, K_max=%d, max_stack=%d\n",
           hdr.M_prob, K_max_actual, hdr.max_stack);

    fclose(f);
    free(nt); free(nv); free(ci); free(metas); free(Karr); free(nlen);
    return 0;

eof:
    fprintf(stderr, "unexpected EOF\n");
    fclose(f);
    return 1;
}
