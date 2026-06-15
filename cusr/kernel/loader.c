// loader.c — load/free pop.bin.

#include "loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_n(FILE *f, void *buf, size_t bytes, const char *what) {
    size_t got = fread(buf, 1, bytes, f);
    if (got != bytes) {
        fprintf(stderr, "load_pop_bin: short read on %s (got %zu / %zu)\n", what, got, bytes);
        return -1;
    }
    return 0;
}

int load_pop_bin(const char *path, PopData *out) {
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_pop_bin: fopen %s failed\n", path);
        return -1;
    }
    if (read_n(f, &out->header, sizeof(out->header), "header")) { fclose(f); return -1; }
    if (out->header.magic != POP_MAGIC) {
        fprintf(stderr, "load_pop_bin: bad magic 0x%08x (want 0x%08x)\n",
                out->header.magic, POP_MAGIC);
        fclose(f); return -1;
    }
    if (out->header.version != POP_VERSION) {
        fprintf(stderr, "load_pop_bin: version %d != %d\n",
                out->header.version, POP_VERSION);
        fclose(f); return -1;
    }
    const PopHeader *h = &out->header;

    out->nt     = (int*)malloc((size_t)h->total_nodes * sizeof(int));
    out->nv     = (float*)malloc((size_t)h->total_nodes * sizeof(float));
    out->ci     = (int*)malloc((size_t)h->total_nodes * sizeof(int));
    out->metas  = (TreeMeta*)malloc((size_t)h->M_prob * sizeof(TreeMeta));
    out->c_init = (float*)malloc((size_t)h->total_c * sizeof(float));
    out->xs     = (float*)malloc((size_t)h->N * h->n_vars * sizeof(float));
    out->ym     = (float*)malloc((size_t)h->M_prob * h->N * sizeof(float));
    if (!out->nt || !out->nv || !out->ci || !out->metas || !out->c_init || !out->xs || !out->ym) {
        fprintf(stderr, "load_pop_bin: malloc failed\n");
        fclose(f); return -1;
    }

    if (read_n(f, out->nt,     (size_t)h->total_nodes * sizeof(int), "nt")) goto err;
    if (read_n(f, out->nv,     (size_t)h->total_nodes * sizeof(float), "nv")) goto err;
    if (read_n(f, out->ci,     (size_t)h->total_nodes * sizeof(int), "ci")) goto err;
    if (read_n(f, out->metas,  (size_t)h->M_prob * sizeof(TreeMeta), "metas")) goto err;
    if (read_n(f, out->c_init, (size_t)h->total_c * sizeof(float), "c_init")) goto err;
    if (read_n(f, out->xs,     (size_t)h->N * h->n_vars * sizeof(float), "xs")) goto err;
    if (read_n(f, out->ym,     (size_t)h->M_prob * h->N * sizeof(float), "ym")) goto err;

    // 一致性 sanity
    int K_sum = 0, node_sum = 0;
    for (int i = 0; i < h->M_prob; i++) { K_sum += out->metas[i].K; node_sum += out->metas[i].n_nodes; }
    if (K_sum != h->total_c) {
        fprintf(stderr, "load_pop_bin: sum(K)=%d != header.total_c=%d\n", K_sum, h->total_c);
        goto err;
    }
    if (node_sum != h->total_nodes) {
        fprintf(stderr, "load_pop_bin: sum(n_nodes)=%d != header.total_nodes=%d\n",
                node_sum, h->total_nodes);
        goto err;
    }

    fclose(f);
    return 0;

err:
    fclose(f);
    return -1;
}

void free_pop_data(PopData *p) {
    if (!p) return;
    free(p->nt); free(p->nv); free(p->ci);
    free(p->metas); free(p->c_init);
    free(p->xs); free(p->ym);
    memset(p, 0, sizeof(*p));
}
