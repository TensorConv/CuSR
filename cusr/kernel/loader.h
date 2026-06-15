// loader.h — 把 pop.bin 解析成内存里的 PopData. 没有 CUDA 依赖, batch_lm.cu
// 可以 #include 这个 + 用 host 指针.

#ifndef BATCH_LM_LOADER_H
#define BATCH_LM_LOADER_H

#include "pop_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PopData {
    PopHeader header;
    // 全部 host-side malloc, free_pop_data() 一并释放
    int   *nt;       // [total_nodes]
    float *nv;       // [total_nodes]
    int   *ci;       // [total_nodes]
    TreeMeta *metas; // [M_prob]
    float *c_init;   // [total_c]
    float *xs;       // [N * n_vars]
    float *ym;       // [N]
} PopData;

// 成功返回 0, 失败 != 0 并打印到 stderr. 失败时不释放 (caller free 即可).
int load_pop_bin(const char *path, PopData *out);

void free_pop_data(PopData *p);

#ifdef __cplusplus
}
#endif

#endif
