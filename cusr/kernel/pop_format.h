// pop_format.h — Python/C 共用的 EvoGP forest dump 格式.
//
// 字节序: 写死 little-endian (本机 x86_64), 不做跨平台.
// 全部 int32 / float32, 没有指针, 没有 padding (PopHeader 64 字节, TreeMeta 16 字节).
//
// 文件 layout:
//   [PopHeader  64 B]
//   [nt_all     total_nodes * 4 B]    int32_t node_type
//   [nv_all     total_nodes * 4 B]    float32_t node_value
//   [ci_all     total_nodes * 4 B]    int32_t const_idx (-1 if not CONST)
//   [metas      M_prob * 16 B]        TreeMeta{node_offset, n_nodes, c_offset, K}
//   [c_init     total_c * 4 B]        float32_t 初始 c (EvoGP forest 当前常数)
//   [xs         N * n_vars * 4 B]     float32_t, layout xs[i*n_vars + k]
//   [ym         M_prob * N * 4 B]     float32_t per-tree y, layout ym[m*N + i]
//     (real EvoGP 跑: 所有树共享同一份 y, dump 端复制 M_prob 份; fixture: 每棵树可
//      独立 y. 4MB @ M=N=1000, MVP 不优化空间)
//
// Python 端: 用 struct.pack 写, 见 dump_evogp.py / tests/pop_fixture_gen.py.
// C 端: load_pop_bin() 读 (loader.c).

#ifndef POP_FORMAT_H
#define POP_FORMAT_H

#include <stdint.h>

// 'M' '4' 'L' 'M' 4 字节, 在 x86 little-endian 上当 uint32 读 = 0x4D4C344D.
#define POP_MAGIC   0x4D4C344D
#define POP_VERSION 1

// 'M', '4', 'L', 'M' 当 char[4] 也认得 (写 / 读用 int32 sentinel, 这是注释方便看).
// 64 字节 header.
typedef struct PopHeader {
    int32_t magic;            // == POP_MAGIC
    int32_t version;          // == POP_VERSION
    int32_t M_prob;           // 树数量
    int32_t total_nodes;      // sum of n_nodes
    int32_t total_c;          // sum of K
    int32_t N;                // 数据点数
    int32_t n_vars;           // 输入变量数
    int32_t K_max;            // max(K) across trees
    int32_t max_stack;        // max stack depth needed
    int32_t reserved[7];      // 凑 64 B, 留给将来
} PopHeader;

// 16 字节 per tree.
typedef struct TreeMeta {
    int32_t node_offset;
    int32_t n_nodes;
    int32_t c_offset;
    int32_t K;
} TreeMeta;

#endif
