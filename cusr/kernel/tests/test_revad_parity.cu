// test_revad_parity.cu — Reverse-AD T3 PRIMARY PARITY GATE: 在**真实 population .bin**
// 上, rev_jacobian_kernel 的 d_J 必须与 forward ad_jacobian_kernel 的 d_J 逐元素一致。
//
// 这是 reverse-AD 的主闸门 (build_jtj_jtr_kernel 逐字节依赖 d_J 契约)。**只** include
// revad_interp.cuh —— transitively 拉 ad_interp.cuh (forward 参照 kernel)。链接 loader.c
// 读 .bin (PopData host 指针), 上传与 batch_lm_ad.cu 完全一致。
//
//   argv[1] = pop .bin 路径 (默认 synth_inner-const-heavy_M4000_N1000_seed0.bin)。
//   跑 BOTH kernel (forward + reverse), 同 launch args (warp-per-tree, block=256, wpb=8, grid),
//   把两份 d_J 拷回 host, 对每棵树每个 j<K_m 每个 i<N 逐元素 diff:
//     - 两边都有限:           rel = |rev-fwd| / (|fwd| + abs_floor),  rel<1e-3 才算等。
//     - 两边都非有限 (NaN/Inf): 视作相等/OK (真奇点处 forward 已 NaN, reverse 也 NaN = 一致)。
//     - fwd 非有限 / rev 有限:  forward-AD 的 NaN 污染缺陷 (NaN*0=NaN 把整条切向量毒化), 在**值有限**
//                              的点上 forward 给 nan 而 reverse 给正确有限值 —— 经 scipy fp64 oracle
//                              证实 reverse 正确 (68/68; 见 experiments/revad_v5/FINDINGS_parity_scipy.md)。
//                              故**接受**, 单独计数, 不算失败 (2026-06-22 用户批准的 oracle 修正)。
//     - fwd 有限 / rev 非有限:  reverse 反而更差 → FAIL (不靠放松容差掩盖)。
//     - 两边有限 rel 超标:      FAIL。
//   打印: 比对元素总数、各类计数、max rel err 及位置、both-non-finite、fwd-nan/rev-finite、max n_nodes。
//
// 容差 rel<1e-3, 最后一行 RESULT: PASS / RESULT: FAIL, exit code 反映之。
//
// build/gate:
//   source /home/weish/hao/CuSR/scripts/env.sh && cd /home/weish/hao/CuSR/cusr/kernel && \
//   nvcc -O2 -arch=sm_80 -std=c++17 --use_fast_math -o tests/test_revad_parity tests/test_revad_parity.cu loader.c && \
//   CUDA_VISIBLE_DEVICES=0 ./tests/test_revad_parity ; echo EXIT=$?

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../revad_interp.cuh"   // reverse-AD impl (transitively pulls ad_interp.cuh + forward kernel)
#include "../loader.h"           // load_pop_bin / PopData (extern "C" 已在头内守卫)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

static const char *DEFAULT_BIN =
    "/home/weish/hao/CuSR/data/workload/synth/synth_inner-const-heavy_M4000_N1000_seed0.bin";

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : DEFAULT_BIN;
    printf("[test_revad_parity] forward AD vs reverse AD d_J element-wise (tol rel < 1e-3)\n");
    printf("  pop = %s\n", path);

    // ---------------- load real population ----------------
    PopData pop;
    if (load_pop_bin(path, &pop)) {
        fprintf(stderr, "load_pop_bin failed for %s\n", path);
        printf("RESULT: FAIL\n");
        return 2;
    }
    const PopHeader *h = &pop.header;
    int M_prob = h->M_prob, N = h->N, n_vars = h->n_vars, K_max = h->K_max;
    int total_nodes = h->total_nodes, total_c = h->total_c;
    printf("  M_prob=%d total_nodes=%d total_c=%d N=%d n_vars=%d K_max=%d max_stack=%d\n",
           M_prob, total_nodes, total_c, N, n_vars, K_max, h->max_stack);

    // max n_nodes over metas. reverse tape d1/d2 是 [MAX_NODES] local 数组, n_nodes>MAX_NODES
    // 会在 launch 后静默越界写 local mem (UB)。**launch 前**就 fail-fast (镜像驱动 batch_lm_revad.cu
    // 的 load 期断言); 任何直接调 rev_jacobian_kernel 的测试/调用者都该有这道守卫 (Codex review #4)。
    int max_nn = 0;
    for (int m = 0; m < M_prob; m++) if (pop.metas[m].n_nodes > max_nn) max_nn = pop.metas[m].n_nodes;
    printf("  max n_nodes over metas = %d\n", max_nn);
    if (max_nn > MAX_NODES) {
        fprintf(stderr, "max n_nodes=%d > MAX_NODES=%d — reverse tape would overflow; refusing to launch.\n",
                max_nn, MAX_NODES);
        free_pop_data(&pop);
        printf("RESULT: FAIL\n");
        return 2;
    }

    if (K_max > MAX_K) {
        fprintf(stderr, "K_max=%d > MAX_K=%d (corpus 超出编译期上界)\n", K_max, MAX_K);
        free_pop_data(&pop);
        printf("RESULT: FAIL\n");
        return 2;
    }

    // ---------------- upload (镜像 batch_lm_ad.cu) ----------------
    int *d_nt, *d_ci; float *d_nv, *d_xs, *d_c, *d_J_fwd, *d_J_rev; TreeMeta *d_metas;
    CK(cudaMalloc(&d_nt,    (size_t)total_nodes * sizeof(int)));
    CK(cudaMalloc(&d_nv,    (size_t)total_nodes * sizeof(float)));
    CK(cudaMalloc(&d_ci,    (size_t)total_nodes * sizeof(int)));
    CK(cudaMalloc(&d_metas, (size_t)M_prob * sizeof(TreeMeta)));
    CK(cudaMalloc(&d_xs,    (size_t)N * n_vars * sizeof(float)));
    CK(cudaMalloc(&d_c,     (size_t)total_c * sizeof(float)));
    CK(cudaMalloc(&d_J_fwd, (size_t)M_prob * K_max * N * sizeof(float)));
    CK(cudaMalloc(&d_J_rev, (size_t)M_prob * K_max * N * sizeof(float)));

    CK(cudaMemcpy(d_nt,    pop.nt,     (size_t)total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_nv,    pop.nv,     (size_t)total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci,    pop.ci,     (size_t)total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_metas, pop.metas,  (size_t)M_prob * sizeof(TreeMeta),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xs,    pop.xs,     (size_t)N * n_vars * sizeof(float),  cudaMemcpyHostToDevice));
    // 用 pop.bin 自带的初始常数 c_init (与 batch_lm_ad.cu 初始 d_call 一致)。
    CK(cudaMemcpy(d_c,     pop.c_init, (size_t)total_c * sizeof(float),     cudaMemcpyHostToDevice));
    // 两份 J 预填 NaN, 确认 kernel 真写了每个 j<K 槽。
    CK(cudaMemset(d_J_fwd, 0xFF, (size_t)M_prob * K_max * N * sizeof(float)));
    CK(cudaMemset(d_J_rev, 0xFF, (size_t)M_prob * K_max * N * sizeof(float)));

    // ---------------- launch BOTH (SAME launch args) ----------------
    int block = 256, wpb = block / 32;
    int grid = (M_prob + wpb - 1) / wpb;
    ad_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
                                        d_xs, n_vars, N, d_c, K_max, d_J_fwd);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    rev_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
                                         d_xs, n_vars, N, d_c, K_max, d_J_rev);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> Jf((size_t)M_prob * K_max * N);
    std::vector<float> Jr((size_t)M_prob * K_max * N);
    CK(cudaMemcpy(Jf.data(), d_J_fwd, (size_t)M_prob * K_max * N * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Jr.data(), d_J_rev, (size_t)M_prob * K_max * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- element-wise diff across trees (only j < K_m, i < N) ----------------
    const float TOL = 1e-3f;
    const float ABS_FLOOR = 1e-5f;
    long long n_cmp = 0;
    long long n_fail_rel = 0;       // 两边有限但 rel 超标
    long long n_fail_mismatch = 0;  // 一边有限一边非有限
    long long n_both_nonfinite = 0;     // 两边都 NaN/Inf (OK)
    long long n_fwd_nan_rev_finite = 0; // forward NaN 污染、reverse 给正确有限值 (scipy 证实 OK)
    double max_rel = 0.0;
    int mr_m = -1, mr_j = -1, mr_i = -1; float mr_f = 0, mr_r = 0;

    for (int m = 0; m < M_prob; m++) {
        int K = pop.metas[m].K;
        if (K == 0) continue;   // K==0 树两 kernel 都直接 return, 无写, 不比
        size_t base = (size_t)m * K_max * N;
        for (int j = 0; j < K; j++) {
            for (int i = 0; i < N; i++) {
                float fwd = Jf[base + (size_t)j * N + i];
                float rev = Jr[base + (size_t)j * N + i];
                n_cmp++;
                bool f_fin = isfinite(fwd), r_fin = isfinite(rev);
                if (!f_fin && !r_fin) { n_both_nonfinite++; continue; }      // both non-finite = equal/OK
                if (f_fin != r_fin) {
                    // forward-AD NaN 污染: 值有限但某 local partial 非有限的点上, forward 的 chunked
                    // JVP 用 NaN*0=NaN 毒化整条切向量, 给 sibling 列吐 nan; reverse 按列隔离给出正确有
                    // 限梯度 —— 经 scipy fp64 oracle 证实 (68/68, FINDINGS_parity_scipy.md)。接受, 不算败。
                    if (!f_fin && r_fin) {
                        n_fwd_nan_rev_finite++;
                        if (n_fwd_nan_rev_finite <= 10)
                            fprintf(stderr, "  FWD-NAN(rev correct, scipy-validated) m=%d j=%d i=%d  fwd=%g rev=%g\n", m, j, i, fwd, rev);
                        continue;
                    }
                    // rev 非有限而 fwd 有限 = reverse 反而更差 → 真失败。
                    n_fail_mismatch++;
                    if (n_fail_mismatch <= 10)
                        fprintf(stderr, "  REV-WORSE m=%d j=%d i=%d  fwd=%g rev=%g\n", m, j, i, fwd, rev);
                    continue;
                }
                double rel = fabs((double)rev - (double)fwd) / (fabs((double)fwd) + ABS_FLOOR);
                if (rel > max_rel) { max_rel = rel; mr_m = m; mr_j = j; mr_i = i; mr_f = fwd; mr_r = rev; }
                if (rel >= TOL) {
                    n_fail_rel++;
                    if (n_fail_rel <= 10)
                        fprintf(stderr, "  RELFAIL m=%d j=%d i=%d  fwd=%g rev=%g rel=%.3e\n",
                                m, j, i, fwd, rev, rel);
                }
            }
        }
    }

    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_c); cudaFree(d_J_fwd); cudaFree(d_J_rev);
    free_pop_data(&pop);

    long long n_fail = n_fail_rel + n_fail_mismatch;
    printf("\n[test_revad_parity] compared %lld elements across %d trees\n", n_cmp, M_prob);
    printf("  both-non-finite (OK):                %lld\n", n_both_nonfinite);
    printf("  fwd-nan/rev-finite (fwd defect, scipy-validated OK): %lld\n", n_fwd_nan_rev_finite);
    printf("  FAIL rel>=%.0e:                      %lld\n", TOL, n_fail_rel);
    printf("  FAIL rev-worse (rev nan/fwd finite): %lld\n", n_fail_mismatch);
    printf("  max rel err = %.3e", max_rel);
    if (mr_m >= 0) printf("  at m=%d j=%d i=%d (fwd=%g rev=%g)", mr_m, mr_j, mr_i, mr_f, mr_r);
    printf("\n");

    if (n_fail) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
