// batch_lm.cu — GPU batched Levenberg-Marquardt for SR populations.
//
// 读 pop.bin (一批 GP 候选表达式树 + 各自 c_init + 共享 (xs, ym)),
// 对每棵树并行跑 LM 拟合常数 c, 输出:
//   - status.bin (int32, M_prob 个)
//   - c_final.bin (float32, total_c 个, jagged by per-tree K)
//
// 数值算法:
//   - Gauss-Newton + Marquardt 阻尼 (λ adaptive)
//   - finite-difference Jacobian, ε = 1e-3
//   - K×K Cholesky 解 (warp-per-problem layout, 1 thread/problem 解 K 维方程)
//   - host 端做 accept/reject + λ 调度, kernel 跑数值热路径
//
// Compile-time bounds:
//   - MAX_K=32 (K_max per tree), MAX_STACK=64 (interpreter stack)
//
// usage: ./batch_lm <pop.bin> [<out_dir>] [--max-iter N] [--quiet]
//   out_dir 不传时默认 = dirname(pop_path).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <cuda_runtime.h>

#include "lm_core.cuh"     // shared: CUDA_CHECK, STATUS_*, enum NTypeE / F_* / MAX_STACK / MAX_K,
                           // eval_tree_d, eval_kernel_batched, residual_kernel, loss_kernel,
                           // build_jtj_jtr_kernel, write_blob (single source of truth)
#include "loader.h"

// NTypeE / FuncE / MAX_STACK / MAX_K / STATUS_* / CUDA_CHECK / eval_tree_d /
// eval_kernel_batched / residual_kernel / loss_kernel / build_jtj_jtr_kernel /
// write_blob 现由 lm_core.cuh 提供 — 删此处副本避免重定义 (codegen drift 防护).

__global__ void solve_kernel(
    const float *JtJ, const float *JtR, const TreeMeta *metas,
    int M_prob, int K_max, const float *lambda_arr,
    float *delta, int *status)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M_prob) return;
    int K = metas[m].K;
    float lam = lambda_arr[m];

    float A[MAX_K * MAX_K];
    float b[MAX_K];

    for (int j1 = 0; j1 < K; j1++) {
        b[j1] = -JtR[m*K_max + j1];
        for (int j2 = 0; j2 < K; j2++)
            A[j1*K + j2] = JtJ[m*K_max*K_max + j1*K_max + j2];
        A[j1*K + j1] *= (1.0f + lam);
    }
    for (int j = 0; j < K; j++) {
        float s = A[j*K + j];
        for (int k = 0; k < j; k++) s -= A[j*K+k] * A[j*K+k];
        if (s <= 0.0f) { status[m] = -1; for (int i=0;i<K;i++) delta[m*K_max+i]=0.0f; return; }
        A[j*K + j] = sqrtf(s);
        for (int i = j+1; i < K; i++) {
            float t = A[i*K + j];
            for (int k = 0; k < j; k++) t -= A[i*K+k] * A[j*K+k];
            A[i*K + j] = t / A[j*K + j];
        }
    }
    for (int i = 0; i < K; i++) {
        float s = b[i];
        for (int k = 0; k < i; k++) s -= A[i*K+k] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = K-1; i >= 0; i--) {
        float s = b[i];
        for (int k = i+1; k < K; k++) s -= A[k*K+i] * b[k];
        b[i] = s / A[i*K + i];
    }
    for (int i = 0; i < K; i++) delta[m*K_max + i] = b[i];
    status[m] = 0;
}

// write_blob 现由 lm_core.cuh 提供 (single source of truth) — 删此处副本。

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <pop.bin> [out_dir] [--max-iter N] [--quiet]\n", argv[0]);
        return 1;
    }
    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    const char *pop_path = argv[1];
    const char *out_dir = NULL;
    int max_iter = 50;
    int quiet = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--max-iter") == 0 && i+1 < argc) { max_iter = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--quiet") == 0) { quiet = 1; }
        else if (argv[i][0] != '-') { out_dir = argv[i]; }
    }
    // 没显式传 out_dir → 从 pop_path 提 dirname. ./batch_lm data/pop.bin → data/, ./batch_lm pop.bin → ./
    static char dir_buf[1024];
    if (out_dir == NULL) {
        const char *slash = strrchr(pop_path, '/');
        if (slash) {
            size_t len = (size_t)(slash - pop_path);
            if (len >= sizeof(dir_buf)) len = sizeof(dir_buf) - 1;
            memcpy(dir_buf, pop_path, len);
            dir_buf[len] = '\0';
            out_dir = dir_buf;
        } else {
            out_dir = ".";
        }
    }

    PopData pop;
    if (load_pop_bin(pop_path, &pop)) {
        fprintf(stderr, "load failed.\n"); return 1;
    }
    const PopHeader *h = &pop.header;
    if (!quiet) {
        printf("[batch-lm] loaded %s: M_prob=%d total_nodes=%d total_c=%d N=%d n_vars=%d K_max=%d max_stack=%d\n",
               pop_path, h->M_prob, h->total_nodes, h->total_c, h->N, h->n_vars, h->K_max, h->max_stack);
    }

    if (h->K_max > MAX_K) {
        fprintf(stderr, "[batch-lm] K_max=%d > MAX_K=%d (compile-time). 增 MAX_K 重编.\n",
                h->K_max, MAX_K);
        free_pop_data(&pop); return 2;
    }
    if (h->max_stack > MAX_STACK) {
        fprintf(stderr, "[batch-lm] max_stack=%d > MAX_STACK=%d (compile-time). 增 MAX_STACK 重编.\n",
                h->max_stack, MAX_STACK);
        free_pop_data(&pop); return 2;
    }

    int M_prob = h->M_prob, N = h->N, n_vars = h->n_vars, K_max = h->K_max;
    int total_nodes = h->total_nodes, total_c = h->total_c;

    // ---- host bufs ----
    float *h_c = (float*)malloc(total_c * sizeof(float));        // c_curr
    float *h_c_try = (float*)malloc(total_c * sizeof(float));
    float *h_c_pert = (float*)malloc(total_c * sizeof(float));
    memcpy(h_c, pop.c_init, total_c * sizeof(float));            // 从 pop.bin 来

    // ---- D allocs ----
    int *d_nt; float *d_nv; int *d_ci; float *d_call; TreeMeta *d_metas;
    float *d_xs, *d_ym, *d_y, *d_yp, *d_r, *d_J, *d_JtJ, *d_JtR, *d_delta, *d_loss, *d_lam;
    int *d_stat;
    CUDA_CHECK(cudaMalloc(&d_nt, total_nodes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nv, total_nodes*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ci, total_nodes*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_call, total_c*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_metas, M_prob*sizeof(TreeMeta)));
    CUDA_CHECK(cudaMalloc(&d_xs, (size_t)N*n_vars*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ym, (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yp, (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r,  (size_t)M_prob*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_J,  (size_t)M_prob*K_max*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_JtJ,(size_t)M_prob*K_max*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_JtR,(size_t)M_prob*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta,(size_t)M_prob*K_max*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss, M_prob*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lam, M_prob*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stat, M_prob*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nt, pop.nt, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nv, pop.nv, total_nodes*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, pop.ci, total_nodes*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, pop.metas, M_prob*sizeof(TreeMeta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xs, pop.xs, (size_t)N*n_vars*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ym, pop.ym, (size_t)M_prob*N*sizeof(float), cudaMemcpyHostToDevice));

    int block = 256, wpb = block/32;
    int grid = (M_prob + wpb - 1) / wpb;

    // ---- Host LM state ----
    float *h_lam = (float*)malloc(M_prob * sizeof(float));
    float *h_loss = (float*)malloc(M_prob * sizeof(float));
    float *h_loss_try = (float*)malloc(M_prob * sizeof(float));
    int   *h_finished = (int*)calloc(M_prob, sizeof(int));
    int   *h_iter = (int*)calloc(M_prob, sizeof(int));
    int   *h_rejected = (int*)calloc(M_prob, sizeof(int));
    for (int m = 0; m < M_prob; m++) h_lam[m] = 1e-3f;

    // K=0 跳过: 没常数可优化, 立即标 finished + status=3
    int n_k0_pre = 0;
    for (int m = 0; m < M_prob; m++) {
        if (pop.metas[m].K == 0) { h_finished[m] = 3; n_k0_pre++; }
    }

    // FD 扰动: 相对步长 (MINPACK lmdif 同款). 绝对 eps 在常数量级远离 1 时毁掉
    // Jacobian (|c|>>1: fp32 舍入吞掉差分 → J 列全零 → Cholesky 挂; |c|<<1: 割线
    // 远离切线 → 假收敛到错误常数). 见 tests/test_fixture_scale.py.
    // h = eps_rel*|c| (c==0 退化为 eps_rel), 除数取 fp32 实际可表示步长.
    // eps_rel 选 1e-3 而非教科书的 sqrt(FLT_EPSILON)≈3.45e-4: --use_fast_math 下
    // eval 误差大于 1 ULP, 3.45e-4 在真实 pop 上尾部退化 (within-10× 100%→98.3%),
    // 1e-3 对 |c|~1 等价旧绝对步长 (经 parity 对拍选定, 见 data/verify_report.md).
    const float eps_rel = 1e-3f;
    // xtol=1e-6 (≈10 ULP @ fp32): 1e-5 会在慢收敛树上提前停在更差的 loss
    // (within-10× 99.3%→99.8%, 1.05× 91.9%→92.4%), 代价 +25% 迭代耗时;
    // 再紧就进 fp32 噪声了. 经 verify.py parity 对拍选定.
    const float xtol = 1e-6f;

    // ---- Initial eval + loss ----
    CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));
    eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
    CUDA_CHECK(cudaGetLastError());
    residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
    CUDA_CHECK(cudaGetLastError());
    loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, M_prob*sizeof(float), cudaMemcpyDeviceToHost));

    // 初始 loss NaN/Inf → 整棵树没救, 直接标 fail. c_init 都 produce NaN 说明这棵树
    // 没法被 LM 拯救.
    for (int m = 0; m < M_prob; m++) {
        if (h_finished[m]) continue;
        if (!isfinite(h_loss[m])) h_finished[m] = 2;
    }

    float *h_yb = (float*)malloc((size_t)M_prob*N*sizeof(float));
    float *h_yp = (float*)malloc((size_t)M_prob*N*sizeof(float));
    float *h_J  = (float*)malloc((size_t)M_prob*K_max*N*sizeof(float));
    float *h_delta = (float*)malloc(M_prob*K_max*sizeof(float));
    int   *h_solve_stat = (int*)malloc(M_prob * sizeof(int));  // 必须读 solve_kernel 的 status (避免假阳性 converge, 见 F 分支 1)
    float *h_heps = (float*)malloc(M_prob * sizeof(float));    // 当前列每棵树的实际 FD 步长

    // ---- LM loop ----
    int last_print = -10;
    for (int it = 0; it < max_iter; it++) {
        int all_done = 1;
        for (int m = 0; m < M_prob; m++) if (!h_finished[m]) { all_done = 0; break; }
        if (all_done) break;

        // A. previous d_y is baseline (init at start; G refreshes after iter)
        CUDA_CHECK(cudaMemcpy(h_yb, d_y, (size_t)M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));

        // B. FD Jacobian (K_max launches)
        for (int k = 0; k < K_max; k++) {
            memcpy(h_c_pert, h_c, total_c*sizeof(float));
            for (int m = 0; m < M_prob; m++) {
                if (h_finished[m]) continue;
                if (k < pop.metas[m].K) {
                    float cv = h_c[pop.metas[m].c_offset+k];
                    float h = eps_rel * fabsf(cv);
                    if (h == 0.0f) h = eps_rel;              // c==0 fallback
                    float cp = cv + h;
                    h_heps[m] = cp - cv;                     // fp32 实际步长 (差分除数)
                    h_c_pert[pop.metas[m].c_offset+k] = cp;
                }
            }
            CUDA_CHECK(cudaMemcpy(d_call, h_c_pert, total_c*sizeof(float), cudaMemcpyHostToDevice));
            eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_yp, d_yp, (size_t)M_prob*N*sizeof(float), cudaMemcpyDeviceToHost));
            for (int m = 0; m < M_prob; m++) {
                if (h_finished[m]) continue;
                if (k >= pop.metas[m].K) continue;
                for (int i = 0; i < N; i++)
                    h_J[(size_t)m*K_max*N + k*N + i] = (h_yp[m*N+i] - h_yb[m*N+i]) / h_heps[m];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_J, h_J, (size_t)M_prob*K_max*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));

        // C. Build JtJ + JtR
        build_jtj_jtr_kernel<<<grid, block>>>(d_J, d_r, d_metas, M_prob, N, K_max, d_JtJ, d_JtR);
        CUDA_CHECK(cudaGetLastError());

        // D. Solve
        CUDA_CHECK(cudaMemcpy(d_lam, h_lam, M_prob*sizeof(float), cudaMemcpyHostToDevice));
        solve_kernel<<<(M_prob+255)/256, 256>>>(d_JtJ, d_JtR, d_metas, M_prob, K_max, d_lam, d_delta, d_stat);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_delta, d_delta, M_prob*K_max*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_solve_stat, d_stat, M_prob*sizeof(int), cudaMemcpyDeviceToHost));

        // E. Apply step, eval, loss_try
        memcpy(h_c_try, h_c, total_c*sizeof(float));
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            for (int k = 0; k < pop.metas[m].K; k++)
                h_c_try[pop.metas[m].c_offset+k] += h_delta[m*K_max+k];
        }
        CUDA_CHECK(cudaMemcpy(d_call, h_c_try, total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_yp);
        CUDA_CHECK(cudaGetLastError());
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_yp, d_ym, M_prob*N, d_r);
        CUDA_CHECK(cudaGetLastError());
        loss_kernel<<<grid, block>>>(d_r, M_prob, N, d_loss);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_loss_try, d_loss, M_prob*sizeof(float), cudaMemcpyDeviceToHost));

        // F. Per-tree accept/reject + λ + convergence
        // h_finished 编码 (internal, == status_out 的 enum):
        //   0 = 还在跑
        //   1 = converged
        //   2 = FAIL_NAN          (loss_try NaN/Inf)
        //   3 = K=0 skip          (pre-loop set)
        //   4 = FAIL_CHOLESKY     (Cholesky breakdown 反复, λ 爆)
        // 主循环看 == 0 决定是否继续. 末尾 maxiter 用还在 0 的项填.
        for (int m = 0; m < M_prob; m++) {
            if (h_finished[m]) continue;
            h_iter[m]++;
            int K = pop.metas[m].K;

            // Cholesky breakdown: solve_kernel 设 status[m]=-1, delta=0.
            // 必须显式读 d_stat → 否则 δ=0 让 d_norm=0 假阳性 "converge". Treat as reject.
            if (h_solve_stat[m] != 0) {
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 4;  // FAIL_CHOLESKY
                continue;
            }

            // NaN/Inf: loss_try 出错说明 c_try 把 tree eval 推到 Inf/NaN 了.
            // 不要 reject 后重试 (c 没被污染, 但下一步 lam *= 10 后还会同一种 c_try 再算
            // — 因为 δ 由 J / r 决定, 都跟 c 有关. 直接放弃这棵).
            if (!isfinite(h_loss_try[m])) {
                h_finished[m] = 2;  // FAIL_NAN
                continue;
            }

            float d_norm_sq = 0, c_norm_sq = 0;
            for (int k = 0; k < K; k++) {
                d_norm_sq += h_delta[m*K_max+k] * h_delta[m*K_max+k];
                c_norm_sq += h_c_try[pop.metas[m].c_offset+k] * h_c_try[pop.metas[m].c_offset+k];
            }
            float d_norm = sqrtf(d_norm_sq), c_norm = sqrtf(c_norm_sq);

            if (d_norm < xtol * (c_norm + xtol)) {
                // accept + 收敛 (K=0 不走这条 — 已经 pre-skip)
                for (int k = 0; k < K; k++)
                    h_c[pop.metas[m].c_offset+k] = h_c_try[pop.metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                h_finished[m] = 1;
            } else if (h_loss_try[m] < h_loss[m]) {
                for (int k = 0; k < K; k++)
                    h_c[pop.metas[m].c_offset+k] = h_c_try[pop.metas[m].c_offset+k];
                h_loss[m] = h_loss_try[m];
                h_lam[m] *= 0.1f;
                if (h_lam[m] < 1e-12f) h_lam[m] = 1e-12f;
            } else {
                h_lam[m] *= 10.0f;
                h_rejected[m]++;
                if (h_lam[m] > 1e12f) h_finished[m] = 2;  // FAIL_NAN bucket: lam-blow-up without Cholesky 是真的 NaN-ish 路径
            }
        }

        // G. Refresh d_y at h_c (accepted state)
        CUDA_CHECK(cudaMemcpy(d_call, h_c, total_c*sizeof(float), cudaMemcpyHostToDevice));
        eval_kernel_batched<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob, d_xs, n_vars, N, d_call, d_y);
        CUDA_CHECK(cudaGetLastError());
        residual_kernel<<<(M_prob*N+127)/128, 128>>>(d_y, d_ym, M_prob*N, d_r);
        CUDA_CHECK(cudaGetLastError());

        if (!quiet && (it - last_print >= 5 || all_done)) {
            int n_done = 0; for (int m = 0; m < M_prob; m++) if (h_finished[m]) n_done++;
            printf("  it %2d: done=%d/%d\n", it, n_done, M_prob);
            last_print = it;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

#ifndef NO_FP64_GUARD
    // fp64-honesty boundary audit: deliver c_init for any tree whose honest fp64
    // loss got worse (fast-math fake-improvement). See lm_core.cuh.
    {
        int n_rev = fp64_boundary_audit(grid, block, d_nt, d_nv, d_ci, d_metas,
            M_prob, d_xs, n_vars, N, d_ym, total_c, pop.metas, pop.c_init,
            h_c, h_finished, h_loss);
        if (n_rev < 0) { fprintf(stderr, "[fp64-guard] CUDA error during boundary audit\n"); return 1; }
        if (!quiet) printf("[fp64-guard] reverted %d/%d trees (fp64 c_final non-finite or worse than c_init)\n",
                           n_rev, M_prob);
    }
#endif

    // ---- Final: 把内部 h_finished {0,1,2,3,4} 映射成外部 status_out ----
    int n_conv = 0, n_maxiter = 0, n_fail_nan = 0, n_k0 = 0, n_fail_chol = 0;
    int *status_out = (int*)malloc(M_prob * sizeof(int));
    for (int m = 0; m < M_prob; m++) {
        if      (h_finished[m] == 1) { status_out[m] = STATUS_CONVERGED;     n_conv++; }
        else if (h_finished[m] == 2) { status_out[m] = STATUS_FAIL_NAN;      n_fail_nan++; }
        else if (h_finished[m] == 3) { status_out[m] = STATUS_K0_SKIP;       n_k0++; }
        else if (h_finished[m] == 4) { status_out[m] = STATUS_FAIL_CHOLESKY; n_fail_chol++; }
        else                         { status_out[m] = STATUS_MAXITER;       n_maxiter++; }
    }

    char buf[2048];  // > dir_buf[1024] + "/c_final.bin" 留余地, 避 snprintf fortify 警告
    snprintf(buf, sizeof(buf), "%s/status.bin", out_dir);
    write_blob(buf, status_out, (size_t)M_prob * sizeof(int));
    snprintf(buf, sizeof(buf), "%s/c_final.bin", out_dir);
    write_blob(buf, h_c, (size_t)total_c * sizeof(float));

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_s = (t_end.tv_sec - t_start.tv_sec)
                     + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    if (!quiet) {
        printf("[batch-lm] 完成: converged=%d (%.1f%%)  maxiter=%d  fail_nan=%d  fail_cholesky=%d  k0_skip=%d  (M_prob=%d)\n",
               n_conv, 100.0 * n_conv / M_prob, n_maxiter, n_fail_nan, n_fail_chol, n_k0, M_prob);
        printf("[batch-lm] 写出 %s/status.bin (%zu B) + %s/c_final.bin (%zu B)\n",
               out_dir, (size_t)M_prob * sizeof(int),
               out_dir, (size_t)total_c * sizeof(float));
        printf("[batch-lm] 总耗时 %.3f 秒 (%.0f trees/s, M_prob=%d)\n",
               elapsed_s, (double)M_prob / elapsed_s, M_prob);
    }

    // ---- cleanup ----
    free(h_c); free(h_c_try); free(h_c_pert);
    free(h_lam); free(h_loss); free(h_loss_try);
    free(h_finished); free(h_iter); free(h_rejected);
    free(h_yb); free(h_yp); free(h_J); free(h_delta);
    free(h_solve_stat); free(h_heps);
    free(status_out);
    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_call); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_ym); cudaFree(d_y); cudaFree(d_yp); cudaFree(d_r);
    cudaFree(d_J); cudaFree(d_JtJ); cudaFree(d_JtR); cudaFree(d_delta);
    cudaFree(d_loss); cudaFree(d_lam); cudaFree(d_stat);
    free_pop_data(&pop);
    return 0;
}
