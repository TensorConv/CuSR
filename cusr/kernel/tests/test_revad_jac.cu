// test_revad_jac.cu — Reverse-AD T2 GPU 单测: rev_jacobian_kernel vs (a) 手推解析
// Jacobian, (b) forward ad_jacobian_kernel (同树同点)。
//
// 镜像 tests/test_ad_jac.cu (forward 版) 逐树逐覆盖, 但被测对象是 reverse-mode
// rev_jacobian_kernel(SAME launch args as ad_jacobian_kernel)。**只** include
// revad_interp.cuh —— 它 transitively 拉 ad_interp.cuh, 故 forward ad_jacobian_kernel
// (做参照) 也可用。
//
// 直接构造 token 数组 (nt/nv/ci/metas/xs/c), 一次 launch rev_jacobian_kernel 拷回
// d_J_rev; 再一次 launch forward ad_jacobian_kernel 拷回 d_J_fwd。对每个 (tree, j, i):
//   (a) rev vs 手推解析 J,  rel<1e-3 (abs floor 1e-5);
//   (b) rev vs fwd,        rel<1e-3 (abs floor 1e-5)。
// 两类比对中任何 NaN/Inf (在应写的 j<K 槽) 也强制 FAIL。
//
// 覆盖树 (与 test_ad_jac.cu 完全相同, 含 T3 K=9 / T4 K=16 跨 AD_W=8 chunk 边界):
//   T0  y = c0*x0 + c1            K=2
//   T1  y = c0*sin(c1*x0)         K=2
//   T2  y = exp(c0*x0)            K=1
//   T3  y = sum_{j=0..8} cj*xj    K=9   (线性, ∂/∂cj=xj)
//   T4  y = sum_{k=0..15} sin(ck*x0)  K=16  (非线性, ∂/∂ck=x0*cos(ck*x0); 两满桶)
//
// d_J 布局必须与 fd_jacobian / ad_jacobian 一致: Jm = J + m*K_max*N; Jm[j*N + i]; j<K 才写。
//
// build/gate:
//   source /home/weish/hao/CuSR/scripts/env.sh && cd /home/weish/hao/CuSR/cusr/kernel && \
//   nvcc -O2 -arch=sm_80 -std=c++17 --use_fast_math -o tests/test_revad_jac tests/test_revad_jac.cu && \
//   CUDA_VISIBLE_DEVICES=0 ./tests/test_revad_jac ; echo EXIT=$?

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../revad_interp.cuh"   // reverse-AD impl (transitively pulls ad_interp.cuh + forward kernel)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); }} while(0)

// ---- batch builder: 把每棵树 (pre-order token + K) 追加到全局扁平数组 ----------
struct Tree {
    std::vector<int>   nt;
    std::vector<float> nv;
    std::vector<int>   ci;   // const_idx, -1 if not CONST
    int K;                   // 常数个数
    const char *name;
};

int main() {
    printf("[test_revad_jac] rev_jacobian_kernel vs analytic + vs forward AD (tol rel < 1e-3)\n");

    // ---------------- 数据点 (N 个), n_vars 个变量 ----------------
    // T0..T2 只用 x0; T3 用 x0..x8。一个共享 xs (N*n_vars)。与 test_ad_jac.cu 完全相同。
    const int n_vars = 9;
    const int N = 6;
    float xs[N * 9] = {
        // x0    x1    x2    x3    x4    x5    x6    x7    x8
         0.7f, 1.0f,-0.5f, 0.3f, 2.0f,-1.2f, 0.9f, 1.5f,-0.8f,
        -0.4f, 0.6f, 1.1f,-1.0f, 0.2f, 0.8f,-0.3f, 1.3f, 0.5f,
         1.2f,-0.7f, 0.4f, 0.9f,-1.5f, 1.0f, 0.6f,-0.9f, 0.7f,
         0.1f, 1.4f,-1.1f, 0.5f, 0.7f,-0.6f, 1.2f, 0.3f,-1.3f,
        -1.0f, 0.8f, 0.6f,-0.2f, 1.1f, 0.4f,-0.7f, 0.9f, 1.0f,
         0.5f,-1.2f, 0.9f, 1.3f,-0.4f, 0.7f, 0.2f,-1.1f, 0.6f,
    };

    // ---------------- 构造 5 棵树 (pre-order) + 各自常数 (与 test_ad_jac.cu 一致) ----------------
    std::vector<Tree>  trees;
    std::vector<float> cvals;

    // T0: y = c0*x0 + c1.  pre-order: [ADD, MUL(c0,x0), c1]
    {
        Tree T;
        T.nt = {N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST};
        T.nv = {(float)F_ADD, (float)F_MUL, 0.0f, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1, 1};
        T.K = 2; T.name = "c0*x0+c1";
        trees.push_back(T);
        cvals.push_back(2.5f);
        cvals.push_back(-1.7f);
    }
    // T1: y = c0*sin(c1*x0).  pre-order: [MUL, c0, SIN(MUL(c1,x0))]
    {
        Tree T;
        T.nt = {N_BFUNC, N_CONST, N_UFUNC, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)F_MUL, 0.0f, (float)F_SIN, (float)F_MUL, 0.0f, 0.0f};
        T.ci = {-1, 0, -1, -1, 1, -1};
        T.K = 2; T.name = "c0*sin(c1*x0)";
        trees.push_back(T);
        cvals.push_back(1.3f);
        cvals.push_back(0.9f);
    }
    // T2: y = exp(c0*x0).  pre-order: [EXP, MUL(c0,x0)]
    {
        Tree T;
        T.nt = {N_UFUNC, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)F_EXP, (float)F_MUL, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1};
        T.K = 1; T.name = "exp(c0*x0)";
        trees.push_back(T);
        cvals.push_back(0.6f);
    }
    // T3: y = sum_{j=0..8} cj*xj.  K=9 (> AD_W=8) → 跨 chunk 边界。
    //   右倾嵌套: ADD(c0*x0, ADD(c1*x1, ..., ADD(c7*x7, c8*x8)))
    //   pre-order: [ADD, MUL(c0,x0), ADD, MUL(c1,x1), ..., MUL(c7,x7), MUL(c8,x8)]
    {
        Tree T;
        T.name = "sum cj*xj (K9)";
        T.K = 9;
        for (int j = 0; j < 9; j++) {
            if (j < 8) {
                T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_ADD); T.ci.push_back(-1);
            }
            T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_MUL); T.ci.push_back(-1);
            T.nt.push_back(N_CONST); T.nv.push_back(0.0f);         T.ci.push_back(j);     // cj
            T.nt.push_back(N_VAR);   T.nv.push_back((float)j);     T.ci.push_back(-1);    // xj
        }
        trees.push_back(T);
        float c9[9] = {1.1f, -0.7f, 0.5f, 2.0f, -1.3f, 0.8f, 0.4f, -0.9f, 1.6f};
        for (int j = 0; j < 9; j++) cvals.push_back(c9[j]);
    }
    // T4: y = sum_{k=0..15} sin(ck*x0).  K=16 → 恰好两满桶 (col_lo=0 W=8, col_lo=8 W=8)。
    //   右倾嵌套: ADD(sin(c0*x0), ADD(sin(c1*x0), ..., ADD(sin(c14*x0), sin(c15*x0))))
    //   ck 全部 |ck|<=1.05 → |ck*x0|<=1.26 远离 π/2 (见 test_ad_jac.cu 量级论证)。
    {
        Tree T;
        T.name = "sum sin(ck*x0) (K16)";
        T.K = 16;
        for (int k = 0; k < 16; k++) {
            if (k < 15) {
                T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_ADD); T.ci.push_back(-1);
            }
            T.nt.push_back(N_UFUNC); T.nv.push_back((float)F_SIN); T.ci.push_back(-1);
            T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_MUL); T.ci.push_back(-1);
            T.nt.push_back(N_CONST); T.nv.push_back(0.0f);         T.ci.push_back(k);   // ck
            T.nt.push_back(N_VAR);   T.nv.push_back(0.0f);         T.ci.push_back(-1);  // x0
        }
        trees.push_back(T);
        float c16[16] = { 0.30f, -0.45f,  0.60f, -0.75f,  0.90f, -1.05f,  0.15f, -0.20f,
                          0.50f, -0.65f,  0.80f, -0.95f,  0.35f, -0.55f,  0.70f, -0.85f };
        for (int k = 0; k < 16; k++) cvals.push_back(c16[k]);
    }

    const int M_prob = (int)trees.size();

    // ---------------- 扁平化成 batch 数组 ----------------
    std::vector<int>      nt_all, ci_all;
    std::vector<float>    nv_all;
    std::vector<TreeMeta> metas(M_prob);
    int node_off = 0, c_off = 0, K_max = 0;
    for (int m = 0; m < M_prob; m++) {
        const Tree &T = trees[m];
        metas[m].node_offset = node_off;
        metas[m].n_nodes     = (int)T.nt.size();
        metas[m].c_offset    = c_off;
        metas[m].K           = T.K;
        for (size_t k = 0; k < T.nt.size(); k++) {
            nt_all.push_back(T.nt[k]);
            nv_all.push_back(T.nv[k]);
            ci_all.push_back(T.ci[k]);
        }
        node_off += (int)T.nt.size();
        c_off    += T.K;
        if (T.K > K_max) K_max = T.K;
    }
    const int total_nodes = node_off;
    const int total_c     = c_off;
    if (K_max > MAX_K) {
        fprintf(stderr, "K_max=%d > MAX_K=%d\n", K_max, MAX_K); return 2;
    }

    // ---------------- upload ----------------
    int *d_nt, *d_ci; float *d_nv, *d_xs, *d_c, *d_J_rev, *d_J_fwd; TreeMeta *d_metas;
    CK(cudaMalloc(&d_nt,    total_nodes * sizeof(int)));
    CK(cudaMalloc(&d_nv,    total_nodes * sizeof(float)));
    CK(cudaMalloc(&d_ci,    total_nodes * sizeof(int)));
    CK(cudaMalloc(&d_metas, M_prob * sizeof(TreeMeta)));
    CK(cudaMalloc(&d_xs,    (size_t)N * n_vars * sizeof(float)));
    CK(cudaMalloc(&d_c,     total_c * sizeof(float)));
    CK(cudaMalloc(&d_J_rev, (size_t)M_prob * K_max * N * sizeof(float)));
    CK(cudaMalloc(&d_J_fwd, (size_t)M_prob * K_max * N * sizeof(float)));

    CK(cudaMemcpy(d_nt,    nt_all.data(),  total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_nv,    nv_all.data(),  total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci,    ci_all.data(),  total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_metas, metas.data(),   M_prob * sizeof(TreeMeta),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xs,    xs,             (size_t)N * n_vars * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_c,     cvals.data(),   total_c * sizeof(float),     cudaMemcpyHostToDevice));
    // 两个 J 都预填 NaN, 确认 kernel 真写了每个 j<K 槽 (防"读到旧值蒙混")。
    CK(cudaMemset(d_J_rev, 0xFF, (size_t)M_prob * K_max * N * sizeof(float)));
    CK(cudaMemset(d_J_fwd, 0xFF, (size_t)M_prob * K_max * N * sizeof(float)));

    // ---------------- launch (warp-per-tree, 256 thread/block) — SAME args 两个 kernel ----------------
    int block = 256, wpb = block / 32;
    int grid = (M_prob + wpb - 1) / wpb;
    // 被测: reverse-mode Jacobian
    rev_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
                                         d_xs, n_vars, N, d_c, K_max, d_J_rev);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    // 参照: forward-mode Jacobian (同 launch args)
    ad_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
                                        d_xs, n_vars, N, d_c, K_max, d_J_fwd);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> Jr((size_t)M_prob * K_max * N);
    std::vector<float> Jf((size_t)M_prob * K_max * N);
    CK(cudaMemcpy(Jr.data(), d_J_rev, (size_t)M_prob * K_max * N * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Jf.data(), d_J_fwd, (size_t)M_prob * K_max * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- 比对 ----------------
    // J[m,j,i] = J[m*K_max*N + j*N + i]。
    auto JrAt = [&](int m, int j, int i) -> float { return Jr[(size_t)m * K_max * N + (size_t)j * N + i]; };
    auto JfAt = [&](int m, int j, int i) -> float { return Jf[(size_t)m * K_max * N + (size_t)j * N + i]; };
    const float TOL = 1e-3f;
    const float ABS_FLOOR = 1e-5f;
    int g_fail = 0;

    for (int m = 0; m < M_prob; m++) {
        const Tree &T = trees[m];
        const float *c = cvals.data() + metas[m].c_offset;

        // (a) rev vs 手推解析 ------------------------------------------------
        float worst_a = 0.0f; int wa_j = -1, wa_i = -1; float wa_got = 0, wa_want = 0;
        // (b) rev vs forward AD ---------------------------------------------
        float worst_b = 0.0f; int wb_j = -1, wb_i = -1; float wb_got = 0, wb_ref = 0;

        for (int i = 0; i < N; i++) {
            const float *xi = xs + (size_t)i * n_vars;
            // 手推解析 ∂y_i/∂c_j (与 test_ad_jac.cu 的 want[] 逐字一致)。
            float want[MAX_K];
            if (m == 0) {                 // c0*x0 + c1
                want[0] = xi[0];
                want[1] = 1.0f;
            } else if (m == 1) {          // c0*sin(c1*x0)
                float s = sinf(c[1] * xi[0]);
                float cc = cosf(c[1] * xi[0]);
                want[0] = s;
                want[1] = c[0] * cc * xi[0];
            } else if (m == 2) {          // exp(c0*x0)
                want[0] = xi[0] * expf(c[0] * xi[0]);
            } else if (m == 3) {          // sum cj*xj, ∂/∂cj = xj
                for (int j = 0; j < T.K; j++) want[j] = xi[j];
            } else {                      // m==4: sum sin(ck*x0), ∂/∂ck = x0*cos(ck*x0)
                for (int k = 0; k < T.K; k++) want[k] = xi[0] * cosf(c[k] * xi[0]);
            }
            for (int j = 0; j < T.K; j++) {
                float got = JrAt(m, j, i);
                // (a) vs analytic
                {
                    float rel = fabsf(got - want[j]) / (fabsf(want[j]) + ABS_FLOOR);
                    bool bad = !isfinite(got);   // 应写的 j<K 槽若 NaN/Inf → 强制 FAIL
                    if (bad || rel > worst_a) {
                        worst_a = bad ? HUGE_VALF : rel;
                        wa_j = j; wa_i = i; wa_got = got; wa_want = want[j];
                    }
                }
                // (b) vs forward AD (参照值由 forward kernel 给; 比的是两实现一致)
                {
                    float ref = JfAt(m, j, i);
                    float rel = fabsf(got - ref) / (fabsf(ref) + ABS_FLOOR);
                    // 两边都非有限 = 等价/OK; 一边有限一边不 = FAIL; 否则比 rel。
                    bool g_fin = isfinite(got), r_fin = isfinite(ref);
                    bool bad;
                    float userel;
                    if (!g_fin && !r_fin) { bad = false; userel = 0.0f; }       // both non-finite = equal
                    else if (g_fin != r_fin) { bad = true; userel = HUGE_VALF; } // one-finite-one-not = FAIL
                    else { bad = false; userel = rel; }
                    if (bad || userel > worst_b) {
                        worst_b = bad ? HUGE_VALF : userel;
                        wb_j = j; wb_i = i; wb_got = got; wb_ref = ref;
                    }
                }
            }
        }

        bool ok_a = (worst_a < TOL);
        bool ok_b = (worst_b < TOL);
        if (!ok_a || !ok_b) g_fail++;
        printf("  T%d %-20s K=%d  rev_vs_analytic=%.3e %s  rev_vs_fwd=%.3e %s\n",
               m, T.name, T.K,
               worst_a, ok_a ? "PASS" : "FAIL",
               worst_b, ok_b ? "PASS" : "FAIL");
        if (!ok_a)
            printf("      [analytic worst] j=%d i=%d: got=%g want=%g\n", wa_j, wa_i, wa_got, wa_want);
        if (!ok_b)
            printf("      [fwd-parity worst] j=%d i=%d: rev=%g fwd=%g\n", wb_j, wb_i, wb_got, wb_ref);
    }

    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_c); cudaFree(d_J_rev); cudaFree(d_J_fwd);

    printf("\n[test_revad_jac] %d/%d trees PASS\n", M_prob - g_fail, M_prob);
    if (g_fail) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
