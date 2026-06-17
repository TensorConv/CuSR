// test_ad_jac.cu — Rung 2 GPU 单测: ad_jacobian_kernel vs 手推解析 Jacobian。
//
// 见 docs/kernel/AD_JACOBIAN_V4.md §6 Rung 2。**只** include ad_interp.cuh
// (完全不碰 batch_lm_ad.cu — 它仍是 FD 拷贝, 独立编译单元, 枚举各一份, 无冲突)。
//
// 直接构造 token 数组 (nt/nv/ci/metas/xs/c) — 不需要 .bin loader。把若干棵树
// 打成一个 batch (共享 n_vars / xs), 一次 launch ad_jacobian_kernel, 拷回 d_J,
// 对每个 (tree, j, i) 槽比对 rel err < 1e-3 (abs floor 1e-5)。
//
// 覆盖树 (硬推解析 J):
//   T0  y = c0*x0 + c1            K=2   ∂/∂c0 = x0,           ∂/∂c1 = 1
//   T1  y = c0*sin(c1*x0)         K=2   ∂/∂c0 = sin(c1*x0),   ∂/∂c1 = c0*cos(c1*x0)*x0
//   T2  y = exp(c0*x0)            K=1   ∂/∂c0 = x0*exp(c0*x0)
//   T3  y = sum_{j=0..8} cj*xj    K=9   ∂/∂cj = xj            (K>AD_W → 跨 chunk 边界)
//
// d_J 布局必须与 fd_jacobian 一致: Jm = J + m*K_max*N; Jm[j*N + i]; j<K 才写。
//
// build/gate:
//   source scripts/env.sh && cd cusr/kernel && \
//   nvcc -O2 -arch=sm_80 -std=c++17 --use_fast_math -o tests/test_ad_jac tests/test_ad_jac.cu && \
//   CUDA_VISIBLE_DEVICES=0 ./tests/test_ad_jac ; echo EXIT=$?

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "../ad_interp.cuh"

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
    printf("[test_ad_jac] ad_jacobian_kernel vs analytic Jacobian (tol rel < 1e-3)\n");

    // ---------------- 数据点 (N 个), n_vars 个变量 ----------------
    // T0..T2 只用 x0; T3 用 x0..x8。一个共享 xs (N*n_vars), 多余列 T0..T2 忽略。
    const int n_vars = 9;
    const int N = 6;
    // 每个点的 9 个变量值 (非平凡, 含负值避免对称巧合)。
    float xs[N * 9] = {
        // x0    x1    x2    x3    x4    x5    x6    x7    x8
         0.7f, 1.0f,-0.5f, 0.3f, 2.0f,-1.2f, 0.9f, 1.5f,-0.8f,
        -0.4f, 0.6f, 1.1f,-1.0f, 0.2f, 0.8f,-0.3f, 1.3f, 0.5f,
         1.2f,-0.7f, 0.4f, 0.9f,-1.5f, 1.0f, 0.6f,-0.9f, 0.7f,
         0.1f, 1.4f,-1.1f, 0.5f, 0.7f,-0.6f, 1.2f, 0.3f,-1.3f,
        -1.0f, 0.8f, 0.6f,-0.2f, 1.1f, 0.4f,-0.7f, 0.9f, 1.0f,
         0.5f,-1.2f, 0.9f, 1.3f,-0.4f, 0.7f, 0.2f,-1.1f, 0.6f,
    };

    // ---------------- 构造 4 棵树 (pre-order) + 各自常数 ----------------
    std::vector<Tree>  trees;
    std::vector<float> cvals;   // 各树常数顺序拼接 (c_offset 取自这里)

    // T0: y = c0*x0 + c1.  pre-order: [ADD, MUL(c0,x0), c1]
    {
        Tree T;
        T.nt = {N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST};
        T.nv = {(float)F_ADD, (float)F_MUL, 0.0f, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1, 1};   // 局部 const idx: c0=0, c1=1
        T.K = 2; T.name = "c0*x0+c1";
        trees.push_back(T);
        cvals.push_back(2.5f);   // c0
        cvals.push_back(-1.7f);  // c1
    }
    // T1: y = c0*sin(c1*x0).  pre-order: [MUL, c0, SIN(MUL(c1,x0))]
    {
        Tree T;
        T.nt = {N_BFUNC, N_CONST, N_UFUNC, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)F_MUL, 0.0f, (float)F_SIN, (float)F_MUL, 0.0f, 0.0f};
        T.ci = {-1, 0, -1, -1, 1, -1};   // c0=0, c1=1
        T.K = 2; T.name = "c0*sin(c1*x0)";
        trees.push_back(T);
        cvals.push_back(1.3f);   // c0
        cvals.push_back(0.9f);   // c1
    }
    // T2: y = exp(c0*x0).  pre-order: [EXP, MUL(c0,x0)]
    {
        Tree T;
        T.nt = {N_UFUNC, N_BFUNC, N_CONST, N_VAR};
        T.nv = {(float)F_EXP, (float)F_MUL, 0.0f, 0.0f};
        T.ci = {-1, -1, 0, -1};   // c0=0
        T.K = 1; T.name = "exp(c0*x0)";
        trees.push_back(T);
        cvals.push_back(0.6f);   // c0
    }
    // T3: y = sum_{j=0..8} cj*xj.  K=9 (> AD_W=8) → 跨 chunk 边界。
    //   左倾累加: ADD(ADD(...ADD(c0*x0, c1*x1)..., c7*x7), c8*x8)。
    //   pre-order: 8 个 ADD 在前, 然后 9 个 (MUL cj xj) 项 —— 但需保证弹栈顺序得到
    //   每项乘积。栈机 reverse 遍历 = 把数组当 pre-order 后序求值, 每个 ADD 取其
    //   两棵子树。下面按 "向左展开" 写: token = [ADD x8] 后跟项 0..8 的 MUL 子树,
    //   但这样结构其实是右倾。为正确性只需 J 与符号无关 (求和可交换), ∂/∂cj = xj
    //   对任何结合方式都成立。这里用最直白的右倾嵌套:
    //     ADD(c0*x0, ADD(c1*x1, ADD(..., ADD(c7*x7, c8*x8))))
    //   pre-order: [ADD, MUL(c0,x0), ADD, MUL(c1,x1), ADD, ..., MUL(c7,x7), MUL(c8,x8)]
    {
        Tree T;
        T.name = "sum cj*xj (K9)";
        T.K = 9;
        for (int j = 0; j < 9; j++) {
            if (j < 8) {                       // 前 8 项前面带一个 ADD
                T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_ADD); T.ci.push_back(-1);
            }
            // MUL(cj, xj)
            T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_MUL); T.ci.push_back(-1);
            T.nt.push_back(N_CONST); T.nv.push_back(0.0f);         T.ci.push_back(j);     // cj (局部 idx j)
            T.nt.push_back(N_VAR);   T.nv.push_back((float)j);     T.ci.push_back(-1);    // xj
        }
        trees.push_back(T);
        // 非零常数 (各不相同)。
        float c9[9] = {1.1f, -0.7f, 0.5f, 2.0f, -1.3f, 0.8f, 0.4f, -0.9f, 1.6f};
        for (int j = 0; j < 9; j++) cvals.push_back(c9[j]);
    }
    // T4: y = sum_{k=0..15} sin(ck*x0).  K=16 → 恰好两满桶 (col_lo=0 W=8, col_lo=8 W=8)。
    //   与 T3 不同: 每列 NONLINEAR (∂/∂ck = x0*cos(ck*x0), 依赖 ck), 真正卡跨 chunk 的
    //   非线性切向量数学 (T3 纯线性 ∂/∂cj=xj 与 ck 无关, 掩盖了非线性 chunk 行为)。
    //   右倾嵌套: ADD(sin(c0*x0), ADD(sin(c1*x0), ..., ADD(sin(c14*x0), sin(c15*x0)))).
    //   pre-order per term: [SIN, MUL(ck,x0), ...]; 16 项前置 15 个 ADD。共用 x0 (n_vars 不变)。
    //   ck 选值: 所有 |ck*x0| 远离 π/2≈1.571 (x0∈[-1.0,1.2] 跨 6 点), 使 cos 不过零、
    //   want=x0*cos 量级 > ~1e-2, fast-math 误差被 1e-3 容差吸收。|ck|<=1.05 → 最坏
    //   |ck*x0|<=1.26, cos>=0.305。各 ck 不同 → 每列切向量数值不同。
    {
        Tree T;
        T.name = "sum sin(ck*x0) (K16)";
        T.K = 16;
        for (int k = 0; k < 16; k++) {
            if (k < 15) {                      // 前 15 项前面带一个 ADD
                T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_ADD); T.ci.push_back(-1);
            }
            // SIN(MUL(ck, x0))
            T.nt.push_back(N_UFUNC); T.nv.push_back((float)F_SIN); T.ci.push_back(-1);
            T.nt.push_back(N_BFUNC); T.nv.push_back((float)F_MUL); T.ci.push_back(-1);
            T.nt.push_back(N_CONST); T.nv.push_back(0.0f);         T.ci.push_back(k);   // ck (局部 idx k)
            T.nt.push_back(N_VAR);   T.nv.push_back(0.0f);         T.ci.push_back(-1);  // x0 (共用)
        }
        trees.push_back(T);
        // 16 个互异常数, 全部 |ck|<=1.05 (见上方量级论证)。
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
    int *d_nt, *d_ci; float *d_nv, *d_xs, *d_c, *d_J; TreeMeta *d_metas;
    CK(cudaMalloc(&d_nt,    total_nodes * sizeof(int)));
    CK(cudaMalloc(&d_nv,    total_nodes * sizeof(float)));
    CK(cudaMalloc(&d_ci,    total_nodes * sizeof(int)));
    CK(cudaMalloc(&d_metas, M_prob * sizeof(TreeMeta)));
    CK(cudaMalloc(&d_xs,    (size_t)N * n_vars * sizeof(float)));
    CK(cudaMalloc(&d_c,     total_c * sizeof(float)));
    CK(cudaMalloc(&d_J,     (size_t)M_prob * K_max * N * sizeof(float)));

    CK(cudaMemcpy(d_nt,    nt_all.data(),  total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_nv,    nv_all.data(),  total_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ci,    ci_all.data(),  total_nodes * sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_metas, metas.data(),   M_prob * sizeof(TreeMeta),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xs,    xs,             (size_t)N * n_vars * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_c,     cvals.data(),   total_c * sizeof(float),     cudaMemcpyHostToDevice));
    // J 预填 NaN, 确认 kernel 真写了每个 j<K 槽 (未写的列我们不检, 但有意防"读到旧值蒙混")。
    CK(cudaMemset(d_J, 0xFF, (size_t)M_prob * K_max * N * sizeof(float)));

    // ---------------- launch (warp-per-tree, 256 thread/block) ----------------
    int block = 256, wpb = block / 32;
    int grid = (M_prob + wpb - 1) / wpb;
    ad_jacobian_kernel<<<grid, block>>>(d_nt, d_nv, d_ci, d_metas, M_prob,
                                        d_xs, n_vars, N, d_c, K_max, d_J);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> J((size_t)M_prob * K_max * N);
    CK(cudaMemcpy(J.data(), d_J, (size_t)M_prob * K_max * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- 手推解析 J, 比对 ----------------
    // 取出 J[m,j,i] = J[m*K_max*N + j*N + i]。rel err = |got-want|/(|want|+1e-5)。
    auto Jat = [&](int m, int j, int i) -> float {
        return J[(size_t)m * K_max * N + (size_t)j * N + i];
    };
    const float TOL = 1e-3f;
    const float ABS_FLOOR = 1e-5f;
    int g_fail = 0;

    for (int m = 0; m < M_prob; m++) {
        const Tree &T = trees[m];
        const float *c = cvals.data() + metas[m].c_offset;
        float worst = 0.0f;
        int worst_j = -1, worst_i = -1;
        float worst_got = 0, worst_want = 0;

        for (int i = 0; i < N; i++) {
            const float *xi = xs + (size_t)i * n_vars;   // 该点的 9 个变量
            // 按树名给出解析 ∂y_i/∂c_j。
            float want[MAX_K];
            if (m == 0) {                 // c0*x0 + c1
                want[0] = xi[0];          // ∂/∂c0
                want[1] = 1.0f;           // ∂/∂c1
            } else if (m == 1) {          // c0*sin(c1*x0)
                float s = sinf(c[1] * xi[0]);
                float cc = cosf(c[1] * xi[0]);
                want[0] = s;                       // ∂/∂c0
                want[1] = c[0] * cc * xi[0];       // ∂/∂c1
            } else if (m == 2) {          // exp(c0*x0)
                want[0] = xi[0] * expf(c[0] * xi[0]);   // ∂/∂c0
            } else if (m == 3) {          // sum cj*xj, ∂/∂cj = xj (线性, 与 ck 无关)
                for (int j = 0; j < T.K; j++) want[j] = xi[j];
            } else {                      // m==4: sum sin(ck*x0), ∂/∂ck = x0*cos(ck*x0)
                for (int k = 0; k < T.K; k++) want[k] = xi[0] * cosf(c[k] * xi[0]);
            }
            for (int j = 0; j < T.K; j++) {
                float got = Jat(m, j, i);
                float rel = fabsf(got - want[j]) / (fabsf(want[j]) + ABS_FLOOR);
                // NaN/Inf 守卫: rel 为 NaN 时 "rel > worst" 恒 false 会让坏槽蒙混过关,
                // 强制判 FAIL (HUGE_VALF), 守住 "any failure → exit nonzero"。
                bool bad = !isfinite(got);
                if (bad || rel > worst) {
                    worst = bad ? HUGE_VALF : rel;
                    worst_j = j; worst_i = i; worst_got = got; worst_want = want[j];
                }
            }
        }
        bool ok = (worst < TOL);
        if (!ok) g_fail++;
        printf("  T%d %-18s K=%d  max_rel_err=%.3e  %s",
               m, T.name, T.K, worst, ok ? "PASS" : "FAIL");
        if (!ok)
            printf("  (worst at j=%d i=%d: got=%g want=%g)", worst_j, worst_i, worst_got, worst_want);
        printf("\n");
    }

    cudaFree(d_nt); cudaFree(d_nv); cudaFree(d_ci); cudaFree(d_metas);
    cudaFree(d_xs); cudaFree(d_c); cudaFree(d_J);

    printf("\n[test_ad_jac] %d/%d trees PASS\n", M_prob - g_fail, M_prob);
    if (g_fail) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
