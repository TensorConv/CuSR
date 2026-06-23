# batch-lm-sr — GPU batched LM for SR populations

对 **1000 棵真实 EvoGP 候选树** 跑常数拟合 (`pop.bin`, K ∈ [0, 8], 每棵 N=1000 个数据点).

## 吞吐对比 / Throughput

| 后端 (backend) | 硬件 | 墙钟 (wall time) | 吞吐 (throughput) |
|---|---|---:|---:|
| **batch-lm-sr** (GPU) | NVIDIA GPU | 0.740 s | 1351 trees/s |
| scipy LM (CPU 并行) | 24 workers | 87.5 s | 9.6 trees/s |
| **加速比 (speedup)** | — | — | **~118×** |

**超参数注**: batch-lm-sr 用 xtol=1e-5 / max_iter=50; scipy 用 xtol=1e-10 / max_nfev=200. 这些都是 placeholder, 后续调参会改, 把这组数字看作 baseline, 不是稳态值。

## 状态分布 / Status distribution

| 状态 (status) | batch-lm-sr | scipy |
|---|---:|---:|
| converged (收敛) | 468 | 755 |
| maxiter (跑满迭代) | 253 | 83 |
| fail (NaN/Inf 在 eval) | 7 | 2 |
| fail (Cholesky breakdown, JᵀJ 退化) | 112 | (scipy 不暴露此分类) |
| K=0 skip (无常数可优化) | 160 | 160 |

**联合统计 (排除 K=0)**:

- 两边都收敛: **447**
- 只 batch-lm-sr 收敛: 21
- 只 scipy 收敛: 308
- 两边都没收敛: 64

## 正确性诊断 / Correctness diagnostics

三个独立视角, **都是诊断指标, 不是 acceptance gate**. 超参数 (xtol, max_iter, FD eps, λ) 后续都会调, 这些数字会跟着动。

### (1) loss 是不是真的降了 / Did the loss actually go down?

K>0 的树里, `final_loss < initial_loss(c_init)` 的比例: **93.6%** (780/833).

这是最松的 sanity check — "LM 到底跑没跑". 这一条过, 说明求解器没在空转。

### (2) 跟 scipy 比, 落在 k× envelope 内的比例

两边都收敛 subset 上对比:

| envelope | 落入比例 | count |
|---|---:|---:|
| `gpu_loss ≤ 1.05× sc_loss` (跟 scipy 几乎等价, ≡ "fitness-equivalent") | **89.9%** | 402/447 |
| `gpu_loss ≤ 2× sc_loss` (差距 2 倍以内) | **92.8%** | 415/447 |
| `gpu_loss ≤ 10× sc_loss` (无 blow-up) | **100.0%** | 447/447 |

### (3) 两边都收敛子集的分布诊断

`c_rel_err = ||gpu_c − sc_c|| / ||sc_c||` (常数向量相对误差):

- p50 = 4.589e-03, p90 = 1.015e+00, p99 = 9.860e+01, max = 2.094e+04

`fitness_rel_err = |gpu_loss − sc_loss| / max(gpu_loss, sc_loss, 1e-10)` (loss 对称相对误差):

- p50 = 1.466e-07, p90 = 2.473e-01, p99 = 7.242e-01

**关于大 c_rel_err 但小 fitness_rel_err**: 进化产出的树常有**identifiability 退化** (例如 `c0 + c1 − c1` 里 c1 不影响 loss), 导致 c 向量不唯一但 loss 等价。所以 c_rel_err 在这种树上不公正, fitness_rel_err 才是公平指标。下面 Top divergences 表里能看到这些案例。

## Top 10 c_rel_err 离群案例 / Top divergences (两边都收敛)

| m | K | n_nodes | c_rel | gpu_loss | sc_loss | gpu_c | sc_c | c_init |
|---|---|---|---|---|---|---|---|---|
| 733 | 2 | 7 | 2.094e+04 | 2.663e+04 | 2.663e+04 | 7.103e+02,-5.166e+03 | -3.389e-02,2.468e-01 | -2.000e+00,1.000e+00 |
| 341 | 5 | 32 | 7.765e+02 | 2.742e+04 | 2.742e+04 | 9.427e-01,1.092e+03,-2.822e+02,-1.004e+00,4.372e-01 | 9.424e-01,-1.510e-01,3.909e-02,-1.003e+00,4.371e-01 | 2.000e+00,1.000e+00,2.000e+00,5.000e-01,5.000e-01 |
| 474 | 1 | 16 | 1.217e+02 | 1.691e-10 | 1.528e-10 | 1.014e-06 | -8.400e-09 | 0.000e+00 |
| 756 | 1 | 12 | 9.860e+01 | 2.134e-10 | 1.528e-10 | -6.350e-07 | -6.376e-09 | 1.000e+00 |
| 436 | 4 | 23 | 9.161e+01 | 5.211e+03 | 5.189e+03 | -6.791e+01,4.866e+02,1.668e+00,2.913e-01 | 4.816e+00,-1.046e+00,2.131e+00,3.711e-01 | -1.000e+00,2.000e+00,1.000e+00,2.000e+00 |
| 170 | 1 | 14 | 8.359e+01 | 3.327e-10 | 1.527e-10 | 2.280e-08 | 2.695e-10 | 0.000e+00 |
| 398 | 1 | 13 | 4.023e+01 | 2.517e-10 | 1.527e-10 | -2.503e-07 | -6.071e-09 | 2.000e+00 |
| 525 | 1 | 12 | 3.968e+01 | 4.070e-10 | 1.526e-10 | 9.882e-08 | 2.429e-09 | 1.000e+00 |
| 670 | 1 | 12 | 3.968e+01 | 4.070e-10 | 1.526e-10 | 9.882e-08 | 2.429e-09 | 1.000e+00 |
| 45 | 1 | 8 | 3.905e+01 | 4.375e-10 | 1.526e-10 | 1.059e-06 | 2.643e-08 | 1.000e+00 |
---

_本报告无 acceptance gate, 三个指标并列展示。超参数 (xtol, max_iter, FD eps, λ) 后续会调, 这组数字视为 **baseline 测量**, 不是稳态值。_