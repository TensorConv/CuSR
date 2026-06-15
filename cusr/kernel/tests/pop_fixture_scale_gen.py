"""pop_fixture_scale_gen.py — 常数量级扫描 fixture (W0: 相对 eps_fd 修复的验收测试).

动机: batch_lm.cu 的 FD 扰动是绝对值 eps_fd=1e-3. 常数量级远离 1 时:
- |c| >> 1: 相对扰动过小, fp32 下 f(c+h)-f(c) 被舍入噪声淹没 → J 是垃圾;
- |c| << 1: h 是 c 的几十倍, 割线斜率完全不在切线附近 → J 错误.
SRSD realistic-range 有 1e-11 量级常数, 这个 bug 不修该类实验报废.

设计: 同一条曲线按尺度 s 重参数化 (c1 → c1*s, x → x/s), 数学上与 s=1 完全
同构 —— 唯一随 s 变的是 "常数量级 vs FD 扰动" 的相互作用. 注意 pop.bin 的
xs 是全 population 共享的, 所以每个尺度写一个独立的 bin 文件.

两档树:
- tier1 (单常数 K=1, 硬断言): exp_pure  y=exp(-(c1*x)),  c1=0.4*s
                              sin_pure  y=sin(c1*x),     c1=2.0*s
  只有一列 Jacobian, 无条件数混淆 —— 纯测 FD eps. 修复后必须全 PASS.
- tier2 (混合量级 K=3, 诊断): expsat    y=c0*exp(-c1*x)+c2, c=(3.0, 0.4*s, 0.7)
  c1 与 c0/c2 量级悬殊 → JtJ 条件数 ~ s^±2, fp32 Cholesky 可能挂.
  这是 eps 之外的第二个质量问题 (列缩放), 先观测不断言.

附: zero_init 守卫树 (s=1 文件里): quadratic c_init=(0,0,0), 测 c==0 时
相对 eps 的 fallback 路径不炸 (线性参数, FD 对任意 h 精确, 修复前后都该过).

scales: 1e-6, 1e-3, 1, 1e3, 1e6
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# pop_format.h
POP_MAGIC = 0x4D4C344D
POP_VERSION = 1

N_VAR, N_CONST, N_UFUNC, N_BFUNC, N_TFUNC = 0, 1, 2, 3, 4
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_SIN, F_COS, F_LOG, F_EXP, F_NEG, F_SQRT = 14, 15, 20, 22, 25, 27

SCALES = [1e-6, 1e-3, 1.0, 1e3, 1e6]
N = 64
HERE = Path(__file__).resolve().parent


def make_trees(s: float) -> list:
    """每个尺度的树清单: (name, tier, nt, nv, ci, c_true, c_init, y_func)."""
    trees = [
        # tier1: K=1, 纯 FD-eps 测试
        ("exp_pure", 1,
         [N_UFUNC, N_UFUNC, N_BFUNC, N_CONST, N_VAR],
         [F_EXP,   F_NEG,   F_MUL,   0.0,    0.0  ],
         [-1, -1, -1, 0, -1],
         [0.4 * s], None,
         lambda x, c: np.exp(-(c[0] * x))),

        ("sin_pure", 1,
         [N_UFUNC, N_BFUNC, N_CONST, N_VAR],
         [F_SIN,   F_MUL,   0.0,    0.0  ],
         [-1, -1, 0, -1],
         [2.0 * s], None,
         lambda x, c: np.sin(c[0] * x)),

        # tier2: 混合量级, 诊断 (eps × 条件数纠缠)
        ("expsat_mixed", 2,
         [N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST],
         [F_ADD,   F_MUL,   0.0,    F_EXP,   F_NEG,   F_MUL,   0.0,    0.0,   0.0   ],
         [-1, -1, 0, -1, -1, -1, 1, -1, 2],
         [3.0, 0.4 * s, 0.7], None,
         lambda x, c: c[0] * np.exp(-(c[1] * x)) + c[2]),
    ]
    if s == 1.0:
        # zero_init 守卫: c_init 全 0, 测相对 eps 的 c==0 fallback. 线性参数.
        trees.append(
            ("zero_init_quad", 1,
             [N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_VAR],
             [F_ADD,   0.0,    F_ADD,   F_MUL,   0.0,    0.0,    F_MUL,   0.0,    F_MUL,   0.0,   0.0  ],
             [-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1],
             [0.5, -1.2, 0.3], [0.0, 0.0, 0.0],
             lambda x, c: c[0] + c[1] * x + c[2] * x * x))
    return trees


def write_pop(s: float, out_path: Path) -> list[dict]:
    trees = make_trees(s)
    # x grid 随尺度反向缩放: [0.1, 5.0] / s
    xs = np.array([(0.1 + 4.9 * i / (N - 1)) / s for i in range(N)], dtype=np.float32)

    nt_all, nv_all, ci_all, metas, c_init_all, ym, fixtures = [], [], [], [], [], [], []
    node_offset = c_offset = 0
    K_max = max_stack = 0
    for name, tier, nt, nv, ci, c_true, c_init, y_func in trees:
        K = len(c_true)
        n_nodes = len(nt)
        nt_all.extend(nt); nv_all.extend(nv); ci_all.extend(ci)
        metas.append((node_offset, n_nodes, c_offset, K))
        ci_vals = c_init if c_init is not None else [0.9 * v for v in c_true]
        c_init_all.extend(ci_vals)
        fixtures.append({"name": name, "tier": tier, "K": K,
                         "c_offset": c_offset, "c_true": list(c_true)})
        node_offset += n_nodes; c_offset += K
        K_max = max(K_max, K)
        sp = ms = 0
        for i in reversed(range(n_nodes)):
            t = nt[i]
            if t in (N_VAR, N_CONST): sp += 1
            elif t == N_UFUNC: sp = sp - 1 + 1
            elif t == N_BFUNC: sp = sp - 2 + 1
            else: raise ValueError(f"bad type {t}")
            ms = max(ms, sp)
        max_stack = max(max_stack, ms)
        y = y_func(xs.astype(np.float64), np.asarray(c_true, dtype=np.float64))
        ym.append(y.astype(np.float32))

    with open(out_path, "wb") as f:
        header = np.zeros(16, dtype=np.int32)
        header[0] = POP_MAGIC; header[1] = POP_VERSION
        header[2] = len(trees); header[3] = node_offset; header[4] = c_offset
        header[5] = N; header[6] = 1  # n_vars
        header[7] = K_max; header[8] = max_stack
        f.write(header.tobytes())
        f.write(np.array(nt_all, dtype=np.int32).tobytes())
        f.write(np.array(nv_all, dtype=np.float32).tobytes())
        f.write(np.array(ci_all, dtype=np.int32).tobytes())
        f.write(np.array(metas, dtype=np.int32).reshape(-1, 4).tobytes())
        f.write(np.array(c_init_all, dtype=np.float32).tobytes())
        f.write(xs.tobytes())
        f.write(np.concatenate(ym).astype(np.float32).tobytes())
    return fixtures


def main():
    manifest = []
    for s in SCALES:
        tag = f"{s:.0e}".replace("+0", "").replace("-0", "-")  # 1e-06→1e-6, 1e+03→1e3
        bin_path = HERE / f"pop_fixture_scale_{tag}.bin"
        fixtures = write_pop(s, bin_path)
        manifest.append({"scale": s, "bin": bin_path.name, "fixtures": fixtures})
        print(f"wrote {bin_path.name} ({bin_path.stat().st_size} B, {len(fixtures)} trees)")
    truth_path = HERE / "pop_fixture_scale_truth.json"
    truth_path.write_text(json.dumps({"files": manifest}, indent=2))
    print(f"wrote {truth_path.name}")


if __name__ == "__main__":
    main()
