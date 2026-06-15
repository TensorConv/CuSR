"""pop_fixture_op_gen.py — op-coverage fixture: TAN/INV/SINH/COSH/TANH/ABS + 多变量.

每棵树验一个新 op (或多变量 layout). xs 是 [N, 3] — 3 vars 共享, 但每棵树自己用哪几个.
ym 是 per-tree 算出来的.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POP_MAGIC = 0x4D4C344D
POP_VERSION = 1

N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_MAX, F_MIN, F_LT, F_GT, F_LE, F_GE = 8, 9, 10, 11, 12, 13
F_SIN, F_COS, F_TAN = 14, 15, 16
F_SINH, F_COSH, F_TANH = 17, 18, 19
F_LOG, F_EXP, F_INV = 20, 22, 23
F_NEG, F_ABS, F_SQRT = 25, 26, 27


# 每棵树:
# - nt, nv, ci, c_true
# - y_func(x_vec, c) where x_vec is shape (N, n_vars_total=3); 选用哪些列由 tree 决定
FIXTURES = [
    # (1) TAN: c0 * tan(x0)        小 x 避奇点
    ("tan_K1",
     [N_BFUNC, N_CONST, N_UFUNC, N_VAR],
     [F_MUL,   0.0,    F_TAN,   0.0],
     [-1, 0, -1, -1],
     [1.5],
     lambda x, c: c[0] * np.tan(x[:, 0])),

    # (2) INV: c0 * inv(x0)  (= c0/x0)
    ("inv_K1",
     [N_BFUNC, N_CONST, N_UFUNC, N_VAR],
     [F_MUL,   0.0,    F_INV,   0.0],
     [-1, 0, -1, -1],
     [2.5],
     lambda x, c: c[0] / x[:, 0]),

    # (3) SINH: c0 * sinh(c1*x0)
    ("sinh_K2",
     [N_BFUNC, N_CONST, N_UFUNC, N_BFUNC, N_CONST, N_VAR],
     [F_MUL,   0.0,    F_SINH,  F_MUL,   0.0,    0.0],
     [-1, 0, -1, -1, 1, -1],
     [1.2, 0.7],
     lambda x, c: c[0] * np.sinh(c[1] * x[:, 0])),

    # (4) COSH: c0 + c1*cosh(x0)
    ("cosh_K2",
     [N_BFUNC, N_CONST, N_BFUNC, N_CONST, N_UFUNC, N_VAR],
     [F_ADD,   0.0,    F_MUL,   0.0,    F_COSH,  0.0],
     [-1, 0, -1, 1, -1, -1],
     [0.3, 0.8],
     lambda x, c: c[0] + c[1] * np.cosh(x[:, 0])),

    # (5) TANH: c0 * tanh(c1*x0)
    ("tanh_K2",
     [N_BFUNC, N_CONST, N_UFUNC, N_BFUNC, N_CONST, N_VAR],
     [F_MUL,   0.0,    F_TANH,  F_MUL,   0.0,    0.0],
     [-1, 0, -1, -1, 1, -1],
     [2.0, 1.5],
     lambda x, c: c[0] * np.tanh(c[1] * x[:, 0])),

    # (6) ABS: c0 * abs(x0) + c1  (abs at x>0 is smooth, no kink issues)
    ("abs_K2",
     [N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_VAR, N_CONST],
     [F_ADD,   F_MUL,   0.0,    F_ABS,   0.0,   0.0],
     [-1, -1, 0, -1, -1, 1],
     [1.7, 0.4],
     lambda x, c: c[0] * np.abs(x[:, 0]) + c[1]),

    # (7) 2-var: c0*x0 + c1*x1
    ("twovar_K2",
     [N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_BFUNC, N_CONST, N_VAR],
     [F_ADD,   F_MUL,   0.0,    0.0,   F_MUL,   0.0,    1.0],
     [-1, -1, 0, -1, -1, 1, -1],
     [2.0, -1.3],
     lambda x, c: c[0] * x[:, 0] + c[1] * x[:, 1]),

    # (8) 3-var: c0 * x0 * x1 * sin(x2)  (这就是 feynman/I.18.12 单常数版本)
    ("threevar_I1812",
     [N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_BFUNC, N_VAR, N_UFUNC, N_VAR],
     [F_MUL,   0.0,    F_MUL,   0.0,   F_MUL,   1.0,   F_SIN,   2.0],
     [-1, 0, -1, -1, -1, -1, -1, -1],
     [1.7],
     lambda x, c: c[0] * x[:, 0] * x[:, 1] * np.sin(x[:, 2])),
]


def main():
    out_path = Path(__file__).resolve().parent / "pop_fixture_op.bin"
    truth_path = Path(__file__).resolve().parent / "pop_fixture_op_truth.json"

    N = 128  # 多变量 + 非线性, 多一点 sample
    n_vars = 3
    M_prob = len(FIXTURES)

    rng = np.random.default_rng(42)
    # x0 ∈ [0.5, 1.2] (TAN/INV/SINH 安全), x1 ∈ [-1.5, 1.5], x2 ∈ [-π/2, π/2]
    xs = np.zeros((N, n_vars), dtype=np.float32)
    xs[:, 0] = rng.uniform(0.5, 1.2, N)
    xs[:, 1] = rng.uniform(-1.5, 1.5, N)
    xs[:, 2] = rng.uniform(-np.pi/2, np.pi/2, N)

    nt_all = []
    nv_all = []
    ci_all = []
    metas = []
    c_init = []
    ym_chunks = []
    node_offset = 0; c_offset = 0
    K_max = 0; max_stack = 0
    for name, nt, nv, ci, c_true, y_func in FIXTURES:
        K = len(c_true)
        n_nodes = len(nt)
        nt_all.extend(nt)
        nv_all.extend(nv)
        ci_all.extend(ci)
        metas.append((node_offset, n_nodes, c_offset, K))
        c_init.extend([0.9 * v for v in c_true])
        node_offset += n_nodes
        c_offset += K
        if K > K_max: K_max = K
        # stack 模拟
        sp = 0; ms = 0
        for i in reversed(range(n_nodes)):
            t = nt[i]
            if t in (N_VAR, N_CONST): sp += 1
            elif t == N_UFUNC: sp = sp - 1 + 1
            elif t == N_BFUNC: sp = sp - 2 + 1
            else: raise ValueError(f"bad type {t}")
            if sp > ms: ms = sp
        if ms > max_stack: max_stack = ms

        y = y_func(xs.astype(np.float64), np.asarray(c_true, dtype=np.float64))
        ym_chunks.append(y.astype(np.float32))

    total_nodes = node_offset
    total_c = c_offset

    nt_arr = np.array(nt_all, dtype=np.int32)
    nv_arr = np.array(nv_all, dtype=np.float32)
    ci_arr = np.array(ci_all, dtype=np.int32)
    metas_arr = np.array(metas, dtype=np.int32).reshape(-1, 4)
    c_arr = np.array(c_init, dtype=np.float32)
    ym_arr = np.concatenate(ym_chunks).astype(np.float32)

    with open(out_path, "wb") as f:
        header = np.zeros(16, dtype=np.int32)
        header[0] = POP_MAGIC
        header[1] = POP_VERSION
        header[2] = M_prob
        header[3] = total_nodes
        header[4] = total_c
        header[5] = N
        header[6] = n_vars
        header[7] = K_max
        header[8] = max_stack
        f.write(header.tobytes())
        f.write(nt_arr.tobytes())
        f.write(nv_arr.tobytes())
        f.write(ci_arr.tobytes())
        f.write(metas_arr.tobytes())
        f.write(c_arr.tobytes())
        f.write(xs.tobytes())
        f.write(ym_arr.tobytes())

    truth = {
        "M_prob": M_prob, "total_c": total_c, "K_max": K_max, "max_stack": max_stack,
        "n_vars": n_vars, "N": N,
        "fixtures": [
            {"name": name, "K": len(c_true), "c_offset": metas[i][2], "c_true": c_true}
            for i, (name, _, _, _, c_true, _) in enumerate(FIXTURES)
        ],
    }
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} B), truth {truth_path}")


if __name__ == "__main__":
    main()
