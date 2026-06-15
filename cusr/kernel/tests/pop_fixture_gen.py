"""pop_fixture_gen.py — 3 棵经典 NLS archetype 序列化成 pop_fixture.bin.

3 棵 fixture:
- powerlaw_K2:  c0 * x0^c1                     c_true=(2.5, 1.3)
- quadratic_K3: c0 + c1*x0 + c2*x0^2            c_true=(0.5, -1.2, 0.3)
- expsat_K3:    c0 * exp(-c1*x0) + c2          c_true=(3.0, 0.4, 0.7)

x_grid: i ∈ [0,64), x0 = 0.1 + 4.9 * i/63.
c_init: c_true * 0.9 (近真值, 验证算法基线行为, 不测远初值收敛)
y per-tree: 不同 (3 个不同函数). pop_format.h ym layout 是 [M_prob*N], 共 3*64=192 个值.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

# pop_format.h
POP_MAGIC = 0x4D4C344D
POP_VERSION = 1

# Op enum 跟 EvoGP 对齐
N_VAR, N_CONST, N_UFUNC, N_BFUNC, N_TFUNC = 0, 1, 2, 3, 4
F_ADD, F_SUB, F_MUL, F_DIV, F_POW = 1, 2, 3, 4, 6
F_SIN, F_COS, F_LOG, F_EXP, F_NEG, F_SQRT = 14, 15, 20, 22, 25, 27


# tree spec: (name, nt[], nv[], ci[], c_true[], y_func) — y_func 算 ym
FIXTURES = [
    ("powerlaw_K2",
     [N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST],
     [F_MUL,   0.0,    F_POW,   0.0,   0.0   ],
     [-1, 0, -1, -1, 1],
     [2.5, 1.3],
     lambda x, c: c[0] * x**c[1]),

    ("quadratic_K3",
     [N_BFUNC, N_CONST, N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_VAR],
     [F_ADD,   0.0,    F_ADD,   F_MUL,   0.0,    0.0,    F_MUL,   0.0,    F_MUL,   0.0,   0.0  ],
     [-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, -1],
     [0.5, -1.2, 0.3],
     lambda x, c: c[0] + c[1]*x + c[2]*x*x),

    ("expsat_K3",
     [N_BFUNC, N_BFUNC, N_CONST, N_UFUNC, N_UFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST],
     [F_ADD,   F_MUL,   0.0,    F_EXP,   F_NEG,   F_MUL,   0.0,    0.0,   0.0   ],
     [-1, -1, 0, -1, -1, -1, 1, -1, 2],
     [3.0, 0.4, 0.7],
     lambda x, c: c[0] * np.exp(-c[1] * x) + c[2]),
]


def main():
    out_path = Path(__file__).resolve().parent / "pop_fixture.bin"
    truth_path = Path(__file__).resolve().parent / "pop_fixture_truth.json"

    N = 64
    n_vars = 1
    M_prob = len(FIXTURES)

    # x grid: N=64 均匀 [0.1, 5.0]
    xs = np.array([0.1 + 4.9 * i / (N - 1) for i in range(N)], dtype=np.float32)

    nt_all = []
    nv_all = []
    ci_all = []
    metas = []
    c_init = []
    ym = []
    node_offset = 0
    c_offset = 0
    c_trues = []
    K_max = 0
    max_stack = 0
    for name, nt, nv, ci, c_true, y_func in FIXTURES:
        K = len(c_true)
        n_nodes = len(nt)
        nt_all.extend(nt)
        nv_all.extend(nv)
        ci_all.extend(ci)
        metas.append((node_offset, n_nodes, c_offset, K))
        c_init.extend([0.9 * v for v in c_true])  # c_init = 0.9 * c_true (近真值)
        c_trues.extend(c_true)
        node_offset += n_nodes
        c_offset += K
        if K > K_max:
            K_max = K
        # 模拟 stack
        sp = 0; ms = 0
        for i in reversed(range(n_nodes)):
            t = nt[i]
            if t in (N_VAR, N_CONST): sp += 1
            elif t == N_UFUNC: sp = sp - 1 + 1
            elif t == N_BFUNC: sp = sp - 2 + 1
            else: raise ValueError(f"bad type {t}")
            if sp > ms: ms = sp
        if ms > max_stack: max_stack = ms

        # 算 ym (per-tree, 64 个)
        y = y_func(xs.astype(np.float64), np.asarray(c_true, dtype=np.float64))
        ym.append(y.astype(np.float32))

    total_nodes = node_offset
    total_c = c_offset

    nt_arr = np.array(nt_all, dtype=np.int32)
    nv_arr = np.array(nv_all, dtype=np.float32)
    ci_arr = np.array(ci_all, dtype=np.int32)
    metas_arr = np.array(metas, dtype=np.int32).reshape(-1, 4)
    c_arr = np.array(c_init, dtype=np.float32)
    xs_arr = xs.astype(np.float32)  # 单变量, layout [N,1]
    ym_arr = np.concatenate(ym).astype(np.float32)  # [M_prob*N]

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
        f.write(xs_arr.tobytes())
        f.write(ym_arr.tobytes())

    # truth sidecar: 给 test_fixture.py 用
    import json
    truth = {
        "M_prob": M_prob, "total_c": total_c, "K_max": K_max, "max_stack": max_stack,
        "fixtures": [
            {"name": name, "K": len(c_true), "c_offset": metas[i][2], "c_true": c_true}
            for i, (name, _, _, _, c_true, _) in enumerate(FIXTURES)
        ],
    }
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"wrote {out_path}  ({out_path.stat().st_size} B), truth {truth_path}")


if __name__ == "__main__":
    main()
