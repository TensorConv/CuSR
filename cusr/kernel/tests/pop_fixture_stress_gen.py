"""pop_fixture_stress_gen.py — 边界情况 stress fixture.

3 棵:
- (1) K=0 tree: sin(x0) — 没 CONST, 应 STATUS_K0_SKIP (3)
- (2) K=2 healthy tree: c0 * x0 + c1 — 应 CONVERGED
- (3) K=1 NaN tree: log(x0 - c0) with c0_init too high → log negative → NaN.
  应 STATUS_FAIL (2). 即使 ym 写 0 (反正不可能拟合), 目的是触发 NaN 路径.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POP_MAGIC = 0x4D4C344D
POP_VERSION = 1
N_VAR, N_CONST, N_UFUNC, N_BFUNC = 0, 1, 2, 3
F_ADD = 1; F_SUB = 2; F_MUL = 3
F_SIN = 14; F_LOG = 20


FIXTURES = [
    # (1) K=0 sin(x0)
    ("sin_K0",
     [N_UFUNC, N_VAR],
     [F_SIN,   0.0],
     [-1, -1],
     [],                  # c_true 为空
     lambda x, c: np.sin(x[:, 0])),

    # (2) K=2 healthy linear  c0*x0 + c1
    ("linear_K2",
     [N_BFUNC, N_BFUNC, N_CONST, N_VAR, N_CONST],
     [F_ADD,   F_MUL,   0.0,    0.0,   0.0],
     [-1, -1, 0, -1, 1],
     [1.5, 0.7],
     lambda x, c: c[0] * x[:, 0] + c[1]),

    # (3) K=1 NaN-prone:  log(x0 - c0)  with x0 ∈ [0.5, 1.5], c0_init=1.5 (= 0.9 * 1.7)
    #     ⇒ x0 - c0_init ∈ [-1.0, 0.0] → log(neg) = NaN. LM 初始就 NaN.
    ("log_sub_K1",
     [N_UFUNC, N_BFUNC, N_VAR, N_CONST],
     [F_LOG,   F_SUB,   0.0,   0.0],
     [-1, -1, -1, 0],
     [1.7],   # c_true=1.7 → c_init = 0.9*1.7 = 1.53
     lambda x, c: np.log(x[:, 0] - c[0])),  # 用 c_true 算 y, 没事
]


def main():
    out_path = Path(__file__).resolve().parent / "pop_fixture_stress.bin"
    truth_path = Path(__file__).resolve().parent / "pop_fixture_stress_truth.json"

    N = 64
    n_vars = 1
    M_prob = len(FIXTURES)

    # x0 ∈ [0.5, 1.5] (LOG tree 的 NaN trigger 关键)
    rng = np.random.default_rng(7)
    xs = rng.uniform(0.5, 1.5, (N, n_vars)).astype(np.float32)

    nt_all = []; nv_all = []; ci_all = []; metas = []
    c_init = []; ym_chunks = []
    node_offset = 0; c_offset = 0
    K_max = 0; max_stack = 0
    for name, nt, nv, ci, c_true, y_func in FIXTURES:
        K = len(c_true)
        n_nodes = len(nt)
        nt_all.extend(nt); nv_all.extend(nv); ci_all.extend(ci)
        metas.append((node_offset, n_nodes, c_offset, K))
        c_init.extend([0.9 * v for v in c_true])
        node_offset += n_nodes; c_offset += K
        if K > K_max: K_max = K
        sp = 0; ms = 0
        for i in reversed(range(n_nodes)):
            t = nt[i]
            if t in (N_VAR, N_CONST): sp += 1
            elif t == N_UFUNC: sp = sp - 1 + 1
            elif t == N_BFUNC: sp = sp - 2 + 1
            if sp > ms: ms = sp
        if ms > max_stack: max_stack = ms

        if K > 0:
            y = y_func(xs.astype(np.float64), np.asarray(c_true, dtype=np.float64))
        else:
            y = y_func(xs.astype(np.float64), None)
        ym_chunks.append(y.astype(np.float32))

    nt_arr = np.array(nt_all, dtype=np.int32)
    nv_arr = np.array(nv_all, dtype=np.float32)
    ci_arr = np.array(ci_all, dtype=np.int32)
    metas_arr = np.array(metas, dtype=np.int32).reshape(-1, 4)
    c_arr = np.array(c_init, dtype=np.float32)
    ym_arr = np.concatenate(ym_chunks).astype(np.float32)

    with open(out_path, "wb") as f:
        header = np.zeros(16, dtype=np.int32)
        header[0] = POP_MAGIC; header[1] = POP_VERSION
        header[2] = M_prob; header[3] = node_offset; header[4] = c_offset
        header[5] = N; header[6] = n_vars; header[7] = K_max; header[8] = max_stack
        f.write(header.tobytes())
        f.write(nt_arr.tobytes()); f.write(nv_arr.tobytes()); f.write(ci_arr.tobytes())
        f.write(metas_arr.tobytes()); f.write(c_arr.tobytes())
        f.write(xs.tobytes())
        f.write(ym_arr.tobytes())

    truth = {
        "M_prob": M_prob, "fixtures": [
            {"name": name, "K": len(c_true), "c_offset": metas[i][2],
             "c_true": c_true,
             "expect_status": (3 if len(c_true) == 0 else (2 if name == "log_sub_K1" else 0))}
            for i, (name, _, _, _, c_true, _) in enumerate(FIXTURES)
        ],
    }
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} B), truth {truth_path}")


if __name__ == "__main__":
    main()
