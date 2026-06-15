"""popio.py — pop.bin (008 格式) 读 / 写 / 构造 / 切片.

格式 (pop_format.h, version 1, 全 little-endian):
  header 16×int32: magic, version, M, total_nodes, total_c, N, n_vars, K_max, max_stack, 0...
  nt    int32[total_nodes]    节点类型 (0=VAR 1=CONST 2=UFUNC 3=BFUNC)
  nv    float32[total_nodes]  VAR→变量下标, FUNC→op 枚举, CONST→0
  ci    int32[total_nodes]    CONST 节点→树内常数局部 rank, 其余 -1
  metas int32[M,4]            (node_offset, n_nodes, c_offset, K)
  c_init float32[total_c]
  xs    float32[N, n_vars]    全 pop 共享
  ym    float32[M, N]         每树目标
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

POP_MAGIC = 0x4D4C344D
POP_VERSION = 1

NTYPE_VAR, NTYPE_CONST, NTYPE_UFUNC, NTYPE_BFUNC = 0, 1, 2, 3


def load_pop_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        header = struct.unpack("<16i", f.read(64))
        magic, version, M, total_n, total_c, N, n_vars, K_max, max_stack = header[0:9]
        if magic != POP_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x} (want 0x{POP_MAGIC:08x})")
        nt = np.frombuffer(f.read(total_n * 4), dtype=np.int32).copy()
        nv = np.frombuffer(f.read(total_n * 4), dtype=np.float32).copy()
        ci = np.frombuffer(f.read(total_n * 4), dtype=np.int32).copy()
        metas = np.frombuffer(f.read(M * 16), dtype=np.int32).reshape(M, 4).copy()
        c_init = np.frombuffer(f.read(total_c * 4), dtype=np.float32).copy()
        xs = np.frombuffer(f.read(N * n_vars * 4), dtype=np.float32).reshape(N, n_vars).copy()
        ym = np.frombuffer(f.read(M * N * 4), dtype=np.float32).reshape(M, N).copy()
    return dict(M=M, total_nodes=total_n, total_c=total_c, N=N, n_vars=n_vars,
                K_max=K_max, max_stack=max_stack,
                nt=nt, nv=nv, ci=ci, metas=metas, c_init=c_init, xs=xs, ym=ym)


def save_pop_bin(pop: dict, path: Path) -> None:
    header = np.zeros(16, dtype=np.int32)
    header[0:9] = (POP_MAGIC, POP_VERSION, pop["M"], pop["total_nodes"], pop["total_c"],
                   pop["N"], pop["n_vars"], pop["K_max"], pop["max_stack"])
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(pop["nt"], np.int32).tobytes())
        f.write(np.ascontiguousarray(pop["nv"], np.float32).tobytes())
        f.write(np.ascontiguousarray(pop["ci"], np.int32).tobytes())
        f.write(np.ascontiguousarray(pop["metas"], np.int32).tobytes())
        f.write(np.ascontiguousarray(pop["c_init"], np.float32).tobytes())
        f.write(np.ascontiguousarray(pop["xs"], np.float32).tobytes())
        f.write(np.ascontiguousarray(pop["ym"], np.float32).tobytes())


def _sim_stack_depth(nt) -> int:
    sp = ms = 0
    for i in reversed(range(len(nt))):
        t = int(nt[i])
        if t in (NTYPE_VAR, NTYPE_CONST):
            sp += 1
        elif t == NTYPE_UFUNC:
            pass  # -1 +1
        elif t == NTYPE_BFUNC:
            sp -= 1
        else:
            raise ValueError(f"unsupported node_type {t}")
        ms = max(ms, sp)
    return ms


def build_pop(trees, xs, ym) -> dict:
    """trees = [(nt, nv, ci, c_init_per_tree)], xs (N,n_vars), ym (M,N) → pop dict.

    offsets / K_max / max_stack 在这里算, 调用方只管单棵树. (W4 合成生成器也走这入口.)
    """
    xs = np.atleast_2d(np.asarray(xs, np.float32))
    nt_all, nv_all, ci_all, metas, c_all = [], [], [], [], []
    node_off = c_off = 0
    K_max = max_stack = 0
    for nt, nv, ci, c_init in trees:
        nt, ci = np.asarray(nt, np.int32), np.asarray(ci, np.int32)
        K = int(np.sum(nt == NTYPE_CONST))
        # 必须按 prefix 扫描序 (非仅置换): PySRBFGS 读回依赖
        # 「ci rank 序 == 前序 DFS 序」这一不变量 (dump_evogp 同样保证)
        ranks = [int(r) for r in ci[ci >= 0]]
        if ranks != list(range(K)):
            raise ValueError(f"ci ranks 必须按 prefix 扫描序 0..{K-1}, got {ranks}")
        if len(np.asarray(c_init).ravel()) != K:
            raise ValueError(f"c_init len {len(c_init)} != K={K}")
        metas.append((node_off, len(nt), c_off, K))
        nt_all.append(nt); nv_all.append(np.asarray(nv, np.float32)); ci_all.append(ci)
        c_all.append(np.asarray(c_init, np.float32).ravel())
        node_off += len(nt); c_off += K
        K_max = max(K_max, K)
        max_stack = max(max_stack, _sim_stack_depth(nt))
    ym = np.asarray(ym, np.float32).reshape(len(trees), xs.shape[0])
    return dict(M=len(trees), total_nodes=node_off, total_c=c_off,
                N=xs.shape[0], n_vars=xs.shape[1], K_max=K_max, max_stack=max_stack,
                nt=np.concatenate(nt_all), nv=np.concatenate(nv_all),
                ci=np.concatenate(ci_all), metas=np.asarray(metas, np.int32).reshape(-1, 4),
                c_init=(np.concatenate(c_all) if c_off else np.zeros(0, np.float32)),
                xs=xs, ym=ym)


def slice_pop(pop: dict, limit: int) -> dict:
    """前 limit 棵树的子 pop (offsets 重排, 给 --limit / warmup 用)."""
    if limit >= pop["M"]:
        return pop
    trees = []
    for m in range(limit):
        node_off, n_nodes, c_off, K = pop["metas"][m].tolist()
        trees.append((pop["nt"][node_off:node_off + n_nodes],
                      pop["nv"][node_off:node_off + n_nodes],
                      pop["ci"][node_off:node_off + n_nodes],
                      pop["c_init"][c_off:c_off + K]))
    return build_pop(trees, pop["xs"], pop["ym"][:limit])
