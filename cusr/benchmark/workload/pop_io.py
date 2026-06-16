"""pop_io.py — pop.bin writer/reader for the workload corpus.

The WRITER (``write_pop_bin``) is factored out of ``dump_evogp._write_pop_bin``
so the on-disk byte format is produced in exactly one place (spec §1): same
header field order + dtypes, same big-flat-array concatenation order
(nt, nv, ci, metas, c_init, xs, ym), same ``ym`` replication (one shared ``y``
tiled ``M_prob`` times), same ``max_stack`` computation. ``cusr/kernel/inspect``
must PASS on its output.

The READER (``read_pop_bin``) is a pure-python mirror of ``loader.c`` for
byte-level test verification: it parses the header + every array back into numpy
and re-checks the loader's consistency invariants (sum(K)==total_c,
sum(n_nodes)==total_nodes).

Also exposes ``compute_max_stack`` (the reverse-prefix stack-depth simulation,
identical to ``dump_evogp._compute_stack_depth``) and the ``NType`` / ``Func``
enums used throughout the kernel + adapter.

Byte layout (pop_format.h, version 1, all little-endian, no padding):
  [PopHeader 64 B]  16×int32: magic, version, M_prob, total_nodes, total_c,
                    N, n_vars, K_max, max_stack, reserved[7]
  [nt_all   total_nodes * int32]    node types
  [nv_all   total_nodes * float32]  node values (VAR->col idx, FUNC->func id, CONST->0)
  [ci_all   total_nodes * int32]    const index (-1 if not CONST else 0..K-1)
  [metas    M_prob * 16 B]          TreeMeta{node_offset, n_nodes, c_offset, K}
  [c_init   total_c * float32]      initial constants (forward-prefix order)
  [xs       N * n_vars * float32]   xs[i*n_vars + k]
  [ym       M_prob * N * float32]   ym[m*N + i]  (all trees share one y -> tiled)
"""
from __future__ import annotations

import struct
from enum import IntEnum
from pathlib import Path

import numpy as np

# pop_format.h: POP_MAGIC = 0x4D4C344D ('M','4','L','M' 4 字节 LE)
POP_MAGIC = 0x4D4C344D
POP_VERSION = 1

# batch_lm.cu compile-time bound (spec: K>32 is ~0%, validated safe).
MAX_K = 32


class NType(IntEnum):
    """Node type tag (matches inspect.c / batch_lm.cu / EvoGP tree/utils.py)."""
    VAR = 0
    CONST = 1
    UFUNC = 2
    BFUNC = 3
    TFUNC = 4  # ternary (IF) — never produced under the restricted grammar


class Func(IntEnum):
    """Func id (subset used by the restricted grammar; full enum in inspect.c)."""
    IF = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    POW = 6
    SIN = 14
    COS = 15
    TAN = 16
    LOG = 20
    EXP = 22
    INV = 23
    NEG = 25
    ABS = 26
    SQRT = 27


def compute_max_stack(nt) -> int:
    """Max stack depth over the reverse-prefix walk (mirrors the kernel).

    VAR/CONST push (+1); UFUNC pops 1 pushes 1 (net 0); BFUNC pops 2 pushes 1
    (net -1); TFUNC pops 3 pushes 1 (net -2). Identical algorithm to
    ``dump_evogp._compute_stack_depth``.
    """
    n = len(nt)
    sp = 0
    max_sp = 0
    for i in reversed(range(n)):
        t = int(nt[i])
        if t == NType.VAR or t == NType.CONST:
            sp += 1
        elif t == NType.UFUNC:
            sp = sp - 1 + 1
        elif t == NType.BFUNC:
            sp = sp - 2 + 1
        elif t == NType.TFUNC:
            sp = sp - 3 + 1
        else:
            raise ValueError(f"unknown type {t}")
        if sp > max_sp:
            max_sp = sp
    return max_sp


def write_pop_bin(
    out_path,
    trees: list[tuple],  # [(nt, nv, ci, c_init), ...]  each a 1-D array/sequence
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Write pop.bin in pop_format.h order. Returns a stats dict.

    Factored from ``dump_evogp._write_pop_bin`` — keep byte-for-byte identical.
    ``trees`` is a list of ``(nt, nv, ci, c_init)`` arrays; ``X`` is ``(N, n_vars)``
    and ``y`` is ``(N,)`` shared by every tree (tiled M_prob times into ``ym``).
    """
    out = Path(out_path)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    M_prob = len(trees)
    total_nodes = sum(len(t[0]) for t in trees)
    total_c = sum(len(t[3]) for t in trees)
    N, n_vars = X.shape
    K_max = max((len(t[3]) for t in trees), default=0)
    max_stack = max((compute_max_stack(t[0]) for t in trees), default=0)

    # build big flat arrays
    nt_all = np.concatenate([np.asarray(t[0]) for t in trees]) if trees else np.array([], np.int32)
    nv_all = np.concatenate([np.asarray(t[1]) for t in trees]) if trees else np.array([], np.float32)
    ci_all = np.concatenate([np.asarray(t[2]) for t in trees]) if trees else np.array([], np.int32)
    c_all = np.concatenate([np.asarray(t[3]) for t in trees]) if trees else np.array([], np.float32)

    # metas: node_offset, n_nodes, c_offset, K
    metas = np.zeros((M_prob, 4), dtype=np.int32)
    nod_off = 0
    c_off = 0
    for i, (nt, _, _, ci) in enumerate(trees):
        K = len(ci)
        metas[i] = [nod_off, len(nt), c_off, K]
        nod_off += len(nt)
        c_off += K

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        # PopHeader 64 字节 = 9 named + 7 reserved = 16 int32
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
        # header[9..15] reserved, 全 0
        f.write(header.tobytes())
        f.write(nt_all.astype(np.int32).tobytes())
        f.write(nv_all.astype(np.float32).tobytes())
        f.write(ci_all.astype(np.int32).tobytes())
        f.write(metas.tobytes())
        f.write(c_all.astype(np.float32).tobytes())
        f.write(X.astype(np.float32).tobytes())
        # ym layout: [M_prob, N] per-tree; all trees share one y -> tile M_prob 份.
        ym_per_tree = np.tile(y.astype(np.float32), M_prob)
        f.write(ym_per_tree.tobytes())

    return dict(M_prob=M_prob, total_nodes=total_nodes, total_c=total_c,
                N=N, n_vars=n_vars, K_max=K_max, max_stack=max_stack)


def read_pop_bin(path) -> dict:
    """Pure-python mirror of ``loader.c::load_pop_bin`` for byte-level tests.

    Parses the header + every array back into numpy and re-checks the same
    consistency invariants the C loader enforces (bad magic / version,
    sum(K)==total_c, sum(n_nodes)==total_nodes). Returns a dict with the header
    fields plus ``nt, nv, ci, metas, c_init, xs, ym`` numpy arrays.
    """
    with open(path, "rb") as f:
        raw = f.read(64)
        if len(raw) != 64:
            raise ValueError(f"short read on header (got {len(raw)} / 64)")
        header = struct.unpack("<16i", raw)
        magic, version, M, total_n, total_c, N, n_vars, K_max, max_stack = header[0:9]
        if magic != POP_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x} (want 0x{POP_MAGIC:08x})")
        if version != POP_VERSION:
            raise ValueError(f"version {version} != {POP_VERSION}")

        def _read(n_bytes, what):
            buf = f.read(n_bytes)
            if len(buf) != n_bytes:
                raise ValueError(f"short read on {what} (got {len(buf)} / {n_bytes})")
            return buf

        nt = np.frombuffer(_read(total_n * 4, "nt"), dtype=np.int32).copy()
        nv = np.frombuffer(_read(total_n * 4, "nv"), dtype=np.float32).copy()
        ci = np.frombuffer(_read(total_n * 4, "ci"), dtype=np.int32).copy()
        metas = np.frombuffer(_read(M * 16, "metas"), dtype=np.int32).reshape(M, 4).copy()
        c_init = np.frombuffer(_read(total_c * 4, "c_init"), dtype=np.float32).copy()
        xs = np.frombuffer(_read(N * n_vars * 4, "xs"), dtype=np.float32).reshape(N, n_vars).copy()
        ym = np.frombuffer(_read(M * N * 4, "ym"), dtype=np.float32).reshape(M, N).copy()

    # consistency sanity (mirrors loader.c)
    K_sum = int(metas[:, 3].sum()) if M else 0
    node_sum = int(metas[:, 1].sum()) if M else 0
    if K_sum != total_c:
        raise ValueError(f"sum(K)={K_sum} != header.total_c={total_c}")
    if node_sum != total_n:
        raise ValueError(f"sum(n_nodes)={node_sum} != header.total_nodes={total_n}")

    return dict(magic=magic, version=version, M_prob=M, total_nodes=total_n,
                total_c=total_c, N=N, n_vars=n_vars, K_max=K_max, max_stack=max_stack,
                nt=nt, nv=nv, ci=ci, metas=metas, c_init=c_init, xs=xs, ym=ym)
