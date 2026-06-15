"""interp.py — 向量化 numpy 后缀解释器 (oracle / loss 重算 / scipy 后端共用).

与 008 tree_interpreter.py 同语义 (LOOSE op 同样退化成标准 op, NaN/Inf 自然传播),
但跨 N 个数据点向量化 — 栈上放 (N,) 数组而不是标量. 已在 test_harness.py 里
对 tree_interpreter 逐点交叉验证.

另带 to_infix(): bytecode → infix 字符串, **不经 sympy** —— sympy 默认自动化简
(c0+c1-c1 直接坍缩), 会悄悄改掉退化树的优化问题; 字符串打印保持树原样,
给将来 pyoperon / PySR 后端当输入.
"""
from __future__ import annotations

import numpy as np

NTYPE_VAR, NTYPE_CONST, NTYPE_UFUNC, NTYPE_BFUNC = 0, 1, 2, 3


class F:
    ADD = 1; SUB = 2; MUL = 3; DIV = 4
    LOOSE_DIV = 5; POW = 6; LOOSE_POW = 7
    MAX = 8; MIN = 9
    LT = 10; GT = 11; LE = 12; GE = 13
    SIN = 14; COS = 15; TAN = 16
    SINH = 17; COSH = 18; TANH = 19
    LOG = 20; LOOSE_LOG = 21
    EXP = 22
    INV = 23; LOOSE_INV = 24
    NEG = 25; ABS = 26
    SQRT = 27; LOOSE_SQRT = 28


UNARY = {
    F.SIN: np.sin, F.COS: np.cos, F.TAN: np.tan,
    F.SINH: np.sinh, F.COSH: np.cosh, F.TANH: np.tanh,
    F.LOG: np.log, F.LOOSE_LOG: lambda a: np.log(np.abs(a)),
    F.EXP: np.exp,
    F.INV: lambda a: 1.0 / a, F.LOOSE_INV: lambda a: 1.0 / a,
    F.NEG: np.negative, F.ABS: np.abs,
    F.SQRT: np.sqrt, F.LOOSE_SQRT: lambda a: np.sqrt(np.abs(a)),
}

BINARY = {
    F.ADD: np.add, F.SUB: np.subtract, F.MUL: np.multiply,
    F.DIV: np.divide, F.LOOSE_DIV: np.divide,
    F.POW: np.power, F.LOOSE_POW: lambda l, r: np.abs(l) ** r,
    F.MAX: np.maximum, F.MIN: np.minimum,
    F.LT: lambda l, r: (l < r).astype(float), F.GT: lambda l, r: (l > r).astype(float),
    F.LE: lambda l, r: (l <= r).astype(float), F.GE: lambda l, r: (l >= r).astype(float),
}


def eval_tree_arrays(nt, nv, ci, xs, c, dtype=np.float64):
    """单棵树在全部 N 点上求值. nt/nv/ci 为该树切片, xs (N,n_vars), c (K,) → y (N,)."""
    xs = np.asarray(xs, dtype)
    c = np.asarray(c, dtype)
    N = xs.shape[0]
    stack: list[np.ndarray] = []
    with np.errstate(all="ignore"):
        for i in reversed(range(len(nt))):
            t = int(nt[i])
            if t == NTYPE_VAR:
                stack.append(xs[:, int(nv[i])])
            elif t == NTYPE_CONST:
                stack.append(np.full(N, c[int(ci[i])], dtype))
            elif t == NTYPE_UFUNC:
                stack.append(UNARY[int(nv[i])](stack.pop()))
            elif t == NTYPE_BFUNC:
                l = stack.pop(); r = stack.pop()
                stack.append(BINARY[int(nv[i])](l, r))
            else:
                raise ValueError(f"unsupported node_type {t} at node {i}")
    if len(stack) != 1:
        raise RuntimeError(f"malformed tree: stack depth {len(stack)} != 1")
    return stack[0]


def tree_view(pop: dict, m: int):
    """pop 里第 m 棵树的 (nt, nv, ci, y, K) 切片视图."""
    node_off, n_nodes, c_off, K = pop["metas"][m].tolist()
    sl = slice(node_off, node_off + n_nodes)
    return pop["nt"][sl], pop["nv"][sl], pop["ci"][sl], pop["ym"][m], K


def eval_pop_tree(pop: dict, m: int, c, dtype=np.float64):
    nt, nv, ci, _, _ = tree_view(pop, m)
    return eval_tree_arrays(nt, nv, ci, pop["xs"], c, dtype)


def loss_pop(pop: dict, c_final: np.ndarray) -> np.ndarray:
    """每树 fp64 loss = 0.5·Σr² (协议 §3: 统一重算, 不信后端自报). 非有限 → inf."""
    M = pop["M"]
    out = np.full(M, np.inf)
    for m in range(M):
        _, _, c_off, K = pop["metas"][m].tolist()
        try:
            y = eval_pop_tree(pop, m, c_final[c_off:c_off + K])
        except Exception:  # noqa: BLE001 — 坏树记 inf, 不炸整批
            continue
        if np.all(np.isfinite(y)):
            r = y - pop["ym"][m].astype(np.float64)
            out[m] = 0.5 * float(np.sum(r * r))
    return out


def prefix_span(nt, i: int) -> int:
    """prefix 编码下, 以 i 为根的子树节点数 (纯计算)."""
    t = int(nt[i])
    if t in (NTYPE_VAR, NTYPE_CONST):
        return 1
    if t == NTYPE_UFUNC:
        return 1 + prefix_span(nt, i + 1)
    if t == NTYPE_BFUNC:
        ls = prefix_span(nt, i + 1)
        return 1 + ls + prefix_span(nt, i + 1 + ls)
    raise ValueError(f"unsupported node_type {t}")


# ---------------------------------------------------------------- infix 打印

_UNARY_NAME = {
    F.SIN: "sin", F.COS: "cos", F.TAN: "tan", F.SINH: "sinh", F.COSH: "cosh",
    F.TANH: "tanh", F.LOG: "log", F.LOOSE_LOG: "logabs", F.EXP: "exp",
    F.INV: "inv", F.LOOSE_INV: "inv", F.NEG: "neg", F.ABS: "abs",
    F.SQRT: "sqrt", F.LOOSE_SQRT: "sqrtabs",
}
_BINARY_SYM = {F.ADD: "+", F.SUB: "-", F.MUL: "*", F.DIV: "/", F.LOOSE_DIV: "/",
               F.POW: "^", F.LOOSE_POW: "^|", F.MAX: "max", F.MIN: "min"}


def to_infix(nt, nv, ci, const_fmt="c{}", var_fmt="x{}") -> str:
    """bytecode → infix 字符串 (全括号, 无化简). 主要给外部后端 / 调试用."""
    stack: list[str] = []
    for i in reversed(range(len(nt))):
        t = int(nt[i])
        if t == NTYPE_VAR:
            stack.append(var_fmt.format(int(nv[i])))
        elif t == NTYPE_CONST:
            stack.append(const_fmt.format(int(ci[i])))
        elif t == NTYPE_UFUNC:
            stack.append(f"{_UNARY_NAME[int(nv[i])]}({stack.pop()})")
        elif t == NTYPE_BFUNC:
            l = stack.pop(); r = stack.pop()
            sym = _BINARY_SYM[int(nv[i])]
            if sym in ("max", "min"):
                stack.append(f"{sym}({l}, {r})")
            else:
                stack.append(f"({l} {sym} {r})")
        else:
            raise ValueError(f"unsupported node_type {t}")
    return stack[0]
