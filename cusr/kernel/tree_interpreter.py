"""EvoGP tree-tensor interpreter (forward eval only).

吃 EvoGP 内部的三元组 (node_type, node_value, subtree_size), 计算 f(x; c).
用 stack-machine 做 reverse-prefix 遍历. enum 跟 EvoGP 的 tree/utils.py 对齐.

LOOSE_* op (EvoGP 在边界 clamp 到 1e9 以避奇点) 在这里**直接退化成标准 op**:
- LOOSE_DIV → DIV, LOOSE_LOG → log|x|, LOOSE_INV → 1/x, LOOSE_SQRT → sqrt|x|, LOOSE_POW → |x|^y
让 NaN/Inf 自然传播, 由外层 LM solver 拒绝坏 step.

这个模块只做 forward eval (给 scipy / verify.py 用作 oracle).
"""

import math

import numpy as np


# ============================================================
# Enums (mirror EvoGP's tree/utils.py)
# ============================================================
class NType:
    VAR = 0
    CONST = 1
    UFUNC = 2
    BFUNC = 3
    TFUNC = 4


class Func:
    # binary (BFUNC)
    ADD = 1; SUB = 2; MUL = 3; DIV = 4
    LOOSE_DIV = 5; POW = 6; LOOSE_POW = 7
    MAX = 8; MIN = 9
    LT = 10; GT = 11; LE = 12; GE = 13
    # unary (UFUNC)
    SIN = 14; COS = 15; TAN = 16
    SINH = 17; COSH = 18; TANH = 19
    LOG = 20; LOOSE_LOG = 21
    EXP = 22
    INV = 23; LOOSE_INV = 24
    NEG = 25; ABS = 26
    SQRT = 27; LOOSE_SQRT = 28


UNARY = {
    Func.SIN: math.sin,    Func.COS: math.cos,    Func.TAN: math.tan,
    Func.SINH: math.sinh,  Func.COSH: math.cosh,  Func.TANH: math.tanh,
    Func.LOG: math.log,    Func.LOOSE_LOG: lambda a: math.log(abs(a)),
    Func.EXP: math.exp,
    Func.INV: lambda a: 1.0 / a,        Func.LOOSE_INV: lambda a: 1.0 / a,
    Func.NEG: lambda a: -a,             Func.ABS: abs,
    Func.SQRT: math.sqrt,               Func.LOOSE_SQRT: lambda a: math.sqrt(abs(a)),
}

BINARY = {
    Func.ADD: lambda l, r: l + r,
    Func.SUB: lambda l, r: l - r,
    Func.MUL: lambda l, r: l * r,
    Func.DIV: lambda l, r: l / r,
    Func.LOOSE_DIV: lambda l, r: l / r,
    Func.POW: lambda l, r: l ** r,
    Func.LOOSE_POW: lambda l, r: abs(l) ** r,
    Func.MAX: max,           Func.MIN: min,
    Func.LT: lambda l, r: float(l < r),
    Func.GT: lambda l, r: float(l > r),
    Func.LE: lambda l, r: float(l <= r),
    Func.GE: lambda l, r: float(l >= r),
}


# ============================================================
# Interpreter
# ============================================================
def build_const_index(node_type, n):
    """Map: tree node index → c_vec index (prefix-order rank of CONST nodes)."""
    idx = {}
    k = 0
    for i in range(n):
        if int(node_type[i]) == NType.CONST:
            idx[i] = k
            k += 1
    return idx, k


def eval_tree(node_type, node_value, subtree_size, x_vec, c_vec, const_idx=None):
    """Forward eval one tree on one data point.

    Args:
        node_type, node_value, subtree_size: EvoGP tree arrays (len ≥ subtree_size[0])
        x_vec: input variables, indexable by int (e.g. np.ndarray[n_vars])
        c_vec: constants, indexable by int (len == #CONST nodes in tree)
        const_idx: optional pre-built {node_index → c_vec_index} for speed

    Returns:
        scalar f(x; c)
    """
    n = int(subtree_size[0])
    if const_idx is None:
        const_idx, _ = build_const_index(node_type, n)

    stack = []
    for i in reversed(range(n)):
        t = int(node_type[i])
        v = node_value[i]
        if t == NType.VAR:
            stack.append(float(x_vec[int(v)]))
        elif t == NType.CONST:
            stack.append(float(c_vec[const_idx[i]]))
        elif t == NType.UFUNC:
            a = stack.pop()
            stack.append(UNARY[int(v)](a))
        elif t == NType.BFUNC:
            l = stack.pop()  # 左孩子(prefix 里在 operator 之后第一个出现)
            r = stack.pop()  # 右孩子
            stack.append(BINARY[int(v)](l, r))
        else:
            raise ValueError(f"unsupported node_type {t} at node {i}")

    if len(stack) != 1:
        raise RuntimeError(f"malformed tree: stack depth {len(stack)} != 1")
    return stack[0]


def eval_batch(node_type, node_value, subtree_size, xs, c_vec):
    """Forward eval one tree on N data points. Returns y_pred[N]."""
    n = int(subtree_size[0])
    const_idx, _ = build_const_index(node_type, n)
    xs = np.atleast_2d(xs)
    if xs.ndim == 1:
        xs = xs[:, None]
    N = xs.shape[0]
    y_pred = np.empty(N, dtype=float)
    for i in range(N):
        y_pred[i] = eval_tree(node_type, node_value, subtree_size,
                              xs[i], c_vec, const_idx)
    return y_pred


# ============================================================
# Self-test fixtures (run `python tree_interpreter.py`)
# ============================================================
def _fixture_1():
    """c0 * x[0] + c1  --  prefix: + * c0 x0 c1"""
    nt = np.array([NType.BFUNC, NType.BFUNC, NType.CONST, NType.VAR, NType.CONST])
    nv = np.array([Func.ADD,    Func.MUL,    0.0,         0.0,        0.0])
    ss = np.array([5])

    c = [2.5, 1.3]
    x = [3.0]
    got = eval_tree(nt, nv, ss, x, c)
    expected = c[0] * x[0] + c[1]  # 2.5 * 3.0 + 1.3 = 8.8
    assert abs(got - expected) < 1e-12, f"F1: {got} vs {expected}"
    print(f"F1  c0*x0+c1                got={got:.10f}  expected={expected:.10f}  PASS")


def _fixture_2():
    """sin(x[0]) * c0 + c1 * exp(x[1])
    prefix: + * sin x0 c0 * c1 exp x1"""
    nt = np.array([NType.BFUNC, NType.BFUNC, NType.UFUNC, NType.VAR, NType.CONST,
                   NType.BFUNC, NType.CONST, NType.UFUNC, NType.VAR])
    nv = np.array([Func.ADD,    Func.MUL,    Func.SIN,    0.0,       0.0,
                   Func.MUL,    0.0,         Func.EXP,    1.0])
    ss = np.array([9])

    c = [2.0, 0.5]
    x = [math.pi / 4, 1.0]
    got = eval_tree(nt, nv, ss, x, c)
    expected = math.sin(x[0]) * c[0] + c[1] * math.exp(x[1])
    assert abs(got - expected) < 1e-12, f"F2: {got} vs {expected}"
    print(f"F2  sin(x0)*c0+c1*exp(x1)   got={got:.10f}  expected={expected:.10f}  PASS")


def _fixture_3_batch():
    """Batch eval of fixture 2 on 3 data points."""
    nt = np.array([NType.BFUNC, NType.BFUNC, NType.UFUNC, NType.VAR, NType.CONST,
                   NType.BFUNC, NType.CONST, NType.UFUNC, NType.VAR])
    nv = np.array([Func.ADD,    Func.MUL,    Func.SIN,    0.0,       0.0,
                   Func.MUL,    0.0,         Func.EXP,    1.0])
    ss = np.array([9])
    c = [2.0, 0.5]
    xs = np.array([
        [0.0,           0.0],
        [math.pi / 2,   1.0],
        [math.pi,       2.0],
    ])
    got = eval_batch(nt, nv, ss, xs, c)
    expected = np.array([
        math.sin(0.0) * 2.0 + 0.5 * math.exp(0.0),       # 0 + 0.5 = 0.5
        math.sin(math.pi/2) * 2.0 + 0.5 * math.exp(1.0), # 2 + 0.5*e
        math.sin(math.pi) * 2.0 + 0.5 * math.exp(2.0),   # 0 + 0.5*e²
    ])
    assert np.allclose(got, expected, atol=1e-12), f"F3: {got} vs {expected}"
    print(f"F3  batch N=3              got={np.round(got, 6)}  expected={np.round(expected, 6)}  PASS")


def _fixture_4_div_sqrt():
    """c0 / sqrt(x[0] + c1) — div + sqrt + add, non-commutative
    prefix: / c0 sqrt + x0 c1"""
    nt = np.array([NType.BFUNC, NType.CONST, NType.UFUNC, NType.BFUNC, NType.VAR, NType.CONST])
    nv = np.array([Func.DIV,    0.0,         Func.SQRT,   Func.ADD,    0.0,       0.0])
    ss = np.array([6])

    c = [3.0, 1.0]
    x = [3.0]
    got = eval_tree(nt, nv, ss, x, c)
    expected = c[0] / math.sqrt(x[0] + c[1])  # 3 / 2 = 1.5
    assert abs(got - expected) < 1e-12, f"F4: {got} vs {expected}"
    print(f"F4  c0/sqrt(x0+c1)         got={got:.10f}  expected={expected:.10f}  PASS")


def _fixture_5_sympy_crosscheck():
    """Use sympy as oracle. Fixture 2 expression but at random (x, c)."""
    try:
        import sympy as sp
    except ImportError:
        print("F5  sympy not available, skipping")
        return

    nt = np.array([NType.BFUNC, NType.BFUNC, NType.UFUNC, NType.VAR, NType.CONST,
                   NType.BFUNC, NType.CONST, NType.UFUNC, NType.VAR])
    nv = np.array([Func.ADD,    Func.MUL,    Func.SIN,    0.0,       0.0,
                   Func.MUL,    0.0,         Func.EXP,    1.0])
    ss = np.array([9])

    x0, x1, c0, c1 = sp.symbols("x0 x1 c0 c1")
    expr = sp.sin(x0) * c0 + c1 * sp.exp(x1)
    f = sp.lambdify((x0, x1, c0, c1), expr, modules="math")

    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(-2, 2, 2)
        c = rng.uniform(-2, 2, 2)
        got = eval_tree(nt, nv, ss, x, c)
        expected = f(x[0], x[1], c[0], c[1])
        assert abs(got - expected) < 1e-12, f"F5 mismatch: {got} vs {expected}"
    print(f"F5  sympy crosscheck × 20  all match to 1e-12  PASS")


if __name__ == "__main__":
    _fixture_1()
    _fixture_2()
    _fixture_3_batch()
    _fixture_4_div_sqrt()
    _fixture_5_sympy_crosscheck()
    print("\nPASS — interpreter forward eval matches closed form + sympy oracle.")
