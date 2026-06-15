"""backends.py — CO 后端契约 + ScipyPop / TorchPop / CudaKernelPop.

契约: fit_pop(pop dict) -> FitResult. 后端只负责把常数拟出来 + 自报核心耗时;
loss 一律由 runner 用 interp.loss_pop (fp64) 统一重算 (协议 §3), tier 判定不在这里.

status 编码 (统一): 0=converged(本家准则) 1=iter-limit 2=failed 3=K0-skip.
n_iter 是后端原生计数 (scipy=nfev 含 FD evals, torch=outer iter, kernel=未导出→-1),
只作辅助诊断, 不跨后端比较 (协议 §4).
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import interp
from . import popio

# 进程池一律 spawn: PySRBFGS 把 Julia 载进主进程, 而 Julia 不是 fork-safe —
# fork 出的池子 worker 继承 Julia 运行时状态会死锁 (实测: pysr 测试之后的
# 第一个 fork Pool 卡死). spawn 干净 (fork+exec), 代价是池启动 ~秒级 →
# 池做成 per-backend 持久 (lazy 创建, 复用跨 fit_pop), 启动成本由 warmup 吃掉.
_MP_CTX = mp.get_context("spawn")
_POOL_CACHE: dict = {}          # nproc → Pool; ScipyPop/OperonLM 同 nproc 共用
_POOL_TIMEOUT = 300.0           # 单个结果的最长等待 (秒), 远大于单树预算


def _shared_pool(nproc: int):
    """按 nproc 缓存共享池: 防止 sweep 场景 (多次 run_bench / 多 backend 实例)
    每次新建 2×ncpu 个 spawn worker 只增不减地泄漏."""
    pool = _POOL_CACHE.get(nproc)
    if pool is None:
        pool = _MP_CTX.Pool(processes=nproc)
        _POOL_CACHE[nproc] = pool
    return pool


def _scrap_pool(nproc: int):
    pool = _POOL_CACHE.pop(nproc, None)
    if pool is not None:
        pool.terminate()


def _imap_drain(pool, fn, payloads, on_result, nproc: int) -> bool:
    """imap_unordered + 单结果超时. worker 被 native 崩溃杀死 (segfault/OOM-kill)
    时 mp.Pool 会补 worker 但不重提交任务也不抛错 → 无超时会永久挂死.
    超时则废掉该池 (缓存剔除 + terminate), 返回 False; 调用方已把 payload 树
    预标 failed, 丢失的结果自然落在 failed 上.
    chunksize 必须为 1: chunksize>1 时 imap_unordered 返回的是展平生成器,
    没有 next(timeout) (实测 AttributeError); 1 才是 IMapUnorderedIterator."""
    it = pool.imap_unordered(fn, payloads, chunksize=1)
    for _ in range(len(payloads)):
        try:
            on_result(it.next(_POOL_TIMEOUT))
        except mp.TimeoutError:
            _scrap_pool(nproc)
            return False
        except StopIteration:
            break
    return True


@atexit.register
def _cleanup_pools():
    for p in list(_POOL_CACHE.values()):
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class FitResult:
    backend: str
    c_final: np.ndarray          # (total_c,) float64
    status: np.ndarray           # (M,) int — 见模块 docstring
    n_iter: np.ndarray           # (M,) int, -1=后端不导出
    wall_e2e: float              # 进 fit_pop → 常数回内存, 含全部 marshal
    wall_core: float             # 后端自报核心段; 拿不到 = nan
    meta: dict = field(default_factory=dict)


class KernelRunError(RuntimeError):
    pass


# ---------------------------------------------------------------- ScipyPop

_ORACLE_SETTINGS = dict(xtol=1e-10, ftol=1e-10, max_nfev=200)  # 协议 §3, 改动要 bump version


def _scipy_worker(payload):
    """top-level, 可 pickle. 返回 (m, c_final, status, nfev)."""
    from scipy.optimize import least_squares

    m, nt, nv, ci, xs, y, c0 = payload
    y64 = y.astype(np.float64)

    def residual(c):
        r = interp.eval_tree_arrays(nt, nv, ci, xs, c) - y64
        # 非有限逐点替 1e6 哨兵 (同 verify.py): scipy LM 见 NaN 直接挂
        return np.where(np.isfinite(r), r, 1e6)

    try:
        res = least_squares(residual, c0.astype(np.float64), method="lm",
                            **_ORACLE_SETTINGS)
        c = np.asarray(res.x, np.float64)
        if not np.all(np.isfinite(c)):
            return m, c0.astype(np.float64), 2, int(res.nfev)
        return m, c, (0 if res.status >= 1 else 1), int(res.nfev)
    except Exception:  # noqa: BLE001 — 单树失败不炸整批
        return m, c0.astype(np.float64), 2, 0


class ScipyPop:
    """scipy LM, fp64, 进程池满核. 设置 = oracle 设置 (协议 §3: 它就是参照点)."""

    name = "scipy"

    def __init__(self, nproc: int | None = None):
        self.nproc = nproc or (os.cpu_count() or 1)

    def fit_pop(self, pop: dict) -> FitResult:
        t0 = time.perf_counter()
        M = pop["M"]
        c_final = pop["c_init"].astype(np.float64).copy()
        status = np.full(M, 3, np.int32)
        n_iter = np.zeros(M, np.int32)
        payloads = []
        for m in range(M):
            nt, nv, ci, y, K = interp.tree_view(pop, m)
            if K == 0:
                continue
            status[m] = 2          # 预标 failed: 结果丢失 (worker 硬死) 时不冒充 K0
            _, _, c_off, _ = pop["metas"][m].tolist()
            payloads.append((m, nt, nv, ci, pop["xs"], y,
                             pop["c_init"][c_off:c_off + K]))
        pool = _shared_pool(self.nproc) if payloads else None  # 首次启动成本由 warmup 吃

        def on_result(res):
            m, c, st, nfev = res
            _, _, c_off, K = pop["metas"][m].tolist()
            c_final[c_off:c_off + K] = c
            status[m] = st
            n_iter[m] = nfev

        t_core = time.perf_counter()
        if payloads:
            _imap_drain(pool, _scipy_worker, payloads, on_result, self.nproc)
        wall_core = time.perf_counter() - t_core
        return FitResult(self.name, c_final, status, n_iter,
                         time.perf_counter() - t0, wall_core,
                         meta=dict(dtype="float64", nproc=self.nproc,
                                   criteria=f"scipy lm {_ORACLE_SETTINGS}"))


# ---------------------------------------------------------------- TorchPop

def _torch_maps():
    import torch
    unary = {
        interp.F.SIN: torch.sin, interp.F.COS: torch.cos, interp.F.TAN: torch.tan,
        interp.F.SINH: torch.sinh, interp.F.COSH: torch.cosh, interp.F.TANH: torch.tanh,
        interp.F.LOG: torch.log, interp.F.LOOSE_LOG: lambda a: torch.log(torch.abs(a)),
        interp.F.EXP: torch.exp,
        interp.F.INV: lambda a: 1.0 / a, interp.F.LOOSE_INV: lambda a: 1.0 / a,
        interp.F.NEG: torch.neg, interp.F.ABS: torch.abs,
        interp.F.SQRT: torch.sqrt, interp.F.LOOSE_SQRT: lambda a: torch.sqrt(torch.abs(a)),
    }
    binary = {
        interp.F.ADD: torch.add, interp.F.SUB: torch.sub, interp.F.MUL: torch.mul,
        interp.F.DIV: torch.div, interp.F.LOOSE_DIV: torch.div,
        interp.F.POW: torch.pow, interp.F.LOOSE_POW: lambda l, r: torch.abs(l) ** r,
        interp.F.MAX: torch.maximum, interp.F.MIN: torch.minimum,
    }
    return unary, binary


def _eval_torch(nt, nv, ci, xs_t, c_t, unary, binary):
    """后缀求值, 栈上是可广播 tensor (CONST 推 0-dim 保持 autograd 图)."""
    import torch
    stack = []
    for i in reversed(range(len(nt))):
        t = int(nt[i])
        if t == 0:
            stack.append(xs_t[:, int(nv[i])])
        elif t == 1:
            stack.append(c_t[int(ci[i])])
        elif t == 2:
            stack.append(unary[int(nv[i])](stack.pop()))
        elif t == 3:
            l = stack.pop(); r = stack.pop()
            op = int(nv[i])
            if op in binary:
                stack.append(binary[op](l, r))
            else:  # 比较类 op (LT/GT/LE/GE)
                cmp = {10: torch.lt, 11: torch.gt, 12: torch.le, 13: torch.ge}[op]
                stack.append(cmp(l, r).to(c_t.dtype))
        else:
            raise ValueError(f"unsupported node_type {t}")
    out = stack[0]
    return torch.broadcast_to(out, (xs_t.shape[0],))


def eval_pop_tree_torch(pop: dict, m: int, c, device="cpu", dtype=None):
    """测试入口: 单树 torch 求值, 返回 cpu tensor."""
    import torch
    dtype = dtype or torch.float64
    nt, nv, ci, _, _ = interp.tree_view(pop, m)
    xs_t = torch.tensor(np.asarray(pop["xs"], np.float64), dtype=dtype, device=device)
    c_t = torch.tensor(np.asarray(c, np.float64), dtype=dtype, device=device)
    unary, binary = _torch_maps()
    return _eval_torch(nt, nv, ci, xs_t, c_t, unary, binary).detach().cpu()


class TorchPop:
    """GPU 原生 LM, autograd Jacobian, 逐树循环 (每树内跨 N 向量化).
    LM 循环逻辑移植自 009 co_backend.TorchLM, model 换成 bytecode 解释器."""

    name = "torch"

    def __init__(self, device: str | None = None, dtype: str = "float64",
                 max_iter: int = 50, lambda0: float = 1e-3,
                 grad_tol: float = 1e-10, step_tol: float = 1e-12):
        self.device = device
        self.dtype = dtype
        self.max_iter = max_iter
        self.lambda0 = lambda0
        self.grad_tol = grad_tol
        self.step_tol = step_tol

    def fit_pop(self, pop: dict) -> FitResult:
        import torch

        t0 = time.perf_counter()
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = getattr(torch, self.dtype)
        unary, binary = _torch_maps()
        xs_t = torch.tensor(np.asarray(pop["xs"], np.float64), dtype=dt, device=dev)
        M = pop["M"]
        c_final = pop["c_init"].astype(np.float64).copy()
        status = np.full(M, 3, np.int32)
        n_iter = np.zeros(M, np.int32)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t_core = time.perf_counter()
        for m in range(M):
            nt, nv, ci, y, K = interp.tree_view(pop, m)
            if K == 0:
                continue
            _, _, c_off, _ = pop["metas"][m].tolist()
            y_t = torch.tensor(y.astype(np.float64), dtype=dt, device=dev)

            def resid(c_t):
                r = _eval_torch(nt, nv, ci, xs_t, c_t, unary, binary) - y_t
                # 哨兵 1e6 同 ScipyPop/verify.py; nan_to_num 的 autograd 在哨兵点给 0 梯度
                return torch.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

            try:
                c, st, it = self._fit_one(resid, pop["c_init"][c_off:c_off + K],
                                          dev, dt)
                c_final[c_off:c_off + K] = c
                status[m], n_iter[m] = st, it
            except Exception:  # noqa: BLE001
                status[m], n_iter[m] = 2, 0
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        wall_core = time.perf_counter() - t_core
        return FitResult(self.name, c_final, status, n_iter,
                         time.perf_counter() - t0, wall_core,
                         meta=dict(dtype=self.dtype, device=dev,
                                   criteria=f"LM grad_tol={self.grad_tol} "
                                            f"step_tol={self.step_tol} "
                                            f"max_iter={self.max_iter}"))

    def _fit_one(self, resid, c0, dev, dt):
        import torch

        nc = len(c0)
        c = torch.tensor(c0.astype(np.float64), dtype=dt, device=dev)
        r = resid(c)
        loss = float(torch.sum(r ** 2).item())
        lam = self.lambda0
        eye = torch.eye(nc, dtype=dt, device=dev)
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):
            J = torch.autograd.functional.jacobian(resid, c, vectorize=True)
            J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            g = J.T @ r
            H = J.T @ J
            if float(torch.max(torch.abs(g)).item()) < self.grad_tol:
                converged = True
                break
            accepted = False
            for _ in range(30):
                try:
                    delta = torch.linalg.solve(H + lam * eye, -g)
                except Exception:  # noqa: BLE001 — 奇异, 加大阻尼
                    lam *= 10.0
                    continue
                c_new = c + delta
                r_new = resid(c_new)
                loss_new = float(torch.sum(r_new ** 2).item())
                if np.isfinite(loss_new) and loss_new < loss:
                    step = float(torch.max(torch.abs(delta)).item())
                    c, r, loss = c_new, r_new, loss_new
                    lam = max(lam / 10.0, 1e-12)
                    accepted = True
                    if step < self.step_tol:
                        converged = True
                    break
                lam *= 10.0
                if lam > 1e12:
                    break
            if not accepted or converged:
                break
        st = 0 if converged else (1 if it >= self.max_iter else 2)
        # 未收敛但 lam 爆掉提前退出的, 算 iter-limit 之外的"卡住" — 仍记 1, 图注披露
        if not converged and it < self.max_iter:
            st = 1
        return c.detach().cpu().numpy().astype(np.float64), st, it


# ---------------------------------------------------------------- OperonLM

# 我们的 op 枚举 → pyoperon NodeType 名 (None = 树级不支持, 标 failed).
# MAX/MIN: 0.6.1 wheel 的 Fmax dispatch 实测算成 min — 不映射.
# LOOSE_*: Operon 原生有 Logabs/Sqrtabs/Powabs, 语义正好对上 008 解释器的退化.
_OPERON_UNARY = {
    interp.F.SIN: "Sin", interp.F.COS: "Cos", interp.F.TAN: "Tan",
    interp.F.SINH: "Sinh", interp.F.COSH: "Cosh", interp.F.TANH: "Tanh",
    interp.F.LOG: "Log", interp.F.LOOSE_LOG: "Logabs", interp.F.EXP: "Exp",
    interp.F.ABS: "Abs", interp.F.SQRT: "Sqrt", interp.F.LOOSE_SQRT: "Sqrtabs",
}
_OPERON_BINARY = {
    interp.F.ADD: "Add", interp.F.SUB: "Sub", interp.F.MUL: "Mul",
    interp.F.DIV: "Div", interp.F.LOOSE_DIV: "Div",
    interp.F.POW: "Pow", interp.F.LOOSE_POW: "Powabs",
}


def build_operon_nodes(op_, nt, nv, ci, c_init, var_hashes):
    """我们的 prefix bytecode → Operon postfix 节点表.

    Operon 节点序约定 (0.6.1 实测): 二元 op 的**左操作数紧贴父节点** —
    发射顺序 = [右子树..., 左子树..., 父]. 常数发射顺序 ≠ 我们的 rank 顺序,
    用 rank_order 显式回映. NEG/INV 没有原生 NodeType, 用冻结常数
    (Optimize=False) 的 Sub(0,a)/Div(1,a) 表示, 不进可优化参数.
    返回 (nodes, rank_order); 不支持的 op 抛 ValueError.
    """
    rank_order: list[int] = []

    def frozen(v):
        n = op_.Node.Constant(float(v))
        n.Optimize = False
        return n

    def span(i):
        """子树节点数 (我们的编码), 纯计算无副作用."""
        t = int(nt[i])
        if t in (0, 1):
            return 1
        if t == 2:
            return 1 + span(i + 1)
        if t == 3:
            ls = span(i + 1)
            return 1 + ls + span(i + 1 + ls)
        raise ValueError(f"unsupported node_type {t}")

    def emit(i, out):
        """发射以 prefix 下标 i 为根的子树, 返回该子树的 span (节点数, 按我们的编码)."""
        t = int(nt[i])
        if t == 0:    # VAR
            n = op_.Node(op_.NodeType.Variable)
            n.Value = 1.0
            n.HashValue = var_hashes[int(nv[i])]
            n.Optimize = False        # 冻结 Operon 的变量权重 w*X
            out.append(n)
            return 1
        if t == 1:    # CONST
            out.append(op_.Node.Constant(float(c_init[int(ci[i])])))
            rank_order.append(int(ci[i]))
            return 1
        fcode = int(nv[i])
        if t == 2:    # UFUNC
            if fcode == interp.F.NEG:
                s = emit(i + 1, out)
                out.append(frozen(0.0)); out.append(op_.Node.Sub())
                return 1 + s
            if fcode in (interp.F.INV, interp.F.LOOSE_INV):
                s = emit(i + 1, out)
                out.append(frozen(1.0)); out.append(op_.Node.Div())
                return 1 + s
            name = _OPERON_UNARY.get(fcode)
            if name is None:
                raise ValueError(f"unsupported unary op {fcode}")
            s = emit(i + 1, out)
            out.append(op_.Node(getattr(op_.NodeType, name)))
            return 1 + s
        if t == 3:    # BFUNC: prefix = [op, L..., R...]; operon 要 [R..., L..., op]
            name = _OPERON_BINARY.get(fcode)
            if name is None:
                raise ValueError(f"unsupported binary op {fcode}")
            l_start = i + 1
            l_span = span(l_start)
            # 发射顺序 = 节点表顺序 (右→左→父), rank_order 跟着发射顺序走 —
            # GetCoefficients/SetCoefficients 按节点表序枚举常数, 两边必须一致
            r_span = emit(l_start + l_span, out)
            emit(l_start, out)
            out.append(op_.Node(getattr(op_.NodeType, name)))
            return 1 + l_span + r_span
        raise ValueError(f"unsupported node_type {t}")

    nodes: list = []
    emit(0, nodes)
    return nodes, rank_order


_OP_WORKER_STATE: dict = {}   # 每 worker 进程缓存 (module, DispatchTable)


def _operon_worker(payload):
    """top-level, 可 pickle. 返回 (m, c_final_our_order, status, n_iter)."""
    m, nt, nv, ci, xs, y, c0, max_iter = payload
    try:
        if not _OP_WORKER_STATE:
            import pyoperon as op_
            _OP_WORKER_STATE["op"] = op_
            _OP_WORKER_STATE["dtable"] = op_.DispatchTable()
        op_ = _OP_WORKER_STATE["op"]
        dtable = _OP_WORKER_STATE["dtable"]

        N = xs.shape[0]
        data = np.asfortranarray(
            np.column_stack([xs.astype(np.float64), y.astype(np.float64)]))
        ds = op_.Dataset(data)
        dsvars = sorted(ds.Variables, key=lambda v: v.Index)
        var_hashes = [v.Hash for v in dsvars[:-1]]

        nodes, rank_order = build_operon_nodes(op_, nt, nv, ci, c0, var_hashes)
        tree = op_.Tree(nodes)
        tree.UpdateNodes()
        got = np.asarray(tree.GetCoefficients(), dtype=np.float64)
        want = np.asarray([c0[r] for r in rank_order], dtype=np.float32)
        if len(got) != len(c0) or not np.allclose(got, want, rtol=1e-6):
            raise RuntimeError(f"coefficient mapping mismatch: {got} vs {want}")

        problem = op_.Problem(ds)
        problem.TrainingRange = op_.Range(0, N)
        problem.TestRange = op_.Range(max(N - 1, 0), N)
        problem.Target = dsvars[-1]
        problem.InputHashes = var_hashes
        # 防雷 (W2 冒烟): lm 必须保活引用, CoefficientOptimizer 存裸指针
        lm = op_.LMOptimizer(dtable, problem, max_iter=max_iter, batch_size=N)
        co = op_.CoefficientOptimizer(lm)
        rng = op_.RandomGenerator(42)
        t_opt, summary = co(rng, tree)

        params = np.asarray(t_opt.GetCoefficients(), dtype=np.float64)
        c_final = np.asarray(c0, np.float64).copy()
        for k, r in enumerate(rank_order):
            c_final[r] = params[k]
        return m, c_final, (0 if summary.Success else 1), int(summary.Iterations)
    except Exception:  # noqa: BLE001 — 不支持的 op / binding 异常 → failed
        return m, np.asarray(c0, np.float64), 2, 0


class OperonLM:
    """pyoperon (Operon C++ SR) 的 LM 常数优化算子, 进程池满核.
    wheel 为单精度构建; GIL 不释放 → 并行只能 multiprocessing (W2 冒烟实测)."""

    name = "operon"

    def __init__(self, nproc: int | None = None, max_iter: int = 50):
        self.nproc = nproc or (os.cpu_count() or 1)
        self.max_iter = max_iter

    def fit_pop(self, pop: dict) -> FitResult:
        import pyoperon  # noqa: F401 — 父进程 fail-fast (未安装时立刻报错而非 worker 内静默 status=2)
        t0 = time.perf_counter()
        M = pop["M"]
        c_final = pop["c_init"].astype(np.float64).copy()
        status = np.full(M, 3, np.int32)
        n_iter = np.zeros(M, np.int32)
        payloads = []
        for m in range(M):
            nt, nv, ci, y, K = interp.tree_view(pop, m)
            if K == 0:
                continue
            status[m] = 2          # 预标 failed (见 ScipyPop)
            _, _, c_off, _ = pop["metas"][m].tolist()
            payloads.append((m, nt, nv, ci, pop["xs"], y,
                             pop["c_init"][c_off:c_off + K], self.max_iter))
        pool = _shared_pool(self.nproc) if payloads else None

        def on_result(res):
            m, c, st, it = res
            _, _, c_off, K = pop["metas"][m].tolist()
            c_final[c_off:c_off + K] = c
            status[m] = st
            n_iter[m] = it

        t_core = time.perf_counter()
        if payloads:
            _imap_drain(pool, _operon_worker, payloads, on_result, self.nproc)
        wall_core = time.perf_counter() - t_core
        return FitResult(self.name, c_final, status, n_iter,
                         time.perf_counter() - t0, wall_core,
                         meta=dict(dtype="float32 (wheel)", nproc=self.nproc,
                                   criteria=f"Operon LMOptimizer (Eigen LM) "
                                            f"max_iter={self.max_iter}"))


# ---------------------------------------------------------------- PySRBFGS

# PySR (SymbolicRegression.jl) 的 CO 算子原样抽取 — Julia BFGS (K=1 时它内部
# 静默换 Newton), 有限差分梯度, optimizer_iterations/nrestarts 外控.
# 防雷规则全集见 smoke_pysr_co.py 顶部注释 (W3 day-1 冒烟).
_JULIA_BATCH_MODULE = r"""
module OpBenchPySR

using SymbolicRegression: Options, Dataset, PopMember, Node,
    parse_expression, get_scalar_constants, eval_loss,
    safe_pow, safe_log, safe_sqrt
using SymbolicRegression.ConstantOptimizationModule: optimize_constants
import Random

# parse_expression 在 DynamicExpressions.ParseModule.EmptyModule (只有 Base)
# 里解析函数符号 — safe_* 是 SR 内部名, 解析不到. 把函数对象直接赋值进去
# (不能 import: EmptyModule 属于 DE 包, 包环境里没有 SR, 会循环依赖).
# 版本被 pysr 1.5.10 pin 死 (SR.jl v1.11.3 / DE gNHdy); fit_batch 里每棵树
# 的 baked 初值断言是这类内部依赖变化的兜底.
let pm = parentmodule(parse_expression)
    Core.eval(pm.EmptyModule, :(const safe_pow = $(safe_pow)))
    Core.eval(pm.EmptyModule, :(const safe_log = $(safe_log)))
    Core.eval(pm.EmptyModule, :(const safe_sqrt = $(safe_sqrt)))
end

function make_options(; iterations::Integer=8, nrestarts::Integer=2)
    return Options(;
        binary_operators=[+, -, *, /, safe_pow, max, min],
        unary_operators=[sin, cos, tan, sinh, cosh, tanh,
                         safe_log, exp, -, abs, safe_sqrt, inv],
        optimizer_iterations=iterations,
        optimizer_nrestarts=nrestarts,
        deterministic=false,
    )
end

# 批量入口: 串行循环 (CO 单棵 ~0.5ms, 线程安全性未验, 先不 @threads — meta 披露).
# exprs 过桥是 PyList{Any}, x0s 是 list of ndarray — 宽接收, 内部转 (防雷 #11).
function fit_batch(exprs, Xpy, Ypy, x0s, options, seed::Integer)
    X = Matrix{Float64}(Xpy)            # (nfeatures, N)
    Y = Matrix{Float64}(Ypy)            # (M_payload, N)
    M = length(exprs)
    strs = String[string(s) for s in exprs]
    inits = [Vector{Float64}(x) for x in x0s]
    vnames = String["x$(i)" for i in 1:size(X, 1)]
    consts = Vector{Vector{Float64}}(undef, M)
    status = fill(2, M)                 # 0=improved 1=no-improve 2=failed
    evals = zeros(Float64, M)
    t0 = time_ns()
    for i in 1:M
        consts[i] = inits[i]
        try
            Random.seed!(seed + i)      # nrestarts 的随机扰动可复现
            ex = parse_expression(Meta.parse(strs[i]);
                                  operators=options.operators,
                                  variable_names=vnames,
                                  node_type=Node{Float64})
            x0, _ = get_scalar_constants(ex)
            # 字面量折叠/解析歧义防线: 解析出的常数必须与 baked 初值逐位对上
            if length(x0) != length(inits[i]) ||
               !all(isapprox.(x0, inits[i]; rtol=1e-6, atol=1e-12))
                continue                # status stays 2
            end
            dataset = Dataset(X, Vector{Float64}(Y[i, :]))
            member = PopMember(dataset, ex, options;
                               deterministic=options.deterministic)
            loss0 = member.loss
            member, ne = optimize_constants(dataset, member, options)
            xopt, _ = get_scalar_constants(member.tree)
            lossf = eval_loss(member.tree, dataset, options; regularization=false)
            consts[i] = Vector{Float64}(xopt)
            evals[i] = ne
            status[i] = lossf < loss0 ? 0 : 1
        catch e
            e isa InterruptException && rethrow()   # Ctrl-C 别吞成 status=2
        end
    end
    wall = (time_ns() - t0) / 1e9
    return (consts=consts, status=status, evals=evals, wall=wall)
end

end # module
"""

_PYSR_STATE: dict = {}


def _pysr_mod_and_options(iterations: int, nrestarts: int):
    if "mod" not in _PYSR_STATE:
        from pysr.julia_import import jl  # Julia 启动 + 包加载, 一次性 (秒级)
        _PYSR_STATE["jl"] = jl
        _PYSR_STATE["mod"] = jl.seval(_JULIA_BATCH_MODULE)
        _PYSR_STATE["options"] = {}
    key = (iterations, nrestarts)
    if key not in _PYSR_STATE["options"]:
        _PYSR_STATE["options"][key] = _PYSR_STATE["mod"].make_options(
            iterations=iterations, nrestarts=nrestarts)
    return _PYSR_STATE["mod"], _PYSR_STATE["options"][key]


_PYSR_UNARY = {
    interp.F.SIN: "sin({})", interp.F.COS: "cos({})", interp.F.TAN: "tan({})",
    interp.F.SINH: "sinh({})", interp.F.COSH: "cosh({})", interp.F.TANH: "tanh({})",
    interp.F.LOG: "safe_log({})", interp.F.LOOSE_LOG: "safe_log(abs({}))",
    interp.F.EXP: "exp({})", interp.F.ABS: "abs({})",
    interp.F.SQRT: "safe_sqrt({})", interp.F.LOOSE_SQRT: "safe_sqrt(abs({}))",
    interp.F.INV: "inv({})", interp.F.LOOSE_INV: "inv({})",
    interp.F.NEG: "-({})",
}
_PYSR_BINARY = {
    interp.F.ADD: "({} + {})", interp.F.SUB: "({} - {})",
    interp.F.MUL: "({} * {})",
    interp.F.DIV: "({} / {})", interp.F.LOOSE_DIV: "({} / {})",
    interp.F.POW: "safe_pow({}, {})", interp.F.LOOSE_POW: "safe_pow(abs({}), {})",
    interp.F.MAX: "max({}, {})", interp.F.MIN: "min({}, {})",
}


def build_pysr_expr(nt, nv, ci, c_init):
    """prefix bytecode → Julia 表达式字符串, 常数 baked 成字面量.

    SR.jl 把树里每个数值字面量都当可优化常数 — 所以映射不得引入结构性
    字面量 (NEG 用一元负号, INV 用 inv()). 唯一的坑: Meta.parse 会把
    -(字面量) 折叠成负字面量 → NEG(CONST) 直接 emit 负值, signs 记 -1,
    读回时 c_final[rank] = sign * param. get_scalar_constants 序 = 前序
    DFS = 我们的 ci rank 序 (W3 冒烟验证), Julia 侧另有 baked 初值逐位
    断言兜底. 返回 (expr_str, signs); 不支持的 op 抛 ValueError.
    """
    signs = np.ones(len(c_init))

    def fmt(v):
        return repr(float(np.float32(v)))

    def emit(i):
        t = int(nt[i])
        if t == 0:
            return f"x{int(nv[i]) + 1}"
        if t == 1:
            return fmt(c_init[int(ci[i])])
        fcode = int(nv[i])
        if t == 2:
            if fcode == interp.F.NEG and int(nt[i + 1]) == 1:   # NEG(CONST) 折叠
                rank = int(ci[i + 1])
                signs[rank] = -1.0
                return fmt(-c_init[rank])
            pat = _PYSR_UNARY.get(fcode)
            if pat is None:
                raise ValueError(f"unsupported unary op {fcode}")
            return pat.format(emit(i + 1))
        if t == 3:
            pat = _PYSR_BINARY.get(fcode)
            if pat is None:
                raise ValueError(f"unsupported binary op {fcode}")
            l_start = i + 1
            ls = interp.prefix_span(nt, l_start)
            return pat.format(emit(l_start), emit(l_start + ls))
        raise ValueError(f"unsupported node_type {t}")

    return emit(0), signs


class PySRBFGS:
    """PySR 的 CO 算子 (SymbolicRegression.jl, Optim.jl BFGS + FD 梯度).
    fp64; Julia 侧串行批循环 (juliacall 持 GIL, 线程化留待需要时做)."""

    def __init__(self, iterations: int = 8, nrestarts: int = 2,
                 name: str = "pysr", seed: int = 0):
        self.iterations = iterations
        self.nrestarts = nrestarts
        self.name = name
        self.seed = seed

    def fit_pop(self, pop: dict) -> FitResult:
        t0 = time.perf_counter()
        mod, options = _pysr_mod_and_options(self.iterations, self.nrestarts)
        M = pop["M"]
        c_final = pop["c_init"].astype(np.float64).copy()
        status = np.full(M, 3, np.int32)
        n_iter = np.zeros(M, np.int32)
        idxs, strs, x0s, sign_list = [], [], [], []
        for m in range(M):
            nt, nv, ci, y, K = interp.tree_view(pop, m)
            if K == 0:
                continue
            _, _, c_off, _ = pop["metas"][m].tolist()
            c0 = pop["c_init"][c_off:c_off + K]
            try:
                s, signs = build_pysr_expr(nt, nv, ci, c0)
            except ValueError:
                status[m] = 2
                continue
            idxs.append(m)
            strs.append(s)
            x0s.append(signs * c0.astype(np.float64))   # baked 字面量值
            sign_list.append(signs)
        wall_core = 0.0
        if idxs:
            X = pop["xs"].astype(np.float64).T                       # (nvars, N)
            Y = pop["ym"][idxs].astype(np.float64)                   # (P, N)
            res = mod.fit_batch(strs, X, Y, x0s, options, self.seed)
            wall_core = float(res.wall)
            st = np.asarray(res.status, np.int32)
            ev = np.asarray(res.evals, np.float64)
            for j, m in enumerate(idxs):
                _, _, c_off, K = pop["metas"][m].tolist()
                params = np.asarray(res.consts[j], np.float64)
                if len(params) == K:
                    c_final[c_off:c_off + K] = sign_list[j] * params
                status[m] = st[j]
                n_iter[m] = int(ev[j])
        return FitResult(self.name, c_final, status, n_iter,
                         time.perf_counter() - t0, wall_core,
                         meta=dict(dtype="float64", julia_threads=1,
                                   criteria=f"Optim BFGS (K=1→Newton), FD 梯度, "
                                            f"iters={self.iterations} "
                                            f"nrestarts={self.nrestarts}; "
                                            f"status: 0=improved 1=no-improve; "
                                            f"n_iter=Optim f_calls (不含 FD 探针)"))


# ---------------------------------------------------------------- CudaKernelPop

_KERNEL_TIME_RE = re.compile(r"总耗时 ([\d.]+) 秒")
_KERNEL_STATUS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2}  # 4=FAIL_CHOLESKY → failed


class CudaKernelPop:
    """008 kernel 二进制, subprocess 调用. wall_e2e 含 pop.bin 落盘 + 进程启动 +
    读回 (协议 §5: in-loop 真实成本); wall_core = binary 自报 GPU 段."""

    def __init__(self, binary: Path, max_iter: int | None = None, name: str | None = None):
        self.binary = Path(binary)
        self.max_iter = max_iter
        self.name = name or f"kernel:{self.binary.name}"

    def fit_pop(self, pop: dict) -> FitResult:
        if not self.binary.exists():
            raise KernelRunError(f"binary not found: {self.binary}")
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="opbench_kernel_") as td:
            pop_path = Path(td) / "pop.bin"
            popio.save_pop_bin(pop, pop_path)
            cmd = [str(self.binary), str(pop_path), td]
            if self.max_iter is not None:
                cmd += ["--max-iter", str(self.max_iter)]
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8")
            if r.returncode != 0:
                raise KernelRunError(f"exit={r.returncode}: {(r.stderr or r.stdout)[-500:]}")
            raw_status = np.fromfile(Path(td) / "status.bin", dtype=np.int32)
            c_final = np.fromfile(Path(td) / "c_final.bin", dtype=np.float32).astype(np.float64)
        m_time = _KERNEL_TIME_RE.search(r.stdout or "")
        wall_core = float(m_time.group(1)) if m_time else float("nan")
        status = np.asarray([_KERNEL_STATUS_MAP.get(int(s), 2) for s in raw_status],
                            np.int32)
        raw_counts = {int(k): int(v) for k, v in
                      zip(*np.unique(raw_status, return_counts=True))}
        return FitResult(self.name, c_final, status,
                         np.full(pop["M"], -1, np.int32),
                         time.perf_counter() - t0, wall_core,
                         meta=dict(dtype="float32", binary=str(self.binary),
                                   binary_mtime=self.binary.stat().st_mtime,
                                   raw_status_counts=raw_counts,
                                   criteria="kernel xtol=1e-6 max_iter=50 (内建)"))
