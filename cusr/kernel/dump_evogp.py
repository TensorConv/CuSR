"""dump_evogp.py — 跑 EvoGP N 代, dump 当代 forest 成 pop.bin (batch_lm 的输入).

⚠️ 需要 EvoGP + PyTorch 环境 (见 README 的 "重新生成 pop.bin" 一节).
仓库自带的 data/pop.bin 是用本脚本预跑好的, 不重新 dump 也能跑 ./batch_lm + verify.py.

自己 spin up `GeneticProgramming` + `StandardPipeline` 跑 N 代, 跑完遍历
`algorithm.forest[i]` 把每棵 Tree 转成 pop.bin 里的扁平数组.

Dump 时的处理:
1. LOOSE_DIV/LOG/INV/POW/SQRT → 对应的标准 op (跟 tree_interpreter.py 的 LOOSE_*
   退化注释一致). 解算端不再需要识别 LOOSE_*.
2. 含 TFUNC (IF) 节点的树跳过 + 计数. 当前算法不支持三元节点.
3. 每棵树计算 K (CONST 数), c_init (从 node_value 取 CONST 位置), max_stack
   (栈模拟).

输出 binary 跟 pop_format.h 一致, little-endian, 字段顺序见该 header.

题目库 (PROBLEMS dict 里) 内置了 Feynman SR Database 的几道题, 想加新题直接改这个
dict 即可 (skeleton_expr 用 sympy 语法; 见现有条目).

CLI:
    uv run python dump_evogp.py --dataset=feynman/I.18.12 --gen=20 --pop=1000 \\
        -o data/pop.bin [--N=1000] [--seed=0]
"""
from __future__ import annotations

import argparse
import datetime
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# EvoGP imports — 触发 CUDA 初始化, 失败时 import 阶段就会暴露
from evogp.algorithm import (
    DefaultCrossover, DefaultMutation, DefaultSelection, GeneticProgramming,
)
from evogp.pipeline import StandardPipeline
from evogp.problem import SymbolicRegression
from evogp.tree import Forest, GenerateDescriptor
from evogp.tree.utils import Func, NType

USING_FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0, "sin": 0.5, "cos": 0.5, "tan": 0.5}
CONST_SAMPLES = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]

# EvoGP run config — single source of truth; recorded verbatim into each snapshot
# manifest so a harvest is fully reproducible from the committed JSON alone.
GP_CONFIG = {
    "max_tree_len": 32,
    "init_max_layer_cnt": 4,
    "mutation_rate": 0.2,
    "mutation_max_layer_cnt": 3,
    "survival_rate": 0.3,
    "elite_rate": 0.01,
    "using_funcs": USING_FUNCS,
    "const_samples": CONST_SAMPLES,
}


# 题目库 — Feynman Symbolic Regression Database 的子集 (Udrescu & Tegmark 2020).
# 想加题目把 (id, skeleton_expr, variables, constants, ground_truth_constants,
# sampling_ranges) 加进来即可. 仓库自带的 pop.bin 是用 I.18.12 跑出来的;
# 其它几道附在这里是为了方便友人试不同 K + 不同算子组合.
PROBLEMS = {
    "feynman/I.12.1": {
        # F = μ·N  (最简单的 K=1, 纯乘法)
        "skeleton_expr": "c0*x0*x1",
        "variables": ["x0", "x1"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0)],
        "_note": "I.12.1  mu*Nn  — K=1 baseline",
    },
    "feynman/I.18.12": {
        # τ = r·F·sin(θ)  (默认题, 仓库 pop.bin 来源)
        "skeleton_expr": "c0*x0*x1*sin(x2)",
        "variables": ["x0", "x1", "x2"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (0.0, 5.0)],
        "_note": "I.18.12  r*F*sin(theta)  — torque magnitude, K=1 with SIN",
    },
    "feynman/I.27.6": {
        # 薄透镜公式  1 / (1/d1 + n/d2)  (嵌套 DIV)
        "skeleton_expr": "c0/(x2/x1 + 1/x0)",
        "variables": ["x0", "x1", "x2"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        "_note": "I.27.6  1/(1/d1 + n/d2)  — thin-lens, K=1 with nested DIV",
    },
    "feynman/I.6.2": {
        # 标准正态  exp(-(θ/σ)²/2) / (√(2π)·σ)  (高 K, SQRT + EXP + 平方)
        "skeleton_expr": "c0*sqrt(c1)*exp(c3*x1**2/x0**2)/(sqrt(c2)*x0)",
        "variables": ["x0", "x1"],
        "constants": ["c0", "c1", "c2", "c3"],
        "ground_truth_constants": [0.5, 2.0, 3.141592653589793, -0.5],
        "sampling_ranges": [(1.0, 3.0), (1.0, 3.0)],
        "_note": "I.6.2  Gaussian exp(-(theta/sigma)^2/2)/(sqrt(2pi)*sigma)  — K=4, SQRT+EXP+POW",
    },

    # --- more multivariate Feynman (K=1, div/inv-heavy; different n_vars) ---
    "feynman/I.12.2": {
        # Coulomb F = q1*q2/(4π ε0 r²)
        "skeleton_expr": "c0*x0*x1/x2**2",
        "variables": ["x0", "x1", "x2"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        "_note": "I.12.2  Coulomb q1*q2/(4pi eps r^2)  — 3 var, K=1, DIV+sq",
    },
    "feynman/I.13.12": {
        # gravitational PE  U = G*m1*m2*(1/r2 - 1/r1)
        "skeleton_expr": "c0*x0*x1*(1/x3 - 1/x2)",
        "variables": ["x0", "x1", "x2", "x3"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        "_note": "I.13.12  G*m1*m2*(1/r2-1/r1)  — 4 var, K=1, INV",
    },
    "feynman/II.3.24": {
        # radiated flux  φ = P/(4π r²)
        "skeleton_expr": "c0*x0/x1**2",
        "variables": ["x0", "x1"],
        "constants": ["c0"],
        "ground_truth_constants": [1.0],
        "sampling_ranges": [(1.0, 5.0), (1.0, 5.0)],
        "_note": "II.3.24  flux P/(4pi r^2)  — 2 var, K=1, DIV+sq",
    },

    # --- Nguyen GP-SR benchmark (Uy et al. 2011). Targets are constant-free
    # (K=0 in ground truth); the GP discovers constants, so harvested pops still
    # have K>0 to optimize. GP funcset = {+,-,*,/,sin,cos,tan}; log/sqrt targets
    # (N7/N8) are only approximable — kept for workload variety, same situation as
    # the SQRT/EXP Feynman entries (skeleton only generates y, not a GP constraint). ---
    "nguyen/1": {"skeleton_expr": "x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-1  x^3+x^2+x"},
    "nguyen/2": {"skeleton_expr": "x0**4 + x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-2  x^4+..+x"},
    "nguyen/3": {"skeleton_expr": "x0**5 + x0**4 + x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-3  x^5+..+x"},
    "nguyen/4": {"skeleton_expr": "x0**6 + x0**5 + x0**4 + x0**3 + x0**2 + x0", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-4  x^6+..+x"},
    "nguyen/5": {"skeleton_expr": "sin(x0**2)*cos(x0) - 1", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-5  sin(x^2)cos(x)-1"},
    "nguyen/6": {"skeleton_expr": "sin(x0) + sin(x0 + x0**2)", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(-1.0, 1.0)], "_note": "Nguyen-6  sin(x)+sin(x+x^2)"},
    "nguyen/7": {"skeleton_expr": "log(x0 + 1) + log(x0**2 + 1)", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(0.0, 2.0)], "_note": "Nguyen-7  ln(x+1)+ln(x^2+1) (approx)"},
    "nguyen/8": {"skeleton_expr": "sqrt(x0)", "variables": ["x0"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(0.0, 4.0)], "_note": "Nguyen-8  sqrt(x) (approx)"},
    "nguyen/9": {"skeleton_expr": "sin(x0) + sin(x1**2)", "variables": ["x0", "x1"],
                 "constants": [], "ground_truth_constants": [],
                 "sampling_ranges": [(0.0, 1.0), (0.0, 1.0)], "_note": "Nguyen-9  sin(x)+sin(y^2)"},
    "nguyen/10": {"skeleton_expr": "2*sin(x0)*cos(x1)", "variables": ["x0", "x1"],
                  "constants": [], "ground_truth_constants": [],
                  "sampling_ranges": [(0.0, 1.0), (0.0, 1.0)], "_note": "Nguyen-10  2sin(x)cos(y)"},
}


# LOOSE → 标准 op 退化映射. dump 端做完, batch_lm 端不识别 LOOSE_*.
DEGRADE_OP = {
    Func.LOOSE_DIV:  Func.DIV,
    Func.LOOSE_LOG:  Func.LOG,
    Func.LOOSE_INV:  Func.INV,
    Func.LOOSE_POW:  Func.POW,
    Func.LOOSE_SQRT: Func.SQRT,
}


# pop_format.h: POP_MAGIC = 0x4D4C344D ('M','4','L','M' 4 字节 LE)
POP_MAGIC = 0x4D4C344D
POP_VERSION = 1


def _load_feynman_problem(dataset_id: str) -> dict:
    """查 PROBLEMS dict 拿 skeleton + 范围 + GT 常数."""
    if dataset_id not in PROBLEMS:
        raise KeyError(
            f"{dataset_id} not in PROBLEMS. Known: {sorted(PROBLEMS.keys())}. "
            f"加新题目: 在 dump_evogp.py 的 PROBLEMS dict 里加一条."
        )
    return PROBLEMS[dataset_id]


def _sample_xy(prob: dict, N: int, seed: int, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """uniform sample N 个数据点, y = skeleton(c_true, X). noise>0: 加 RMS 相对高斯噪声."""
    import sympy as sp
    rng = np.random.default_rng(seed)
    n_vars = len(prob["variables"])
    X = np.empty((N, n_vars), dtype=np.float64)
    for i, (lo, hi) in enumerate(prob["sampling_ranges"]):
        X[:, i] = rng.uniform(lo, hi, size=N)
    var_syms = tuple(sp.Symbol(v) for v in prob["variables"])
    const_syms = tuple(sp.Symbol(c) for c in prob["constants"])
    locals_ = {str(s): s for s in (*var_syms, *const_syms)}
    expr = sp.sympify(prob["skeleton_expr"], locals=locals_)
    subs = {c: float(v) for c, v in zip(const_syms, prob["ground_truth_constants"])}
    expr = expr.subs(subs)
    f = sp.lambdify(var_syms, expr, modules="numpy")
    y = np.asarray(f(*X.T), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError("sampled y has non-finite — check ranges")
    if noise > 0.0:
        sigma = noise * float(np.sqrt(np.mean(y ** 2)))  # RMS-relative Gaussian noise
        y = y + rng.normal(0.0, sigma, size=y.shape)
    return X.astype(np.float32), y.astype(np.float32)


def _compute_stack_depth(node_type: np.ndarray) -> int:
    """模拟 reverse-prefix stack walk 算 max stack depth.

    跟 tree_interpreter.py 同算法: var/const push, ufunc/bfunc/tfunc pop k push 1.
    """
    n = len(node_type)
    sp = 0
    max_sp = 0
    for i in reversed(range(n)):
        t = int(node_type[i])
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


def _has_tfunc(node_type: np.ndarray) -> bool:
    return bool((node_type == NType.TFUNC).any())


def _extract_tree(tree, problem_n_vars: int) -> tuple | None:
    """读一棵 Tree, 返回 (nt[], nv[], ci[], c_init[]) 或 None (TFUNC skip).

    nt: int32 array of node types (TYPE_MASK 已应用 — TFUNC_OUT/BFUNC_OUT/UFUNC_OUT 都
        映射回基础类型, 因为我们的 interpreter 不区分 OUT bit).
    nv: float32 array of node values (operator id 或 var index, 跟 m1 一致). LOOSE_*
        已退化.
    ci: int32 array, -1 if not CONST else c_vec index (forward-prefix rank of CONSTs).
    c_init: float32 array of CONST values 从 node_value 抠出来, 长度 K.

    返回 None: 含 TFUNC 节点跳过.
    """
    n = int(tree.subtree_size[0].item())
    nt_raw = tree.node_type[:n].detach().cpu().numpy().astype(np.int32)
    nv_raw = tree.node_value[:n].detach().cpu().numpy().astype(np.float32)
    # 屏蔽 OUT bit, 跟 EvoGP 内部 TYPE_MASK = 0x7F 一致
    nt = (nt_raw & NType.TYPE_MASK).astype(np.int32)
    if _has_tfunc(nt):
        return None

    # degrade LOOSE_* → 标准 op (按 tree_interpreter 同语义)
    nv = nv_raw.copy()
    for i in range(n):
        t = int(nt[i])
        if t == NType.UFUNC or t == NType.BFUNC:
            fid = int(nv[i])
            if fid in DEGRADE_OP:
                nv[i] = float(DEGRADE_OP[fid])

    # CONST index map: prefix-forward rank
    ci = np.full(n, -1, dtype=np.int32)
    c_init = []
    k = 0
    for i in range(n):
        if int(nt[i]) == NType.CONST:
            ci[i] = k
            c_init.append(float(nv_raw[i]))  # CONST 值在原始 node_value 里
            k += 1
    c_init_arr = np.asarray(c_init, dtype=np.float32)

    return nt, nv, ci, c_init_arr


def _build_evogp(X_t: torch.Tensor, y_t: torch.Tensor, *, pop: int, n_vars: int, seed: int,
                 max_tree_len: int | None = None):
    max_tree_len = max_tree_len or GP_CONFIG["max_tree_len"]
    desc = GenerateDescriptor(
        max_tree_len=max_tree_len, input_len=n_vars, output_len=1,
        using_funcs=USING_FUNCS, max_layer_cnt=GP_CONFIG["init_max_layer_cnt"],
        const_samples=CONST_SAMPLES,
    )
    torch.manual_seed(seed); np.random.seed(seed)
    forest = Forest.random_generate(pop_size=pop, descriptor=desc)
    problem = SymbolicRegression(datapoints=X_t, labels=y_t)
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=GP_CONFIG["mutation_rate"],
                                 descriptor=desc.update(max_layer_cnt=GP_CONFIG["mutation_max_layer_cnt"])),
        selection=DefaultSelection(survival_rate=GP_CONFIG["survival_rate"],
                                   elite_rate=GP_CONFIG["elite_rate"]),
    )
    pipeline = StandardPipeline(
        algorithm=algo, problem=problem,
        generation_limit=1,  # 我们自己手动 step
        is_show_details=False,
    )
    return algo, pipeline


def _write_pop_bin(
    out: Path,
    trees: list[tuple],  # [(nt, nv, ci, c_init), ...]
    X: np.ndarray, y: np.ndarray,
) -> dict:
    """按 pop_format.h 顺序写文件. 返回 stats dict 给 caller 打印."""
    M_prob = len(trees)
    total_nodes = sum(len(t[0]) for t in trees)
    total_c = sum(len(t[3]) for t in trees)
    N, n_vars = X.shape
    K_max = max((len(t[3]) for t in trees), default=0)
    max_stack = max((_compute_stack_depth(t[0]) for t in trees), default=0)

    # build big flat arrays
    nt_all = np.concatenate([t[0] for t in trees]) if trees else np.array([], np.int32)
    nv_all = np.concatenate([t[1] for t in trees]) if trees else np.array([], np.float32)
    ci_all = np.concatenate([t[2] for t in trees]) if trees else np.array([], np.int32)
    c_all = np.concatenate([t[3] for t in trees]) if trees else np.array([], np.float32)

    # metas
    metas = np.zeros((M_prob, 4), dtype=np.int32)  # node_offset, n_nodes, c_offset, K
    nod_off = 0; c_off = 0
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
        # ym layout: [M_prob, N] per-tree. Real EvoGP 跑里所有树共享同一份 y, 复制 M_prob
        # 份. fixture 路径会 per-tree 不同 y. pop_format.h §"ym layout" 有说明.
        ym_per_tree = np.tile(y.astype(np.float32), M_prob)
        f.write(ym_per_tree.tobytes())

    return dict(M_prob=M_prob, total_nodes=total_nodes, total_c=total_c,
                N=N, n_vars=n_vars, K_max=K_max, max_stack=max_stack)


def _extract_forest(forest, pop: int, n_vars: int) -> tuple[list[tuple], int, int]:
    """遍历 forest 抽全部树, 返回 (trees, n_tfunc_skip, n_kover)."""
    trees = []
    n_tfunc_skip = 0
    n_kover = 0
    K_compile = 32  # 跟 batch_lm 的 MAX_K 一致
    for i in range(pop):
        tree = forest[i]
        result = _extract_tree(tree, n_vars)
        if result is None:
            n_tfunc_skip += 1
            continue
        nt, nv, ci, c_init = result
        if len(c_init) > K_compile:
            n_kover += 1
            continue
        trees.append((nt, nv, ci, c_init))
    return trees, n_tfunc_skip, n_kover


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="feynman/I.18.12")
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--pop", type=int, default=1000)
    ap.add_argument("--N", type=int, default=1000, help="data points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).resolve().parent / "data" / "pop.bin")
    ap.add_argument("--checkpoint-every", type=int, default=0, metavar="G",
                    help="每 G 代额外 dump 一份 pop_gen{g:04d}.bin 到 --out 同目录 "
                         "(含 gen 0 初始种群); 0=关 (默认, 只 dump 末代)")
    ap.add_argument("--checkpoint-gens", default="", metavar="LIST",
                    help="显式 checkpoint 世代列表 (逗号分隔, e.g. 0,1,2,4,8,16,32,64,100); "
                         "给了就覆盖 --checkpoint-every, 并跑到列表最大世代")
    ap.add_argument("--noise", type=float, default=0.0, metavar="REL",
                    help="加 RMS 相对高斯噪声 (e.g. 0.01 = 1%%); 0=无噪 (默认)")
    ap.add_argument("--max-tree-len", type=int, default=0, metavar="N",
                    help="覆盖 GP_CONFIG.max_tree_len (节点上限); 0=用默认")
    args = ap.parse_args(argv)

    ckpt_gens = sorted({int(x) for x in args.checkpoint_gens.split(",") if x.strip()}) or None
    total_gens = max(ckpt_gens) if ckpt_gens else args.gen
    max_tree_len = args.max_tree_len or GP_CONFIG["max_tree_len"]
    print(f"[dump] dataset={args.dataset}  gens={total_gens}  pop={args.pop}  N={args.N}  "
          f"seed={args.seed}  noise={args.noise}  max_tree_len={max_tree_len}  "
          f"ckpt={ckpt_gens if ckpt_gens else ('every %d' % args.checkpoint_every)}", flush=True)

    # ---- data ----
    prob = _load_feynman_problem(args.dataset)
    n_vars = len(prob["variables"])
    X_np, y_np = _sample_xy(prob, args.N, args.seed, noise=args.noise)
    X_t = torch.from_numpy(np.ascontiguousarray(X_np)).cuda()
    y_t = torch.from_numpy(np.ascontiguousarray(y_np.reshape(-1, 1))).cuda()

    # ---- EvoGP run ----
    algo, pipeline = _build_evogp(X_t, y_t, pop=args.pop, n_vars=n_vars, seed=args.seed,
                                  max_tree_len=max_tree_len)

    snap_records = []  # per-snapshot workload stats -> manifest.json

    def _snap_record(g, out_path, trees, n_tf, n_ko, stats):
        Ks = np.array([len(t[3]) for t in trees], dtype=float)
        Ns = np.array([len(t[0]) for t in trees], dtype=float)
        return dict(gen=g, file=out_path.name, M=stats["M_prob"],
                    total_nodes=stats["total_nodes"], total_c=stats["total_c"],
                    N=stats["N"], n_vars=stats["n_vars"], K_max=stats["K_max"],
                    max_stack=stats["max_stack"],
                    mean_K=round(float(Ks.mean()), 3) if len(Ks) else 0.0,
                    mean_nodes=round(float(Ns.mean()), 3) if len(Ns) else 0.0,
                    max_nodes=int(Ns.max()) if len(Ns) else 0,
                    n_tfunc_skip=n_tf, n_kover=n_ko, bytes=out_path.stat().st_size)

    def _checkpoint(g: int):
        """dump 当前 algo.forest 成 pop_gen{g:04d}.bin (g = 已完成代数)."""
        ck_out = args.out.parent / f"pop_gen{g:04d}.bin"
        ck_trees, ck_tf, ck_ko = _extract_forest(algo.forest, args.pop, n_vars)
        ck_stats = _write_pop_bin(ck_out, ck_trees, X_np, y_np)
        snap_records.append(_snap_record(g, ck_out, ck_trees, ck_tf, ck_ko, ck_stats))
        print(f"[dump] checkpoint gen={g}: kept {len(ck_trees)}/{args.pop} "
              f"(TFUNC-skip {ck_tf}, K-over {ck_ko})  total_nodes={ck_stats['total_nodes']}  "
              f"K_max={ck_stats['K_max']}  -> {ck_out}", flush=True)

    def want_ckpt(g):
        return (g in ckpt_gens) if ckpt_gens else (
            args.checkpoint_every > 0 and (g == 0 or g % args.checkpoint_every == 0))

    if want_ckpt(0):
        _checkpoint(0)  # 初始种群
    for gen in range(total_gens):
        pipeline.step()
        if want_ckpt(gen + 1):
            _checkpoint(gen + 1)
    best_fitness = float(pipeline.best_fitness)
    print(f"[dump] EvoGP ran {total_gens} gens, best_fitness={best_fitness:.4e}", flush=True)

    # ---- per-tree extract ----
    trees, n_tfunc_skip, n_kover = _extract_forest(algo.forest, args.pop, n_vars)
    K_compile = 32
    print(f"[dump] kept {len(trees)} / {args.pop} trees  (TFUNC-skipped: {n_tfunc_skip}, K>{K_compile}-skipped: {n_kover})", flush=True)

    # ---- write ----
    stats = _write_pop_bin(args.out, trees, X_np, y_np)
    print(f"[dump] wrote {args.out}  ({args.out.stat().st_size:,} bytes)", flush=True)
    print(f"[dump] stats: M_prob={stats['M_prob']}  total_nodes={stats['total_nodes']}  "
          f"total_c={stats['total_c']}  N={stats['N']}  n_vars={stats['n_vars']}  "
          f"K_max={stats['K_max']}  max_stack={stats['max_stack']}", flush=True)

    # final pop.bin as its own snapshot record (unless it's already a checkpoint)
    if not want_ckpt(total_gens):
        snap_records.append(_snap_record(total_gens, args.out, trees, n_tfunc_skip, n_kover, stats))

    # sidecar: tfunc skip 计数 (inspect 不重跑 evogp, 它只看 .bin)
    sidecar = args.out.with_suffix(args.out.suffix + ".meta.txt")
    sidecar.write_text(
        f"dataset={args.dataset}\ngen={total_gens}\npop_requested={args.pop}\n"
        f"n_tfunc_skipped={n_tfunc_skip}\nn_kover_skipped={n_kover}\n"
        f"n_kept={len(trees)}\nseed={args.seed}\n"
    )
    print(f"[dump] sidecar: {sidecar}", flush=True)

    # manifest.json — committed reproducibility record (the .bin files are gitignored).
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=args.out.parent, text=True).strip()
    except Exception:
        git_sha = "unknown"
    prob_rec = {k: prob[k] for k in
                ("skeleton_expr", "variables", "constants",
                 "ground_truth_constants", "sampling_ranges") if k in prob}
    evogp_cfg = dict(GP_CONFIG); evogp_cfg["max_tree_len"] = max_tree_len
    manifest = dict(
        harvest_date=datetime.date.today().isoformat(), git_sha=git_sha,
        dataset=args.dataset, problem=prob_rec, N=args.N, noise=args.noise,
        seed=args.seed, pop=args.pop, gens_run=total_gens, max_tree_len=max_tree_len,
        checkpoint_gens=ckpt_gens, checkpoint_every=args.checkpoint_every,
        evogp_config=evogp_cfg, best_fitness=best_fitness, snapshots=snap_records,
    )
    manifest_path = args.out.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[dump] manifest: {manifest_path}  ({len(snap_records)} snapshots)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
