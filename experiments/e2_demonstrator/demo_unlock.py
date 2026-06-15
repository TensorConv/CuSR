"""demo_unlock.py — kernel-CO "解锁能力" 演示 (HPEC demonstrator, 构造题).

构造一撮目标, 关键常数 (频率/衰减率/系数) 全落在 EvoGP 离散常数表
{0, ±1, ±2, 0.5} **之外** → stock GP (p=0) 的 terminal/mutation 够不到那个常数,
结构对了也拟不准 → 找不回; kernel-CO (p=1, 008 batch_lm via bridge) 把内部常数拟出来
→ 找回. 直接展示 "kernel 在环里解锁了 stock GP 拿不到的发现".

口径: 这是**按构造对 CO 有利**的 demonstrator (实验没人做过, cherry-pick 由用户定),
不是 SR benchmark 成绩. 真实 benchmark 是另一回事, 不在这里报.

关键: 用**离散常数表** (不是 run_bench/run_pilot 的 [-5,5]×10000 富采样 —— 那个 stock
能撞到常数, 对比就没了). backend = 真 kernel CudaKernelLM, 不是 TorchLM.

跑: source scripts/env.sh
    uv run python experiments/009_sr_benchmark/demo_unlock.py --smoke      # 1 题×1 seed×2 臂
    uv run python experiments/009_sr_benchmark/demo_unlock.py --seeds 5    # 全套
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent

from cusr.demonstrator import judge
from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.co_backend import CudaKernelLM
from cusr.demonstrator.pipeline import MemeticPipeline

# EvoGP 默认离散常数表: 表外常数 stock 够不到, 只有 CO 能拟到
CONST_PALETTE = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]
FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0, "sin": 0.5, "cos": 0.5, "exp": 0.3, "sqrt": 0.3}
POP, N_GEN, TOP_K = 600, 60, 16

# 构造题: 关键常数全在 CONST_PALETTE 之外 (2.5 / 1.3 / 3.0 / 0.7 / 3.5 / 1.7 / 0.6 ...)
TARGETS = [
    dict(id="u_sin25",    expr="2.0*sin(2.5*x0)",          nv=1, rng=((0.0, 3.0),),            nc=2, ni=1),
    dict(id="u_sincos",   expr="sin(2.5*x0)*cos(1.3*x1)",  nv=2, rng=((0.0, 3.0), (0.0, 3.0)), nc=2, ni=2),
    dict(id="u_expdecay", expr="3.0*exp(-0.7*x0)",         nv=1, rng=((0.0, 3.0),),            nc=2, ni=1),
    dict(id="u_quad",     expr="x0*x0 + 3.5*x0",           nv=1, rng=((-3.0, 3.0),),           nc=1, ni=0),
    dict(id="u_sinphase", expr="sin(0.7*x0 + 1.3)",        nv=1, rng=((0.0, 6.0),),            nc=2, ni=2),
    dict(id="u_linear",   expr="1.7*x0 + 0.6",             nv=1, rng=((-3.0, 3.0),),           nc=2, ni=0),
]


def make_prob(t):
    return sb.Problem(id=t["id"], source="constructed", true_expr=t["expr"], n_vars=t["nv"],
                      var_ranges=t["rng"], n_consts=t["nc"], n_inner_consts=t["ni"],
                      difficulty="demo", is_control=False)


def run(prob, backend, co_on, seed):
    from evogp.algorithm import (DefaultCrossover, DefaultMutation,
                                 DefaultSelection, GeneticProgramming)
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    X_np, y_np = sb.generate(prob, seed=seed, n_samples=200)
    torch.manual_seed(seed)
    np.random.seed(seed)
    desc = GenerateDescriptor(max_tree_len=32, input_len=prob.n_vars, output_len=1,
                              using_funcs=FUNCS, max_layer_cnt=4, const_samples=CONST_PALETTE)
    forest = Forest.random_generate(pop_size=POP, descriptor=desc)
    problem = SymbolicRegression(
        datapoints=torch.from_numpy(X_np.astype(np.float32)).cuda(),
        labels=torch.from_numpy(y_np.astype(np.float32).reshape(-1, 1)).cuda())
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.2, descriptor=desc.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01))
    p = MemeticPipeline(algo, problem, backend, problem_n_vars=prob.n_vars, X_np=X_np, y_np=y_np,
                        top_k=TOP_K, co_every=2, co_probability=(1.0 if co_on else 0.0),
                        co_rng_seed=seed, generation_limit=N_GEN)
    for _ in range(N_GEN):
        p.step()
    rec, expr = False, None
    if p.best_tree is not None:
        try:
            be = p.best_tree.to_sympy_expr()
            expr = str(be)
            rec = bool(judge.recovery_columns(be, prob.true_expr, X=X_np)["lenient"])
        except Exception as e:  # noqa: BLE001
            expr = f"<err: {e}>"
    return dict(best_fit=float(p.best_fitness), rec=rec, expr=expr,
                co_fits=p.n_co_fits, gens=N_GEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 题 × 1 seed × 2 臂")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=50)
    args = ap.parse_args()

    targets = TARGETS[:1] if args.smoke else TARGETS
    n_seeds = 1 if args.smoke else args.seeds
    backend = CudaKernelLM(max_iter=args.max_iter)

    out = HERE / "runs" / "demo_unlock.jsonl"
    out.parent.mkdir(exist_ok=True)
    sink = out.open("w")
    print(f"kernel-CO 解锁演示  题={[t['id'] for t in targets]}  臂 p=[0,1]  seeds={n_seeds}")
    print(f"  (离散常数表 {CONST_PALETTE} → 表外常数 stock 够不到; backend=真 kernel)\n")
    print(f"{'id':12}{'n_in':>5}{'p=0 rec':>9}{'p=1 rec':>9}{'unlock':>8}   true_expr")

    summary = []
    for t in targets:
        prob = make_prob(t)
        rec0 = rec1 = 0
        for seed in range(n_seeds):
            backend.cum_stats.clear()
            r0 = run(prob, backend, co_on=False, seed=seed)
            r1 = run(prob, backend, co_on=True, seed=seed)
            rec0 += int(r0["rec"]); rec1 += int(r1["rec"])
            for arm, r in (("p0", r0), ("p1", r1)):
                rec = dict(id=t["id"], arm=arm, seed=seed, **r, true_expr=t["expr"])
                sink.write(json.dumps(rec, default=str) + "\n"); sink.flush()
        unlocked = "✓" if (rec1 > rec0) else ("=" if rec1 == rec0 else "✗")
        summary.append((t["id"], t["ni"], rec0, rec1, n_seeds))
        print(f"{t['id']:12}{t['ni']:>5}{f'{rec0}/{n_seeds}':>9}{f'{rec1}/{n_seeds}':>9}{unlocked:>8}   {t['expr']}")

    tot0 = sum(s[2] for s in summary); tot1 = sum(s[3] for s in summary)
    tot = sum(s[4] for s in summary)
    print(f"\n  stock (p=0): {tot0}/{tot} recovered   kernel-CO (p=1): {tot1}/{tot} recovered")
    print(f"  → kernel-CO 解锁 {tot1 - tot0}/{tot} (这些题 stock 因常数在表外必然找不回)")
    print(f"\nwrote {out}")
    sink.close()


if __name__ == "__main__":
    main()
