"""Smoke test: existing EvoGP memetic pipeline -> 009 judge, end-to-end.

Proves the whole loop runs IN THIS ENV before any GPU-LM work:
  EvoGP evolve (CUDA fitness) -> per-gen extract skeleton + CPU scipy LM
  (_MemeticTopKPipeline) -> best tree -> sympy -> judge.recovered.

Pick an easy problem we EXPECT to recover (Nguyen-1, +-*/ only) so a True
verdict confirms the judge wiring isn't silently dead. The CPU LM here is the
thing Stage 2 will swap for a GPU batched LM; this isolates that as the only
remaining unknown.

Run:  uv run python experiments/009_sr_benchmark/smoke_evogp.py
"""
from __future__ import annotations

import time

import numpy as np
import sympy
import torch

from cusr.demonstrator import judge
from cusr.demonstrator import seed_bench as sb
from cusr.bench.sources.evogp import _MemeticTopKPipeline
from evogp.algorithm import (
    DefaultCrossover, DefaultMutation, DefaultSelection, GeneticProgramming,
)
from evogp.problem import SymbolicRegression  # noqa: E402
from evogp.tree import Forest, GenerateDescriptor  # noqa: E402

POP, N_GEN, TOP_K = 500, 40, 16
NLS_EVERY, NLS_MAX_NFEV = 1, 100
FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0}  # start simple, no transcendentals
CONST_SAMPLES = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5]
EARLY_STOP_R2 = 0.9999
SEED = 0


def main():
    prob = {q.id: q for q in sb.SEED}["nguyen_1"]
    X_np, y_np = sb.generate(prob, seed=SEED, n_samples=200)
    print(f"problem={prob.id}  true={prob.true_expr}  n={len(y_np)}  vars={prob.n_vars}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    desc = GenerateDescriptor(
        max_tree_len=32, input_len=prob.n_vars, output_len=1,
        using_funcs=FUNCS, max_layer_cnt=4, const_samples=CONST_SAMPLES,
    )
    forest = Forest.random_generate(pop_size=POP, descriptor=desc)
    X_t = torch.from_numpy(np.ascontiguousarray(X_np.astype(np.float32))).cuda()
    y_t = torch.from_numpy(np.ascontiguousarray(y_np.astype(np.float32).reshape(-1, 1))).cuda()
    problem = SymbolicRegression(datapoints=X_t, labels=y_t)
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.2, descriptor=desc.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )
    p = _MemeticTopKPipeline(
        algorithm=algo, problem=problem,
        top_k=TOP_K, problem_n_vars=prob.n_vars,
        X_np=X_np, y_np=y_np,
        nls_max_nfev=NLS_MAX_NFEV, nls_every=NLS_EVERY,
        generation_limit=N_GEN, is_show_details=False,
    )

    var_y = float(np.var(y_np))
    early = -(1.0 - EARLY_STOP_R2) * var_y
    t0 = time.time()
    gens_run = 0
    for _ in range(N_GEN):
        p.step()
        gens_run += 1
        if p.best_fitness >= early:
            break
    wall = time.time() - t0

    best_expr = None
    cols = {"strict": False, "lenient": False, "srbench": False}
    if p.best_tree is not None:
        try:
            be = p.best_tree.to_sympy_expr()
            best_expr = str(be)
            cols = judge.recovery_columns(be, prob.true_expr, X=X_np)
        except Exception as e:  # noqa: BLE001
            best_expr = f"<err: {e}>"

    # rough compute accounting
    n_nls = len(p.nls_records)
    nls_iters = sum((r.get("n_iter") or 0) for r in p.nls_records)
    fitness_evals = gens_run * POP

    print(f"\n--- result ---")
    print(f"gens_run={gens_run}  wall={wall:.1f}s  best_fitness={float(p.best_fitness):.3e}")
    print(f"best_expr = {best_expr}")
    print(f"recovered: strict={cols['strict']}  lenient={cols['lenient']}  srbench={cols['srbench']}")
    print(f"compute: fitness_evals~{fitness_evals}  nls_calls={n_nls}  nls_total_iter={nls_iters}")
    ok = cols["lenient"]
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'} (expected recovery on Nguyen-1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
