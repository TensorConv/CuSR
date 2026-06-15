"""Integration test: 009 MemeticPipeline drives a CO backend IN the EvoGP loop.

This is the actual "接入" — the GPU LM (TorchLM) running per-generation inside
EvoGP, not just as a standalone backend. Proves it on Nguyen-1 (easy, expect
recovery) for BOTH backends, so the pipeline is confirmed backend-agnostic.

Needs a GPU (EvoGP fitness eval). Run:
  uv run pytest experiments/009_sr_benchmark/test_pipeline.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cusr.demonstrator import judge
from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.co_backend import ScipyLM, TorchLM
from cusr.demonstrator.pipeline import MemeticPipeline

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU for EvoGP")

BACKENDS = [ScipyLM(), TorchLM(dtype="float64")]
IDS = [b.name for b in BACKENDS]


def _run(backend, prob, *, pop=500, n_gen=40, top_k=16, seed=0):
    from evogp.algorithm import (DefaultCrossover, DefaultMutation,
                                 DefaultSelection, GeneticProgramming)
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    X_np, y_np = sb.generate(prob, seed=seed, n_samples=200)
    torch.manual_seed(seed)
    np.random.seed(seed)
    desc = GenerateDescriptor(
        max_tree_len=32, input_len=prob.n_vars, output_len=1,
        using_funcs={"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0},
        max_layer_cnt=4, const_samples=[0.0, 1.0, -1.0, 2.0, -2.0, 0.5],
    )
    forest = Forest.random_generate(pop_size=pop, descriptor=desc)
    problem = SymbolicRegression(
        datapoints=torch.from_numpy(X_np.astype(np.float32)).cuda(),
        labels=torch.from_numpy(y_np.astype(np.float32).reshape(-1, 1)).cuda(),
    )
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.2, descriptor=desc.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )
    p = MemeticPipeline(
        algo, problem, backend,
        problem_n_vars=prob.n_vars, X_np=X_np, y_np=y_np,
        top_k=top_k, co_every=1, generation_limit=n_gen,
    )
    early = -(1.0 - 0.9999) * float(np.var(y_np))
    for _ in range(n_gen):
        p.step()
        if p.best_fitness >= early:
            break
    return p, X_np


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_nguyen1_recovers_in_loop(backend):
    prob = {q.id: q for q in sb.SEED}["nguyen_1"]
    p, X_np = _run(backend, prob)
    assert p.best_tree is not None
    be = p.best_tree.to_sympy_expr()
    assert judge.recovered(be, prob.true_expr, X=X_np) is True, f"best={be}"
    assert p.n_co_fits > 0  # the CO backend was actually exercised in-loop
