"""End-to-end benchmark runner: catalogue problem -> generate -> EvoGP+CO
(MemeticPipeline) -> calibrated judge -> recorded row.

This is the harness that produces recovery numbers. A pilot (a few problems,
one backend) is enough to prove the whole chain runs and the table is sensible;
the full sweep (all 113 problems x seeds x baseline-vs-recipe arms) is a bigger
experiment run.

Usage:
  uv run python experiments/009_sr_benchmark/run_bench.py            # pilot
  uv run python experiments/009_sr_benchmark/run_bench.py id1 id2 .. # specific ids
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent

from cusr.demonstrator import judge
from cusr.demonstrator import problems as P
from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.co_backend import TorchLM
from cusr.demonstrator.pipeline import MemeticPipeline

PILOT_IDS = ["nguyen_1", "nguyen_5", "korns_12"]
# Fixed union op-set (SRBench style). MUST cover every operator in the catalogue's
# true_exprs — audit found log (nguyen_7, feynman_I_44_4) and tanh (feynman_II_35_21)
# are needed; without them those problems are silently unrecoverable.
FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0,
         "sin": 0.5, "cos": 0.5, "exp": 0.3, "log": 0.3, "sqrt": 0.3, "tanh": 0.2}
POP, N_GEN, TOP_K, CO_EVERY = 600, 60, 16, 2
N_SAMPLES, EARLY_R2 = 200, 0.9999


def stratified_subset():
    """A representative cut: all controls + all Korns + 3 Feynman per n_inner bucket."""
    cat = P.load_catalogue()
    nguyen = sorted(p.id for p in cat if p.source == "nguyen")
    korns = sorted(p.id for p in cat if p.source == "korns")
    by_inner = {}
    for p in cat:
        if p.source == "feynman":
            by_inner.setdefault(p.n_inner_consts, []).append(p.id)
    feyn = []
    for k in sorted(by_inner):
        feyn += sorted(by_inner[k])[:2]
    return nguyen + korns + feyn


def run_one(prob, backend, seed=0, co_probability=1.0, sample_cnt=10000, const_samples=None):
    from evogp.algorithm import (DefaultCrossover, DefaultMutation,
                                 TournamentSelection, GeneticProgramming)
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    X_np, y_np = sb.generate(prob, seed=seed, n_samples=N_SAMPLES)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Config aligned with EvoGP's official SR example (uci_sr.py / custom_sr.py) on the
    # axes that drive STRUCTURE search — rich const sampling [-5,5]x10000 (not 6 fixed
    # values) and TournamentSelection — but tree size right-sized to OUR hardest true_expr
    # (~30 nodes) + slack, NOT EvoGP's 512. EvoGP minimizes MSE on black-box UCI data with
    # no parsimony (enable_pareto_front=False), so 512 bloats by design; for symbolic
    # recovery bloat is the enemy, so 64/depth-6 caps it naturally.
    const_kw = (dict(const_samples=const_samples) if const_samples is not None
                else dict(const_range=[-5.0, 5.0], sample_cnt=sample_cnt))
    desc = GenerateDescriptor(
        max_tree_len=64, input_len=prob.n_vars, output_len=1,
        using_funcs=FUNCS, max_layer_cnt=6, layer_leaf_prob=0.3, **const_kw,
    )
    forest = Forest.random_generate(pop_size=POP, descriptor=desc)
    problem = SymbolicRegression(
        datapoints=torch.from_numpy(X_np.astype(np.float32)).cuda(),
        labels=torch.from_numpy(y_np.astype(np.float32).reshape(-1, 1)).cuda())
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.1, descriptor=desc.update(max_layer_cnt=4)),
        selection=TournamentSelection(tournament_size=20, survivor_rate=0.5, elite_rate=0.1))
    p = MemeticPipeline(algo, problem, backend, problem_n_vars=prob.n_vars,
                        X_np=X_np, y_np=y_np, top_k=TOP_K, co_every=CO_EVERY,
                        co_probability=co_probability, co_rng_seed=seed,
                        generation_limit=N_GEN)
    early = -(1.0 - EARLY_R2) * float(np.var(y_np))
    t0 = time.time()
    gens = 0
    for _ in range(N_GEN):
        p.step()
        gens += 1
        if p.best_fitness >= early:
            break
    wall = time.time() - t0

    cols = {"strict": False, "lenient": False, "srbench": False}
    best_expr = None
    if p.best_tree is not None:
        # >50-node trees are never a clean recovery of these tiny targets -> skip the
        # symbolic check (cheap pre-filter, symmetric across arms). For the rest, the
        # tree->sympy CONSTRUCTION runs here (bounded), but the EQUIVALENCE check
        # (sympy.simplify, which can hang in C code SIGALRM can't kill -- 009: one
        # tree spun 13h) runs in a hard-timeout subprocess. See judge.recovery_columns_timeout.
        try:
            n_nodes = int(p.best_tree.subtree_size[0].item())
        except Exception:  # noqa: BLE001
            n_nodes = 0
        if n_nodes > 50:
            best_expr = f"<{n_nodes} nodes (bloated) -> skip sympy, non-recovery>"
        else:
            try:
                best_expr = str(p.best_tree.to_sympy_expr())
            except Exception as e:  # noqa: BLE001
                best_expr = f"<to_sympy err: {e}>"
            if not best_expr.startswith("<"):
                cols = judge.recovery_columns_timeout(best_expr, prob.true_expr, X=X_np, timeout=20)
    return {
        "id": prob.id, "source": prob.source, "n_inner": prob.n_inner_consts,
        "strict": cols["strict"], "lenient": cols["lenient"], "srbench": cols["srbench"],
        "best_fit": float(p.best_fitness), "gens": gens, "co_fits": p.n_co_fits,
        "rollbacks": p.n_rollbacks, "evals": p.n_evals, "wall_s": round(wall, 1),
        "best_expr": best_expr, "true_expr": prob.true_expr,
    }


def main(ids):
    cat = {p.id: p for p in P.load_catalogue()}
    backend = TorchLM(dtype="float32")
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    fn = out / "subset_results.jsonl"
    sink = fn.open("w")
    rows = []
    print(f"backend={backend.name}  POP={POP} N_GEN={N_GEN} TOP_K={TOP_K} CO_EVERY={CO_EVERY}\n")
    print(f"{'id':14}{'n_in':>5}{'strict':>7}{'lenient':>8}{'srbench':>8}{'gens':>5}{'co_fits':>8}{'wall':>7}  expr")
    for pid in ids:
        if pid not in cat:
            print(f"  {pid}: NOT IN CATALOGUE"); continue
        r = run_one(cat[pid], backend)
        rows.append(r)
        sink.write(json.dumps(r) + "\n"); sink.flush()  # stream-write: partial survives a timeout
        ex = (r["best_expr"] or "")[:42]
        print(f"{r['id']:14}{r['n_inner']:>5}{str(r['strict']):>7}{str(r['lenient']):>8}"
              f"{str(r['srbench']):>8}{r['gens']:>5}{r['co_fits']:>8}{r['wall_s']:>6}s  {ex}")
    sink.close()

    # quick recovery summary by n_inner bucket (lenient column)
    print("\n=== recovery by n_inner (lenient / total) ===")
    buckets = {}
    for r in rows:
        b = buckets.setdefault(r["n_inner"], [0, 0])
        b[0] += int(r["lenient"]); b[1] += 1
    for k in sorted(buckets):
        rec, tot = buckets[k]
        print(f"  n_inner={k}: {rec}/{tot}")
    print(f"  TOTAL lenient: {sum(int(r['lenient']) for r in rows)}/{len(rows)}")
    print(f"\nwrote {fn} ({len(rows)} rows)")


if __name__ == "__main__":
    args = sys.argv[1:]
    ids = stratified_subset() if not args else args
    main(ids)
