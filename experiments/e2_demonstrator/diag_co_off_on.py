"""Diagnostic: is each failure stuck on STRUCTURE or on CONSTANTS?

For each problem, run the SAME config CO-off (co_every=inf) vs CO-on, same
seed, and compare final best fitness + recovery:
  - CO-on >> CO-off  -> constants were the bottleneck (CO/recipe is the lever)
  - CO-on ~= CO-off   -> structure was the bottleneck (need bigger GP, not CO)

probe_sin25 (2*sin(2.5*x0), freq not in const_samples) is a POSITIVE CONTROL:
CO is known to help there, so it validates the comparison is wired right.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent

from cusr.demonstrator import judge
from cusr.demonstrator import problems as P
from cusr.demonstrator import seed_bench as sb
from cusr.demonstrator.co_backend import TorchLM
from cusr.demonstrator.pipeline import MemeticPipeline

FUNCS = {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0, "sin": 0.5, "cos": 0.5, "exp": 0.3, "sqrt": 0.3}
POP, N_GEN, TOP_K = 600, 60, 16
SEED = 0


def run(prob, co_on):
    from evogp.algorithm import (DefaultCrossover, DefaultMutation,
                                 DefaultSelection, GeneticProgramming)
    from evogp.problem import SymbolicRegression
    from evogp.tree import Forest, GenerateDescriptor

    X_np, y_np = sb.generate(prob, seed=SEED, n_samples=200)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    desc = GenerateDescriptor(max_tree_len=32, input_len=prob.n_vars, output_len=1,
                             using_funcs=FUNCS, max_layer_cnt=4,
                             const_samples=[0.0, 1.0, -1.0, 2.0, -2.0, 0.5])
    forest = Forest.random_generate(pop_size=POP, descriptor=desc)
    problem = SymbolicRegression(
        datapoints=torch.from_numpy(X_np.astype(np.float32)).cuda(),
        labels=torch.from_numpy(y_np.astype(np.float32).reshape(-1, 1)).cuda())
    algo = GeneticProgramming(
        initial_forest=forest, crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.2, descriptor=desc.update(max_layer_cnt=3)),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01))
    p = MemeticPipeline(algo, problem, TorchLM(dtype="float32"), problem_n_vars=prob.n_vars,
                        X_np=X_np, y_np=y_np, top_k=TOP_K,
                        co_every=(2 if co_on else 10**9), generation_limit=N_GEN)
    for _ in range(N_GEN):
        p.step()
    rec = False
    try:
        rec = judge.recovered(p.best_tree.to_sympy_expr(), prob.true_expr, X=X_np)
    except Exception:
        pass
    return float(p.best_fitness), bool(rec)


def main():
    cat = {p.id: p for p in P.load_catalogue()}
    probe = sb.Problem(id="probe_sin25", source="probe", true_expr="2.0*sin(2.5*x0)",
                       n_vars=1, var_ranges=((0.0, 3.0),), n_consts=2, n_inner_consts=1,
                       difficulty="medium", is_control=False)
    targets = [probe]
    for pid in ["korns_7", "korns_12", "feynman_II_10_9", "feynman_II_11_3"]:
        if pid in cat:
            targets.append(cat[pid])

    out = HERE / "runs" / "co_off_on.jsonl"
    out.parent.mkdir(exist_ok=True)
    sink = out.open("w")
    print(f"{'problem':18}{'n_in':>5}{'fit_off':>12}{'fit_on':>12}{'Δ(on-off)':>12}{'rec_off':>8}{'rec_on':>8}  verdict")
    for prob in targets:
        t0 = time.time()
        foff, roff = run(prob, co_on=False)
        fon, ron = run(prob, co_on=True)
        delta = fon - foff
        verdict = "CONSTANTS" if (delta > 0.05 * (abs(foff) + 1e-9) or (ron and not roff)) else "structure"
        rec = {"id": prob.id, "n_inner": prob.n_inner_consts, "fit_off": foff, "fit_on": fon,
               "delta": delta, "rec_off": roff, "rec_on": ron, "verdict": verdict,
               "wall_s": round(time.time() - t0, 1)}
        sink.write(json.dumps(rec) + "\n"); sink.flush()
        print(f"{prob.id:18}{prob.n_inner_consts:>5}{foff:>12.3e}{fon:>12.3e}{delta:>12.3e}"
              f"{str(roff):>8}{str(ron):>8}  {verdict}")
    sink.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
