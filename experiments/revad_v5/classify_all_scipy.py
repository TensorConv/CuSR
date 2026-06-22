#!/usr/bin/env python3
"""EXTENDED scipy classifier — reproduces Codex review finding #1/#2.

compare_scipy.py only classified DISPUTED elements (fwd/rev differ in finiteness).
It NEVER examined the SHARED-NON-FINITE set (both fwd AND rev non-finite), which the
parity test auto-accepts as `both_nonfinite -> OK`. This script classifies EVERY
element against the scipy fp64 oracle, with special attention to the shared-non-finite
bucket: how many are TRUE singularities (scipy also non-finite -> NaN defensible) vs
how many have a FINITE true derivative (both AD modes wrong -> residual NaN contamination).

Reuses eval_tree + scipy_grad from compare_scipy.py (same dir).
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_scipy import eval_tree, scipy_grad, close

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "jac_sample.jsonl")
    trees = [json.loads(line) for line in open(path) if line.strip()]
    print(f"loaded {len(trees)} sample trees from {path}\n")

    n_total = 0
    n_both_finite = 0
    n_disputed = 0
    n_shared_nonfinite = 0
    # shared-non-finite breakdown vs scipy:
    sh_true_singular = 0      # scipy also non-finite -> genuinely undefined, NaN defensible
    sh_both_wrong_finite = 0  # value finite & scipy finite -> BOTH AD modes WRONG (residual contamination)
    sh_both_wrong_zero = 0    # ...of which scipy == 0 exactly (0*inf signature)
    sh_both_wrong_nonzero = 0 # ...of which scipy != 0 (worse: a real nonzero deriv lost)
    sh_valnan = 0             # tree value itself non-finite -> undefined, NaN defensible
    # overall reverse correctness vs scipy across ALL finite-truth elements:
    rev_ok = rev_bad = 0      # among elements where scipy is finite & value finite
    examples = []

    for T in trees:
        nt, nv, ci, c = T["nt"], T["nv"], T["ci"], np.array(T["c"], dtype=np.float64)
        K = T["K"]
        for P in T["pts"]:
            x = P["x"]
            fwd = np.array(P["fwd"], dtype=np.float64)
            rev = np.array(P["rev"], dtype=np.float64)
            val = float(eval_tree(nt, nv, ci, x, c))
            ref = scipy_grad(nt, nv, ci, x, c)
            val_finite = np.isfinite(val)
            for k in range(K):
                f, r, g = fwd[k], rev[k], ref[k]
                n_total += 1
                ff, rf = np.isfinite(f), np.isfinite(r)
                # overall reverse-vs-truth accounting (only where truth is finite & value finite)
                if val_finite and np.isfinite(g):
                    if close(r, g): rev_ok += 1
                    else:           rev_bad += 1
                if ff and rf:
                    n_both_finite += 1
                elif ff != rf:
                    n_disputed += 1
                else:  # both non-finite -- the UNEXAMINED bucket
                    n_shared_nonfinite += 1
                    if not val_finite:
                        sh_valnan += 1
                    elif not np.isfinite(g):
                        sh_true_singular += 1
                    else:
                        sh_both_wrong_finite += 1
                        if g == 0.0: sh_both_wrong_zero += 1
                        else:        sh_both_wrong_nonzero += 1
                        if len(examples) < 20:
                            examples.append((T["m"], P["i"], k, val, f, r, g))

    print("=== ALL elements, by finiteness category ===")
    print(f"  total                = {n_total}")
    print(f"  both finite          = {n_both_finite}")
    print(f"  disputed (1 finite)  = {n_disputed}")
    print(f"  shared non-finite    = {n_shared_nonfinite}   <-- auto-accepted by parity, unexamined by compare_scipy\n")

    print("=== SHARED-NON-FINITE bucket vs scipy fp64 oracle ===")
    print(f"  tree value non-finite (undefined; NaN defensible)        = {sh_valnan}")
    print(f"  scipy also non-finite (true singularity; NaN defensible) = {sh_true_singular}")
    print(f"  value finite & scipy FINITE -> BOTH AD MODES WRONG       = {sh_both_wrong_finite}")
    print(f"        of which scipy == 0 exactly (0*inf signature)      = {sh_both_wrong_zero}")
    print(f"        of which scipy != 0 (a real nonzero deriv lost)    = {sh_both_wrong_nonzero}\n")

    print("=== overall reverse-AD vs scipy, across elements with finite value & finite truth ===")
    print(f"  rev matches scipy = {rev_ok}")
    print(f"  rev WRONG vs scipy = {rev_bad}   (these are the residual reverse NaN-contamination)\n")

    print("=== sample BOTH-WRONG-FINITE elements  [m,i,k] value fwd rev scipy ===")
    for (m, i, k, val, f, r, g) in examples:
        print(f"  m={m:<5} i={i:<4} k={k}  val={val:<12.5g} fwd={f:<8.4g} rev={r:<8.4g} scipy={g:<12.5g}")

    print()
    if sh_both_wrong_finite > 0:
        print(f"FINDING CONFIRMED: {sh_both_wrong_finite} shared-NaN elements have a FINITE true derivative "
              f"({sh_both_wrong_zero} are exactly 0). Reverse-AD still NaN-contaminates here "
              f"(0*inf in the backward pass). Parity test + compare_scipy.py never checked these.")
    else:
        print("No shared-NaN elements with finite true derivative — Codex finding NOT reproduced.")

if __name__ == "__main__":
    main()
